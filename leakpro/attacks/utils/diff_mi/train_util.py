import copy
import functools
import io
import os

import blobfile as bf
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from . import dist_util#, logger
from .dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP, model_wrapper
from .fp16_util import MixedPrecisionTrainer
from .losses import p_reg_loss, topk_loss
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

from leakpro.utils.logger import logger

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class PreTrain:
    def __init__(
        self,
        model,
        diffusion,
        data,
        args,
        save_path=None,
        schedule_sampler=None,
    ):
        """Pre-train the diffusion model.

        Args:
            model: The diffusion model to be pre-trained.
            diffusion: The diffusion process.
            data: The training data.
            args: The training arguments.
            save_path: The path to save checkpoints.
            schedule_sampler: The schedule sampler for training.

        Returns:
            None

        """
        dist_util.setup_dist()
        self.model = model.to(dist_util.dev())
        self.diffusion = diffusion

        # Set up data generator
        data.dataset.return_cond = True

        self.data = generator_from_dataloader(data)
        self.batch_size = args.batch_size
        self.microbatch = args.microbatch if args.microbatch > 0 else self.batch_size
        self.lr = args.lr
        self.ema_rate = (
            [args.ema_rate]
            if isinstance(args.ema_rate, float)
            else [float(x) for x in args.ema_rate.split(",")]
        )
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.save_path = "./" if save_path is None else save_path
        self.save_name = args.save_name if args.save_name else "pretrain"

        # logger.info(f"Saving checkpoints to {self.save_path}")
        logger.info(f"Saving checkpoints to {self.save_path}")

        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = args.use_fp16
        self.fp16_scale_growth = args.fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.max_steps = args.max_steps if args.max_steps else None
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()


        self.sync_cuda = False # th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:

            print("RESUME CHECKPOINT: ", resume_checkpoint)

            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.info(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.info(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.info(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            # If max steps is set, stop training after that many steps.
            if self.max_steps and (self.step > self.max_steps):
                logger.info(f"Max steps {self.max_steps} reached, saving model and stopping training.")
                break
            batch, cond = next(self.data)
            if np.random.rand() < 0.1:
                cond["y"] = th.ones_like(cond["y"]) * (self.model.num_classes - 1)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                # logger.dumpkvs()
                self._dump_loggs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        # self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            self.loss_loggs = log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def _dump_loggs(self):
        logger.info("***** Training statistics *****")
        logger.info(f"Step: {self.step + self.resume_step}")
        logger.info(f"Samples: {(self.step + self.resume_step + 1) * self.global_batch}")
    
        for key, value in self.loss_loggs.items():
            logger.info(f"loss: {key}: {value}")

        for key, value in self.mp_trainer.loggs.items():
            logger.info(f"mp: {key}: {value}")

    # def log_step(self):
        # logger.infokv("step", self.step + self.resume_step)
        # logger.infokv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        # self.loggs["step"] = self.step + self.resume_step
        # self.loggs["samples"] = (self.step + self.resume_step + 1) * self.global_batch

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.info(f"saving model {rate}...")
                if not rate:
                    filename = f"{self.save_name}.pt"
                else:
                    filename = f"ema_{rate}_{self.save_name}.pt"
                print(self.save_path, filename)
                with bf.BlobFile(bf.join(self.save_path, filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(self.save_path, f"opt_{self.save_name}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

class FineTune:

    def __init__(
        self,
        # *,
        args,
        target,
        model,
        diffusion,
        p_reg,
        save_path=None,
        schedule_sampler=None,
    ):
        """Fine-tune the diffusion model with classifier guidance.

        Args:
        ----
            self: Self
            args: The arguments for fine-tuning.
            target (Module): The target model to guide the diffusion model.
            model (Module): The diffusion model to be fine-tuned.
            diffusion: The diffusion process.
            p_reg (Tensor): The regularization tensor.
            save_path: The path to save checkpoints.
            schedule_sampler: The schedule sampler for training.

        Returns:
            None

        """

        dist_util.setup_dist()
        self.threshold = args.threshold
        self.epochs = args.epochs
        self.target = target.to(dist_util.dev())
        self.model = model.to(dist_util.dev())
        self.diffusion = diffusion
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.resume_checkpoint = args.resume_checkpoint
        # self.save_path = get_blob_logdir() if save_path is None else save_path
        self.save_path = "./" if save_path is None else save_path
        print(self.save_path, get_blob_logdir())
        logger.info(f"Saving checkpoints to {self.save_path}")

        self.use_fp16 = args.use_fp16
        self.fp16_scale_growth = args.fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = args.weight_decay

        self.global_batch = self.batch_size * dist.get_world_size()
        self.sync_cuda = th.cuda.is_available()
        self._load_and_sync_parameters()

        self.mp_trainer_cls = MixedPrecisionTrainer(
            model=self.model,
            layers=[["label_emb.weight",
                     "middle_block.0.in_layers.2",
                     "middle_block.0.out_layers.3",
                     "middle_block.2.in_layers.2",
                     "middle_block.2.out_layers.3",
                     "middle_block.1.proj"], []],
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )
        self.opt_cls = AdamW(
            self.mp_trainer_cls.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        # Configuration for dpm_solver
        b0, bT, timesteps = 1e-4, 2e-2, 1000
        betas = th.tensor(np.linspace(b0, bT, timesteps), dtype=float)
        self.noise_schedule = NoiseScheduleVP(schedule="discrete", betas=betas)

        # Load for p_reg loss
        self.p_reg = p_reg

        self.aug = kornia.augmentation.container.ImageSequential(
            kornia.augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            kornia.augmentation.RandomHorizontalFlip(),
            kornia.augmentation.RandomRotation(5),
        )

    def _load_and_sync_parameters(self):
        if resume_checkpoint := find_resume_checkpoint() or self.resume_checkpoint:
            if dist.get_rank() == 0:
                logger.info(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                    # , strict=False
                )
        dist_util.sync_params(self.model.parameters())

    def run(self, guidance_scale=3.0, aug_times=2):

        acc, acc_mean, best_mean_acc, iter = [], 0.0, 0.0, 0
        labels = th.tensor(np.arange(0, 300)).to(dist_util.dev())
        label_dataset = TensorDataset(labels)

        while (acc_mean < self.threshold):

            # ME
            loss_agg = 0.0
            loss1_agg = 0.0
            loss2_agg = 0.0

            bar = tqdm(DataLoader(dataset=label_dataset, batch_size=self.batch_size, shuffle=True))
            for classes in bar:
                bs = classes[0].shape[0]
                bar.set_description(f"Epoch {iter}")
                self.mp_trainer_cls.zero_grad()
                model_fn = model_wrapper(
                    self.model,
                    self.noise_schedule,
                    model_type="noise",
                    guidance_type="classifier-free",
                    condition=classes[0],
                    unconditional_condition=th.ones_like(classes[0])*1000,
                    guidance_scale=guidance_scale)
                dpm_solver = DPM_Solver(model_fn, self.noise_schedule, algorithm_type="dpmsolver++")

                x_T = th.randn((bs, 3, 64, 64)).to(dist_util.dev())
                x0_pred_list = dpm_solver.sample(
                    x_T,
                    steps=10,
                    order=2,
                    method="multistep",
                    skip_type="time_uniform",
                    return_pred_x0=True)
                samples = th.cat(x0_pred_list)

                img_input_batch = []
                for _ in range(aug_times):
                    img_input = samples
                    img_input = self.aug(img_input).clamp(-1,1)
                    img_input_batch.append(img_input)
                img_input_batch = th.cat(img_input_batch)

                feats, logits = self.target((img_input_batch + 1) / 2)
                loss1 = topk_loss(logits, classes[0].repeat(len(x0_pred_list)*aug_times), k=20)
                loss2 = 1.0 * p_reg_loss(feats, classes[0].repeat(len(x0_pred_list)*aug_times), self.p_reg)
                loss = loss1 + loss2

                self.mp_trainer_cls.backward(loss)
                self.mp_trainer_cls.optimize(self.opt_cls)

                acc.append(th.eq(th.topk(logits[-bs * aug_times:], k=1)[1], classes[0].repeat(aug_times).view(-1,1)).float().mean().item())
                bar.set_postfix({"Loss1": loss1.item(), "Loss2": loss2.item(), "Loss": loss.item()})

                loss_agg += loss.item()
                loss1_agg += loss1.item()
                loss2_agg += loss2.item()
            logger.info(f"Epoch {iter}: loss {loss_agg/len(bar):.4f}, loss1 {loss1_agg/len(bar):.4f}, loss2 {loss2_agg/len(bar):.4f}")

            # Save the fine-tuned model
            with th.no_grad():
                acc_mean = np.mean(acc)
                logger.info(f"The mean acc in iteration {iter} is {acc_mean:.2%}")

                if acc_mean > self.best_mean_acc:
                    self.best_mean_acc = acc_mean
                    self.state_dict = self.mp_trainer_cls.master_params_to_state_dict(self.mp_trainer_cls.model_params)
                    self.save()
                    acc, iter = [], (iter + 1)
                if iter >= self.epochs:
                    logger.info(f"Training completed with best mean acc: {self.best_mean_acc:.2%}")
                    break

    def save(self):
        filename = f"{self.save_name}.pt"
        with bf.BlobFile(bf.join(self.save_path, filename), "wb") as f:
            th.save(self.state_dict, f)
            logger.info(f"New best mean acc: {self.best_mean_acc:.2%}. Saving model to {f.name}")


def generator_from_dataloader(dataloader: DataLoader):
    """Create a generator that yields batches from a DataLoader indefinitely.

    Args:
    ----
        dataloader (DataLoader): The DataLoader to yield batches from.

    Returns:
        generator: A generator that yields batches from the DataLoader.

    """
    while True:
        for batch in dataloader:
            yield batch

def parse_resume_step_from_filename(filename):
    """Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    loggs = {}
    for key, values in losses.items():
        # logger.infokv_mean(key, values.mean().item())
        loggs[key] = values.mean().item()
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            # logger.infokv_mean(f"{key}_q{quartile}", sub_loss)
            loggs[f"{key}_q{quartile}"] = sub_loss
    return loggs

def visualize(img, label):
    sample = ((img + 1) * 127.5).clamp(0, 255).to(th.uint8)
    arr = np.array([i.permute(1, 2, 0).cpu().numpy() for i in sample])
    N = min(int(np.sqrt(len(arr))), 4)

    fig, axes = plt.subplots(N, N, figsize=(4, 4))
    plt.subplots_adjust(wspace=0, hspace=0.5)

    for i in range(N * N):
        plt.subplot(N, N, i + 1)
        plt.imshow(arr[i])
        plt.title(label[i].item())
        plt.axis("off")

    # Save the Matplotlib figure to a BytesIO object
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()

    # Convert the BytesIO object to a PIL Image
    img_pil = Image.open(img_bytes)

    return img_pil

def show_images_side_by_side(image1, image2, title1="Reconstructed Images (Diff-MI)", title2="Private Images (Ground-Truth)"):
    """Display two PIL images side by side in a Jupyter Notebook.

    Parameters
    ----------
    - image1: PIL image object for the first image
    - image2: PIL image object for the second image
    - title1: Title for the first image (default is 'Image 1')
    - title2: Title for the second image (default is 'Image 2')

    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image1)
    axes[0].set_title(title1)

    axes[1].imshow(image2)
    axes[1].set_title(title2)

    for ax in axes:
        ax.axis("off")

    plt.show()
