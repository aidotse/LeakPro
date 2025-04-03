from torch import cuda, device, optim, no_grad, save, set_default_device, backends
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from leakpro import AbstractInputHandler
from leakpro.schemas import TrainingOutput,EvalOutput
import kornia
import time


class CelebA_InputHandler(AbstractInputHandler):
    """Class to handle the user input for the CelebA dataset for plgmi attack."""
        
    def train(
        self,
        dataloader: DataLoader,
        model: Module = None,
        criterion: Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> TrainingOutput:
        """Model training procedure."""

        if epochs is None:
            raise ValueError("epochs not found in configs")

        # prepare training
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)

        accuracy_history = []
        loss_history = []
        
        # training loop
        for epoch in range(epochs):
            train_loss, train_acc, total_samples = 0, 0, 0
            model.train()
            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                labels = labels.long()
                inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), labels.to(gpu_or_cpu, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                pred = outputs.argmax(dim=1) 
                loss.backward()
                optimizer.step()

                # Accumulate performance of shadow model
                train_acc += pred.eq(labels.view_as(pred)).sum().item()
                total_samples += labels.size(0)
                train_loss += loss.item() * labels.size(0)
                
            avg_train_loss = train_loss / total_samples
            train_accuracy = train_acc / total_samples 
            
            accuracy_history.append(train_accuracy) 
            loss_history.append(avg_train_loss)
        
        model.to("cpu")

        results = EvalOutput(accuracy = train_accuracy,
                             loss = avg_train_loss,
                             extra = {"accuracy_history": accuracy_history, "loss_history": loss_history})
        return TrainingOutput(model = model, metrics=results)
    
    def eval(self, loader, model, criterion):
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)
        model.eval()
        loss, acc = 0, 0
        total_samples = 0
        with no_grad():
            for data, target in loader:
                data, target = data.to(gpu_or_cpu), target.to(gpu_or_cpu)
                target = target.view(-1) 
                output = model(data)
                loss += criterion(output, target).item() * target.size(0)
                pred = output.argmax(dim=1) 
                acc += pred.eq(target).sum().item()
                total_samples += target.size(0)
            loss /= total_samples
            acc = float(acc) / total_samples
            
        output_dict = {"accuracy": acc, "loss": loss}
        return EvalOutput(**output_dict)


    def train_gan(self,
                    pseudo_loader: DataLoader,
                    gen: Module,
                    dis: Module,
                    gen_criterion: callable,
                    dis_criterion: callable,
                    inv_criterion: callable,
                    target_model: Module,
                    opt_gen: optim.Optimizer,
                    opt_dis: optim.Optimizer,
                    n_iter: int,
                    n_dis: int,
                    device: device,
                    alpha: float,
                    log_interval: int,
                    sample_from_generator: callable
                  ) -> None:
        """Train the GAN model. Inspired by cGAN from https://github.com/LetheSec/PLG-MI-Attack.
        
            Args:
                pseudo_loader: DataLoader for the pseudo data.
                gen: Generator model.
                dis: Discriminator model.
                gen_criterion: Generator criterion.
                dis_criterion: Discriminator criterion.
                inv_criterion: Inverted criterion.
                target_model: Target model.
                opt_gen: Generator optimizer.
                opt_dis: Discriminator optimizer.
                n_iter: Number of iterations.
                n_dis: Number of discriminator updates per generator update.
                device: Device to run the training.
                alpha: Alpha value for the invariance loss.
                log_interval: Log interval.
                sample_from_generator: Function to sample from the generator.
        """
        set_default_device(device)
        backends.cudnn.benchmark = True
        gen_losses = []
        dis_losses = []
        inv_losses = []
        
        # Augmentations for generated images. TODO: Move this to a image modality extension and have it as an input
        aug_list = kornia.augmentation.container.ImageSequential(
            kornia.augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            kornia.augmentation.RandomHorizontalFlip(),
            kornia.augmentation.RandomRotation(5),
        ).to(device)

        target_model.to(device)
        gen.to(device)
        dis.to(device)
        # Training loop
        for i in range(n_iter):
            _l_g = .0
            cumulative_inv_loss = 0.
            cumulative_loss_dis = .0

            cumulative_target_acc = .0
            target_correct = 0
            count = 0
            
            # Number of discriminator updates per generator update
            for j in range(n_dis):
                if j == 0:
                    # Generator update
                    fake, fake_labels, _ = sample_from_generator()
                    fake_aug = aug_list(fake).to(device)
                    dis_fake = dis(fake_aug, fake_labels)
                    inv_loss = inv_criterion(target_model(fake_aug), fake_labels)

                    inv_losses.append(inv_loss.item())
                    dis_real = None

                    loss_gen = gen_criterion(dis_fake, dis_real)
                    gen_losses.append(loss_gen.item())
                    loss_all = loss_gen + inv_loss*alpha

                    gen.zero_grad()
                    loss_all.backward()
                    opt_gen.step()
                    _l_g += loss_gen.item()
                    cumulative_inv_loss += inv_loss.item()
                
                # Discriminator update
                fake, fake_labels, _ = sample_from_generator()

                real, real_labels = next(iter(pseudo_loader))
                real, real_labels = real.to(device), real_labels.to(device)

                dis_fake = dis(fake, fake_labels)
                dis_real = dis(real, real_labels)

                loss_dis = dis_criterion(dis_fake, dis_real)
            
                dis.zero_grad()
            
                loss_dis.backward()
                opt_dis.step()

                cumulative_loss_dis += loss_dis.item()
                dis_losses.append(cumulative_loss_dis/n_dis)
                
                # Evaluate target model accuracy for training progress monitoring TODO: make optional for efficiency
                with no_grad():
                    count += fake.shape[0]
                    T_logits = target_model(fake)
                    T_preds = T_logits.max(1, keepdim=True)[1]
                    target_correct += T_preds.eq(fake_labels.view_as(T_preds)).sum().item()
                    cumulative_target_acc += round(target_correct / count, 4)

            if i % log_interval == 0:
                print(
                        'iteration: {:05d}/{:05d}, loss gen: {:05f}, loss dis {:05f}, inv loss {:05f}, target acc {:04f}, time {}'.format(
                            i, n_iter, _l_g, cumulative_loss_dis / n_dis, cumulative_inv_loss,
                            cumulative_target_acc / n_dis, time.strftime("%H:%M:%S")))

        save(gen.state_dict(), './gen.pth')
        save(dis.state_dict(), './dis.pth')

    class UserDataset(AbstractInputHandler.UserDataset):
        def __init__(self, x, y, transform=None,  indices=None):
            """
            Dataset for celebA.

            Args:
                x (torch.Tensor): Tensor of input images.
                y (torch.Tensor): Tensor of labels.
                transform (callable, optional): Optional transform to be applied on the image tensors.
            """
            self.x = x
            self.y = y
            self.transform = transform
            self.indices = indices

        def __len__(self):
            """Return the total number of samples."""
            return len(self.y)

        def __getitem__(self, idx):
            """Retrieve the image and its corresponding label at index 'idx'."""
            image = self.x[idx]
            label = self.y[idx]

            # Apply transformations to the image if any
            if self.transform:
                image = self.transform(image)

            return image, label
        
        def get_classes(self):
            return len(self.y.unique())
        
        

        @classmethod
        def from_celebA(cls, config, subfolder):
            re_size = 64
            crop_size = 108
            offset_height = (218 - crop_size) // 2
            offset_width = (178 - crop_size) // 2
            crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]    
            
            data_dir = config["data"]["data_dir"]
            train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(crop),
            transforms.ToPILImage(),
            transforms.Resize((re_size, re_size)),
            transforms.ToTensor(),
            ])

            train_dataset = datasets.ImageFolder(os.path.join(data_dir, subfolder), train_transform)
        
            train_dataset.class_to_idx = {cls_name: int(cls_name) for cls_name in train_dataset.class_to_idx.keys()}

            # Prepare data loader to iterate over combined_dataset
            loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

            # Collect all data and targets
            data_list = []
            target_list = []
            for data, target in loader:
                data_list.append(data)  # Remove batch dimension
                target_list.append(target)

            # Concatenate data and targets into large tensors
            data = cat(data_list, dim=0)  # Shape: (N, C, H, W)
            targets = cat(target_list, dim=0)  # Shape: (N,)


            return cls(data, targets)


        def subset(self, indices):
            """Return a subset of the dataset based on the given indices."""
            return celebADataset(self.x[indices], self.y[indices], transform=self.transform)