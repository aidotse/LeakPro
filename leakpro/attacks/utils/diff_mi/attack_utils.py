"""Utility functions for the Diff-MI attack."""

import copy
import math
import os
import statistics

import kornia.augmentation as K
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as augmentation
from robustness import model_utils
from torch.utils.data import TensorDataset
from tqdm import tqdm

from leakpro.utils.logger import logger

from .diffusion import GaussianDiffusion, InferenceModel
from .losses import p_reg_loss, topk_loss


def Iterative_Image_Reconstruction(args, diff_net, classifier, classes, p_reg, iter=None, batch_num=None, device="cuda"):
    """Perform iterative image reconstruction using diffusion model and classifier guidance.

    Args:
    ----
        args: Argument parser containing hyperparameters.
        diff_net: Pre-trained diffusion model.
        classifier: Pre-trained classifier for guidance.
        classes: Target class labels for reconstruction.
        p_reg: Per-class regularization features.
        iter: Current iteration number (for logging).
        batch_num: Total number of batches (for logging).
        device: Device to run the computations on.

    Returns:
    -------
        Reconstructed: Images.

    """

    diffusion = GaussianDiffusion(T=1000, schedule="linear")
    model = InferenceModel(batch_size=classes.shape[0]).to(device=device)
    model.train()

    # Inference procedure steps
    steps = args.steps
    opt = torch.optim.Adamax(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=1.0, total_iters=steps)

    norm_track = 0

    aug = K.container.ImageSequential(
        K.RandomHorizontalFlip(),
        K.ColorJitter(brightness=0.2, p=0.5),
        K.RandomGaussianBlur((7, 7), (3, 3), p=0.5),
    )
    # print(f"--------------------- Epoch {iter}/{batch_num} ---------------------")
    bar = range(steps) #tqdm(range(steps))
    for i, _ in enumerate(bar):

        # bar.set_description(f'Epoch {iter}/{batch_num}')

        # Select t
        t = ((steps-i)/1.5 + (steps-i)/3*math.cos(3/(10*math.pi)*i))/steps*800 + 200 # Linearly decreasing + cosine
        t = np.array([t + np.random.randint(-50, 51) for _ in range(1)]).astype(int) # Add noise to t
        t = np.clip(t, 1, diffusion.T)

        # Denoise
        sample_img = model.encode()
        xt, epsilon = diffusion.sample(sample_img, t)
        t = torch.from_numpy(t).float().view(1)
        eps = diff_net(xt.float().to(device=device), t.to(device=device), classes.to(device=device))
        nonEps = diff_net(xt.float().to(device=device), t.to(device=device), torch.ones_like(classes, device=device) * (diff_net.num_classes - 1))
        epsilon_pred = args.w * eps - (args.w - 1) * nonEps

        # Compute diffusion loss: ||epsilon - epsilon_theta||^2
        loss = 1 * F.mse_loss(epsilon_pred, epsilon)

        opt.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad_norm = torch.linalg.norm(model.img.grad)
            if i > 0:
                alpha = 0.5
                norm_track = alpha*norm_track + (1-alpha)*grad_norm
            else:
                norm_track = grad_norm
        opt.step()

        attr_input_batch = []
        for _ in range(args.aug_times):
            attr_input = model.encode()
            # attr_input = args.aug(attr_input).clamp(-1,1)
            attr_input = aug(attr_input).clamp(-1,1)
            attr_input_batch.append(attr_input)

        attr_input_batch = torch.cat(attr_input_batch, dim=0).to(device=device)
        feats, logits = classifier.forward((attr_input_batch+1)/2)

        # topk loss + p_reg loss
        loss = topk_loss(logits, classes.repeat(args.aug_times), k=args.k) + \
               args.alpha * p_reg_loss(feats, classes.repeat(args.aug_times), p_reg)

        opt.zero_grad()
        loss.backward()

        # Clip attribute loss gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.25 * norm_track)
        opt.step()
        scheduler.step()

    with torch.no_grad():
        if args.ddim_step == 1000:
            t = np.array([args.ddim_step]).astype(int)
            xt = model.encode()
        else:
            t = np.array([args.ddim_step]).astype(int)
            xt, _ = diffusion.sample(model.encode(), t)
        fine_tuned = diffusion.inverse_ddim(diff_net, x=xt, start_t=t[0], w=args.w, y=classes, device=device)

    return (fine_tuned + 1) / 2

def get_PGD(model):
    """Set the model to use PGD adversarial training.

    Args:
    ----
        model: Pre-trained classifier model.

    Returns:
    -------
        PGD_model: Model set up for PGD adversarial training.

    """
    logger.info("Making PGD version of model for the Diff-MI attack.")
    class mean_and_std():
        def __init__(self):
            self.mean = torch.tensor([0.0, 0.0, 0.0])
            self.std = torch.tensor([1.0, 1.0, 1.0])

    # Create a copy of the model to avoid modifying the original
    model_copy = copy.deepcopy(model)

    # Set to False to return only the output logits
    model_copy.return_feature = False

    PGD_model, _ = model_utils.make_and_restore_model(arch=model_copy, dataset=mean_and_std())
    PGD_model.eval()

    return PGD_model

def calc_acc(classifier, imgs, labels, bs=64, anno="", with_success=False, enable_print=True):
    """Calculate top-1 and top-5 accuracy of the classifier on the given images and labels.

    Args:
    ----
        classifier: Pre-trained classifier model.
        imgs: Input images (Tensor).
        labels: True labels (Tensor).
        bs: Batch size for processing images.
        anno: Annotation string for logging.
        with_success: Whether to return indices of successful predictions.
        enable_print: Whether to print the accuracy results.

    Returns:
    -------
        top1_count: Number of correct top-1 predictions.
        top5_count: Number of correct top-5 predictions.
        success_idx (optional): Indices of successful predictions if with_success is True.

    """

    output, img_dataset = [], TensorDataset(imgs)
    for x in torch.utils.data.DataLoader(img_dataset, batch_size=bs, shuffle=False):
        output.append(classifier(x[0])[-1])
    output = torch.cat(output)
    top1_count = torch.eq(torch.topk(output, k=1)[1], labels.view(-1,1)).float()
    top5_count = torch.eq(torch.topk(output, k=5)[1], labels.view(-1,1)).float()
    if enable_print:
        print(f"===> top1_acc: {top1_count.mean().item():.2%}, top5_acc: {5*top5_count.mean().item():.2%} {anno}")
    if with_success:
        success_idx = torch.nonzero((output.max(1)[1] == labels).int()).squeeze(1)
        return top1_count.sum().item(), top5_count.sum().item(), success_idx
    return top1_count.sum().item(), top5_count.sum().item()


def calc_acc_std(imgs, labels, cls, label_num, dims=(64,64)):
    """Calculate top-1 and top-5 accuracy and their standard deviations over groups of images.

    Args:
    ----
        imgs: Input images (Tensor).
        labels: True labels (Tensor).
        cls: Pre-trained classifier model.
        label_num: Number of labels per group.
        dims: Dimensions to resize images for the classifier.

    Returns:
    -------
        acc1: Mean top-1 accuracy.
        acc5: Mean top-5 accuracy.
        var1: Standard deviation of top-1 accuracy.
        var5: Standard deviation of top-5 accuracy.

    """

    top1_list, top5_list = [], []
    assert imgs.shape[0] % label_num == 0
    for i in range(int(imgs.shape[0]/label_num)):
        imgs_ = imgs[i * label_num: (i+1) * label_num]
        labels_ = labels[i * label_num: (i+1) * label_num]
        assert torch.max(labels_) - torch.min(labels_) == label_num - 1
        top1_count, top5_count = calc_acc(cls, augmentation.Resize((dims[0], dims[1]))(imgs_),
                                          labels_, enable_print=False)
        top1_list.append(top1_count/label_num)
        top5_list.append(top5_count/label_num)
    try:
        acc1 = statistics.mean(top1_list)
        acc5 = statistics.mean(top5_list)
    except Exception:
        acc1, acc5 = top1_list[0], top5_list[0]

    try:
        var1 = statistics.stdev(top1_list)
        var5 = statistics.stdev(top5_list)
    except Exception:
        var1, var5 = 0.0, 0.0

    return acc1, acc5, var1, var5

def save_tensor_to_image(imgs, lables, save_path):
    """Save a batch of images as individual image files organized by labels.

    Args:
    ----
        imgs: Input images (Tensor).
        lables: Corresponding labels (Tensor).
        save_path: Directory to save the images.

    Returns:
    -------
        label_paths: List of paths to the saved image files.

    """
    label_paths = []
    fake_dataset = TensorDataset(imgs.cpu(), lables.cpu())
    for i, (x,y) in enumerate(torch.utils.data.DataLoader(fake_dataset, batch_size=1, shuffle=False)):
        label_path = os.path.join(save_path, str(y.item()))
        if not os.path.exists(label_path): os.makedirs(label_path)
        label_paths.append(label_path)
        torchvision.utils.save_image(x.detach()[0,:,:,:], os.path.join(label_path, f"{i}_attack.png"), padding=0)
    return label_paths

# def calc_lpips(fake_images, fake_targets, anno='', device='cuda'):

#     import lpips
#     loss_fn_alex = lpips.LPIPS(net='alex').to(device) # best forward scores
#     loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) # closer to "traditional" perceptual loss, when used for optimization

#     # get target images and labels
#     inferred_image_path = PRIVATE_PATH
#     list_of_idx = os.listdir(inferred_image_path)
#     images_list, targets_list = [], []
#     # load reconstructed images
#     for idx in list_of_idx:
#         for filename in os.listdir(os.path.join(inferred_image_path, idx)):
#             image = Image.open(os.path.join(inferred_image_path, idx, filename))
#             image = T.functional.to_tensor(image)
#             images_list.append(image)
#             targets_list.append(int(idx))
#     real_images = torch.stack(images_list, dim=0).to(device)
#     real_targets = torch.LongTensor(targets_list).to(device)

#     # get fake images and labels
#     fake_images = fake_images.to(device)
#     fake_targets = fake_targets.to(device)
#     bs = fake_targets.size(0)
#     value_a, value_v = 0, 0

#     # calculate metric
#     for i in range(bs):
#         single_value_a, single_value_v = -1, -1
#         idx = torch.nonzero(real_targets == fake_targets[i]).squeeze(1)
#         for j in idx:
#             temp_value_a = loss_fn_alex(fake_images[i].unsqueeze(0), real_images[j].unsqueeze(0))
#             temp_value_v = loss_fn_vgg(fake_images[i].unsqueeze(0), real_images[j].unsqueeze(0))
#             if temp_value_a > single_value_a: single_value_a = temp_value_a
#             if temp_value_v > single_value_v: single_value_v = temp_value_v
#         value_a += single_value_a
#         value_v += single_value_v
#     value_a = (value_a / bs).item()
#     value_v = (value_v / bs).item()
#     print(f"LPIPS : Alex {value_a:.4f} | VGG {value_v:.4f} {anno}")

def calc_knn(fake_imgs, fake_targets,
             private_feats,
             private_idents,
             evaluation_model,
             batch_size=64,
             anno="", device="cuda", dims=(64,64)):
    """Calculate KNN distance between reconstructed images and target images in feature space.

    Args:
    ----
        fake_imgs: Reconstructed images (Tensor).
        fake_targets: Corresponding target labels (Tensor).
        private_feats: Fetures from private dataset (np.ndarray).
        private_idents: Labels corresponding to the private features (np.ndarray).
        evaluation_model: Pre-trained feature extractor model.
        anno: Annotation string for logging.
        path: Path to the directory containing target image features.
        device: Device to run the computations on.
        dims: Dimensions to resize images for the feature extractor.

    Returns:
    -------
        knn: Average Minimum KNN distance value.
        knn_arr: Array of Minimum KNN distances for each reconstructed image.

    """

    # get features of reconstructed images
    infered_feats = None
    for i, images in enumerate(torch.utils.data.DataLoader(fake_imgs, batch_size=batch_size)):
        images = augmentation.Resize(dims)(images).to(device)
        feats = evaluation_model(images)[0]
        if i == 0:
            infered_feats = feats.detach().cpu()
        else:
            infered_feats = torch.cat([infered_feats, feats.detach().cpu()], dim=0)

    # get features of target images
    idens = fake_targets.to(device).long()
    feats = infered_feats.to(device)
    private_feats = torch.from_numpy(private_feats).float().to(device)
    private_idents = torch.from_numpy(private_idents).view(-1).long().to(device)
    bs = feats.size(0)
    knn_dist = 0

    # calculate knn dist
    knn_arr = np.zeros((bs,))
    for i in tqdm(range(bs), desc="Calculating KNN Dist"):
        knn = 1e8
        idx = torch.nonzero(private_idents == idens[i]).squeeze(1)
        fake_feat = feats[i].repeat(idx.shape[0], 1)
        true_feat = private_feats[idx]
        knn = torch.sum(torch.pow(fake_feat - true_feat, 2), dim=1)
        knn_min = torch.min(knn)
        knn_arr[i] = knn_min
        knn_dist += knn_min

    knn = (knn_dist / bs).item()
    print(f"KNN Dist computed on {fake_imgs.shape[0]} attack samples: {knn:.2f} {anno}")

    return knn, knn_arr


# def calc_pytorch_fid(file_path, file_path2=PRIVATE_PATH, anno=""):
#     file_path2 = PRIVATE_PATH+f"/{file_path.split('/')[-1]}"
#     fid_value = calculate_fid_given_paths(paths=[file_path, file_path2], batch_size=1, device='cuda', dims=2048, num_workers=8,)
#     print(f"[pytorch_fid] FID score computed on file {file_path.split('Inversion/')[-1]} is {fid_value:.2f} {anno}")

# from pytorch_fid.inception import InceptionV3
# def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1):
#     """Calculates the FID of two paths"""
#     for p in paths:
#         if not os.path.exists(p):
#             raise RuntimeError('Invalid path: %s' % p)

#     block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

#     model = InceptionV3([block_idx]).to(device)

#     m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
#                                         dims, device, num_workers)
#     m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
#                                         dims, device, num_workers)
#     fid_value = calculate_frechet_distance(m1, s1, m2, s2)

#     return fid_value

# def compute_statistics_of_path(path, model, batch_size, dims, device,
#                                num_workers=1):
#     if path.endswith('.npz'):
#         with np.load(path) as f:
#             m, s = f['mu'][:], f['sigma'][:]
#     else:
#         # path = pathlib.Path(path)
#         subfolders = [pathlib.Path(f.path) for f in os.scandir(path) if f.is_dir()]
#         files = sorted([file for ext in IMAGE_EXTENSIONS for p in subfolders
#                        for file in p.glob('*.{}'.format(ext))])
#         m, s = calculate_activation_statistics(files, model, batch_size,
#                                                dims, device, num_workers)
#     return m, s

# def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
#                                     device='cpu', num_workers=1):
#     """Calculation of the statistics used by the FID.
#     Params:
#     -- files       : List of image files paths
#     -- model       : Instance of inception model
#     -- batch_size  : The images numpy array is split into batches with
#                      batch size batch_size. A reasonable batch size
#                      depends on the hardware.
#     -- dims        : Dimensionality of features returned by Inception
#     -- device      : Device to run calculations
#     -- num_workers : Number of parallel dataloader workers

#     Returns:
#     -- mu    : The mean over samples of the activations of the pool_3 layer of
#                the inception model.
#     -- sigma : The covariance matrix of the activations of the pool_3 layer of
#                the inception model.
#     """
#     act = get_activations(files, model, batch_size, dims, device, num_workers)
#     mu = np.mean(act, axis=0)
#     sigma = np.cov(act, rowvar=False)
#     return mu, sigma

# from torch.nn.functional import adaptive_avg_pool2d
# def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
#                     num_workers=1):
#     """Calculates the activations of the pool_3 layer for all images.

#     Params:
#     -- files       : List of image files paths
#     -- model       : Instance of inception model
#     -- batch_size  : Batch size of images for the model to process at once.
#                      Make sure that the number of samples is a multiple of
#                      the batch size, otherwise some samples are ignored. This
#                      behavior is retained to match the original FID score
#                      implementation.
#     -- dims        : Dimensionality of features returned by Inception
#     -- device      : Device to run calculations
#     -- num_workers : Number of parallel dataloader workers

#     Returns:
#     -- A numpy array of dimension (num images, dims) that contains the
#        activations of the given tensor when feeding inception with the
#        query tensor.
#     """
#     model.eval()

#     if batch_size > len(files):
#         print(('Warning: batch size is bigger than the data size. '
#                'Setting batch size to data size'))
#         batch_size = len(files)

#     dataset = ImagePathDataset(files, transforms=TF.ToTensor())
#     dataloader = torch.utils.data.DataLoader(dataset,
#                                              batch_size=batch_size,
#                                              shuffle=False,
#                                              drop_last=False,
#                                              num_workers=num_workers)

#     pred_arr = np.empty((len(files), dims))

#     start_idx = 0

#     for batch in tqdm(dataloader):
#         batch = batch.to(device)

#         with torch.no_grad():
#             pred = model(batch)[0]

#         # If model output is not scalar, apply global spatial average pooling.
#         # This happens if you choose a dimensionality not equal 2048.
#         if pred.size(2) != 1 or pred.size(3) != 1:
#             pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

#         pred = pred.squeeze(3).squeeze(2).cpu().numpy()

#         pred_arr[start_idx:start_idx + pred.shape[0]] = pred

#         start_idx = start_idx + pred.shape[0]

#     return pred_arr
