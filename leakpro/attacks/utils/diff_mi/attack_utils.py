"""Utility functions for the Diff-MI attack."""

import os
import torch
import torch.nn.functional as F
import copy
import math
import numpy as np
import torchvision


from torch.utils.data import TensorDataset, DataLoader
from leakpro.utils.logger import logger
from leakpro.attacks.utils.diff_mi.diffusion import GaussianDiffusion, InferenceModel
import torchvision.transforms as augmentation
from tqdm import tqdm
import statistics

from robustness import model_utils

def Iterative_Image_Reconstruction(args, diff_net, classifier, classes, p_reg, iter=None, batch_num=None, device='cuda'):

    diffusion = GaussianDiffusion(T=1000, schedule='linear')
    model = InferenceModel(batch_size=classes.shape[0]).to(device=device)
    model.train()

    # Inference procedure steps
    steps = args.steps
    opt = torch.optim.Adamax(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=1.0, total_iters=steps)

    norm_track = 0

    print(f"--------------------- Epoch {iter}/{batch_num} ---------------------")
    bar = tqdm(range(steps))
    for i, _ in enumerate(bar): 

        bar.set_description(f'Epoch {iter}/{batch_num}')

        # Select t
        t = ((steps-i)/1.5 + (steps-i)/3*math.cos(3/(10*math.pi)*i))/steps*800 + 200 # Linearly decreasing + cosine
        t = np.array([t + np.random.randint(-50, 51) for _ in range(1)]).astype(int) # Add noise to t
        t = np.clip(t, 1, diffusion.T)

        # Denoise
        sample_img = model.encode()
        xt, epsilon = diffusion.sample(sample_img, t)
        t = torch.from_numpy(t).float().view(1)
        eps = diff_net(xt.float(), t.to(device), classes)
        nonEps = diff_net(xt.float(), t.to(device), torch.ones_like(classes) * (diff_net.num_classes - 1))
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
            attr_input = args.aug(attr_input).clamp(-1,1)
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
    """Set the model to use PGD adversarial training."""
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

def calc_acc(classifier, imgs, labels, bs=64, anno='', with_success=False, enable_print=True):

    output, img_dataset = [], TensorDataset(imgs)
    for x in torch.utils.data.DataLoader(img_dataset, batch_size=bs, shuffle=False):
        output.append(classifier(x[0])[-1])
    output = torch.cat(output)
    top1_count = torch.eq(torch.topk(output, k=1)[1], labels.view(-1,1)).float()
    top5_count = torch.eq(torch.topk(output, k=5)[1], labels.view(-1,1)).float()
    if enable_print:
        print(f'===> top1_acc: {top1_count.mean().item():.2%}, top5_acc: {5*top5_count.mean().item():.2%} {anno}')
    if with_success:
        success_idx = torch.nonzero((output.max(1)[1] == labels).int()).squeeze(1)
        return top1_count.sum().item(), top5_count.sum().item(), success_idx
    else:
        return top1_count.sum().item(), top5_count.sum().item()


def calc_acc_std(imgs, labels, cls, label_num):
    top1_list, top5_list = [], []
    assert imgs.shape[0] % label_num == 0
    for i in range(int(imgs.shape[0]/label_num)):
        imgs_ = imgs[i * label_num: (i+1) * label_num]
        labels_ = labels[i * label_num: (i+1) * label_num]
        assert torch.max(labels_) - torch.min(labels_) == label_num - 1
        top1_count, top5_count = calc_acc(cls, augmentation.Resize((112, 112))(imgs_), 
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
    label_paths = []
    fake_dataset = TensorDataset(imgs.cpu(), lables.cpu())
    for i, (x,y) in enumerate(torch.utils.data.DataLoader(fake_dataset, batch_size=1, shuffle=False)):
        label_path = os.path.join(save_path, str(y.item()))
        if not os.path.exists(label_path): os.makedirs(label_path)
        label_paths.append(label_path)
        torchvision.utils.save_image(x.detach()[0,:,:,:], os.path.join(label_path, f"{i}_attack.png"), padding=0)
    return label_paths

def calc_lpips(fake_images, fake_targets, anno='', device='cuda'):
    
    import lpips
    loss_fn_alex = lpips.LPIPS(net='alex').to(device) # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) # closer to "traditional" perceptual loss, when used for optimization

    # get target images and labels
    inferred_image_path = PRIVATE_PATH
    list_of_idx = os.listdir(inferred_image_path)
    images_list, targets_list = [], []
    # load reconstructed images
    for idx in list_of_idx:
        for filename in os.listdir(os.path.join(inferred_image_path, idx)):
            image = Image.open(os.path.join(inferred_image_path, idx, filename))
            image = T.functional.to_tensor(image)
            images_list.append(image)
            targets_list.append(int(idx))
    real_images = torch.stack(images_list, dim=0).to(device)
    real_targets = torch.LongTensor(targets_list).to(device)

    # get fake images and labels
    fake_images = fake_images.to(device)
    fake_targets = fake_targets.to(device)
    bs = fake_targets.size(0)
    value_a, value_v = 0, 0

    # calculate metric
    for i in range(bs):
        single_value_a, single_value_v = -1, -1
        idx = torch.nonzero(real_targets == fake_targets[i]).squeeze(1)
        for j in idx:
            temp_value_a = loss_fn_alex(fake_images[i].unsqueeze(0), real_images[j].unsqueeze(0))
            temp_value_v = loss_fn_vgg(fake_images[i].unsqueeze(0), real_images[j].unsqueeze(0))
            if temp_value_a > single_value_a: single_value_a = temp_value_a
            if temp_value_v > single_value_v: single_value_v = temp_value_v
        value_a += single_value_a
        value_v += single_value_v
    value_a = (value_a / bs).item()
    value_v = (value_v / bs).item()
    print(f"LPIPS : Alex {value_a:.4f} | VGG {value_v:.4f} {anno}")

def calc_knn(fake_imgs, fake_targets, E, anno='', path="assets/celeba_private_feats", device='cuda'):

    # get features of reconstructed images
    infered_feats = None
    for i, images in enumerate(torch.utils.data.DataLoader(fake_imgs, batch_size=64)):
        images = augmentation.Resize((112, 112))(images).to(device)
        feats = E(images)[0]
        if i == 0:
            infered_feats = feats.detach().cpu()
        else:
            infered_feats = torch.cat([infered_feats, feats.detach().cpu()], dim=0)

    # get features of target images
    idens = fake_targets.to(device).long()
    feats = infered_feats.to(device)
    true_feats = torch.from_numpy(np.load(os.path.join(path, "private_feats.npy"))).float().to(device)
    info = torch.from_numpy(np.load(os.path.join(path, "private_targets.npy"))).view(-1).long().to(device)
    bs = feats.size(0)
    knn_dist = 0

    def row_mse(a, b):
        c = a - b
        d = torch.pow(c, 2)
        e = torch.sum(d, dim=1)
        return e

    # calculate knn dist
    for i in tqdm(range(bs), desc='Calculating KNN Dist'):
        knn = 1e8
        idx = torch.nonzero(info == idens[i]).squeeze(1)
        fake_feat = feats[i].repeat(idx.shape[0], 1)
        true_feat = true_feats[idx]
        knn = row_mse(fake_feat, true_feat)
        knn_dist += torch.min(knn)
    knn = (knn_dist / bs).item()
    print(f"KNN Dist computed on {fake_imgs.shape[0]} attack samples: {knn:.2f} {anno}")