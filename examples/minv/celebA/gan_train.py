import os
import sys
import yaml


# Path to the dataset zip file
data_folder = "./data"


project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(project_root)

from examples.minv.celebA.utils.celebA_data import get_celebA_dataloader

import torch
from torch.nn import functional as F
import numpy as np
from examples.mia.celebA_HQ.utils.celeb_hq_model import ResNet18


# Load the config.yaml file
with open('train_config.yaml', 'r') as file:
    train_config = yaml.safe_load(file)

# Generate the dataset and dataloaders
path = os.path.join(os.getcwd(), train_config["data"]["data_dir"])

train_loader, test_loader = get_celebA_dataloader(path, train_config)

# We have model_metadata.pkl and target_model.pkl. Load metadata and target model
# Load the model
num_classes = train_loader.dataset.dataset.get_classes()
model = ResNet18(num_classes=num_classes)
model.load_state_dict(torch.load('./target/target_model.pkl'))
model.eval()

from examples.minv.celebA.utils.celebA_data import get_celebA_pseudoloader

pseudo_loader = get_celebA_pseudoloader(path, train_config, shuffle=True)

# GAN training
from examples.minv.celebA.utils.generator import ResNetGenerator
from examples.minv.celebA.utils.discriminator import SNResNetProjectionDiscriminator
import examples.minv.celebA.utils.losses as losses 
import kornia
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dim_z = train_config["gan"]["dim_z"]
n_iter = train_config["gan"]["iterations"]
n_dis = train_config["gan"]["n_dis"]
gen_lr = train_config["gan"]["gen_lr"]
dis_lr = train_config["gan"]["dis_lr"]
beta1 = train_config["gan"]["beta1"]
beta2 = train_config["gan"]["beta2"]
batch_size = train_config["gan"]["batch_size"]
alpha = train_config["gan"]["alpha"]

# Initialize the generator and discriminator
gen = ResNetGenerator(num_classes=num_classes, dim_z=dim_z, activation=F.relu, bottom_width=4).to(device)
dis = SNResNetProjectionDiscriminator(num_classes=num_classes, activation=F.relu).to(device)

# Load optimizers
opt_gen = torch.optim.Adam(gen.parameters(), gen_lr, (beta1, beta2))
opt_dis = torch.optim.Adam(dis.parameters(), dis_lr, (beta1, beta2))

# Adversarial losses
gen_criterion = losses.GenLoss(loss_type='hinge', is_relativistic=False)
dis_criterion = losses.DisLoss(loss_type='hinge', is_relativistic=False)

# Augmentations for generated images
aug_list = kornia.augmentation.container.ImageSequential(
        #kornia.augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.RandomRotation(5),
    ).to(device)

# Training loop
log_interval = 10

model.to(device)
model.eval()


torch.backends.cudnn.benchmark = True

def sample_from_generator(gen, n_classes, batch_size, device):
    """Sample random z and y from the generator"""
    
    z = torch.empty(batch_size, dim_z, dtype=torch.float32, device=device).normal_()
    y = torch.randint(0, n_classes, (batch_size,)).to(device)
    return gen(z, y), y, z


gen_losses = []
dis_losses = []
inv_losses = []

# Training loop
for i in range(n_iter):
    _l_g = .0
    cumulative_inv_loss = 0.
    cumulative_loss_dis = .0

    cumulative_target_acc = .0
    target_correct = 0
    count = 0
    for j in range(n_dis):
        if j == 0:
            fake, fake_labels, _ = sample_from_generator(gen, num_classes, batch_size, device)
            fake_aug = aug_list(fake)
            dis_fake = dis(fake_aug, fake_labels)

            inv_loss = losses.max_margin_loss(model(fake_aug), fake_labels)

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

        fake, fake_labels, _ = sample_from_generator(gen, num_classes, batch_size, device)

        real, real_labels = next(iter(pseudo_loader))
        real, real_labels = real.to(device), real_labels.to(device)

        dis_fake = dis(fake, fake_labels)
        dis_real = dis(real, real_labels)

        loss_dis = dis_criterion(dis_fake, dis_real)
    
        if loss_dis.item() > 0.2*_l_g:
        
            dis.zero_grad()
        
            loss_dis.backward()
            opt_dis.step()

        cumulative_loss_dis += loss_dis.item()
        dis_losses.append(cumulative_loss_dis/n_dis)
        
        with torch.no_grad():
            count += fake.shape[0]
            T_logits = model(fake)
            T_preds = T_logits.max(1, keepdim=True)[1]
            target_correct += T_preds.eq(fake_labels.view_as(T_preds)).sum().item()
            cumulative_target_acc += round(target_correct / count, 4)

    if i % log_interval == 0:
        print(
                'iteration: {:05d}/{:05d}, loss gen: {:05f}, loss dis {:05f}, inv loss {:05f}, target acc {:04f}'.format(
                    i, n_iter, _l_g, cumulative_loss_dis, cumulative_inv_loss,
                    cumulative_target_acc, ))
        
        if cumulative_target_acc > 0.1:
            break

# Save the generator and discriminator
torch.save(gen.state_dict(), 'generator_1.pth')
torch.save(dis.state_dict(), 'discriminator_1.pth')
