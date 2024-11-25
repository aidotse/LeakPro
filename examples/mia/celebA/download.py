from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import CelebA
root =  "./data"
transform = Compose([
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalization for CelebA
            ])
celebA_dataset = CelebA(root=root, split="all", target_type= "identity" , download=True,  transform=transform)