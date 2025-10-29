import os, sys, pdb
import queue
import shutil
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import random

class PublicFFHQ(torch.utils.data.Dataset):
    def __init__(self, root='data/ffhq/thumbnails128x128', transform=None):
        super(PublicFFHQ, self).__init__()
        self.root = root
        self.transform = transform
        self.images = []
        self.path = self.root

        num_classes = len([lists for lists in os.listdir(
            self.path) if os.path.isdir(os.path.join(self.path, lists))])

        for idx in range(num_classes):
            class_path = os.path.join(self.path, str(idx * 1000).zfill(5))
            for _, _, files in os.walk(class_path):
                for img_name in files:
                    self.images.append(os.path.join(class_path, img_name))

    def __getitem__(self, index):

        img_path = self.images[index]
        # print(img_path)
        img = Image.open(img_path)
        if self.transform != None:
            img = self.transform(img)

        return img, img_path

    def __len__(self):
        return len(self.images)


class PublicCeleba(torch.utils.data.Dataset):
    def __init__(self, 
                 file_path='data/celeba/data_files/celeba_ganset.txt',
                 img_root='data/celeba/img_align_celeba',
                 mode_gan=True,
                 transform=None
        ):
        super(PublicCeleba, self).__init__()
        self.file_path = file_path
        self.img_root = img_root
        self.transform = transform
        self.images = []
        self.labels = []
        self.mode_gan = mode_gan
        
        # _, _, image2id, id2image = get_identity_from_file(self.img_idx_path)

        f = open(self.file_path, "r")
        for line in f.readlines():
            if self.mode_gan:
                img_name = line.strip()
                # img_name, iden = line.strip().split(' ')
                # self.labels.append(int(iden))
            else:
                img_name, iden = line.strip().split(' ')
                self.labels.append(int(iden))

            img_path = os.path.join(self.img_root, img_name)
            self.images.append(img_path)

        # if self.file_path:
        #     f = open(self.file_path, "r")
        #     for line in f.readlines():
        #         img_name = line.strip()
        #         self.images.append((os.path.join(self.img_root, img_name), int(image2id[img_name])))

        # else:
        #     # METHOD FOR READING IMAGES AND APPENDING LABELS FROM A FILE
        #     for _class_ in range(classes[0], classes[1]):
        #         images = id2image[str(_class_)]
        #         for _image_ in random.sample(images, min(30, len(images))):
        #             img_path = os.path.join(self.img_root, _image_)
        #             self.images.append((os.path.join(img_path, img_name), _class_))

    def __getitem__(self, index):
        if self.mode_gan:
            img_path = self.images[index]
            img = Image.open(img_path)
            if self.transform != None:
                img = self.transform(img)
            return img, img_path
        else:
            img_path = self.images[index]
            label = self.labels[index]
            img = Image.open(img_path)
            if self.transform != None:
                img = self.transform(img)
            return img, label

    def __len__(self):
        return len(self.images)


class PublicFaceScrub(torch.utils.data.Dataset):
    def __init__(self, file_path='data/data_files/facescrub_ganset.txt',
                 img_root='data/facescrub', transform=None):
        super(PublicFaceScrub, self).__init__()
        self.file_path = file_path
        self.img_root = img_root
        self.transform = transform
        self.images = []

        name_list, label_list = [], []

        f = open(self.file_path, "r")
        for line in f.readlines():
            img_name = line.strip()
            img_path = os.path.join(self.img_root, img_name)
            try:
                if img_path.endswith(".png") or img_path.endswith(".jpg"):
                    img = Image.open(img_path)
                    if img.size != (64, 64):
                        img = img.resize((64, 64), Image.ANTIALIAS)
                    img = img.convert('RGB')
                    self.images.append((img, img_path))
            except:
                continue

    def __getitem__(self, index):

        img, img_path = self.images[index]
        if self.transform != None:
            img = self.transform(img)

        return img, img_path

    def __len__(self):
        return len(self.images)

def _noise_adder(img):
    return torch.empty_like(img, dtype=torch.float32).uniform_(0.0, 1 / 256.0) + img
    
def pathdataset(
        data_name,
        mode_gan=True,
        classes=[0, 1000],
        img_path=None,
        img_set=None,
        img_idx_path=None,
        add_noise=False,
        batch_size=32,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        ):

    if data_name == 'celeba':
        re_size = 64
        crop_size = 108
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
        celeba_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(crop),
            transforms.ToPILImage(),
            transforms.Resize((re_size, re_size)),
            transforms.ToTensor(),
            _noise_adder if add_noise else transforms.Lambda(lambda x: x),
        ])
        data_set = PublicCeleba(file_path=img_set,
                                img_root=img_path,
                                # img_idx_path=img_idx_path,
                                # classes=classes,
                                mode_gan=mode_gan,
                                transform=celeba_transform)
    elif data_name == 'ffhq':
        re_size = 64
        crop_size = 88
        offset_height = (128 - crop_size) // 2
        offset_width = (128 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
        ffhq_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(crop),
            transforms.ToPILImage(),
            transforms.Resize((re_size, re_size)),
            transforms.ToTensor(),
            _noise_adder if add_noise else transforms.Lambda(lambda x: x),
        ])
        data_set = PublicFFHQ(root=img_path,
                              transform=ffhq_transform)
    elif data_name == 'facescrub':
        crop_size = 54
        offset_height = (64 - crop_size) // 2
        offset_width = (64 - crop_size) // 2
        re_size = 64
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

        faceScrub_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(crop),
            transforms.ToPILImage(),
            transforms.Resize((re_size, re_size)),
            transforms.ToTensor(),
            _noise_adder if add_noise else transforms.Lambda(lambda x: x),
        ])
        data_set = PublicFaceScrub(file_path=img_set,
                                img_root=img_path,
                                transform=faceScrub_transform)

    data_loader = data.DataLoader(data_set, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory)

    return data_loader
    # while True:
    #     yield from data_loader

def get_identity_from_file(identity_annot_filename='data/celeba/identity_CelebA.txt'):
    """
    Reads the identity annotations from a file and returns the unique identities, images, and mappings.
    """
    with open(identity_annot_filename, 'r') as file:
        lines = file.readlines()
        ids = set()
        images = []
        image2id = {}
        id2images = {}
        excludes = '202599.jpg' #get_excludes()

        for line in lines:
            line = line.strip()
            if len(line) > 0:
                tokens = line.split(' ')
                image_name = tokens[0].strip()
                if image_name not in excludes and image_name != '202599.jpg':
                    id = tokens[1].strip()
                    ids.add(id)
                    images.append(image_name)
                    image2id[image_name] = int(id)
                    if id in id2images.keys():
                        id2images[id].append(image_name)
                    else:
                        id2images[id] = [image_name]

        return list(ids), sorted(images), image2id, id2images