# Description: This script partitions the CelebA dataset into public and private folders based on the identity of the person in the image.

import pandas as pd
import os
import shutil

# read img_align_celeba/identity_CelebA.txt
df = pd.read_csv('./data/identity_CelebA.txt', sep=' ', header=None)
df.columns = ['img', 'label']

# Parameters
n = 10000 # number of classes to take, max 10177
n_private = 5000 # number of classes to be private, max 10177 and <= n. The rest will be public

#create folders

os.makedirs('./data/celeba_reshaped', exist_ok=True)
os.makedirs('./data/celeba_reshaped/private', exist_ok=True)
os.makedirs('./data/celeba_reshaped/public', exist_ok=True)

# copy images to folders based on label

for i in range(1,n+1):
    if i <= n_private:
        os.makedirs('./data/celeba_reshaped/private/' + str(i), exist_ok=True)
        for img in df[df['label'] == i]['img']:
            shutil.copy('./data/img_align_celeba/' + img, './data/celeba_reshaped/private/' + str(i) + '/' + img)
    else:
        os.makedirs('./data/celeba_reshaped/public/' + str(i), exist_ok=True)
        for img in df[df['label'] == i]['img']:
            shutil.copy('./data/img_align_celeba/' + img, './data/celeba_reshaped/public/' + str(i) + '/' + img)
    if i % 100 == 0:
        print(i / n * 100, 'percent copied')