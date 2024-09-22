
import os
import sys


current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
# Add the project root to sys.path to allow absolute imports
sys.path.insert(0, project_root)
os.chdir('/home/johan/project/LeakPro/examples/gia/cifar10_reconstruction')

from torch.utils.data import DataLoader
from utils.model_preparation import ResNet, train_model
from utils.data_preparation import get_cifar10_dataset

# Load the dataset
path = "./data/"
trainset, testset, pretrainset = get_cifar10_dataset(path)
trainloader = DataLoader(trainset, batch_size=128, shuffle=False, drop_last=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False, drop_last=False)
pretrainloader = DataLoader(pretrainset, batch_size=128, shuffle=False, drop_last=False)

# Load the model
target_model = ResNet(num_classes=10)

# # Pretrain the global model on all training data
#train_model(target_model, pretrainloader, trainloader, testloader, epochs=25)



# Initiate LeakPro
from leakpro import LeakPro
from input_handler import Cifar10GIAInputHandler

# Read the config file
config_path = "./audit.yaml"

# Prepare leakpro object
leakpro = LeakPro(Cifar10GIAInputHandler, config_path)

leakpro.run_audit()