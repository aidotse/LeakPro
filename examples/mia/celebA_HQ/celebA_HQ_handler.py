from torch import cuda, device, optim, no_grad
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from torchvision import datasets, transforms
from torch import cat
import os

from leakpro import AbstractInputHandler
from leakpro.schemas import TrainingOutput, EvalOutput

class CelebAHQInputHandler(AbstractInputHandler):
    """Class to handle the user input for the CelebA_HQ dataset."""

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



    class UserDataset(AbstractInputHandler.UserDataset):
        def __init__(self, data, targets, transform=None, indices=None):
            """
            Custom dataset for CelebA data.

            Args:
                x (torch.Tensor): Tensor of input images.
                y (torch.Tensor): Tensor of labels.
                transform (callable, optional): Optional transform to be applied on the image tensors.
                indices (optional): Indices for custom sampling.
            """
            self.data = data  # Use 'data' as the image data
            self.targets = targets  # Use 'targets' as the labels
            self.transform = transform
            self.indices = indices

        def __len__(self):
            """Return the total number of samples."""
            return len(self.targets)

        def __getitem__(self, idx):
            """Retrieve the image and its corresponding label at index 'idx'."""
            image = self.data[idx]
            label = self.targets[idx]

            # Apply transformations to the image if any
            if self.transform:
                image = self.transform(image)

            return image, label

        @classmethod
        def from_celebHq(cls, config):
            data_dir = config["data"]["data_dir"]
            train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
            test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), test_transform)
            combined_dataset = ConcatDataset([train_dataset, test_dataset])

            # Prepare data loader to iterate over combined_dataset
            loader = DataLoader(combined_dataset, batch_size=1, shuffle=False)

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