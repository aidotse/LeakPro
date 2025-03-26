from torch import Tensor, cuda, device, optim, no_grad, from_numpy
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro import AbstractInputHandler
from leakpro.schemas import TrainingOutput, EvalOutput

class MIMICInputHandler(AbstractInputHandler):

    # def train():
    #     pass

    # def eval():
    #     pass

    def to_3D_tensor(df):
        idx = pd.IndexSlice
        np_3D = np.dstack([df.loc[idx[:, :, :, i], :].values for i in sorted(set(df.index.get_level_values("hours_in")))])
        return from_numpy(np_3D)

    class UserDataset(AbstractInputHandler.UserDataset):
        """
        A custom dataset class for handling user data.

        Args:
            x (torch.Tensor): The input features as a torch tensor.
            y (torch.Tensor): The target labels as a torch tensor.
            
        Methods:
            __len__(): Returns the length of the dataset.
            __getitem__(idx): Returns the item at the given index.
            subset(indices): Returns a subset of the dataset based on the given indices.
        """

        def __init__(self, x, y):
            # Ensure both x and y are converted to tensors (float32 type)
            self.x = Tensor(x).float()  # Convert features to torch tensor and ensure it is float32
            self.y = Tensor(y).float()  # Convert labels to torch tensor and ensure it is float32

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx].squeeze(0)

        # def subset(self, indices):
        #     return MimicDataset(self.x[indices], self.y[indices])