"""Util functions relating to data."""
from data_.data import TrainingBatch
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

from leakpro.fl_utils.data_utils import GiaDataModalityExtension


class GiaNERExtension(GiaDataModalityExtension):

    def get_at_data(self,client_loader: DataLoader, token_used: np.ndarray=None) -> DataLoader:
        """DataLoader with random noise images of the same shape as the client_loader's dataset, using the same labels."""
        reconstruction_dataset = deepcopy(client_loader.dataset)
        reconstruction = []  # Collect embeddings to optimize
        for d in reconstruction_dataset:
            ind = np.where(np.array(d.labels)!=0)[0]#[0:1]
            token_used = torch.tensor(token_used, device=d.embedding.device)
            ind = torch.tensor(ind, device=d.embedding.device)

            d.embedding.index_put_((ind[:, None], token_used), torch.ones_like(d.embedding[ind[:, None],token_used])/54)#54)


            #d.embedding[ind] = (d.embedding[ind].T/torch.sum(d.embedding[ind],1)).T
            d.embedding.requires_grad = True 
            # make tokens with labels != 0 trainable
            mask = torch.zeros_like(d.embedding)
            mask[ind] = 1 
            #d.embedding = PartialTrainableTensor.apply(d.embedding, mask).detach().requires_grad_(True)
            def mask_grad(grad, mask=mask):
                return grad * mask  # Apply the mask to the gradient

            d.embedding.register_hook(mask_grad)

            # Add reference to the embedding for optimization
            reconstruction.append(d.embedding)

        reconstruction_loader = DataLoader(reconstruction_dataset, collate_fn=TrainingBatch, batch_size=1, shuffle=False)
        return reconstruction, reconstruction_loader