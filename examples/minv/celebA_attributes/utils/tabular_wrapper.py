from pytorch_tabular import TabularModel
import cupy as cp
import torch
import numpy as np

class TabularWrapper(TabularModel):
    """Wrapper class for Tabular Model."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def __call__(self, entry):
        """Make the model callable with PyTorch tensors."""
               # Convert the DataFrame into a dataloader (pytorch_tabular handles formatting)
        inference_dataloader = self.datamodule.prepare_inference_dataloader(entry)

        all_logits = []  # List to collect logits

        # No 'torch.no_grad()' here because we want to keep the computation graph for backprop
        for batch in inference_dataloader:
            # Send batch to device            
            # Perform forward pass to get model output
            out = self.model.forward(batch)

            # Collect logits for later backpropagation
            all_logits.append(out["logits"])

        # Concatenate all logits after the loop to avoid repeated allocation
        tensor_output = torch.cat(all_logits, dim=0)

        return tensor_output

    def to(self, device):
        self.model.to(device)
        pass
    
    def eval(self):
        pass