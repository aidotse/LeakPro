"""Train function that keeps the computational graph intact."""
from collections import OrderedDict

from torch import cuda, device
from torch.nn import Module
from torch.utils.data import DataLoader
import tqdm
import torch
from leakpro.fl_utils.gia_module_to_functional import MetaModule
from leakpro.fl_utils.gia_optimizers import MetaOptimizer


def train(
    model: Module,
    data: DataLoader,
    optimizer: MetaOptimizer,
    criterion: Module,
    epochs: int,
) -> list:
    """Model training procedure for GIA.

    This training will create a computational graph through multiple steps, which is necessary
    for backpropagating to an input image.

    Requires a meta optimizer that performs step to a new set of parameters to keep a functioning
    graph.

    Training does not update the original model, but returns a norm of what the update would have been.
    """
   

    #torch.manual_seed(42)  # Set the seed before forward pass
    set_seed(seed=42)
    
    gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
    model.to(gpu_or_cpu)
    #sum_model = sum([torch.sum(l) for l in model.parameters()])
    #print("sum model: ", sum_model)
    patched_model = MetaModule(model)
    outputs = None
    model.train()

    #print("sum patched_model.parameters: ", sum([torch.sum(l) for l in patched_model.parameters.values()]))
    for _ in range(epochs):
        
        
        for X in data:
            
            X['embedding'].to(gpu_or_cpu)
            
            
            X['labels'].to(gpu_or_cpu)
            X['attention_masks'].to(gpu_or_cpu)
            y = X['labels']
            
            y_pred = patched_model(X, patched_model.parameters)
            #print("y_pred sum: ", torch.sum(y_pred))
        
            y_pred = y_pred.permute(0,2,1)
           
            loss = criterion(y_pred, y)
            

            patched_model.parameters = optimizer.step(loss, patched_model.parameters)
    #print("forward loss: ", loss)
    model_delta = OrderedDict((name, param - param_origin)
                                            for ((name, param), (name_origin, param_origin))
                                            in zip(patched_model.parameters.items(),
                                                    OrderedDict(model.named_parameters()).items()))
    #model_sum = sum([torch.sum(l) for l in model_delta.values()])
    #print("model_sum: ", model_sum)
    return list(model_delta.values())



def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using GPU
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Makes CuDNN deterministic
    torch.backends.cudnn.benchmark = False  # Slows down but ensures consistency

