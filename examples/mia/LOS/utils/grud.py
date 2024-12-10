import  math, os, pickle, time, pandas as pd, numpy as np, scipy.stats as ss
from sklearn.metrics import accuracy_score


import torch.utils.data as utils, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from tqdm import tqdm
from torch import save, device, cuda, Tensor, eye, transpose, nn, optim, save, sigmoid, no_grad, from_numpy, cat, zeros, ones, exp, tanh, zeros, ones, max, squeeze, isnan

def to_3D_tensor(df):
    idx = pd.IndexSlice
    np_3D = np.dstack([df.loc[idx[:, :, :, i], :].values for i in sorted(set(df.index.get_level_values('hours_in')))])
    return from_numpy(np_3D)

def prepare_dataloader(df, Ys, batch_size, shuffle=True):
    """
    dfs = (df_train, df_dev, df_test).
    df_* = (subject, hadm, icustay, hours_in) X (level2, agg fn \ni {mask, mean, time})
    Ys_series = (subject, hadm, icustay) => label.
    """
    X     = from_numpy(to_3D_tensor(df).astype(np.float32))
    label = from_numpy(Ys.values.astype(np.int64))
    dataset = utils.TensorDataset(X, label)
    return utils.DataLoader(dataset, batch_size =int(batch_size) , shuffle=shuffle, drop_last = True)

class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, device, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        assert in_features > 1 and out_features > 1, "Passing in nonsense sizes"
        
        self.filter_square_matrix = None
        self.filter_square_matrix = Variable(filter_square_matrix.to(device), requires_grad=False)
        
        self.weight = Parameter(Tensor(out_features, in_features)).to(device)

        if bias: 
            self.bias = Parameter(Tensor(out_features)).to(device)
        else:    
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None: 
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return F.linear(
            x,
            self.filter_square_matrix.mul(self.weight),
            self.bias
        )

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'
        
class GRUD(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size, X_mean, batch_size = 0, output_last = False):
        """
        With minor modifications from https://github.com/zhiyongc/GRU-D/

        Recurrent Neural Networks for Multivariate Times Series with Missing Values
        GRU-D: GRU exploit two representations of informative missingness patterns, i.e., masking and time interval.
        cell_size is the size of cell_state.
        
        Implemented based on the paper: 
        @article{che2018recurrent,
          title={Recurrent neural networks for multivariate time series with missing values},
          author={Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan},
          journal={Scientific reports},
          volume={8},
          number={1},
          pages={6085},
          year={2018},
          publisher={Nature Publishing Group}
        }
        
        GRU-D:
            input_size: variable dimension of each time
            hidden_size: dimension of hidden_state
            mask_size: dimension of masking vector
            X_mean: the mean of the historical input data
        """
        
        super(GRUD, self).__init__()

        # Save init params to a dictionary 
        self.init_params = {
            "input_size": input_size,
            "cell_size": cell_size,
            "hidden_size": hidden_size,
            "X_mean": X_mean,
            "batch_size": batch_size,
            "output_last": output_last
        }
        
        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size

        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.identity = eye(input_size).to(self.device)
        self.X_mean = Variable(Tensor(X_mean).to(self.device))
        
        self.zl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size).to(self.device) # Wz, Uz are part of the same network. the bias is bz
        self.rl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size).to(self.device) # Wr, Ur are part of the same network. the bias is br
        self.hl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size).to(self.device) # W, U are part of the same network. the bias is b
        
        self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, self.identity, self.device)
        
        self.gamma_h_l = nn.Linear(self.delta_size, self.hidden_size).to(self.device) # this was wrong in available version. remember to raise the issue
        
        self.output_last = output_last
        
        self.fc = nn.Linear(self.hidden_size, 2).to(self.device)
        self.bn= nn.BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True).to(self.device)
        self.drop=nn.Dropout(p=0.5, inplace=False)
        

    def step(self, x, x_last_obsv, x_mean, h, mask, delta):
        """
        Inputs:
            x: input tensor
            x_last_obsv: input tensor with forward fill applied
            x_mean: the mean of each feature
            h: the hidden state of the network
            mask: the mask of whether or not the current value is observed
            delta: the tensor indicating the number of steps since the last time a feature was observed.
            
        Returns:
            h: the updated hidden state of the network
        """
        
        # Assert to check for NaNs in x_mean
        assert not isnan(x_mean).any(), "NaN values found in x_mean"

        batch_size = x.size()[0]
        feature_size = x.size()[1]
        zero_x = zeros(batch_size, feature_size).to(self.device)
        zero_h = zeros(batch_size, self.hidden_size).to(self.device)

        
        gamma_x_l_delta = self.gamma_x_l(delta)
        delta_x = exp(-max(zero_x, gamma_x_l_delta))
        
        gamma_h_l_delta = self.gamma_h_l(delta)
        delta_h = exp(-max(zero_h, gamma_h_l_delta))
        
        x_mean = x_mean.repeat(batch_size, 1)

        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)
        h = delta_h * h
        
        combined = cat((x, h, mask), 1)
        # Assert to check for NaNs in combined
        assert not isnan(combined).any(), "NaN values found in combined"
        
        z = sigmoid(self.zl(combined)) #sigmoid(W_z*x_t + U_z*h_{t-1} + V_z*m_t + bz)
        r = sigmoid(self.rl(combined)) #sigmoid(W_r*x_t + U_r*h_{t-1} + V_r*m_t + br)
        combined_new = cat((x, r*h, mask), 1)
        h_tilde = tanh(self.hl(combined_new)) #tanh(W*x_t +U(r_t*h_{t-1}) + V*m_t) + b
        h = (1 - z) * h + z * h_tilde
        
        return h
    
    def forward(self, X, X_last_obsv, Mask, Delta):
       
        batch_size = X.size(0)
        step_size = X.size(1) # num timepoints
        Hidden_State = self.initHidden(batch_size)

        
        outputs = None
        for i in range(step_size):
            Hidden_State = self.step(
                squeeze(X[:,i,:], 1),
                squeeze(X_last_obsv[:,i,:], 1), # At the starting point, this is the same as X
                squeeze(self.X_mean[:,i,:], 1), 
                Hidden_State,
                squeeze(Mask[:,i,:], 1),
                squeeze(Delta[:,i,:], 1), # time of measurment 
            )
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                #TODO: check the order. in the original git repo of GRU it is cat((outputs, Hidden_State.unsqueeze(1)), 1)
                outputs = cat((Hidden_State.unsqueeze(1), outputs), 1)
                
        # we want to predict a binary outcome
        #Apply 50% dropout and batch norm here
        self.drop(self.bn(self.fc(Hidden_State)))
        return self.drop(self.bn(self.fc(Hidden_State)))
                
    def initHidden(self, batch_size):
        Hidden_State = Variable(zeros(batch_size, self.hidden_size)).to(self.device)
        return Hidden_State

# def convert_to_device(x):
#         device_name = device("cuda" if cuda.is_available() else "cpu")
#         return x.to(device_name)


# Focus on the audit 
def gru_trained_model_and_metadata(model,
                                   train_dataloader,
                                   test_dataloader, 
                                   epochs,
                                   patience,
                                   min_delta = 1e-5, 
                                   learning_rate=1e-3,
                                    batch_size=None,
                                    metadata = None):
    

    
    print('Model Structure: ', model)
    print('Start Training ... ')
    device_name = device("cuda" if cuda.is_available() else "cpu")

    if isinstance(model, nn.Sequential):
        output_last = model[-1].output_last
        print('Output type dermined by the last layer')
    else:
        output_last = model.output_last
        print('Output type dermined by the model')
        
    criterion_CEL = nn.CrossEntropyLoss()
    criterion_MSE = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    

    train_losses = []
    test_losses = []
    test_acc = []
    train_acc = []

    
    cur_time = time.time()
    pre_time = time.time()
    
    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0

    model.to(device_name)

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        
        model.train()
        train_loss = 0.0
        
        test_dataloader_iter = iter(test_dataloader)

        for _, (X, labels) in enumerate(tqdm(train_dataloader, desc="Training Batches")):

            X = X.numpy()
            mask        = from_numpy(X[:, np.arange(0, X.shape[1], 3), :].astype(np.float32))
            measurement = from_numpy(X[:, np.arange(1, X.shape[1], 3), :].astype(np.float32))
            time_       = from_numpy(X[:, np.arange(2, X.shape[1], 3), :].astype(np.float32))
            
            mask = transpose(mask, 1, 2)
            measurement = transpose(measurement, 1, 2)
            time_ = transpose(time_, 1, 2)
            measurement_last_obsv = measurement
            
            X = measurement.to(device_name)
            X_last_obsv = measurement_last_obsv.to(device_name)
            Mask = mask.to(device_name)
            Delta = time_.to(device_name)
            labels =labels.long().to(device_name)


            model.zero_grad()
            prediction = model(X, X_last_obsv, Mask, Delta)

            output_last = True
            if output_last:
                loss = criterion_CEL(squeeze(prediction), squeeze(labels))
            else:
                full_labels = cat((X[:,1:,:], labels), dim = 1)
                loss = criterion_MSE(prediction, full_labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        # Convert predictions to class indices
        binary_predictions = to_numpy(prediction).argmax(axis=1)

        # Ensure labels are integer and 1D
        binary_labels = to_numpy(labels).astype(int)
        # Compute accuracy
        train_acc.append(accuracy_score(binary_labels, binary_predictions))

        # test 
        model.eval()
        try: 
            X_test, labels_test = next(test_dataloader_iter)
            X_test = X_test.numpy()
            mask_test        = from_numpy(X_test[:, np.arange(0, X_test.shape[1], 3), :].astype(np.float32))
            measurement_test = from_numpy(X_test[:, np.arange(1, X_test.shape[1], 3), :].astype(np.float32))
            time_test       = from_numpy(X_test[:, np.arange(2, X_test.shape[1], 3), :].astype(np.float32))
        
            mask_test = transpose(mask_test, 1, 2)
            measurement_test = transpose(measurement_test, 1, 2)
            time_test = transpose(time_test, 1, 2)
            measurement_last_obsv_test = measurement_test
        except StopIteration:
            valid_dataloader_iter = iter(test_dataloader)
            X_test, labels_test = next(valid_dataloader_iter)
            X_test = X_test.numpy()
            mask_test        = from_numpy(X_test[:, np.arange(0, X_test.shape[1], 3), :].astype(np.float32))
            measurement_test = from_numpy(X_test[:, np.arange(1, X_test.shape[1], 3), :].astype(np.float32))
            time_test       = from_numpy(X_test[:, np.arange(2, X_test.shape[1], 3), :].astype(np.float32))
        
            mask_test = transpose(mask_test, 1, 2)
            measurement_test = transpose(measurement_test, 1, 2)
            time_test = transpose(time_test, 1, 2)
            measurement_last_obsv_test = measurement_test
        

        X_test = measurement_test.to(device_name)
        X_last_obsv_test = measurement_last_obsv_test.to(device_name)
        Mask_test = mask_test.to(device_name)
        Delta_test = time_test.to(device_name)
        labels_test = labels_test.long().to(device_name)

        model.zero_grad()
        prediction_val = model(X_test, X_last_obsv_test, Mask_test, Delta_test)
        
        if output_last:
            test_loss = criterion_CEL(squeeze(prediction_val), squeeze(labels_test))
        else:
            full_labels_val = cat((X_test[:,1:,:], labels_test), dim = 1)
            test_loss = criterion_MSE(prediction_val, full_labels_val)

        test_losses.append(test_loss)

        # Convert predictions to class indices
        binary_predictions_val = to_numpy(prediction_val).argmax(axis=1)

        # Ensure labels are integer and 1D
        binary_labels_test = to_numpy(labels_test).astype(int)
        # Compute accuracy
        test_acc.append(accuracy_score(binary_labels_test, binary_predictions_val))


        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            min_loss_epoch_valid = 10000.0
            if test_loss < min_loss_epoch_valid:
                min_loss_epoch_valid = test_loss
        else:
            if min_loss_epoch_valid - test_loss > min_delta:
                is_best_model = 1
                min_loss_epoch_valid = test_loss 
                patient_epoch = 0
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break
        
        # Print training parameters
        cur_time = time.time()
        print('Epoch: {}, train_loss: {}, valid_loss: {}, time: {}, best model: {}'.format( \
                    epoch, \
                    np.around(test_loss.cpu().item(), decimals=8),\
                    np.around(test_loss.cpu().item(), decimals=8),\
                    np.around(cur_time - pre_time, decimals=2),\
                    is_best_model))
        pre_time = cur_time
    # Move the model back to the CPU
    # Ensure the target directory exists
    os.makedirs("target", exist_ok=True)
    model.to("cpu")
    with open("target/target_model.pkl", "wb") as f:
        save(model.state_dict(), f)

     # Create metadata and store it
    meta_data = {}
    meta_data["train_indices"] = train_dataloader.dataset.indices
    meta_data["test_indices"] = test_dataloader.dataset.indices
    meta_data["num_train"] = len(meta_data["train_indices"])

    # Write init params
    meta_data["init_params"] = {}
    for key, value in model.init_params.items():
        meta_data["init_params"][key] = value

    # read out optimizer parameters
    meta_data["optimizer"] = {}
    meta_data["optimizer"]["name"] = optimizer.__class__.__name__.lower()
    meta_data["optimizer"]["lr"] = optimizer.param_groups[0].get("lr", 0)
    meta_data["optimizer"]["weight_decay"] = optimizer.param_groups[0].get("weight_decay", 0)
    meta_data["optimizer"]["momentum"] = optimizer.param_groups[0].get("momentum", 0)
    meta_data["optimizer"]["dampening"] = optimizer.param_groups[0].get("dampening", 0)
    meta_data["optimizer"]["nesterov"] = optimizer.param_groups[0].get("nesterov", False)

    # read out criterion parameters
    meta_data["loss"] = {}
    meta_data["loss"]["name"] = criterion_CEL.__class__.__name__.lower()

    meta_data["batch_size"] = train_dataloader.batch_size
    meta_data["epochs"] = epochs
    meta_data["train_acc"] = 0
    meta_data["test_acc"] = 0
    meta_data["train_loss"] = train_loss
    meta_data["test_loss"] = test_loss
    meta_data["dataset"] = "mimiciii"
    


    with open("target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)


    return  train_losses, test_losses, train_acc, test_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    loss, acc = 0, 0
    with no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            target = target.float().unsqueeze(1)
            output = model(data)
            loss += criterion(output, target).item()
            pred = sigmoid(output) >= 0.5
            acc += pred.eq(target.data.view_as(pred)).sum()
        loss /= len(loader)
        acc = float(acc) / len(loader.dataset)
    return loss, acc

def accuracy(model, loader,  criterion, device):
    model.eval()
    loss, acc = 0, 0
    acc_list = []
    with no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            target = target.float().unsqueeze(1)
            output = model(data)
            loss += criterion(output, target).item()
            pred = sigmoid(output) >= 0.5
            acc += pred.eq(target.data.view_as(pred)).sum()
            acc = accuracy_score(target, pred)
            acc_list.append(acc)
    return acc_list

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.is_cuda else tensor.detach().numpy()



