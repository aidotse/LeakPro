import copy, math, os, pickle, time, pandas as pd, numpy as np, scipy.stats as ss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score

import torch, torch.utils.data as utils, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
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
        if self.bias is not None: self.bias.data.uniform_(-stdv, stdv)

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
        
        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size
        self.device = device("cuda" if cuda.is_available() else "cpu")
        #TODO: check if this is the right way to do this
        use_gpu = cuda.is_available()
        if use_gpu:
            self.identity = eye(input_size).cuda()
            self.X_mean = Variable(Tensor(X_mean).cuda())
        else:
            self.identity = eye(input_size)
            self.X_mean = Variable(Tensor(X_mean))
        
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
                outputs = cat((Hidden_State.unsqueeze(1), outputs), 1)
                
        # we want to predict a binary outcome
        #Apply 50% dropout and batch norm here
        self.drop(self.bn(self.fc(Hidden_State)))
        return self.drop(self.bn(self.fc(Hidden_State)))
                
    def initHidden(self, batch_size):
        use_gpu = cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State
        else:
            Hidden_State = Variable(zeros(batch_size, self.hidden_size))
            return Hidden_State

def convert_to_device(x):
        device = device("cuda" if cuda.is_available() else "cpu")
        return x.to(device)


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

    if (type(model) == nn.modules.container.Sequential):
        output_last = model[-1].output_last
        print('Output type dermined by the last layer')
    else:
        output_last = model.output_last
        print('Output type dermined by the model')
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    # losses_train = []
    # losses_valid = []
    # losses_epochs_train = []
    # losses_epochs_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0
    for epoch in tqdm.tqdm(range(epochs), desc="Training Progress"):
        model.train()
        train_acc, train_loss = 0.0, 0.0
        
        for batch_idx, (X, labels) in enumerate(tqdm(train_dataloader, desc="Training Batches")):

            X = X.numpy()
            mask        = from_numpy(X[:, np.arange(0, X.shape[1], 3), :].astype(np.float32))
            measurement = from_numpy(X[:, np.arange(1, X.shape[1], 3), :].astype(np.float32))
            time_       = from_numpy(X[:, np.arange(2, X.shape[1], 3), :].astype(np.float32))
            
            mask = transpose(mask, 1, 2)
            measurement = transpose(measurement, 1, 2)
            time_ = transpose(time_, 1, 2)
            measurement_last_obsv = measurement

            X, X_last_obsv, Mask, Delta, labels = map(convert_to_device, [measurement, measurement_last_obsv, mask, time_, labels])

            model.zero_grad()
            prediction = model(X, X_last_obsv, Mask, Delta)
            loss = criterion(squeeze(prediction), squeeze(labels).long())


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader.dataset)
            
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        test_loss, test_acc = evaluate(model, test_dataloader, criterion, device_name)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

  


             # validation 
            try: 
                X_val, labels_val = next(valid_dataloader_iter)
                X_val = X_val.numpy()
                mask_val        = from_numpy(X_val[:, np.arange(0, X_val.shape[1], 3), :].astype(np.float32))
                measurement_val = from_numpy(X_val[:, np.arange(1, X_val.shape[1], 3), :].astype(np.float32))
                time_val       = from_numpy(X_val[:, np.arange(2, X_val.shape[1], 3), :].astype(np.float32))
            
                mask_val = transpose(mask_val, 1, 2)
                measurement_val = transpose(measurement_val, 1, 2)
                time_val = transpose(time_val, 1, 2)
                measurement_last_obsv_val = measurement_val
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                X_val, labels_val = next(valid_dataloader_iter)
                X_val = X_val.numpy()
                mask_val        = from_numpy(X_val[:, np.arange(0, X_val.shape[1], 3), :].astype(np.float32))
                measurement_val = from_numpy(X_val[:, np.arange(1, X_val.shape[1], 3), :].astype(np.float32))
                time_val       = from_numpy(X_val[:, np.arange(2, X_val.shape[1], 3), :].astype(np.float32))
            
                mask_val = transpose(mask_val, 1, 2)
                measurement_val = transpose(measurement_val, 1, 2)
                time_val = transpose(time_val, 1, 2)
                measurement_last_obsv_val = measurement_val
            

            X_val, X_last_obsv_val, Mask_val, Delta_val, labels_val = map(convert_to_device, 
                                                                          [measurement_val, measurement_last_obsv_val, mask_val, time_val, labels_val])
                
            model.zero_grad()
            
#             outputs_val = model(inputs_val)
            prediction_val = model(X_val, X_last_obsv_val, Mask_val, Delta_val)
    
#             print(labels.shape)
#             print(prediction_val.shape)
            
            # if output_last:
            loss_valid =loss_CEL(squeeze(prediction_val), squeeze(labels_val.long()))
            # else:
                # raise NotImplementedError("Should be output last!")
                # full_labels_val = torch.cat((inputs_val[:,1:,:], labels_val), dim = 1)
                # loss_valid = loss_MSE(outputs_val, full_labels_val)

            losses_valid.append(loss_valid.data)
            losses_epoch_valid.append(loss_valid.data)
            
#             print(sklearn.metrics.roc_auc_score(labels_val.detach().cpu().numpy(), prediction_val.detach().cpu().numpy()[:,1]))
            
            # output
            trained_number += 1
            
        avg_losses_epoch_train = sum(losses_epoch_train).cpu().numpy() / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid).cpu().numpy() / float(len(losses_epoch_valid))
        losses_epochs_train.append(avg_losses_epoch_train)
        losses_epochs_valid.append(avg_losses_epoch_valid)
        
        
        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid 
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
                    np.around(avg_losses_epoch_train, decimals=8),\
                    np.around(avg_losses_epoch_valid, decimals=8),\
                    np.around([cur_time - pre_time] , decimals=2),\
                    is_best_model) )
        pre_time = cur_time
#         if epoch==1:
#             break
    # Move the model back to the CPU
    model.to("cpu")
    with open("target/target_model.pkl", "wb") as f:
        save(model.state_dict(), f)
                
    return best_model, [losses_train, losses_valid, losses_epochs_train, losses_epochs_valid]











def Train_Model(
    model, train_dataloader, valid_dataloader, num_epochs = 300, patience = 3, min_delta = 1e-5, learning_rate=1e-3, batch_size=None
):
    
    print('Model Structure: ', model)
    print('Start Training ... ')
    device = device("cuda" if cuda.is_available() else "cpu")
    
    model
    
    if (type(model) == nn.modules.container.Sequential):
        output_last = model[-1].output_last
        print('Output type dermined by the last layer')
    else:
        output_last = model.output_last
        print('Output type dermined by the model')
        
    loss_MSE = nn.MSELoss()
    loss_CEL = nn.CrossEntropyLoss()
    loss_BCE = nn.BCELoss()
    
#     optimizer = optim.RMSprop(model.parameters(), lr = learning_rate, alpha=0.99)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    use_gpu = cuda.is_available()
    

    losses_train = []
    losses_valid = []
    losses_epochs_train = []
    losses_epochs_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0
    for epoch in range(num_epochs):
        
        trained_number = 0
        
        valid_dataloader_iter = iter(valid_dataloader)
        
        losses_epoch_train = []
        losses_epoch_valid = []
        
        for batch_idx, (X, labels) in enumerate(tqdm(train_dataloader, desc="Training Batches")):

            X = X.numpy()
            mask        = from_numpy(X[:, np.arange(0, X.shape[1], 3), :].astype(np.float32))
            measurement = from_numpy(X[:, np.arange(1, X.shape[1], 3), :].astype(np.float32))
            time_       = from_numpy(X[:, np.arange(2, X.shape[1], 3), :].astype(np.float32))
            
            mask = transpose(mask, 1, 2)
            measurement = transpose(measurement, 1, 2)
            time_ = transpose(time_, 1, 2)
            measurement_last_obsv = measurement

            # assert measurement.size()[0] == batch_size, "Batch Size doesn't match! %s" % str(measurement.size())


            X, X_last_obsv, Mask, Delta, labels = map(convert_to_device, [measurement, measurement_last_obsv, mask, time_, labels])

            model.zero_grad()

#           outputs = model(inputs)
            prediction = model(X, X_last_obsv, Mask, Delta)

            loss_train = loss_CEL(torch.squeeze(prediction), torch.squeeze(labels).long())


            # if output_last:
            #     loss_train = loss_CEL(torch.squeeze(prediction), torch.squeeze(labels))
            # else:
            #     full_labels = torch.cat((X[:,1:,:], labels), dim = 1)
            #     loss_train = loss_MSE(prediction, full_labels)
        
            losses_train.append(loss_train.data)
            losses_epoch_train.append(loss_train.data)
            
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()
            
             # validation 
            try: 
                X_val, labels_val = next(valid_dataloader_iter)
                X_val = X_val.numpy()
                mask_val        = from_numpy(X_val[:, np.arange(0, X_val.shape[1], 3), :].astype(np.float32))
                measurement_val = from_numpy(X_val[:, np.arange(1, X_val.shape[1], 3), :].astype(np.float32))
                time_val       = from_numpy(X_val[:, np.arange(2, X_val.shape[1], 3), :].astype(np.float32))
            
                mask_val = transpose(mask_val, 1, 2)
                measurement_val = transpose(measurement_val, 1, 2)
                time_val = transpose(time_val, 1, 2)
                measurement_last_obsv_val = measurement_val
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                X_val, labels_val = next(valid_dataloader_iter)
                X_val = X_val.numpy()
                mask_val        = from_numpy(X_val[:, np.arange(0, X_val.shape[1], 3), :].astype(np.float32))
                measurement_val = from_numpy(X_val[:, np.arange(1, X_val.shape[1], 3), :].astype(np.float32))
                time_val       = from_numpy(X_val[:, np.arange(2, X_val.shape[1], 3), :].astype(np.float32))
            
                mask_val = transpose(mask_val, 1, 2)
                measurement_val = transpose(measurement_val, 1, 2)
                time_val = transpose(time_val, 1, 2)
                measurement_last_obsv_val = measurement_val
            

            X_val, X_last_obsv_val, Mask_val, Delta_val, labels_val = map(convert_to_device, 
                                                                          [measurement_val, measurement_last_obsv_val, mask_val, time_val, labels_val])
                
            model.zero_grad()
            
#             outputs_val = model(inputs_val)
            prediction_val = model(X_val, X_last_obsv_val, Mask_val, Delta_val)
    
#             print(labels.shape)
#             print(prediction_val.shape)
            
            # if output_last:
            loss_valid =loss_CEL(squeeze(prediction_val), squeeze(labels_val.long()))
            # else:
                # raise NotImplementedError("Should be output last!")
                # full_labels_val = torch.cat((inputs_val[:,1:,:], labels_val), dim = 1)
                # loss_valid = loss_MSE(outputs_val, full_labels_val)

            losses_valid.append(loss_valid.data)
            losses_epoch_valid.append(loss_valid.data)
            
#             print(sklearn.metrics.roc_auc_score(labels_val.detach().cpu().numpy(), prediction_val.detach().cpu().numpy()[:,1]))
            
            # output
            trained_number += 1
            
        avg_losses_epoch_train = sum(losses_epoch_train).cpu().numpy() / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid).cpu().numpy() / float(len(losses_epoch_valid))
        losses_epochs_train.append(avg_losses_epoch_train)
        losses_epochs_valid.append(avg_losses_epoch_valid)
        
        
        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid 
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
                    np.around(avg_losses_epoch_train, decimals=8),\
                    np.around(avg_losses_epoch_valid, decimals=8),\
                    np.around([cur_time - pre_time] , decimals=2),\
                    is_best_model) )
        pre_time = cur_time
#         if epoch==1:
#             break
    # Move the model back to the CPU
    model.to("cpu")
    with open("target/target_model.pkl", "wb") as f:
        save(model.state_dict(), f)
                
    return best_model, [losses_train, losses_valid, losses_epochs_train, losses_epochs_valid]

def predict_proba(model, dataloader):
    """
    Input:
        model: GRU-D model
        test_dataloader: containing batches of measurement, measurement_last_obsv, mask, time_, labels
    Returns:
        predictions: size[num_samples, 2]
        labels: size[num_samples]
    """
    model.eval()
    

    probabilities = []
    labels        = []
    for X, label in dataloader:
        X = X.numpy()
        mask        = from_numpy(X[:, np.arange(0, X.shape[1], 3), :].astype(np.float32))
        measurement = from_numpy(X[:, np.arange(1, X.shape[1], 3), :].astype(np.float32))
        time_       = from_numpy(X[:, np.arange(2, X.shape[1], 3), :].astype(np.float32))

        mask = transpose(mask, 1, 2)
        measurement = transpose(measurement, 1, 2)
        time_ = transpose(time_, 1, 2)
        measurement_last_obsv = measurement            

    

        X, X_last_obsv, Mask, Delta, label = map(convert_to_device, [measurement, measurement_last_obsv, mask, time_, label])

        prob = model(X, X_last_obsv, Mask, Delta)
        probabilities.append(prob.detach().cpu().data.numpy())
        labels.append(label.detach().cpu().data.numpy())

    return probabilities, labels

