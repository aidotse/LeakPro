from ctgan import CTGAN
from ctgan.synthesizers.ctgan import Generator, Discriminator
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from torch import cuda, device

class CustomCTGAN(CTGAN):
    def __init__(self, 
                 embedding_dim=128, 
                 generator_dim=(256, 256), 
                 discriminator_dim=(256, 256), 
                 generator_lr=0.0002, 
                 generator_decay=0.000001, 
                 discriminator_lr=0.0002, 
                 discriminator_decay=0.000001,
                 num_classes=5088,
                 batch_size=1000, 
                 discriminator_steps=1, 
                 log_frequency=True, 
                 verbose=False, 
                 epochs=300, 
                 pac=100, 
                 cuda=True):
        
        self.dim_z = embedding_dim

        super().__init__(embedding_dim, generator_dim, discriminator_dim, 
                         generator_lr, generator_decay, discriminator_lr, 
                         discriminator_decay, batch_size, discriminator_steps, 
                         log_frequency, verbose, epochs, pac, cuda)
        
        self._transformer = DataTransformer()
        
    def to(self, device):
        self._device = device
        self._generator.to(device)

    
    def eval(self):
        self._generator.eval()

        
    def __call__(self, z=None, y=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if self._transformer is None:
            raise ValueError("The transformer has not been initialized. Please call the `fit` method first.")

        condition_column = "pseudo_label"
        condition_values = y.detach().cpu().numpy()

        # TODO: Check this logic
        condition_values = condition_values[:1]
        
        if y is not None:
            # Batch size is length of y
            bs = y.shape[0]
        else:
            bs = self._batch_size

        samples = pd.DataFrame()
        
        for condition_value in condition_values:        
            if condition_column is not None and condition_value is not None:
                try:
                    condition_info = self._transformer.convert_column_name_value_to_id(
                        condition_column, condition_value
                    )
                    global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                        condition_info, bs
                    )
                except ValueError:
                    # If the transformer has not seen the condition value in training, it will raise a ValueError
                    # We still want to be able to sample, so we set the global_condition_vec to None
                    global_condition_vec = None
            else:
                global_condition_vec = None

            data = []
            mean = torch.zeros(bs, self._embedding_dim)
            std = mean + 1
            if z is None:
                fakez = torch.normal(mean=mean, std=std).to(self._device)
            else:
                fakez = z
            
            
            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(bs)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

            data = np.concatenate(data, axis=0)

            sample = self._transformer.inverse_transform(data)
            # add row to samples
            samples = pd.concat([samples, sample])
        return samples

    def fit(self, train_data, target_model, num_classes, inv_criterion, gen_criterion, dis_criterion, alpha = 0.1, discrete_columns=(), use_inv_loss=True):
        """
        Fit the CTGAN model to the training data using pseudo-labeled guidance as in the PLG-MI attack.

        Args:
            train_data (pandas.DataFrame):
                Training data.
            target_model (torch.nn.Module):
                Target model.
            num_classes (int):
                Number of classes.
            inv_criterion (callable):
                Inversion criterion.
            gen_criterion (callable):
                Generator criterion.
            dis_criterion (callable):
                Discriminator criterion.
            alpha (float):
                Alpha value for the inversion loss.
            discrete_columns (list of str):
                List of column names that are discrete.
        """

        epochs = self._epochs
        self._validate_discrete_columns(train_data, discrete_columns)
        self._validate_null_data(train_data, discrete_columns)
        
        #print(train_data["pseudo_label"].unique().shape)
        
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )
        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss', 'Inversion Loss'])
        

        def sample_pseudo_c1(batch_size):
            """
            sample conditional vector c1 from the pseudo-labels
            """
                
            condition_column = "pseudo_label"
            condition_value = np.random.randint(0, num_classes)

            try:
                warnings.filterwarnings('ignore') 

                condition_info = self._transformer.convert_column_name_value_to_id(
                    condition_column, condition_value
                )
                
                global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                    condition_info, batch_size
                )

            except ValueError:
                # If the transformer has not seen the condition value in training, it will raise a ValueError
                # We still want to be able to sample, so we set the global_condition_vec to None
                global_condition_vec = None

            return global_condition_vec
        

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f}) | Inv. ({inv:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0, inv=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    
                    # TODO: Only condition on pseudo-labels
                    
                    fakez = torch.normal(mean=mean, std=std)

                    """
                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]
                    """
                    # Sample conditional vector c1 from the pseudo-labels
                    c1 = sample_pseudo_c1(self._batch_size)
                    
                    if c1 is None:                   
                        c1 = self._data_sampler.sample_original_condvec(self._batch_size)

                    c1 = torch.from_numpy(c1).to(self._device)
                    
                    fakez = torch.cat([fakez, c1], dim=1)
                    
                    # Sample real data
                    real = self._data_sampler.sample_data(train_data, self._batch_size, col=None, opt=None)
                    
                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c1], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact
                    
                    
                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    # TODO: Maybe change this loss
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                
                c1 = sample_pseudo_c1(self._batch_size)
                
                if c1 is None:                   
                    c1 = self._data_sampler.sample_original_condvec(self._batch_size)
                
                c1 = torch.from_numpy(c1).to(self._device)
                
                fakez = torch.cat([fakez, c1], dim=1)
                    

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                # Pseudo label batch is last column of fakeact
                pseudo_label_batch = fakeact[:, -1].long()

                # Fake feature vector
                fakefeat = fakeact
                
                fakefeat = fakefeat.detach().cpu().numpy()
                     
                sample = self._transformer.inverse_transform(fakefeat)
                
                #remove 'pseudo_label' column
                sample = sample.drop(columns=['pseudo_label'])                
                

                if condvec is None or not use_inv_loss:
                    inv_loss = 0
                else:
                    inv_loss  = inv_criterion(target_model(sample), pseudo_label_batch)

                loss_g = gen_criterion(y_fake)
                loss_all = loss_g + inv_loss*alpha

                optimizerG.zero_grad(set_to_none=False)
                loss_all.backward()
                optimizerG.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()
            if use_inv_loss:
                inversion_loss = inv_loss.detach().cpu().item()
            else:
                inversion_loss = 0

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
                'Inversion Loss': [inversion_loss]
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss, inv=inversion_loss)
                )
                
    def save(self, path):
        """Save the model to the specified path."""
        torch.save(self._generator.state_dict(), path)
        #self._transformer.save('transformer_' + path)
        #self.loss_values.to_csv(path + '/loss_values.csv', index=False)
