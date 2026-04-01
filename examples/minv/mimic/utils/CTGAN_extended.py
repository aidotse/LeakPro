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
import torch

class CustomCTGAN(CTGAN):
    def __init__(self, 
                 embedding_dim=256, 
                 generator_dim=(256, 256), 
                 discriminator_dim=(256, 256), 
                 generator_lr = 2e-4,
                 generator_decay=1e-6,
                 discriminator_lr=2e-5,
                 discriminator_decay=1e-7,
                 num_classes=5088,
                 batch_size=500,
                 discriminator_steps=5,
                 log_frequency=True, 
                 verbose=False, 
                 epochs=300, 
                 pac=10,
                 only_pseudo_label_conditioning=True, 
                 cuda=True):
        
        self.dim_z = embedding_dim

        super().__init__(embedding_dim, generator_dim, discriminator_dim, 
                         generator_lr, generator_decay, discriminator_lr, 
                         discriminator_decay, batch_size, discriminator_steps, 
                         log_frequency, verbose, epochs, pac, cuda)
        self.gumbel_seed = 1234
        self._transformer = DataTransformer()

        self.only_psuedo_label_conditioning = only_pseudo_label_conditioning
        
    def to(self, device):
        self._device = device
        self._generator.to(device)

    
    def eval(self):
        self._generator.eval()
        
    def __call__(self, z=None, y=None, df_conv =None):
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
    
        if y is not None:
            # TODO: If desired, support for conditioning on other discrete 
            # columns than pseudo-labels can be implemented here for correct sampling.

            # Batch size is length of y
            bs = y.shape[0]
            condition_values = y.detach().cpu().numpy()
        
            discrete_column_id = np.array([self._data_sampler._n_discrete_columns-1]*bs)
            cond = np.zeros((bs, self._data_sampler._n_categories), dtype='float32')
            category_id = self._data_sampler._discrete_column_cond_st[discrete_column_id] + condition_values
            cond[np.arange(bs), category_id] = 1
            c1 = cond
            
        else:
            bs = z.shape[0]
            c1 = self._data_sampler.sample_original_condvec(bs)

        c1 = torch.from_numpy(c1).to(self._device)
        
        # if z is None:
        #     mean = torch.zeros(bs, self._embedding_dim)
        #     std = mean + 1
        #     fakez = torch.normal(mean=mean, std=std).to(self._device)
        # else:
        #     fakez = z
        
        
        fakez = torch.cat([z, c1], dim=1)

        fake = self._generator(fakez)
        fakeact = self._apply_activate(fake)

        # if df_conv:
        samples = self._transformer.inverse_transform(fakeact.detach().cpu().numpy())
        # else:
        #     self._transformer_only_con = DataTransformerContinuousOnly() 
        #     samples_dis = self._transformer_only_con.inverse_transform_continuous_only(fakeact)

        '''
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
            if z is None:
                mean = torch.zeros(bs, self._embedding_dim)
                std = mean + 1
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

        '''    
        return samples
    
    def sample_condvec(self, batch):
        """Generate the conditional vector for training. Supports conditioning on all discrete columns or only the pseudo label column.
        Args:
            batch (int):
                The batch size.
            only_pseudo_label_conditioning (bool):
                If True, only sample the pseudo label column.
                If False, sample all discrete columns.
        
        Returns:
            cond (batch x #categories):
                The conditional vector.
            mask (batch x #discrete columns):
                A one-hot vector indicating the selected discrete column.
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
        """
        if self._data_sampler._n_discrete_columns == 0:
            return None

        if self.only_psuedo_label_conditioning:
            discrete_column_id = np.array([self._data_sampler._n_discrete_columns-1]*batch)
        else:
            discrete_column_id = np.random.choice(np.arange(self._data_sampler._n_discrete_columns), batch)
        
        cond = np.zeros((batch, self._data_sampler._n_categories), dtype='float32')
        mask = np.zeros((batch, self._data_sampler._n_discrete_columns), dtype='float32')
        mask[np.arange(batch), discrete_column_id] = 1
        category_id_in_col = self._data_sampler._random_choice_prob_index(discrete_column_id)
        category_id = self._data_sampler._discrete_column_cond_st[discrete_column_id] + category_id_in_col
        cond[np.arange(batch), category_id] = 1
        
        return cond, mask, discrete_column_id, category_id_in_col

 
    def fit(self, train_data, target_model, num_classes, inv_criterion, gen_criterion, dis_criterion, n_iter, n_dis, alpha = 0.1, discrete_columns=(), use_inv_loss=True):
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
        epochs = n_iter
        print("Alpha",alpha)
        
        # --- setup once ---
        self._validate_discrete_columns(train_data, discrete_columns)
        self._validate_null_data(train_data, discrete_columns)
        mat = self._prepare_training(train_data, discrete_columns, num_classes)

        # target LightningModule (pytorch_tabular)
        mdl = target_model.model if hasattr(target_model, "model") else target_model
        mdl = mdl.to(self._device).eval()

        self.loss_values = pd.DataFrame(columns=['Epoch','Generator Loss','Discriminator Loss','Inversion Loss','Conditioning Loss (CE)','Accuracy'])
        steps_per_epoch = max(len(mat) // self._batch_size, 1)
        print("step per epoch", steps_per_epoch)

        for epoch in tqdm(range(n_iter), disable=(not self._verbose)):
            for _ in range(steps_per_epoch):

                # --- D updates ---
                for _ in range(n_dis):
                    fakez, real, c1, m1, c2 = self._sample_batch(mat)
                    fake   = self._generator(fakez)
                    fakeact= self._apply_activate(fake)
                    loss_d = self._d_step(fakeact, real, c1, c2)

                # --- G update ---
                fakez, _, c1, m1, _ = self._sample_batch(mat)
                fake    = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                y_fake  = self._discriminator(torch.cat([fakeact, c1], dim=1) if c1 is not None else fakeact)

                # pack inputs for GANDALF (pure torch)
                x_cont, x_cat, pseudo = self._pack_for_gandalf(fakeact)
                batch = {}
                if x_cont is not None: batch["continuous"]  = x_cont.float().to(self._device)
                if x_cat  is not None: batch["categorical"] = x_cat.long().to(self._device)
                T_logits = mdl(batch)["logits"]

                # Inversion loss: only if we're using it AND we sampled a condition (c1)
                if use_inv_loss and c1 is not None:
                    inv_loss = inv_criterion(T_logits, pseudo.to(self._device))
                else:
                    inv_loss = torch.zeros((), device=self._device, dtype=y_fake.dtype)

                # Conditioning loss (CTGAN): only if we sampled a condition
                if c1 is not None:
                    ce_loss = self._cond_loss(fake, c1, m1)
                else:
                    ce_loss = torch.zeros((), device=self._device, dtype=y_fake.dtype)


                loss_g   = gen_criterion(y_fake) + ce_loss
                loss_all = loss_g + alpha * inv_loss
              
                # Log once in a while
                if (epoch % 100 == 0):  # or every N steps
                    # ---- sanity check: does inv_loss contribute gradients to G? ----
                    # 1) Grad norm of G params with the inversion term
                    grads_with = torch.autograd.grad(
                        loss_all,                      # includes alpha * inv_loss
                        self._generator.parameters(),
                        retain_graph=True,
                        allow_unused=True
                    )
                    gn_with = torch.sqrt(sum((g.detach()**2).sum() for g in grads_with if g is not None)).item()

                    # 2) Grad norm of G params without the inversion term (alpha=0)
                    grads_wo = torch.autograd.grad(
                        loss_g,                        # just gen + CE (no inversion)
                        self._generator.parameters(),
                        retain_graph=True,
                        allow_unused=True
                    )
                    gn_wo = torch.sqrt(sum((g.detach()**2).sum() for g in grads_wo if g is not None)).item()

                    # (Optional) also see if inv_loss is connected at all
                    assert inv_loss.requires_grad, "inv_loss is detached from the graph."
                    # -----------------------------------------------------------------


                    print(f"[sanity] gen grad WITH inv: {gn_with:.3e} | WITHOUT inv: {gn_wo:.3e}")


                self.optG.zero_grad(set_to_none=False)
                loss_all.backward()
                self.optG.step()


            # --- metrics/logging from tensors ---
            with torch.no_grad():
                preds = T_logits.argmax(dim=1)
                acc   = (preds == pseudo).float().mean().item()

            row = dict(Epoch=epoch,
                    **{"Generator Loss": loss_g.detach().cpu().item(),
                        "Discriminator Loss": loss_d.detach().cpu().item(),
                        "Inversion Loss": inv_loss.detach().cpu().item() if use_inv_loss else 0.0,
                        "Conditioning Loss (CE)": ce_loss.detach().cpu().item() if c1 is not None else 0.0,
                        "Accuracy": acc})
            self.loss_values = pd.concat([self.loss_values, pd.DataFrame([row])], ignore_index=True)

            if (epoch % 50 == 0):
                tqdm.write(f"Gen {row['Generator Loss']:.2f} | Dis {row['Discriminator Loss']:.2f} | "
                        f"Inv {row['Inversion Loss']:.2f} | CE {row['Conditioning Loss (CE)']:.2f} | "
                        f"Acc {acc:.2f}")
            

    def build_schema_and_params(self, transformer, device, weight_threshold=0.005):
        import numpy as np
        import torch
        import warnings

        def _find_bgm_model(host):
            """Return an object that has sklearn-like attrs: means_, covariances_ or precisions_cholesky_, weights_."""
            # direct hit?
            for cand in (host, getattr(host, "_bgm_transformer", None)):
                if cand is None:
                    continue
                # common inner names
                for name in ("model", "_model", "bgm", "_bgm", "gmm", "_gmm"):
                    m = getattr(cand, name, None)
                    if m is not None and (
                        hasattr(m, "means_") or hasattr(m, "covariances_") or hasattr(m, "precisions_cholesky_")
                    ):
                        return m
                # sometimes the transformer itself is the sklearn model
                if hasattr(cand, "means_") or hasattr(cand, "covariances_") or hasattr(cand, "precisions_cholesky_"):
                    return cand
            return None

        def _get_valid_mask_from_attr(gm):
            for name in ("valid_component_indicator", "_valid_component_indicator"):
                if hasattr(gm, name):
                    return np.asarray(getattr(gm, name), dtype=bool).reshape(-1)
            return None

        def _get_weights(gm):
            m = _find_bgm_model(gm)
            if m is None:
                return None
            for n in ("weights_", "_weights", "mixing_weights_"):
                if hasattr(m, n):
                    w = getattr(m, n)
                    return np.asarray(w, dtype="float32").reshape(-1)
            return None

        def _extract_gmm_params(gm, expected_K=None):
            """
            Return (means, stds) as float32 arrays, after applying either:
            - gm.valid_component_indicator (if present), else
            - weights_ > weight_threshold (if weights exist)
            Finally, ensure the number of kept components matches expected_K (β dim).
            """
            means = stds = None

            # Older RDT shallow attrs
            for n in ("_means", "means_", "means"):
                if hasattr(gm, n):
                    means = getattr(gm, n)
                    break
            for n in ("_stds", "stds_", "stds"):
                if hasattr(gm, n):
                    stds = getattr(gm, n)
                    break

            # Newer: pull from inner BGMM/GMM
            if means is None or stds is None:
                m = _find_bgm_model(gm)
                if m is not None:
                    if hasattr(m, "means_"):
                        means = np.asarray(m.means_).reshape(-1)
                    if hasattr(m, "covariances_"):
                        cov = np.asarray(m.covariances_)
                        if cov.ndim == 3:
                            var = cov[:, 0, 0]
                        else:
                            var = cov.reshape(-1)
                        stds = np.sqrt(var)
                    elif hasattr(m, "precisions_cholesky_"):
                        pc = np.asarray(m.precisions_cholesky_).reshape(-1)
                        var = 1.0 / (pc ** 2)
                        stds = np.sqrt(var)

            if means is None or stds is None:
                raise AttributeError(
                    "Could not locate GMM means/stds on ClusterBasedNormalizer. "
                    f"Available attrs on inner model: {sorted([a for a in dir(getattr(gm,'_bgm_transformer',gm)) if not a.startswith('__')])}"
                )

            means = np.asarray(means, dtype="float32").reshape(-1)
            stds  = np.asarray(stds,  dtype="float32").reshape(-1)

            # 1) Prefer explicit valid mask if present
            valid = _get_valid_mask_from_attr(gm)
            # 2) Else derive from weights_ and threshold (if weights exist)
            if valid is None and weight_threshold is not None and weight_threshold > 0.0:
                weights = _get_weights(gm)
                if weights is not None:
                    K = min(len(weights), len(means), len(stds))
                    weights = weights[:K]; means = means[:K]; stds = stds[:K]
                    valid = weights > float(weight_threshold)

            # Apply mask if we have one
            if valid is not None:
                K = min(valid.size, means.size, stds.size)
                valid = valid[:K]
                means = means[:K][valid]
                stds  = stds[:K][valid]

            # Align to expected_K (the β softmax size from output_info_list)
            if expected_K is not None:
                if len(means) > expected_K:
                    # Too many kept; keep the top-expected_K by weight if possible, else trim head
                    weights = _get_weights(gm)
                    if weights is not None and len(weights) >= len(means):
                        # Build an index set of the currently kept comps
                        if valid is not None:
                            kept_idx = np.flatnonzero(valid[:len(weights)])
                        else:
                            kept_idx = np.arange(len(means))  # already trimmed to len(means)
                        # sort kept by descending weight
                        sub_w = weights[kept_idx]
                        order = kept_idx[np.argsort(-sub_w)][:expected_K]
                        means = means[order - kept_idx.min()] if kept_idx.min() != 0 else means[order]
                        stds  = stds[ order - kept_idx.min()] if kept_idx.min() != 0 else stds[order]
                    else:
                        warnings.warn(
                            f"Kept {len(means)} components but β span expects {expected_K}; trimming to first {expected_K}."
                        )
                        means = means[:expected_K]
                        stds  = stds[:expected_K]
                elif len(means) < expected_K:
                    # Not enough kept; warn and fall back to using the top-weighted components to reach expected_K
                    warnings.warn(
                        f"Kept {len(means)} components but β span expects {expected_K}; "
                        f"padding by selecting top-weighted components."
                    )
                    m = _find_bgm_model(gm)
                    weights = _get_weights(gm)
                    if (weights is not None) and (m is not None):
                        # choose top expected_K overall, aligned with original order
                        K_all = min(len(weights), len(getattr(m, "means_", means)))
                        idx = np.argsort(-weights[:K_all])[:expected_K]
                        all_means = np.asarray(getattr(m, "means_", means)).reshape(-1)[:K_all].astype("float32")
                        if hasattr(m, "covariances_"):
                            cov = np.asarray(m.covariances_)[:K_all]
                            var = cov[:, 0, 0] if cov.ndim == 3 else cov.reshape(-1)
                            all_stds = np.sqrt(var).astype("float32")
                        else:
                            all_stds = stds  # best effort
                        means = all_means[idx]
                        stds  = all_stds[idx]
                    else:
                        # last resort: tile last component
                        means = np.pad(means, (0, expected_K - len(means)), mode="edge")
                        stds  = np.pad(stds,  (0, expected_K - len(stds)),  mode="edge")

            return means, stds

        schema = []
        st = 0
        for info in transformer._column_transform_info_list:
            dim = info.output_dimensions  # for continuous: 1 + K ; for categorical: #cats
            entry = {"name": info.column_name, "type": info.column_type, "st": st, "dim": dim}
            if info.column_type == "continuous":
                gm = info.transform  # ClusterBasedNormalizer
                # expected_K = size of β softmax block for this column
                expected_K = dim - 1
                mu_np, sigma_np = _extract_gmm_params(gm, expected_K=expected_K)
                entry["mu"]         = torch.as_tensor(mu_np,    dtype=torch.float32, device=device)   # [K]
                entry["sigma"]      = torch.as_tensor(sigma_np, dtype=torch.float32, device=device)   # [K]
                entry["alpha_idx"]  = st
                entry["beta_slice"] = slice(st + 1, st + dim)  # K-wide softmax block
            else:
                entry["onehot_slice"] = slice(st, st + dim)
            schema.append(entry)
            st += dim

        # ensure pseudo_label exists
        assert any(c["type"] == "discrete" and c["name"] == "pseudo_label" for c in schema), \
            "pseudo_label span not found in transformer schema"

        return schema




    def _prepare_training(self, train_df, discrete_columns, num_classes):
        assert train_df.columns[-1] == "pseudo_label"
        self._transformer.fit(train_df, discrete_columns)

        # schema with GMM params and span indices
        self._schema = self.build_schema_and_params(self._transformer, self._device)

        # (optional) cont scaler to match target training
        cont_cols = [c for c in train_df.columns if c != "pseudo_label" and c not in discrete_columns]
        self._cont_mean = torch.tensor(train_df[cont_cols].mean().to_numpy(), dtype=torch.float32, device=self._device)
        self._cont_std  = torch.tensor(train_df[cont_cols].std(ddof=0).replace(0,1).to_numpy(), dtype=torch.float32, device=self._device)

        mat = self._transformer.transform(train_df)  # ndarray
        self._data_sampler = DataSampler(mat, self._transformer.output_info_list, self._log_frequency)
        data_dim = self._transformer.output_dimensions

        self._generator = Generator(self._embedding_dim + self._data_sampler.dim_cond_vec(),
                                    self._generator_dim, data_dim).to(self._device)
        self._discriminator = Discriminator(data_dim + self._data_sampler.dim_cond_vec(),
                                            self._discriminator_dim, pac=self.pac).to(self._device)

        self.optG = optim.Adam(self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
                            weight_decay=self._generator_decay)
        self.optD = optim.Adam(self._discriminator.parameters(), lr=self._discriminator_lr, betas=(0.5, 0.9),
                            weight_decay=self._discriminator_decay)

        self._z_mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        self._z_std  = self._z_mean + 1
        self.num_classes = num_classes
        return mat  # transformed train matrix

    def _sample_batch(self, mat):
        """Sample real and condition, prepare fakez (+cond), all on device."""
        fakez = torch.normal(mean=self._z_mean, std=self._z_std)
        condvec = self.sample_condvec(self._batch_size)
        if condvec is None:
            c1 = m1 = col = opt = None
            real = self._data_sampler.sample_data(mat, self._batch_size, col, opt)
            c2 = None
        else:
            c1, m1, col, opt = condvec
            c1 = torch.from_numpy(c1).to(self._device)
            m1 = torch.from_numpy(m1).to(self._device)
            fakez = torch.cat([fakez, c1], dim=1)
            perm = np.random.permutation(self._batch_size)
            real = self._data_sampler.sample_data(mat, self._batch_size, col[perm], opt[perm])
            c2 = c1[perm]
        real = torch.from_numpy(real.astype("float32")).to(self._device)
        return fakez, real, c1, m1, c2
    
    def _d_step(self, fakeact, real, c1, c2):
        """One discriminator update (WGAN-GP)."""
        if c1 is not None:
            fake_cat = torch.cat([fakeact, c1], dim=1)
            real_cat = torch.cat([real,   c2], dim=1)
        else:
            fake_cat, real_cat = fakeact, real
        y_fake = self._discriminator(fake_cat)
        y_real = self._discriminator(real_cat)
        pen = self._discriminator.calc_gradient_penalty(real_cat, fake_cat, self._device, self.pac)
        loss_d = -(torch.mean(y_real) - torch.mean(y_fake))  # WGAN loss
        self.optD.zero_grad(set_to_none=False)
        pen.backward(retain_graph=True)
        loss_d.backward()
        self.optD.step()
        return loss_d

    def _atanh_clamped(self, a, eps=1e-6):
        a = a.clamp(min=-1+eps, max=1-eps)
        return 0.5 * (torch.log1p(a) - torch.log1p(-a))

    def _invert_continuous_columns_torch(self, fakeact, scale=4.0):
        xs = []
        for col in self._schema:
            if col["type"] != "continuous": continue
            alpha = fakeact[:, col["alpha_idx"]:col["alpha_idx"]+1]
            beta  = fakeact[:, col["beta_slice"]]
            mu    = col["mu"].view(1, -1)
            sigma = col["sigma"].view(1, -1)
            # u     = self._atanh_clamped(alpha)
            u     = torch.tanh(alpha)
            x_k   = mu + scale * sigma * u
            x     = (beta * x_k).sum(dim=1, keepdim=True)
            xs.append(x)
        return torch.cat(xs, dim=1) if xs else None

    def _pack_for_gandalf(self, fakeact):
        """Build pytorch-tabular inputs: continuous tensor, categorical indices, and pseudo labels."""
        # pseudo_label from its one-hot
        pl = next(c for c in self._schema if c["type"]=="discrete" and c["name"]=="pseudo_label")
        pseudo = fakeact[:, pl["st"]:pl["st"]+pl["dim"]].argmax(dim=1).long()  # [B]

        # continuous raw (optionally standardize to match target training)
        x_cont = self._invert_continuous_columns_torch(fakeact, scale=4.0)
        # if x_cont is not None and (getattr(self, "_cont_mean", None) is not None) and (getattr(self, "_cont_std", None) is not None):
        #     x_cont = (x_cont - self._cont_mean) / (self._cont_std + 1e-6)

        # categoricals as indices (excluding pseudo_label)
        idxs = []
        for col in self._schema:
            if col["type"]=="discrete" and col["name"]!="pseudo_label":
                idxs.append(fakeact[:, col["onehot_slice"]].argmax(dim=1))
        x_cat = torch.stack(idxs, dim=1) if idxs else None
        return x_cont, x_cat, pseudo
    
    def forward_fakeact_with_labels(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Differentiable forward:
        - builds the correct one-hot condition vector for the pseudo_label column
        - concatenates with z
        - returns fakeact (after _apply_activate), still a torch tensor
        Shapes:
        z: [B, self._embedding_dim]   (or self.dim_z)
        y: [B] long, class ids in [0, num_classes-1]
        """
        assert hasattr(self, "_data_sampler"), "Call _prepare_training(...) before using this."

        device = self._device
        y = y.view(-1).long().to(device)
        B = y.size(0)

        # Build cond one-hot in torch for the *last* discrete column (pseudo_label)
        n_cats_total = int(self._data_sampler._n_categories)
        cond = torch.zeros(B, n_cats_total, device=device)

        # Start offset of the pseudo_label block inside the concatenated cond vector
        pl_col_id = int(self._data_sampler._n_discrete_columns - 1)
        start_idx = int(self._data_sampler._discrete_column_cond_st[pl_col_id])

        idx = start_idx + y                       # [B] long
        cond.scatter_(1, idx.view(-1, 1), 1.0)    # one-hot

        # Concatenate noise + condition and run the generator
        fakez = torch.cat([z, cond], dim=1)       # [B, embed + cond]
        fake   = self._generator(fakez)
        fakeact= self._apply_activate(fake)       # stays in torch (differentiable)

        return fakeact, cond
    
    # def _apply_activate(self, data):
    #     """Apply proper activation function to the output of the generator."""
    #     data_t = []
    #     st = 0
    #     for column_info in self._transformer.output_info_list:
    #         for span_info in column_info:
    #             if span_info.activation_fn == 'tanh':
    #                 ed = st + span_info.dim
    #                 data_t.append(torch.tanh(data[:, st:ed]))
    #                 st = ed
    #             elif span_info.activation_fn == 'softmax':
    #                 ed = st + span_info.dim
    #                 transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
    #                 data_t.append(transformed)
    #                 st = ed
    #             else:
    #                 raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

    #     return torch.cat(data_t, dim=1)
    
    def _gumbel_softmax(self,logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deterministic when `seed` is provided; otherwise unchanged behavior."""
        seed_attr = getattr(self, "gumbel_seed", None)
        if seed_attr is not None:
            # seeded path: sample Gumbel noise ourselves with a fixed generator
            gen = torch.Generator(device=logits.device)
            gen.manual_seed(int(seed_attr))
            for _ in range(10):
                U = torch.rand(logits.shape,generator=gen,device=logits.device,dtype=logits.dtype).clamp_(min=eps, max=1.0 - eps)
                g = -torch.log(-torch.log(U))
                y = torch.softmax((logits + g) / max(tau, eps), dim=dim)
                if hard:
                    idx = y.argmax(dim=dim, keepdim=True)
                    y_hard = torch.zeros_like(y).scatter_(dim, idx, 1.0)
                    y = y_hard - y.detach() + y
                if not torch.isnan(y).any():
                    return y
            raise ValueError('gumbel_softmax returning NaN (seeded).')

        # original path (non-deterministic unless you set global seeds)
        for _ in range(10):
            transformed = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')
