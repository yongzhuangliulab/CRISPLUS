""" 
The drug perturbation encoding module is adapted from chemCPA
"""

import json
import logging
from collections import OrderedDict
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from CRISPLUS.losses import MMDloss, AFMSELoss, loss_adapt

def _move_inputs(*inputs, device="cuda"):
    def mv_input(x):
        if x is None:
            return None
        elif isinstance(x, torch.Tensor):
            return x.to(device)
        else:
            return [mv_input(y) for y in x]

    return [mv_input(x) for x in inputs]

class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(
        self,
        sizes,
        dropout,
        batch_norm=True,
        last_layer_act="linear",
    ):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ]

        layers = [l for l in layers if l is not None][:-2]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        layers_dict = OrderedDict(
            {str(i): module for i, module in enumerate(layers)}
        )

        self.network = torch.nn.Sequential(layers_dict)

    def forward(self, x):
        if self.activation == "ReLU":
            x = self.network(x)
            return self.relu(x)
        return self.network(x)

class PertAE(torch.nn.Module):

    def __init__(
        self,
        num_genes: int, # number of genes
        num_drugs: int, # the number of drugs in whole dataset, including test and ood
        num_celltypes: int, # the number of cell types in training dataset
        num_covariates: int, # the number of each covariate expect from cell type in whole dataset, could be [0]
        drug_embeddings=None, # initialized drug embeddings, if None, model will randomly initialize it
        mmd_co=None, # coefficient of mmd loss item
        celltype_co=None, # coefficient of celltype-specific loss (contrastive learning & cell type classification loss)
        device="cpu",
        seed=0,
        hparams="",
        FM_ndim = 512,
    ):
    
        super(PertAE, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.num_genes = num_genes
        self.num_drugs = num_drugs
        self.num_covariates = num_covariates 
        self.num_celltypes = num_celltypes
        self.device = device
        self.FM_ndim = FM_ndim
        
        if self.num_covariates==[0]:
            self.num_latents = int(2) # drug, genes, celltype
        else:
            self.num_latents = int(len(self.num_covariates)+2)

        # set hyperparameters
        self.set_hparams_(hparams)
        logging.info(f"Effective flow_steps: {self.hparams.get('flow_steps')}")
        if mmd_co is not None:
            self.hparams['mmd'] = mmd_co 
        if celltype_co is not None:
            self.hparams['celltype'] = celltype_co

        # store the variables used for initialization (allows restoring model later).
        self.init_args = {
            "num_genes": num_genes,
            "num_drugs": num_drugs,
            "num_covariates": num_covariates,
            "num_celltypes": num_celltypes,
            "FM_ndim": FM_ndim,
            "hparams": hparams,
        }

        self.encoder_FM = MLP(
            [self.FM_ndim]
            + [self.hparams["encoder_width"]] * self.hparams["encoder_depth"]
            + [self.hparams["lat_dim"]*2],
            dropout=self.hparams['dropout'],
        )

        self.decoder = MLP(
            [self.hparams["lat_dim"] * self.num_latents]
            + [self.hparams["decoder_width"]] * self.hparams["decoder_depth"]
            + [num_genes],
            dropout=self.hparams['dropout'],
            last_layer_act='ReLU',
        )

        cond_dim = self.hparams["lat_dim"] * (self.num_latents - 1)

        self.time_encoder = MLP(
            [1, 64, self.hparams["lat_dim"]],
            dropout=self.hparams["dropout"],
            batch_norm=False,
        )

        self.condition_projector = MLP(
            [cond_dim]
            + [self.hparams["encoder_width"]]
            + [self.hparams["lat_dim"]],
            dropout=self.hparams["dropout"],
            batch_norm=True,
        )

        self.velocity_net = MLP(
            [self.hparams["lat_dim"] * 3]
            + [self.hparams["decoder_width"]] * self.hparams["decoder_depth"]
            + [self.hparams["lat_dim"]],
            dropout=self.hparams["dropout"],
            batch_norm=True,
        )
        
        last_linear = None
        for m in self.velocity_net.network.modules():
            if isinstance(m, nn.Linear):
                last_linear = m

        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

        # if we initialize cell type embedding with FM-encoded embs
        if self.num_celltypes>0:
            self.ct_predictor = MLP(
                [self.hparams["lat_dim"] * self.num_latents]
                + [128] 
                + [self.num_celltypes],
                dropout=self.hparams['dropout'],
            )
        else: 
            self.ct_predictor = None
    
        # initialize drug embeddings from rdkit model        
        if self.num_drugs > 0:
            if drug_embeddings is None:
                self.drug_embeddings = torch.nn.Embedding(
                    self.num_drugs, self.hparams["lat_dim"]
                )
                embedding_requires_grad = True
            else:
                self.drug_embeddings = drug_embeddings
                embedding_requires_grad = False

            self.drug_embedding_encoder = MLP(
                [self.drug_embeddings.embedding_dim]
                + [self.hparams["embedding_encoder_width"]]
                * self.hparams["embedding_encoder_depth"]
                + [self.hparams["lat_dim"]],
                dropout=self.hparams['dropout'],
                last_layer_act="linear",
            )

            self.dosers = MLP(
                [self.drug_embeddings.embedding_dim + 1]
                + [self.hparams["dosers_width"]] * self.hparams["dosers_depth"]
                + [1],
                dropout=self.hparams['dropout'],
            )

        # randomly initialize other covariates except from cell type
        if self.num_covariates == [0]:
            pass
        else:
            assert 0 not in self.num_covariates

            self.covariates_embeddings = (
                []
            ) 
            for num_covariate in self.num_covariates:
                self.covariates_embeddings.append(
                    torch.nn.Embedding(num_covariate, self.hparams["dim"])
                )
            cov_emb_grad = True

        # self.loss_autoencoder = torch.nn.GaussianNLLLoss()
        self.loss_afmse = AFMSELoss()
        self.loss_mse = torch.nn.MSELoss()
        self.loss_cell_pred = torch.nn.CrossEntropyLoss()
        self.contrloss = nn.CosineEmbeddingLoss()
        # self.loss_mmd = SamplesLoss(loss='gaussian',blur=1).to(self.device)
        self.iteration = 0
        self.to(self.device)

        # optimizers
        has_drugs = self.num_drugs > 0
        has_covariates = self.num_covariates[0] > 0
        get_params = lambda model, cond: list(model.parameters()) if cond else []
        _parameters = (
            get_params(self.encoder_FM, True)
            + get_params(self.decoder, True)
            + get_params(self.drug_embeddings, has_drugs and embedding_requires_grad)
            + get_params(self.drug_embedding_encoder, has_drugs)
            + get_params(self.time_encoder, True)
            + get_params(self.condition_projector, True)
            + get_params(self.velocity_net, True)
        )

        if self.num_covariates != [0]:
            for emb in self.covariates_embeddings:
                _parameters.extend(get_params(emb, has_covariates and cov_emb_grad))
    
        if self.ct_predictor is not None:
            cell_parameters=(get_params(self.ct_predictor,True))
                
        self.optimizer_autoencoder = torch.optim.Adam(
            _parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["wd"],
        )

        # ===== DEBUG model.py / __init__: optimizer parameter check =====
        num_velocity_params = sum(p.numel() for p in self.velocity_net.parameters())
        num_time_params = sum(p.numel() for p in self.time_encoder.parameters())
        num_condition_params = sum(p.numel() for p in self.condition_projector.parameters())

        logging.info(f"velocity_net params: {num_velocity_params}")
        logging.info(f"time_encoder params: {num_time_params}")
        logging.info(f"condition_projector params: {num_condition_params}")

        opt_param_ids = set()
        for group in self.optimizer_autoencoder.param_groups:
            for p in group["params"]:
                opt_param_ids.add(id(p))

        assert all(id(p) in opt_param_ids for p in self.velocity_net.parameters()), \
            "velocity_net parameters are not in optimizer_autoencoder"

        assert all(id(p) in opt_param_ids for p in self.time_encoder.parameters()), \
            "time_encoder parameters are not in optimizer_autoencoder"

        assert all(id(p) in opt_param_ids for p in self.condition_projector.parameters()), \
            "condition_projector parameters are not in optimizer_autoencoder"

        self.optimizer_cell = torch.optim.Adam(
            cell_parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["cell_wd"],
        )

        if has_drugs:
            self.optimizer_dosers = torch.optim.Adam(
                self.dosers.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["wd"],
            )

        # learning rate schedulers
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder,
            step_size=self.hparams["step_size_lr"],
            gamma=0.5,
        )
        self.scheduler_cell = torch.optim.lr_scheduler.StepLR(
            self.optimizer_cell,
            step_size=self.hparams["step_size_lr"],
            gamma=0.5,
        )

        if has_drugs:
            self.scheduler_dosers = torch.optim.lr_scheduler.StepLR(
                self.optimizer_dosers,
                step_size=self.hparams["step_size_lr"],
                gamma=0.5,
            )

        self.history = {"epoch": [], "stats_epoch": []}
        
    def set_hparams_(self, hparams):
        self.hparams = {
            "lat_dim": 128,
            "dosers_width": 64,
            "dosers_depth": 3,
            "encoder_width": 256,
            "encoder_depth": 4,
            "decoder_width": 1028,
            "decoder_depth": 4,    
            "embedding_encoder_width": 128,
            "embedding_encoder_depth": 4,        
            "lr": 1e-3,
            "wd": 1e-7,
            "batch_size": 128,
            "step_size_lr": 50,
            "dropout": 0.2,
            "alpha": 0.75,
            "celltype": 1,
            "cell_wd": 0.001,
            "mmd": 0.1,
            "kld_weight": 500,
            "adapt": 0,
            "flow_steps": 8,
            "flow_co": 1.0,
            "gene_co": 1.0,
            "latent_co": 0.1,
            "kld_co": 0.0,
            "debug_predict": False,
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams

    def compute_drug_embeddings_(self, drugs_idx=None, dosages=None, drugs_pre=None):
        """
        Compute sum of drug embeddings, each of them multiplied by its dose-response curve.
        @param drugs_idx: A vector of dim [batch_size]. Each entry contains the index of the applied drug. The
            index is ∈ [0, num_drugs).
        @param dosages: A vector of dim [batch_size]. Each entry contains the dose of the applied drug.
        @return: a tensor of shape [batch_size, drug_embedding_dimension]
        """
        assert (drugs_idx is not None or drugs_pre is not None)
        if drugs_idx is not None:

            drugs_idx, dosages = _move_inputs(
                drugs_idx, dosages, device=self.device
            )

            latent_drugs = self.drug_embeddings.weight

            if len(drugs_idx.size()) == 0:
                drugs_idx = drugs_idx.unsqueeze(0)

            if len(dosages.size()) == 0:
                dosages = dosages.unsqueeze(0)

            assert drugs_idx.shape == dosages.shape and len(drugs_idx.shape) == 1
            # results in a tensor of shape [batchsize, drug_embedding_dimension]
            latent_drugs = latent_drugs[drugs_idx]
        
        else:
            drugs_pre, dosages = _move_inputs(
                drugs_pre, dosages, device=self.device
            )
            latent_drugs = drugs_pre

        scaled_dosages = self.dosers(
            torch.concat([latent_drugs, torch.unsqueeze(dosages, dim=-1)], dim=1)
        ).squeeze()

        # unsqueeze if batch_size is 1
        if len(scaled_dosages.size()) == 0:
            scaled_dosages = scaled_dosages.unsqueeze(0)

        # Transform and adjust dimension to latent dims
        latent_drugs = self.drug_embedding_encoder(latent_drugs)

        # scale latent vector by scalar scaled_dosage
        return torch.einsum("b,be->be", [scaled_dosages, latent_drugs])
    
    def encode_FM(self, cell_embeddings, sample=True):
        output = self.encoder_FM(cell_embeddings)

        mu = output[:, :self.hparams["lat_dim"]]
        logvar = F.relu(output[:, self.hparams["lat_dim"]:]).clamp(max=10)

        if sample:
            eps = torch.randn_like(mu)
            z = eps * torch.exp(0.5 * logvar) + mu
        else:
            z = mu

        return z, mu, logvar

    def get_condition_parts(
        self,
        drugs_idx=None,
        dosages=None,
        covariates=None,
        drugs_pre=None,
    ):
        cond_parts = []

        if self.num_drugs > 0:
            drug_embedding = self.compute_drug_embeddings_(
                drugs_idx=drugs_idx,
                dosages=dosages,
                drugs_pre=drugs_pre,
            )
            cond_parts.append(drug_embedding)

        if self.num_covariates[0] > 0:
            for cov_type, emb_cov in enumerate(self.covariates_embeddings):
                emb_cov = emb_cov.to(self.device)
                cov_idx = covariates[cov_type].argmax(1)
                cov_emb = emb_cov(cov_idx)
                cond_parts.append(cov_emb)

        return cond_parts
    
    def velocity(self, zt, t, cond_parts):
        t_emb = self.time_encoder(t)

        cond_raw = torch.cat(cond_parts, dim=1)
        cond = self.condition_projector(cond_raw)

        v_in = torch.cat([zt, t_emb, cond], dim=1)
        return self.velocity_net(v_in)

    def flow_euler(self, z0, cond_parts, n_steps=None):
        if n_steps is None:
            n_steps = self.hparams.get("flow_steps", 8)

        z = z0
        dt = 1.0 / n_steps

        for k in range(n_steps):
            t = torch.full(
                (z.shape[0], 1),
                float(k) / n_steps,
                device=z.device,
                dtype=z.dtype,
            )
            v = self.velocity(z, t, cond_parts)
            z = z + dt * v

        return z

    def decode_gene(self, z1, cond_parts):
        decoder_input = torch.cat([z1] + cond_parts, dim=1)
        gene_pred = self.decoder(decoder_input)
        return gene_pred, decoder_input

    # def predict(
    #     self,
    #     genes,
    #     cell_embeddings, # FM-encoded paired control embedding 
    #     drugs_idx=None,
    #     dosages=None,
    #     covariates=None,
    #     drugs_pre=None,
    # ):
    #     """
    #     Predict the post-perturbation gene expression profile 
    #     given paired control embedding, drugs, and cell type, etc
    #     """
    #     assert dosages is not None
    #     assert (drugs_idx is not None) or (drugs_pre is not None)
    #     genes, cell_embeddings, drugs_idx, dosages, covariates = _move_inputs(
    #         genes, cell_embeddings, drugs_idx, dosages, covariates, device=self.device
    #     )

    #     output = self.encoder_FM(cell_embeddings)
    #     mu = output[:,0:self.hparams['lat_dim']]
    #     logvar = F.relu(output[:,self.hparams['lat_dim']:]).clamp(max=10)
    #     gaussian_noise = torch.randn(mu.size(0), mu.size(1), device=self.device)
    #     latent_basal = gaussian_noise*torch.exp(logvar*0.5) + mu
    #     # latent_basal = self.encoder_FM(cell_embeddings)

    #     latent_treated = [latent_basal]
        
    #     if self.num_drugs > 0:
    #         drug_embedding = self.compute_drug_embeddings_(
    #             drugs_idx=drugs_idx, dosages=dosages, drugs_pre=drugs_pre
    #         )
    #         latent_treated.append(drug_embedding)

    #     if self.num_covariates[0] > 0:
    #         for cov_type, emb_cov in enumerate(self.covariates_embeddings):
    #             emb_cov = emb_cov.to(self.device)
    #             cov_idx = covariates[cov_type].argmax(1)
    #             cell_emb = emb_cov(cov_idx)
    #             latent_treated.append(cell_emb)

    #     latent_treated = torch.cat(latent_treated,dim=1)
    #     gene_reconstructions = self.decoder(latent_treated)

    #     return gene_reconstructions, latent_treated, mu, logvar
    def predict(
        self,
        genes,
        cell_embeddings,
        drugs_idx=None,
        dosages=None,
        covariates=None,
        drugs_pre=None,
    ):
        assert dosages is not None
        assert (drugs_idx is not None) or (drugs_pre is not None)

        genes, cell_embeddings, drugs_idx, dosages, covariates = _move_inputs(
            genes,
            cell_embeddings,
            drugs_idx,
            dosages,
            covariates,
            device=self.device,
        )

        # inference uses deterministic basal latent
        z0, mu, logvar = self.encode_FM(cell_embeddings, sample=False)

        cond_parts = self.get_condition_parts(
            drugs_idx=drugs_idx,
            dosages=dosages,
            covariates=covariates,
            drugs_pre=drugs_pre,
        )

        z1_pred = self.flow_euler(z0, cond_parts)
        gene_reconstructions, latent_treated = self.decode_gene(z1_pred, cond_parts)

        # # ===== DEBUG 8: predict path =====
        # if self.hparams.get("debug_predict", False):
        #     print("\n[DEBUG model.py / predict()]")
        #     print("cell_embeddings:", cell_embeddings.shape)
        #     print("z0:", z0.shape)
        #     print("z1_pred:", z1_pred.shape)
        #     print("latent_treated / decoder_input:", latent_treated.shape)
        #     print("gene_reconstructions:", gene_reconstructions.shape)

        return gene_reconstructions, latent_treated, mu, logvar


    # def iter_update(
    #     self,
    #     genes,
    #     cell_embeddings,
    #     # paired_mean=None,
    #     # paired_std=None,
    #     drugs_idx=None,
    #     dosages=None,
    #     degs=None,
    #     celltype_idx=None,
    #     covariates=None,
    #     neg_genes=None,
    #     neg_cell_embeddings=None,
    #     # neg_paired_mean=None,
    #     # neg_paired_std=None,
    #     neg_drugs_idx=None,
    #     neg_dosages=None,
    #     neg_degs=None,
    #     neg_celltype_idx=None,
    #     neg_covariates=None,
    # ):
    #     """
    #     Calculate loss and update parameters of model
    #     """
    #     assert drugs_idx is not None and dosages is not None

    #     gene_reconstructions, latent_treated, mu, logvar = self.predict(
    #         genes=genes,
    #         cell_embeddings=cell_embeddings,
    #         drugs_idx=drugs_idx,
    #         dosages=dosages,
    #         covariates=covariates,
    #     )
    #     neg_gene_reconstructions, neg_latent_treated, neg_mu, neg_logvar = self.predict(
    #         genes=neg_genes,
    #         cell_embeddings=neg_cell_embeddings,
    #         drugs_idx=neg_drugs_idx,
    #         dosages=neg_dosages,
    #         covariates=neg_covariates,
    #     )

    #     both_latent = torch.concatenate((latent_treated,neg_latent_treated))
    #     both_celltype = torch.concatenate((celltype_idx,neg_celltype_idx))
    #     both_genes = torch.concatenate((genes,neg_genes))
    #     both_recon = torch.concatenate((gene_reconstructions,neg_gene_reconstructions))
    #     both_degs = torch.concatenate((degs,neg_degs))
    #     # both_paired_mean = torch.concatenate((paired_mean,neg_paired_mean))
    #     # both_paired_std = torch.concatenate((paired_std,neg_paired_std))
    #     both_mu = torch.concatenate((mu,neg_mu))
    #     both_logvar = torch.concatenate((logvar,neg_logvar))

    #     kld_loss = -0.5 * (1 + both_logvar - both_mu**2 - torch.exp(both_logvar)).sum(1).mean()
        
    #     afloss = self.loss_afmse(y=both_genes,pred=both_recon,degs=both_degs)
    #     mseloss = self.loss_mse(both_genes, both_recon)
    #     mmdloss = MMDloss(both_genes, both_recon)

    #     negative_labels = -torch.ones(neg_latent_treated.size(0),device=self.device)
    #     neg_loss = self.contrloss(neg_latent_treated, latent_treated,negative_labels)
    #     # adapt = loss_adapt(pred=both_recon,true=both_genes,mean_ctrl=both_paired_mean,std_ctrl=both_paired_std)

    #     if self.num_celltypes > 0:
    #         self.ct_predictor = self.ct_predictor.to(self.device)
    #         ct_predictions = self.ct_predictor(both_latent)
    #         ct_pred_loss = self.loss_cell_pred(
    #             ct_predictions, both_celltype
    #         )

    #     alpha = self.hparams['alpha']
    #     reconstruction_loss = mseloss * alpha + afloss * (1-alpha)
    #     kld_weight = 1 / (self.hparams['kld_weight'] * both_mu.shape[1])
    #     loss = reconstruction_loss + (ct_pred_loss + neg_loss*0.1) * self.hparams['celltype'] + mmdloss*self.hparams['mmd'] + kld_loss*kld_weight

    #     self.optimizer_autoencoder.zero_grad()
    #     self.optimizer_cell.zero_grad()
    #     if self.num_drugs > 0:
    #         self.optimizer_dosers.zero_grad()

    #     loss.backward()
    #     self.optimizer_autoencoder.step()
    #     self.optimizer_cell.step()
    #     if self.num_drugs > 0:
    #         self.optimizer_dosers.step()
    #     self.iteration += 1

    #     return {
    #         "loss": loss.item(),
    #         "autofocus_loss": afloss.item(),
    #         "mmd_loss": mmdloss.item(),         
    #         "mse_loss": mseloss.item(),
    #         "neg_loss": neg_loss.item(),
    #         "cell_pred": ct_pred_loss.item(),
    #         # "adapt": adapt.item(),
    #         "kld": kld_loss.item(),
    #         "loss_reconstruction": reconstruction_loss.item(),
    #     }
    def iter_update(
        self,
        genes,
        source_embeddings,
        target_embeddings,
        drugs_idx=None,
        dosages=None,
        degs=None,
        celltype_idx=None,
        group_idx=None,
        covariates=None,

        neg_genes=None,
        neg_source_embeddings=None,
        neg_target_embeddings=None,
        neg_drugs_idx=None,
        neg_dosages=None,
        neg_degs=None,
        neg_celltype_idx=None,
        neg_covariates=None,
    ):
        assert drugs_idx is not None and dosages is not None

        (
            genes,
            source_embeddings,
            target_embeddings,
            drugs_idx,
            dosages,
            degs,
            covariates,
        ) = _move_inputs(
            genes,
            source_embeddings,
            target_embeddings,
            drugs_idx,
            dosages,
            degs,
            covariates,
            device=self.device,
        )

        # z0: source latent, from paired control embedding
        z0, mu0, logvar0 = self.encode_FM(source_embeddings, sample=False)

        # z1: target latent, from true treated embedding
        # use deterministic mu as target to reduce noise
        z1, mu1, logvar1 = self.encode_FM(target_embeddings, sample=False)

        cond_parts = self.get_condition_parts(
            drugs_idx=drugs_idx,
            dosages=dosages,
            covariates=covariates,
        )

        # # ===== DEBUG 3: condition parts =====
        # if self.iteration == 0:
        #     print("\n[DEBUG model.py / iter_update() condition]")
        #     print("number of cond_parts:", len(cond_parts))
        #     for j, c in enumerate(cond_parts):
        #         print(f"cond_parts[{j}]:", c.shape, c.dtype, c.device)

        # Flow matching
        z0_cfm = z0.detach()
        z1_cfm = z1.detach()
        t = torch.rand(z0.shape[0], 1, device=self.device, dtype=z0.dtype)
        zt = (1.0 - t) * z0_cfm + t * z1_cfm
        target_v = z1_cfm - z0_cfm

        pred_v = self.velocity(zt, t, cond_parts)

        # # ===== DEBUG 4: flow matching tensors =====
        # if self.iteration == 0:
        #     print("\n[DEBUG model.py / iter_update() flow tensors]")
        #     print("source_embeddings:", source_embeddings.shape)
        #     print("target_embeddings:", target_embeddings.shape)
        #     print("z0:", z0.shape)
        #     print("z1:", z1.shape)
        #     print("t:", t.shape, t.min().item(), t.max().item())
        #     print("zt:", zt.shape)
        #     print("target_v:", target_v.shape)
        #     print("pred_v:", pred_v.shape)

        #     assert z0.shape == z1.shape, "z0 and z1 shape mismatch"
        #     assert zt.shape == z0.shape, "zt and z0 shape mismatch"
        #     assert pred_v.shape == target_v.shape, "pred_v and target_v shape mismatch"
        #     assert torch.isfinite(z0).all(), "z0 has NaN/Inf"
        #     assert torch.isfinite(z1).all(), "z1 has NaN/Inf"
        #     assert torch.isfinite(pred_v).all(), "pred_v has NaN/Inf"
        #     assert torch.isfinite(target_v).all(), "target_v has NaN/Inf"

        flow_loss = F.mse_loss(pred_v, target_v)

        # Endpoint prediction by integration
        z1_pred = self.flow_euler(z0, cond_parts)
        gene_reconstructions, latent_treated = self.decode_gene(z1_pred, cond_parts)

        # # ===== DEBUG 5: endpoint and decoder =====
        # if self.iteration == 0:
        #     print("\n[DEBUG model.py / iter_update() endpoint decoder]")
        #     print("z1_pred:", z1_pred.shape)
        #     print("latent_treated / decoder_input:", latent_treated.shape)
        #     print("gene_reconstructions:", gene_reconstructions.shape)
        #     print("genes:", genes.shape)

        #     assert z1_pred.shape == z1.shape, "z1_pred and z1 shape mismatch"
        #     assert gene_reconstructions.shape == genes.shape, \
        #         "gene_reconstructions and genes shape mismatch"
        #     assert torch.isfinite(z1_pred).all(), "z1_pred has NaN/Inf"
        #     assert torch.isfinite(gene_reconstructions).all(), \
        #         "gene_reconstructions has NaN/Inf"

        # Gene-level reconstruction loss
        afloss = self.loss_afmse(
            y=genes,
            pred=gene_reconstructions,
            degs=degs,
        )
        mseloss = self.loss_mse(genes, gene_reconstructions)

        alpha = self.hparams["alpha"]
        reconstruction_loss = mseloss * alpha + afloss * (1.0 - alpha)

        # Optional endpoint latent loss
        latent_loss = F.mse_loss(z1_pred, z1.detach())

        # Optional KLD on source encoder
        kld_loss = -0.5 * (
            1 + logvar0 - mu0**2 - torch.exp(logvar0)
        ).sum(1).mean()

        loss = (
            self.hparams.get("flow_co", 1.0) * flow_loss
            + self.hparams.get("gene_co", 1.0) * reconstruction_loss
            + self.hparams.get("latent_co", 0.1) * latent_loss
            + self.hparams.get("kld_co", 0.0) * kld_loss
        )

        # # ===== DEBUG 6: losses =====
        # if self.iteration == 0:
        #     print("\n[DEBUG model.py / iter_update() losses]")
        #     print("flow_loss:", flow_loss.item())
        #     print("latent_loss:", latent_loss.item())
        #     print("mseloss:", mseloss.item())
        #     print("afloss:", afloss.item())
        #     print("reconstruction_loss:", reconstruction_loss.item())
        #     print("kld_loss:", kld_loss.item())
        #     print("total loss:", loss.item())

        #     assert torch.isfinite(flow_loss), "flow_loss is NaN/Inf"
        #     assert torch.isfinite(latent_loss), "latent_loss is NaN/Inf"
        #     assert torch.isfinite(mseloss), "mseloss is NaN/Inf"
        #     assert torch.isfinite(afloss), "afloss is NaN/Inf"
        #     assert torch.isfinite(reconstruction_loss), "reconstruction_loss is NaN/Inf"
        #     assert torch.isfinite(loss), "total loss is NaN/Inf"

        self.optimizer_autoencoder.zero_grad()
        if hasattr(self, "optimizer_cell"):
            self.optimizer_cell.zero_grad()
        if self.num_drugs > 0:
            self.optimizer_dosers.zero_grad()

        loss.backward()

        # # ===== DEBUG 7: gradient check =====
        # if self.iteration == 0:
        #     v_grad = 0.0
        #     v_count = 0
        #     for p in self.velocity_net.parameters():
        #         if p.grad is not None:
        #             v_grad += p.grad.abs().mean().item()
        #             v_count += 1

        #     t_grad = 0.0
        #     t_count = 0
        #     for p in self.time_encoder.parameters():
        #         if p.grad is not None:
        #             t_grad += p.grad.abs().mean().item()
        #             t_count += 1

        #     c_grad = 0.0
        #     c_count = 0
        #     for p in self.condition_projector.parameters():
        #         if p.grad is not None:
        #             c_grad += p.grad.abs().mean().item()
        #             c_count += 1

        #     print("\n[DEBUG model.py / iter_update() gradients]")
        #     print("velocity_net grad mean:", v_grad / max(v_count, 1))
        #     print("time_encoder grad mean:", t_grad / max(t_count, 1))
        #     print("condition_projector grad mean:", c_grad / max(c_count, 1))

        #     assert v_count > 0, "velocity_net has no gradients"
        #     assert t_count > 0, "time_encoder has no gradients"
        #     assert c_count > 0, "condition_projector has no gradients"

        self.optimizer_autoencoder.step()
        if hasattr(self, "optimizer_cell"):
            self.optimizer_cell.step()
        if self.num_drugs > 0:
            self.optimizer_dosers.step()

        self.iteration += 1

        return {
            "loss": loss.item(),
            "flow_loss": flow_loss.item(),
            "latent_loss": latent_loss.item(),
            "autofocus_loss": afloss.item(),
            "mse_loss": mseloss.item(),
            "kld": kld_loss.item(),
            "loss_reconstruction": reconstruction_loss.item(),
        }

