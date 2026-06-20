"""
The preprocessing parts of perturbation condition and covariates (except cell type) are adapted from chemCPA.
"""

import logging
from typing import List, Optional, Union

import ot

import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from sklearn.preprocessing import OneHotEncoder

from CRISPLUS.utils import canonicalize_smiles, sample_neg


indx = lambda a, i: a[i] if a is not None else None

def get_group_idx(data, pert_group_key):
    group_name = data.obs[pert_group_key].values
    unique_group_name = np.unique(group_name)
    unique_group_name_to_idx = dict(zip(unique_group_name,range(len(unique_group_name))))
    group_idx = torch.tensor([unique_group_name_to_idx[i] for i in group_name],dtype=torch.long)

    return unique_group_name_to_idx, group_idx

def get_degs(data, pert_group_key, de_genes,var_names):
    number_idx = np.array(range(len(data)))
    degs = torch.zeros((data.shape))
    degs = degs.bool()
    for k,sub_degenes in de_genes.items():
        adata_sub_index = number_idx[data.obs[pert_group_key]==k]
        degs[adata_sub_index,:] = torch.tensor(var_names.isin(sub_degenes)).detach().clone()

    return degs

def get_groups(obs_df,covnames: str,split: str,control_key):
    obs_df['pc_cov_split'] = obs_df[covnames].astype(str) + '_' + obs_df[split].astype(str)
    obs_control = obs_df[obs_df[control_key]==1]
    grouped_df_control = obs_control.groupby('pc_cov_split')
    return grouped_df_control

def get_paired_mean(obs_df, data, control_key, pc_cov, split_key, calc='mean',keep_ctrl=True):

    grouped_df_control = get_groups(obs_df,pc_cov,split_key,control_key)
    index_to_num = dict(zip(obs_df.index,range(len(obs_df))))
    
    # calculate mean of each control group
    def get_mean(group):
        idxnum = torch.tensor([index_to_num[i] for i in group.index])
        sub_x = data[idxnum]
        return torch.mean(sub_x,dim=0)

    def get_std(group):
        idxnum = torch.tensor([index_to_num[i] for i in group.index])
        sub_x = data[idxnum]
        return torch.std(sub_x,dim=0)
    
    if calc=='mean':
        grouped_mean = grouped_df_control.apply(get_mean)
    else:
        grouped_mean = grouped_df_control.apply(get_std)

    group_dict = dict(grouped_mean)
    keys = group_dict.keys()

    if keep_ctrl:
        # choice_control_mean = [group_dict[k] for k in obs_df[obs_df[control_key]==0]['pc_cov_split'].values]
        paired = data.clone()
        treated_index = [index_to_num[i] for i in obs_df[obs_df[control_key]==0].index if (obs_df.loc[i,'pc_cov_split'] in keys)]
        choice_control_mean = [group_dict[obs_df['pc_cov_split'][i]] for i in treated_index]
        paired[treated_index,:] = torch.tensor(np.stack(choice_control_mean,axis=0))
    
    else:
        choice_control_mean = [group_dict[k] for k in obs_df['pc_cov_split'].values if k in keys]
        paired = torch.tensor(np.stack(choice_control_mean,axis=0))
    
    return paired

def _ot_barycentric_pot(
    x_control,
    x_treated,
    reg=0.1,
    num_iter=100,
    stop_thr=1e-6,
    center_cost=True,
):
    """
    Use POT Sinkhorn to compute OT barycentric control embeddings.

    If center_cost=True, the OT cost is computed after centering
    control and treated embeddings separately. This makes OT focus on
    within-population heterogeneity rather than drug-induced global shift.
    """

    n_control = x_control.shape[0]
    n_treated = x_treated.shape[0]

    a = np.ones(n_control, dtype=np.float64) / n_control
    b = np.ones(n_treated, dtype=np.float64) / n_treated

    if center_cost:
        x_control_cost = x_control - x_control.mean(axis=0, keepdims=True)
        x_treated_cost = x_treated - x_treated.mean(axis=0, keepdims=True)
    else:
        x_control_cost = x_control
        x_treated_cost = x_treated

    M = ot.dist(
        x_control_cost,
        x_treated_cost,
        metric="euclidean",
    ).astype(np.float64)

    max_m = M.max()
    if max_m > 0:
        M = M / max_m

    gamma = ot.sinkhorn(
        a,
        b,
        M,
        reg=reg,
        numItermax=num_iter,
        stopThr=stop_thr,
        verbose=False,
    )

    col_mass = gamma.sum(axis=0)
    col_mass = np.maximum(col_mass, 1e-12)

    # 注意：barycenter 仍然用原始 control embedding，而不是 centered embedding
    paired_treated = gamma.T @ x_control
    paired_treated = paired_treated / col_mass[:, None]

    return paired_treated.astype(np.float32)

def get_paired_ot_pot_fast(
    obs_df,
    data,
    control_key,
    pc_cov,
    split_key,
    pert_category,
    train_only=True,
    reg=0.1,
    num_iter=100,
    stop_thr=1e-6,
    min_cells=3,
    max_control=512,
    max_treated_chunk=512,
    seed=0,
    fallback="mean",
):
    """
    Fast, low-memory OT replacement for get_paired_mean().

    This function constructs paired control embeddings for treated cells
    using OT barycentric projection in FM embedding space.

    Important:
    - Control cells are NOT modified.
    - Only treated cells are replaced.
    - OT is computed offline in Dataset.__init__().
    - This does NOT enter the training computation graph.

    Parameters
    ----------
    obs_df:
        adata.obs.copy()

    data:
        torch.Tensor, shape [n_cells, dim]
        Usually self.FM_emb.

    control_key:
        Column name where control_key == 1 indicates control cells.

    pc_cov:
        Column used to identify matched control group.

    split_key:
        Column indicating train/test/ood split.

    pert_category:
        Column indicating perturbation condition,
        e.g. celltype_drug_dose.

    train_only:
        If True, only apply OT replacement to train treated cells.
        Recommended to avoid test/OOD leakage.

    reg:
        Sinkhorn entropy regularization.

    num_iter:
        Maximum Sinkhorn iterations.

    max_control:
        Maximum number of control cells sampled per group.

    max_treated_chunk:
        Maximum number of treated cells per OT computation.

    fallback:
        If OT fails or group is too small, use control mean.
    """

    rng = np.random.default_rng(seed)

    obs_df = obs_df.copy()

    # Start from original FM embeddings.
    # This ensures control cells keep their original FM embeddings.
    paired = data.clone()

    # Same grouping logic as get_paired_mean:
    # pc_cov + split_key define matched control group.
    obs_df["pc_cov_split"] = (
        obs_df[pc_cov].astype(str) + "_" + obs_df[split_key].astype(str)
    )

    index_to_num = dict(zip(obs_df.index, range(len(obs_df))))

    # Build control groups using only control cells.
    control_df = obs_df[obs_df[control_key] == 1]
    control_groups = {
        key: np.array([index_to_num[idx] for idx in group.index], dtype=int)
        for key, group in control_df.groupby("pc_cov_split")
    }

    # Treated cells.
    treated_df = obs_df[obs_df[control_key] != 1]

    # Recommended: only use train treated cells for OT pairing.
    if train_only:
        treated_df = treated_df[treated_df[split_key] == "train"]

    # Group treated cells by matched control group + perturbation condition.
    # This avoids mixing different drugs/doses in the same OT computation.
    treated_groups = treated_df.groupby(["pc_cov_split", pert_category])

    # Move once to CPU numpy source.
    # POT is CPU/NumPy-oriented here; no autograd is needed.
    data_cpu = data.detach().cpu()

    for (pc_key, pert_key), group in treated_groups:
        if pc_key not in control_groups:
            continue

        control_idx_all = control_groups[pc_key]
        treated_idx_all = np.array(
            [index_to_num[idx] for idx in group.index],
            dtype=int,
        )

        # Too few cells: fallback to original mean.
        if len(control_idx_all) < min_cells or len(treated_idx_all) < min_cells:
            if fallback == "mean" and len(control_idx_all) > 0:
                ctrl_mean = data[control_idx_all].mean(dim=0)
                paired[treated_idx_all, :] = ctrl_mean.unsqueeze(0).repeat(
                    len(treated_idx_all), 1
                )
            continue

        # Sample control cells once per treated group.
        # All treated chunks in this group reuse the same control subset.
        if len(control_idx_all) > max_control:
            control_idx = rng.choice(
                control_idx_all,
                size=max_control,
                replace=False,
            )
        else:
            control_idx = control_idx_all

        x_control = data_cpu[control_idx].numpy().astype(np.float32)

        # Process treated cells in chunks to limit memory.
        for start in range(0, len(treated_idx_all), max_treated_chunk):
            treated_idx = treated_idx_all[start:start + max_treated_chunk]
            x_treated = data_cpu[treated_idx].numpy().astype(np.float32)

            try:
                paired_treated = _ot_barycentric_pot(
                    x_control=x_control,
                    x_treated=x_treated,
                    reg=reg,
                    num_iter=num_iter,
                    stop_thr=stop_thr,
                    center_cost=True,
                )

                paired[treated_idx, :] = torch.tensor(
                    paired_treated,
                    dtype=data.dtype,
                    device=data.device,
                )

            except Exception as e:
                print(
                    f"[Warning] OT failed for group "
                    f"pc_key={pc_key}, pert_key={pert_key}. "
                    f"Fallback to mean. Error: {e}"
                )

                if fallback == "mean":
                    ctrl_mean = data[control_idx_all].mean(dim=0)
                    paired[treated_idx, :] = ctrl_mean.unsqueeze(0).repeat(
                        len(treated_idx), 1
                    )

    return paired

def get_cov_gpt(adata, data, control_key, cov_name, split_key, setting='train'):
    adata_control = adata[(adata.obs[control_key]==1) & (adata.obs[split_key]==setting)]
    grouped_adata_control = adata_control.obs.groupby(cov_name)
    index_to_num = dict(zip(adata.obs.index,range(len(adata))))

    def get_mean(group):
        idxnum = torch.tensor([index_to_num[i] for i in group.index])
        sub_x = data[idxnum]
        return torch.mean(sub_x,axis=0)
    grouped_mean = grouped_adata_control.apply(get_mean)
    group_mean_dict = dict(grouped_mean)

    return group_mean_dict

def drug_names_to_once_canon_smiles(
    drug_names: List[str], dataset: sc.AnnData, perturbation_key: str, smiles_key: str
):
    """
    Converts a list of drug names to a list of SMILES. The ordering is of the list is preserved.
    """
    name_to_smiles_map = {
        drug: canonicalize_smiles(smiles)
        for drug, smiles in dataset.obs.groupby(
            [perturbation_key, smiles_key]
        ).groups.keys()
    }
    return [name_to_smiles_map[name] for name in drug_names]

def drug_to_idx(drugs_names):
    drugs_names_unique = set()
    for d in drugs_names:
        [drugs_names_unique.add(i) for i in d.split("+")]

    drugs_names_unique_sorted = np.array(sorted(drugs_names_unique))
    _drugs_name_to_idx = {
        smiles: idx for idx, smiles in enumerate(drugs_names_unique_sorted)
    }
    drugs_idx = [_drugs_name_to_idx[drug] for drug in drugs_names]

    return drugs_idx, drugs_names_unique_sorted,_drugs_name_to_idx

class Dataset:
    covariate_keys: Optional[List[str]]
    drugs: torch.Tensor  # stores the (OneHot * dosage) encoding of drugs / combinations of drugs
    drugs_idx: torch.Tensor  # stores the integer index of the drugs applied to each cell.
    max_num_perturbations: int  # how many drugs are applied to each cell at the same time?
    dosages: torch.Tensor  # shape: (dataset_size, max_num_perturbations)
    drugs_names_unique_sorted: np.ndarray  # sorted list of all drug names in the dataset

    def __init__(
        self,
        data,
        perturbation_key=None,
        dose_key=None,
        celltype_key='cell_type',
        covariate_keys=None,
        smiles_key='SMILES',
        FM_key="X_scGPT",
        degs_key="rank_genes_groups_cov",
        pert_category="cov_drug_name",
        control_key = 'control',
        split_key="split",
        pc_cov='type_donor',
        seed=0,
        use_FM=True,
    ):
        """
        FM_key: The name of FM embedding for cells in adata.obsm. 
            We extract FM embedding before training to avoid the FM's forward process in each iteration thus save time and memory
        celltype_key: The column name of cell type in obs dataframe
        covariate_keys: The column names of other covariates in obs, input as list format.
        perturbation_key: The column name of treatment name (like drug name).
        dose_key: The column name of treatment dosage
        degs_key: The name of DEG's for each celltype_treatment group in adata.uns
        pert_category: The name of column storing group condition for evaluation.
            e.g: cell type + drug name + drug dose. This is used during contrastive sampling and evaluation. Should align with deg dict's keys
        pc_cov: The name of column in adata.obs to identify paired control group, 
            e.g 'celltype_donor' means find paired control group with the same celltype and donor attribute 
        """
        logging.info(f"Starting to read in data: {data}\n...")
        if isinstance(data, AnnData):
            data = data
        else:
            data = sc.read(data)
        logging.info(f"Finished data loading.")

        try:
            self.genes = torch.Tensor(data.X.toarray())
        except:
            self.genes = torch.Tensor(data.X)

        self.var_names = data.var_names
        if use_FM:
            self.FM_emb = torch.tensor(data.obsm[FM_key],dtype=torch.float)
        else: 
            self.FM_emb = self.genes.clone()
        self.control_key = control_key
        obs_df = data.obs.copy()
        
        # data.obs['drug_dose_name'] = adata.obs.condition.astype(str) + '_' + adata.obs.dose_val.astype(str)
        # data.obs['cov_drug_dose_name'] = adata.obs.cell_type.astype(str) + '_' + adata.obs.drug_dose_name.astype(str)
        if 'cov_drug_name' not in data.obs.columns:
            data.obs['cov_drug_name'] = data.obs[celltype_key].astype(str) + '_' + data.obs[perturbation_key].astype(str)

        # find paired control groups and calculate paired control FM embedding and gene expression profile
        # self.paired_cell_embeddings = get_paired_mean(obs_df,self.FM_emb,control_key,pc_cov,split_key)
        self.paired_cell_embeddings = get_paired_ot_pot_fast(
            obs_df=obs_df,
            data=self.FM_emb,
            control_key=control_key,
            pc_cov=pc_cov,
            split_key=split_key,
            pert_category=pert_category,
            train_only=True,
            reg=0.01,
            num_iter=100,
            stop_thr=1e-6,
            min_cells=3,
            max_control=512,
            max_treated_chunk=512,
            seed=seed,
            fallback="mean",
        )
        # self.paired_genes = get_paired_mean(obs_df,self.genes,control_key,celltype_key,split_key,keep_ctrl=True)
        # self.paired_std = get_paired_mean(obs_df,self.genes,control_key,celltype_key,split_key,calc='std',keep_ctrl=True)

        logging.info(f"FM_emb shape: {self.FM_emb.shape}")
        logging.info(f"paired_cell_embeddings shape: {self.paired_cell_embeddings.shape}")

        assert self.paired_cell_embeddings.shape == self.FM_emb.shape
        assert torch.isfinite(self.paired_cell_embeddings).all()

        # identify negative samples with same perturbation condition but different cell type
        self.neg_idx = sample_neg(data, split_key, 'cov_drug_name',perturbation_key,seed)

        # cell type labels
        self.celltype = np.array(data.obs[celltype_key].values)

        # preprocess of drug perturbation and other covariate information
        self.perturbation_key = perturbation_key
        self.dose_key = dose_key
        if isinstance(covariate_keys, str):
            covariate_keys = [covariate_keys]
        self.covariate_keys = covariate_keys

        if perturbation_key is not None:
            if dose_key is None:
                raise ValueError(
                    f"A 'dose_key' is required when provided a 'perturbation_key'({perturbation_key})."
                )
            self.pert_categories = np.array(data.obs[pert_category].values)
            self.de_genes = data.uns[degs_key]
            self.drugs_names = np.array(data.obs[perturbation_key].values)
            self.dose_names = np.array(data.obs[dose_key].values)

            # get unique drugs
            drugs_idx,self.drugs_names_unique_sorted,_ = drug_to_idx(self.drugs_names)
            self.canon_smiles_unique_sorted = drug_names_to_once_canon_smiles(
                list(self.drugs_names_unique_sorted), data, perturbation_key, smiles_key
            )

            self.drugs_idx = torch.tensor(
                drugs_idx,
                dtype=torch.long,
            )

            dosages = [float(dosage) for dosage in self.dose_names]
            self.dosages = torch.tensor(
                dosages,
                dtype=torch.float32,
            )

        else:
            self.pert_categories = None
            self.de_genes = None
            self.drugs_names = None
            self.dose_names = None
            self.drugs_names_unique_sorted = None

        if isinstance(covariate_keys, list) and covariate_keys:
            if not len(covariate_keys) == len(set(covariate_keys)):
                raise ValueError(f"Duplicate keys were given in: {covariate_keys}")
            self.covariate_names = {}
            self.covariate_names_unique = {}
            self.covariates = []
            for cov in covariate_keys:
                self.covariate_names[cov] = np.array(data.obs[cov].values)
                self.covariate_names_unique[cov] = np.unique(self.covariate_names[cov])

                names = self.covariate_names_unique[cov]

                encoder_cov = OneHotEncoder(sparse=False)
                encoder_cov.fit(names.reshape(-1, 1))

                names = self.covariate_names[cov]
                self.covariates.append(
                    torch.Tensor(encoder_cov.transform(names.reshape(-1, 1))).float()
                )
        else:
            self.covariate_names = None
            self.covariate_names_unique = None
            self.covariates = None

        if self.covariates is not None:
            self.num_covariates = [
                len(names) for names in self.covariate_names_unique.values()
            ]
        else:
            self.num_covariates = [0]

        self.num_genes = self.genes.shape[1]
        self.num_drugs = (
            len(self.drugs_names_unique_sorted)
            if self.drugs_names_unique_sorted is not None
            else 0
        )

        # get DEG mask matrix for each sample, indicating which genes are DE genes, used for loss calculation and evaluation
        self.degs = get_degs(data, pert_category, self.de_genes, self.var_names)
        self.unique_group_name_dict, self.group_idxs = get_group_idx(data, pert_category)

        self.indices = {
            "all": list(range(len(self.genes))),
            "control": np.where(data.obs[control_key] == 1)[0].tolist(),
            "treated": np.where(data.obs[control_key] != 1)[0].tolist(),
            "train": np.where(data.obs[split_key] == "train")[0].tolist(),
            "test": np.where(data.obs[split_key] == "test")[0].tolist(),
            "ood": np.where(data.obs[split_key] == "ood")[0].tolist(),
        }

    def subset(self, split, condition="all"):
        idx = list(set(self.indices[split]) & set(self.indices[condition]))
        return SubDataset(self, idx, split)

    def __getitem__(self, i):
        if self.covariates is None:
            return (
                self.genes[i],
                self.paired_cell_embeddings[i],
                indx(self.drugs_idx, i),
                indx(self.dosages, i),
                indx(self.degs, i),
                indx(self.celltype_idx, i),
                indx(self.group_idxs,i),
                None,
            )
        else:
            return (
                self.genes[i],
                self.paired_cell_embeddings[i],
                indx(self.drugs_idx, i),
                indx(self.dosages, i),
                indx(self.degs, i),
                indx(self.celltype_idx, i),
                indx(self.group_idxs,i),
                *[indx(cov, i) for cov in self.covariates],
            )

    def __len__(self):
        return len(self.genes)


class SubDataset:
    """
    Subsets a `Dataset` by selecting the examples given by `indices`.
    """

    def __init__(self, dataset: Dataset, indices, split_set):
        self.perturbation_key = dataset.perturbation_key
        self.dose_key = dataset.dose_key
        self.covariate_keys = dataset.covariate_keys
        self.canon_smiles_unique_sorted = dataset.canon_smiles_unique_sorted

        # self.genes = dataset.genes[indices]
        # # self.paired_genes = dataset.paired_genes[indices]
        # # self.paired_std = dataset.paired_std[indices]
        # self.paired_cell_embeddings = dataset.paired_cell_embeddings[indices]
        self.genes = dataset.genes[indices]
        # z1: target current-cell FM embedding.
        # treated cells: true treated embedding
        # control cells: true control embedding
        self.cell_embeddings = dataset.FM_emb[indices]
        # z0: paired control/source FM embedding
        self.paired_cell_embeddings = dataset.paired_cell_embeddings[indices]

        self.drugs_idx = indx(dataset.drugs_idx, indices)
        self.dosages = indx(dataset.dosages, indices)
        if dataset.covariates is not None:
            self.covariates = [indx(cov, indices) for cov in dataset.covariates]
        else:
            self.covariates = None
        self.drugs_names = indx(dataset.drugs_names, indices)
        self.pert_categories = indx(dataset.pert_categories, indices)
        self.covariate_names = {}

        if self.covariate_keys is not None:
            for cov in self.covariate_keys:
                self.covariate_names[cov] = indx(dataset.covariate_names[cov], indices)
        else:
            self.covariate_names = None
        self.var_names = dataset.var_names
        self.de_genes = dataset.de_genes

        self.num_covariates = dataset.num_covariates
        self.num_genes = dataset.num_genes
        self.num_drugs = dataset.num_drugs

        self.degs = dataset.degs[indices]
        self.group_idxs = indx(dataset.group_idxs, indices)

        self.celltype = indx(dataset.celltype, indices)
        self.unique_celltype = np.array(sorted(set(self.celltype)))
        self.num_celltypes = len(self.unique_celltype)
        self.celltype_to_idx = {ct:idx for idx,ct in enumerate(self.unique_celltype)}
        celltype_idx = [self.celltype_to_idx[ct] for ct in self.celltype]
        self.celltype_idx = torch.tensor(celltype_idx, dtype=torch.long)

        if split_set == 'train':
            neg_idx = dataset.neg_idx
            self.neg_idx = neg_idx

            # self.neg_genes = self.genes[neg_idx]
            # # self.neg_paired_genes = self.paired_genes[neg_idx]
            # # self.neg_paired_std = self.paired_std[neg_idx]
            # self.neg_paired_cell_embeddings = self.paired_cell_embeddings[neg_idx]
            self.neg_genes = self.genes[neg_idx]
            self.neg_cell_embeddings = self.cell_embeddings[neg_idx]
            self.neg_paired_cell_embeddings = self.paired_cell_embeddings[neg_idx]

            self.neg_drugs_idx = indx(self.drugs_idx, neg_idx)
            self.neg_dosages = indx(self.dosages, neg_idx)
            self.neg_degs = self.degs[neg_idx]
            self.neg_celltype_idx = indx(self.celltype_idx, neg_idx)
            if self.covariate_keys is not None:
                for cov in self.covariate_keys:
                    self.neg_covariate_names[cov] = indx(dataset.covariate_names[cov], neg_idx)
            else:
                self.neg_covariate_names = None
        else:
            self.neg_idx = None


    def __getitem__(self, i):
        # # ===== DEBUG data.py / SubDataset.__getitem__() =====
        # # 注意：这个会被 dataloader 多次调用，所以只建议非常临时地开。
        # if i == 0 and not hasattr(self, "_debug_getitem_printed"):
        #     print("\n[DEBUG data.py / SubDataset.__getitem__()]")
        #     print("genes[0]:", self.genes[i].shape)
        #     print("cell_embeddings[0] target z1:", self.cell_embeddings[i].shape)
        #     print("paired_cell_embeddings[0] source z0:", self.paired_cell_embeddings[i].shape)
        #     print("drugs_idx[0]:", indx(self.drugs_idx, i))
        #     print("dosages[0]:", indx(self.dosages, i))
        #     print("degs[0]:", indx(self.degs, i).shape)
        #     print("celltype_idx[0]:", indx(self.celltype_idx, i))
        #     print("group_idxs[0]:", indx(self.group_idxs, i))
        #     self._debug_getitem_printed = True
        if (self.covariates is None):
            # return (
            #     self.genes[i],
            #     self.paired_cell_embeddings[i],
            #     # self.paired_genes[i],
            #     # self.paired_std[i],
            #     indx(self.drugs_idx, i),
            #     indx(self.dosages, i),
            #     indx(self.degs, i),
            #     indx(self.celltype_idx, i),
            #     self.neg_genes[i],
            #     self.neg_paired_cell_embeddings[i],
            #     # self.neg_paired_genes[i],
            #     # self.neg_paired_std[i],
            #     indx(self.neg_drugs_idx, i),
            #     indx(self.neg_dosages, i),
            #     indx(self.neg_degs, i),
            #     indx(self.neg_celltype_idx, i),
            #     None,
            #     None,
            # )
            return (
                self.genes[i],                       # 0: x1 gene target
                self.cell_embeddings[i],             # 1: z1 target FM embedding
                self.paired_cell_embeddings[i],      # 2: z0 source FM embedding
                indx(self.drugs_idx, i),             # 3
                indx(self.dosages, i),               # 4
                indx(self.degs, i),                  # 5
                indx(self.celltype_idx, i),          # 6
                indx(self.group_idxs, i),            # 7

                self.neg_genes[i],                   # 8
                self.neg_cell_embeddings[i],         # 9
                self.neg_paired_cell_embeddings[i],  # 10
                indx(self.neg_drugs_idx, i),         # 11
                indx(self.neg_dosages, i),           # 12
                indx(self.neg_degs, i),              # 13
                indx(self.neg_celltype_idx, i),      # 14

                None,                                # 15 covariates
                None,                                # 16 neg_covariates
            )
        else:
            # return (
            #     self.genes[i],
            #     self.paired_cell_embeddings[i],
            #     # self.paired_genes[i],
            #     indx(self.drugs_idx, i),
            #     indx(self.dosages, i),
            #     indx(self.degs, i),
            #     indx(self.celltype_idx, i),
            #     self.neg_genes[i],
            #     self.neg_paired_cell_embeddings[i],
            #     # self.neg_paired_genes[i],
            #     indx(self.neg_drugs_idx, i),
            #     indx(self.neg_dosages, i),
            #     indx(self.neg_degs, i),
            #     indx(self.neg_celltype_idx, i),
            #     *[indx(cov, i) for cov in self.covariates],
            #     *[indx(cov, i) for cov in self.neg_covariates],
            # )
            return (
                self.genes[i],
                self.cell_embeddings[i],
                self.paired_cell_embeddings[i],
                indx(self.drugs_idx, i),
                indx(self.dosages, i),
                indx(self.degs, i),
                indx(self.celltype_idx, i),
                indx(self.group_idxs, i),

                self.neg_genes[i],
                self.neg_cell_embeddings[i],
                self.neg_paired_cell_embeddings[i],
                indx(self.neg_drugs_idx, i),
                indx(self.neg_dosages, i),
                indx(self.neg_degs, i),
                indx(self.neg_celltype_idx, i),

                *[indx(cov, i) for cov in self.covariates],
                *[indx(cov, i) for cov in self.neg_covariates],
            )

    def __len__(self):
        return len(self.genes)
    
def custom_collate(batch):
    transposed = zip(*batch)
    concat_batch = []
    for samples in transposed:
        if samples[0] is None:
            concat_batch.append(None)
        else:
            concat_batch.append(torch.stack(samples, 0).to("cuda"))
    return concat_batch
