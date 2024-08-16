import anndata as ad
import scanpy as sc

adata_normal = sc.read_h5ad("./data/raw/scRNA_fetalBM_normal.h5ad")

adata_normal = adata_normal[adata_normal.obs.assay == "10x 5' v1"]
adata_normal = adata_normal[
    (adata_normal.obs.development_stage == "12th week post-fertilization human stage")
    | (adata_normal.obs.development_stage == "13th week post-fertilization human stage")
]

adata_t21 = sc.read_h5ad("./data/raw/scRNA_fetalBM_t21.h5ad")
ad.concat([adata_normal, adata_t21]).write_h5ad("./data/raw/scRNA_fetalBM_both.h5ad")
