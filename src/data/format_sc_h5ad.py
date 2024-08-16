import click
import os
import pandas as pd

# import logging
# try:
from src.utils.utils_basic import getlogger

# except:
#     getlogger = lambda: logging.getLogger(__name__)
#     logger = logging.getLogger(__name__)
from src.utils.config import get_cfg

import scanpy as sc


def single_h5ad(file, cfg):
    logger = getlogger(cfg)
    logger.info(f"Read in h5ad file")
    adata = sc.read_h5ad(os.path.join(cfg["ROOT_RAW"], file))

    logger.info(f"Shape before: {adata.shape}")
    logger.info(f"Filter samples with low number of expressed genes")
    sc.pp.filter_cells(
        adata, min_genes=int(adata.shape[1] * cfg["MIN_GENE"]), inplace=True
    )
    logger.info(f"Shape after filtering: {adata.shape}")

    logger.info(f"Filter genes which are lowly present in cells")
    sc.pp.filter_genes(
        adata, min_cells=int(adata.shape[0] * cfg["MIN_PERC_CELLS"]), inplace=True
    )
    logger.info(f"Shape after filtering: {adata.shape}")

    logger.info(f"Filter for highly variable genes")
    sc.pp.highly_variable_genes(
        adata, n_top_genes=cfg["K_FILTER_SC"], subset=True, inplace=True
    )
    logger.info(f"Shape after filtering: {adata.shape}")

    logger.info(f"Convert to pandas")

    for layer in cfg["ANNDATA_LAYER"]:
        match layer:

            case "X":
                df = pd.DataFrame.sparse.from_spmatrix(
                    adata.X, index=adata.obs.index, columns=adata.var.index
                )
                df.sparse.to_dense().to_parquet(
                    os.path.join(
                        cfg["ROOT_RAW"], file.split(".")[0] + cfg["FORM_SUFFIX"]
                    )
                )

            case "obs":
                adata.obs.to_parquet(
                    os.path.join(
                        cfg["ROOT_RAW"],
                        file.split(".")[0] + "_clinical" + cfg["FORM_SUFFIX"],
                    )
                )

            case _:
                if "layers/" in layer:
                    layer_name = layer.split("/")[1]
                    logger.info(f"Processing custom layer: {layer_name}")
                    # print(adata.layers[layer_name].shape)
                    df = pd.DataFrame(
                        adata.layers[layer_name],
                        index=adata.obs.index,
                        columns=adata.var.index,
                    )
                    df.to_parquet(
                        os.path.join(
                            cfg["ROOT_RAW"],
                            file.split(".")[0] + "_velocity" + cfg["FORM_SUFFIX"],
                        )
                    )
                else:
                    logger.warning(
                        f"You provided a not supported layer specification: {layer}"
                    )
                    logger.warning(
                        "Only anndata layers and specification like layers/custom is supported."
                    )

    return df


@click.command()
@click.argument("run_id", type=str)
def main(run_id):
    """Formats hda5 and anndata file types to data frames compatible with pipeline"""
    cfg = get_cfg(run_id)
    logger = getlogger(cfg)
    logger.info("Format single-cell files (h5ad)")

    for filename in cfg["H5AD_FILES"]:
        logger.info(f"Formatting: {filename}")

        single_h5ad(file=filename, cfg=cfg)


if __name__ == "__main__":
    main()
