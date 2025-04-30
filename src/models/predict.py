import warnings

from numba.core.errors import NumbaDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
import os

import click
import pandas as pd
import scanpy as sc

from src.models.build_models import build_stackix, build_vanilla
from src.models.main_translate import predict_translate
from src.utils.config import get_cfg
from src.utils.utils import get_loader
from src.utils.utils_basic import getlogger


def predict(cfg, data_path_keys, model_type, training_type="load_trained"):
    """Runs Autoencoder Model for given data types and returns latent space
    ARGS:
        cfg - (dict): config dictionary
        data_path_keys - (list): list of data type keys as seen in config.yaml
        model_type - (str): model type stackix, varix, vanillix, ontix
        training_type - (str): training type load_tuned, load_trained
    RETURNS:
        latent_space - (np.array): latent space of given data types

    """
    loaders = []
    translation = None
    logger = getlogger(cfg)
    for key in data_path_keys:
        dataloader = get_loader(cfg, key, cfg["PREDICT_SPLIT"])
        loaders.append(dataloader)
    match model_type:
        case "stackix":
            latent_space = build_stackix(
                cfg, data_path_keys, mode=training_type, split=cfg["PREDICT_SPLIT"]
            )
        case "varix":
            latent_space = build_vanilla(
                cfg,
                data_path_keys,
                mode=training_type,
                ae_type=model_type,
                split=cfg["PREDICT_SPLIT"],
            )
        case "vanillix":
            latent_space = build_vanilla(
                cfg,
                data_path_keys,
                mode=training_type,
                ae_type=model_type,
                split=cfg["PREDICT_SPLIT"],
            )
        # case "oae":
        #     latent_space = build_vanilla(
        #         cfg,
        #         data_path_keys,
        #         mode=training_type,
        #         ae_type=model_type,
        #         split=cfg["PREDICT_SPLIT"]
        #     )
        case "ontix":
            latent_space = build_vanilla(
                cfg,
                data_path_keys,
                mode=training_type,
                ae_type=model_type,
                split=cfg["PREDICT_SPLIT"],
            )

        case "x-modalix":
            from_latent_space, to_latent_space, translation = predict_translate(
                cfg, logger
            )
            from_latent_space = from_latent_space.add_prefix("FROM_", axis=0)
            to_latent_space = to_latent_space.add_prefix("TO_", axis=0)
            latent_space = pd.concat([from_latent_space, to_latent_space])
        case _:
            raise ValueError(
                f"Model type {model_type} not supported \n use stackix, varix, vanillix"
            )
    return latent_space, translation


@click.command()
@click.argument("run_id", type=str)
def main(run_id):
    mapping = {"train": "load_trained", "tune": "load_tuned"}
    cfg = get_cfg(run_id)
    logger = getlogger(cfg)
    usefiles = list(cfg["DATA_TYPE"].keys())
    # Remove ANNOTATION Types
    for u in usefiles:
        if cfg["DATA_TYPE"][u]["TYPE"] == "ANNOTATION":
            usefiles.remove(u)
    logger.info(usefiles)
    # check if stackix is used with only one dataseJt
    if cfg["MODEL_TYPE"] == "stackix" and len(usefiles) == 1:
        logger.warning(
            "Stackix is used with only one dataset. \
                       This is not recommended, as the model will not \
                       be able to learn a latent space, defaulting to Varix"
        )
        cfg["MODEL_TYPE"] = "varix"
    latent_space, translate = predict(
        cfg,
        data_path_keys=usefiles,
        model_type=cfg["MODEL_TYPE"],
        training_type=mapping[cfg["TRAIN_TYPE"]],
    )

    # print(latent_space)
    latent_space.columns = latent_space.columns.map(str)
    # print(f'latent space cols after mapping: {latent_space.columns}')
    latent_space.to_parquet(
        os.path.join("reports", run_id, "predicted_latent_space.parquet")
    )
    ## add feature here to store embeddings in h5ad file
    if cfg["WRITE_H5AD"]:
        for adata_file in cfg["H5AD_FILES"]:
            logger.info(f"Writing latent space to {adata_file} as obsm X_{cfg['MODEL_TYPE']}")
            adata = sc.read_h5ad(os.path.join(cfg["ROOT_RAW"], adata_file))
            if cfg["MODEL_TYPE"] != "x-modalix":
                adata.obsm["X_"+cfg["MODEL_TYPE"]] = latent_space.loc[adata.obs_names,:].to_numpy() # Ensure the order of the index is correct
            else:
                adata.obsm["X_x-modalix_FROM"] = latent_space.loc[
                    ["FROM_"+sample for sample in  adata.obs_names] # Add prefix to sample names
                ].to_numpy()
                adata.obsm["X_x-modalix_TO"] = latent_space.loc[
                    ["TO_"+sample for sample in  adata.obs_names] # Add prefix to sample names
                ].to_numpy()

            sc.write(adata=adata, filename=os.path.join(cfg["ROOT_RAW"], adata_file)) # Overwrite file with new embeddings

    # if translate is not None:
    if cfg["RECON_SAVE"] and (translate is not None):
        translate.to_csv(
            os.path.join("reports", run_id, "translated.txt"), sep=cfg["DELIM"]
        )


if __name__ == "__main__":
    main()
