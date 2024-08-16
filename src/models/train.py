from numba.core.errors import NumbaDeprecationWarning
import warnings

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
from src.models.build_models import build_stackix, build_vanilla
from src.utils.utils_basic import getlogger
from src.models.main_translate import train_translate
import click
from src.utils.config import get_cfg


def train_models(data_path_keys, model_type, cfg, train_type="train"):

    """trains autoencoder models for given input data and model type. User can
    wheter to train the model weights or to train the whole model architecture
    ARGS:
        data_path_keys - (list): list of data path keys
        model_type - (str): model type, options are: 'vanillix', 'varix', 'stackix', 'ontix'
        cfg - (dict): config dictionary
        train_type (str): train type options are: 'tune', 'train',
                          default is 'train'
    """
    logger = getlogger(cfg)
    match model_type:
        case "vanillix":
            build_vanilla(
                cfg=cfg,
                data_path_keys=data_path_keys,
                mode=train_type,
                ae_type=model_type,
                split="train",
            )
        case "varix":
            build_vanilla(
                cfg=cfg,
                data_path_keys=data_path_keys,
                mode=train_type,
                ae_type=model_type,
                split="train",
            )
        # case "oae":
        #     build_vanilla(
        #         cfg=cfg,
        #         data_path_keys=data_path_keys,
        #         mode=train_type,
        #         ae_type=model_type,
        #         split="train",
        #     )
        case "ontix":
            build_vanilla(
                cfg=cfg,
                data_path_keys=data_path_keys,
                mode=train_type,
                ae_type=model_type,
                split="train",
            )
        case "stackix":
            build_stackix(
                cfg=cfg,
                data_path_keys=data_path_keys,
                mode=train_type,
                split="train",
            )
        case "x-modalix":
            logger.info("Training translate model")
            train_translate(cfg=cfg, logger=logger)
        # case "translate2":
        #     logger.info("Training translate model uhler")
        #     train_translate(cfg=cfg, logger=logger)
        case _:
            raise ValueError(
                f"model type {model_type} not supported, options are: 'vanillix',\
                'varix', 'stackix', 'x-modalix'"
            )


@click.command()
@click.argument("run_id", type=str)
def main(run_id):
    cfg = get_cfg(run_id)
    logger = getlogger(cfg)
    usefiles = list(cfg["DATA_TYPE"].keys())
    # Remove ANNOTATION Types
    for u in usefiles:
        if cfg["DATA_TYPE"][u]["TYPE"] == "ANNOTATION":
            usefiles.remove(u)

    # check if stackix is used with only one dataset
    if cfg["MODEL_TYPE"] == "stackix" and len(usefiles) == 1:
        logger.warning("Stackix is used with only one dataset, defaulting to Varix")
        cfg["MODEL_TYPE"] = "varix"
    train_models(
        data_path_keys=usefiles,
        model_type=cfg["MODEL_TYPE"],
        cfg=cfg,
        train_type=cfg["TRAIN_TYPE"],
    )


if __name__ == "__main__":
    main()
