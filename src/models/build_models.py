import os

import pandas as pd
import torch

from src.models.tuning.tuning import find_best_model
from src.utils.utils import (
    get_latent_space,
    get_loader,
    get_model,
    make_mask_decoder,
    train_ae_model,
)
from src.utils.utils_basic import getlogger, read_ont_file
from src.visualization.visualize import plot_model_weights


def build_vanilla(
    cfg, data_path_keys, mode="load_trained", ae_type="varix", split="train"
):
    """takes different data modalities and trains a vanilla autoencoder or
    variational vanilla autoencoder
    ARGS:
        cfg - (dict): config dictionary
        data_path_keys - (list): list of config keys as defined in src/config.yaml
                                for example: [RNA_FILE_PATH, CNA_FILE_PATH]
        mode - (str): indicator if the model should be loaded, trained or
                       trained and tuned (optuna). Default is load_trained, options are
                       'load_trained', 'load_tuned', 'train', 'tune'

        ae_type - (str): type of autoencoder (optional, default='varix', options: 'vanillix', 'ontix')
        split - (str): indicator if the model should be trained on the train, val or test split
    RETURNS:
        None

    """
    logger = getlogger(cfg)
    data = []
    valid_data = []
    feature_ids = []
    name_helper = ""
    for k in data_path_keys:
        if k == "IMG":
            logger.info(
                f'image data only supported for translate case not for the case {cfg["MODEL_TYPE"]} '
            )
            continue
        name_helper += f"{k}_"
        trainloader = get_loader(cfg, k, split_type=split)
        validloader = get_loader(cfg, k, split_type="valid")
        data.append(trainloader.dataset.x_tensor)
        valid_data.append(validloader.dataset.x_tensor)
        feature_ids.extend(
            list(trainloader.dataset.get_cols())
        )  # train and valid have similar column names

    model_name = name_helper + ae_type + cfg["RUN_ID"] + ".pt"
    # concat data to one tensor
    data = torch.cat(data, dim=1)
    valid_data = torch.cat(valid_data, dim=1)

    # save datapath in config file to use same logic as for other input data
    cfg["DATA_TYPE"][f"COMBINED-{name_helper}_{ae_type}_INPUT"] = {
        "FILE_PROC": os.path.join(
            f"data/interim/{cfg['RUN_ID']}/combined_{ae_type}_input.parquet"
        ),
        "TYPE": "COMBINED",
    }
    cfg["DATA_TYPE"][f"COMBINED-{name_helper}_{ae_type}_INPUT_VAL"] = {
        "FILE_PROC": os.path.join(
            f"data/interim/{cfg['RUN_ID']}/combined_{ae_type}_input_val.parquet"
        ),
        "TYPE": "COMBINED",
    }

    data = pd.DataFrame(
        data.cpu().numpy(), index=trainloader.dataset.sample_ids, columns=feature_ids
    )
    # data.to_csv(
    #     cfg["DATA_TYPE"][f"COMBINED-{name_helper}_{ae_type}_INPUT"]["FILE_PROC"], sep=cfg['DELIM']
    # )
    data.to_parquet(
        cfg["DATA_TYPE"][f"COMBINED-{name_helper}_{ae_type}_INPUT"]["FILE_PROC"]
    )

    valid_data = pd.DataFrame(
        valid_data.cpu().numpy(),
        index=validloader.dataset.sample_ids,
        columns=feature_ids,
    )
    # valid_data.to_csv(
    #     cfg["DATA_TYPE"][f"COMBINED-{name_helper}_{ae_type}_INPUT_VAL"]["FILE_PROC"], sep=cfg['DELIM']
    # )
    valid_data.to_parquet(
        cfg["DATA_TYPE"][f"COMBINED-{name_helper}_{ae_type}_INPUT_VAL"]["FILE_PROC"]
    )
    combined_loader = get_loader(
        cfg, f"COMBINED-{name_helper}_{ae_type}_INPUT", split_type=split
    )
    valid_combined_loader = get_loader(
        cfg,
        f"COMBINED-{name_helper}_{ae_type}_INPUT_VAL",
        split_type="valid",
    )
    logger.info(f"DATA SIZE:{combined_loader.dataset.shape()}")

    ## if oAE or oVAE calculate masks
    if cfg["MODEL_TYPE"] == "ontix":
        # read in dict
        ont_data_lvl1 = read_ont_file(
            os.path.join("data/processed", cfg["RUN_ID"], cfg["FILE_ONT_LVL1"]),
            sep=cfg["DELIM"],
        )
        # make mask
        mask_1 = make_mask_decoder(
            prev_lay_dim=len(ont_data_lvl1),
            next_lay_dim=len(feature_ids),
            ont_dic=ont_data_lvl1,
            node_list=feature_ids,
            cfg=cfg,
        )
        if not cfg["FILE_ONT_LVL2"] == None:
            ont_data_lvl2 = read_ont_file(
                os.path.join("data/processed", cfg["RUN_ID"], cfg["FILE_ONT_LVL2"]),
                sep=cfg["DELIM"],
            )
            mask_2 = make_mask_decoder(
                prev_lay_dim=len(ont_data_lvl2),
                next_lay_dim=len(ont_data_lvl1),
                ont_dic=ont_data_lvl2,
                node_list=list(ont_data_lvl1.keys()),
                cfg=cfg,
            )
        else:
            mask_2 = None
    else:
        mask_1 = None
        mask_2 = None

    if mode == "load_trained":
        model = get_model(
            cfg,
            data.shape[1],
            f"COMBINED-{name_helper}_{ae_type}_INPUT",
            ae_type,
            mask_1=mask_1,
            mask_2=mask_2,
        )
        assert os.path.exists(os.path.join("models", f"{cfg['RUN_ID']}", model_name))
        model.load_state_dict(
            torch.load(os.path.join("models", f"{cfg['RUN_ID']}", model_name))
        )
    elif mode == "load_tuned":
        assert os.path.exists(
            os.path.join("models", "tuned", f"{cfg['RUN_ID']}", "tuned_" + model_name)
        )
        model = torch.load(
            os.path.join("models", "tuned", f"{cfg['RUN_ID']}", "tuned_" + model_name)
        )

    elif mode == "train":
        model = get_model(
            cfg,
            data.shape[1],
            f"COMBINED-{name_helper}_{ae_type}_INPUT",
            ae_type,
            mask_1=mask_1,
            mask_2=mask_2,
        )

        if cfg["PLOT_WEIGHTS"]:
            logger.info("Plotting model weights init")
            filepath = os.path.join(
                "reports",
                cfg["RUN_ID"],
                f"figures/model_weights_init.png",
            )
            plot_model_weights(filepath=filepath, model=model)
        model, _ = train_ae_model(
            cfg,
            combined_loader,
            model,
            model_name,
            model_type=ae_type,
            valid_loader=valid_combined_loader,
            mask_1=mask_1,
            mask_2=mask_2,
        )
    elif mode == "tune":
        model = find_best_model(
            cfg,
            model_name="tuned_" + model_name,
            path_key=f"COMBINED-{name_helper}_{ae_type}_INPUT",
            dataloader=combined_loader,
            valloader=valid_combined_loader,
            model_type=ae_type,
            mask_1=mask_1,
            mask_2=mask_2,
        )

    else:
        raise ValueError("Mode not recognized (load, load_trained, train, load_tuned)")
    logger.info(f"Model structure:")
    logger.info(f"{model}")
    latent_space, recon_x = get_latent_space(
        cfg, model, combined_loader, recon_calc=cfg["RECON_SAVE"]
    )

    if cfg["PLOT_WEIGHTS"]:
        logger.info("Plotting model weights")
        filepath = os.path.join(
            "reports",
            cfg["RUN_ID"],
            f"figures/model_weights.png",
        )

        plot_model_weights(filepath=filepath, model=model)
    return latent_space


def build_stackix(cfg, data_path_keys, mode="load_trained", split="train"):
    """Takes up to five different data modalities and trains a concatenated stackix
    and returns latent space and saves trained models

    ARGS:
        cfg - (dict): config dictionary
        data_path_keys - (list): list of config keys as defined in src/config.yaml
                                for example: [RNA_FILE_PATH, CNA_FILE_PATH]
        mode - (bool): indicator if the model should be loaded, trained or
                       trained and tuned (optuna). Default is load, options are
                       'load_trained', 'load_tuned', 'train', 'tune'
        split - (str): indicator if the model should be trained on the train, valid or test split
    RETURNS:
        concatenated_latent_space - (torch.tensor)

    """
    logger = getlogger(cfg)
    latent_spaces = []
    valid_latent_spaces = []
    name_helper = ""
    data_path_keys.sort()  # sort to ensure same order of data

    for k in data_path_keys:
        if k == "IMG":
            logger.info(
                f'image data only supported for translate case not for the case {cfg["MODEL_TYPE"]} '
            )
            continue
        name_helper += f"{k}_"  # for combined model name
        dataloader = get_loader(
            cfg=cfg, path_key=k, split_type=split
        )  # train or test depending on split
        validloader = get_loader(cfg=cfg, path_key=k, split_type="valid")

        # dense layer size of VAE per data modality
        if "DENSE_SIZE" in cfg:
            dense_size = cfg["DENSE_SIZE"]
        else:
            dense_size = int(dataloader.dataset.input_size() / 8)
        # print("[1] Dense size: " + str(dense_size))

        model = get_model(
            cfg,
            dataloader.dataset.input_size(),
            input_type=k,
            model_type="varix",
            latent_dim=dense_size,
        )
        model_name = "stackix_base_" + k + "_" + cfg["RUN_ID"] + ".pt"
        if mode == "load_trained":
            logger.info(f"Loading model {model_name}")
            assert os.path.exists(
                os.path.join("models", f"{cfg['RUN_ID']}", model_name)
            )
            model.load_state_dict(
                torch.load(os.path.join("models", f"{cfg['RUN_ID']}", model_name))
            )
            logger.info(f"model loaded: {model}")
        elif mode == "train":
            logger.info(f"Training model {model_name}")
            model, _ = train_ae_model(
                cfg,
                dataloader,
                model,
                model_name,
                model_type="varix",
                valid_loader=validloader,
            )
            logger.info(f"model trained: {model}")
        elif mode == "load_tuned":
            logger.info(f"Training and tuning model {model_name}")
            logger.info(f"Loading model {model_name}")

            assert os.path.exists(
                os.path.join("models", f"{cfg['RUN_ID']}", "tuned_" + model_name)
            )
            model = torch.load(
                os.path.join(
                    "models", "tuned", f"{cfg['RUN_ID']}", "tuned_" + model_name
                )
            )
            logger.info(f"model loaded tuned: {model}")
        elif mode == "tune":
            model = find_best_model(
                cfg,
                model_name="tuned_" + model_name,
                path_key=k,
                dataloader=dataloader,
                valloader=validloader,
                model_type="varix",
                latent_dim=dense_size,
            )
            logger.info(f"model tuned trained: {model}")
        else:
            raise ValueError(
                "Mode: {mode} not recognized (load, load_trained, train, load_tuned)"
            )
        cur_latent_space, _ = get_latent_space(
            cfg, model, dataloader, recon_calc=cfg["RECON_SAVE"]
        )
        val_cur_latent_space, _ = get_latent_space(
            cfg, model, validloader, recon_calc=cfg["RECON_SAVE"]
        )
        valid_latent_spaces.append(val_cur_latent_space)
        latent_spaces.append(cur_latent_space)
        """ Get latent space for validation data, so we need to use each of the
        models and get the latent space for the validation data """

    # HVAE part ---------------------------------------------------------------
    concat_name = "stackix_concat_" + name_helper + cfg["RUN_ID"] + ".pt"
    concated_latent_space = pd.concat(latent_spaces, axis=1)
    df = pd.DataFrame(concated_latent_space, index=dataloader.dataset.sample_ids)
    valid_concated_latent_space = pd.concat(valid_latent_spaces, axis=1)
    valid_df = pd.DataFrame(
        valid_concated_latent_space, index=validloader.dataset.sample_ids
    )

    valid_temp_path_key = f"CONCAT-{name_helper}_VALID_LATENT_SPACE"
    # valid_df.to_csv(
    #     os.path.join(f"data/interim/{cfg['RUN_ID']}/valid_latent_space.txt"), sep=cfg['DELIM']
    # )
    # print(f'valid df cols: {valid_df.columns}')
    valid_df.to_parquet(
        os.path.join(f"data/interim/{cfg['RUN_ID']}/valid_latent_space.parquet")
    )

    temp_path_key = f"CONCAT-{name_helper}_LATENT_SPACE"
    # df.to_csv(
    #     os.path.join(f"data/interim/{cfg['RUN_ID']}/latent_space.txt"), sep=cfg['DELIM']
    # )
    df.to_parquet(os.path.join(f"data/interim/{cfg['RUN_ID']}/latent_space.parquet"))

    cfg["DATA_TYPE"][temp_path_key] = {
        "FILE_PROC": os.path.join(f"data/interim/{cfg['RUN_ID']}/latent_space.parquet"),
        "TYPE": "CONCAT",
    }
    cfg["DATA_TYPE"][valid_temp_path_key] = {
        "FILE_PROC": os.path.join(
            f"data/interim/{cfg['RUN_ID']}/valid_latent_space.parquet"
        ),
        "TYPE": "CONCAT",
    }

    concat_loader = get_loader(cfg, temp_path_key, split_type=split)
    valid_concat_loader = get_loader(
        cfg,
        f"CONCAT-{name_helper}_VALID_LATENT_SPACE",  # temp_path_key for valid
        split_type="valid",  # does not matter for concat cases, since correct csv is loaded via temp_path_key
    )

    if mode == "train" or mode == "load_trained":
        model = get_model(
            cfg,
            concat_loader.dataset.input_size(),
            input_type=temp_path_key,
            model_type="stackix",
        )
        if mode == "train":
            model, _ = train_ae_model(
                cfg,
                concat_loader,
                model,
                concat_name,
                model_type="stackix",
                valid_loader=valid_concat_loader,
            )
            logger.info(f"(concatenated) model trained: {model}")
        elif mode == "load_trained":
            assert os.path.exists(
                os.path.join("models", f"{cfg['RUN_ID']}", concat_name)
            )
            model.load_state_dict(
                torch.load(os.path.join("models", f"{cfg['RUN_ID']}", concat_name))
            )
            logger.info(f"(concatenated) model loaded: {model}")
        else:
            raise ValueError(
                f"Mode: {mode} not recognized (load, load_trained, train, load_tuned)"
            )
    elif mode == "tune":
        if mode == "tune":
            model = find_best_model(
                cfg,
                model_name="tuned_" + concat_name,
                path_key=temp_path_key,
                dataloader=concat_loader,
                valloader=valid_concat_loader,
                model_type="stackix",
            )
            logger.info(f"(concatenated) model tuned trained: {model}")
    elif mode == "load_tuned":
        assert os.path.exists(
            os.path.join(
                "models", "tuned", f"{cfg['RUN_ID']}", "tuned_" + concat_name
            )
        )
        model = torch.load(
            os.path.join(
                "models", "tuned", f"{cfg['RUN_ID']}", "tuned_" + concat_name
            )
        )
        logger.info(f"(concatenated) model loaded tuned: {model}")
    else:
        raise ValueError(
            f"Mode: {mode} not recognized (load_trained, train, load_tuned, tune)"
        )

    logger.info(f"Getting concatenated latent space")
    concat_latent_space, recon_x = get_latent_space(
        cfg, model, concat_loader, recon_calc=cfg["RECON_SAVE"]
    )

    if cfg["PLOT_WEIGHTS"]:
        logger.info("Plotting model weights")
        filepath = os.path.join(
            "reports",
            cfg["RUN_ID"],
            f"figures/model_weights.png",
        )

        plot_model_weights(filepath=filepath, model=model)

    return concat_latent_space
