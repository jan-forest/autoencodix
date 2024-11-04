import glob
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import r2_score

from src.features.build_features import (
    CrossModaleDataset,
    ImageDataset,
    UnlabelledNumericDataset,
)
from src.models.models import ImageVAE, Ontix, Stackix, Vanillix, Varix
from src.models.tuning.models_for_tuning import (
    ImageVAETune,
    OntixTune,
    StackixTune,
    VanillixTune,
    VarixTune,
)
from src.utils.utils_basic import annealer, get_device, getlogger, total_correlation
from src.visualization.visualize import dim_red, plot_latent_2D, plot_latent_simple

# from math import exp


global g
g = torch.Generator()


def seed_worker(worker_id):
    """A function to seed workers for reproducibility"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_model_for_tuning(
    cfg,
    trial,
    input_dim,
    input_type,
    model_type,
    mask_1=None,
    mask_2=None,
    latent_dim=None,
):
    """returns pytorch model depending on input dataset

    ARGS:
        cfg - (dict): config dictionary
        trial - (optuna.trial): optuna trial object
        input_dim - (int): input dimension
        input_type - (str): input type, options are: 'NUMERIC', 'MIXED',
                            'IMG', 'COMBINED'
        model_type - (str): model type, options are: 'vanillix', 'varix', 'stackix', 'ontix', 'x-modalix'
    RETURNS:
        model - (torch.nn.Module): pytorch model

    """

    if not input_type in cfg["DATA_TYPE"].keys():
        raise ValueError(
            f"input_type {input_type} not in config.yaml DATA_TYPE, use RNA, CNA, CLIN and METH"
        )

    input_type = cfg["DATA_TYPE"][input_type]["TYPE"]
    match input_type:
        case "NUMERIC":
            match model_type:
                case "varix":
                    return VarixTune(
                        trial, input_dim=input_dim, cfg=cfg, latent_dim=latent_dim
                    )
                case "vanillix":
                    return VanillixTune(trial, input_dim=input_dim, cfg=cfg)
                case "stackix":
                    return StackixTune(trial, input_dim=input_dim, cfg=cfg)
                case "x-modalix":
                    return VarixTune(trial, input_dim=input_dim, cfg=cfg)
                case "ontix":
                    model = OntixTune(
                        trial=trial,
                        input_dim=input_dim,
                        latent_dim=cfg["LATENT_DIM_FIXED"],
                        mask_1=mask_1,
                        mask_2=mask_2,
                        cfg=cfg,
                    )
                    return model

                case _:
                    raise NotImplementedError(
                        "model_type not known use 'vanillix', 'varix', 'stackix', 'ontix', 'x-modalix'"
                    )
        case "MIXED":
            match model_type:
                case "varix":
                    return VarixTune(
                        trial, input_dim=input_dim, cfg=cfg, latent_dim=latent_dim
                    )
                case "vanillix":
                    return VanillixTune(trial, input_dim=input_dim, cfg=cfg)
                case "stackix":
                    return StackixTune(trial, input_dim=input_dim, cfg=cfg)
                case "ontix":
                    model = OntixTune(
                        trial=trial,
                        input_dim=input_dim,
                        latent_dim=cfg["LATENT_DIM_FIXED"],
                        mask_1=mask_1,
                        mask_2=mask_2,
                        cfg=cfg,
                    )
                    return model

                case "x-modalix":
                    return VarixTune(trial, input_dim=input_dim, cfg=cfg)
                case _:
                    raise NotImplementedError(
                        "model_type not known use 'vanillix', 'varix', 'stackix', 'ontix', 'x-modalix'"
                    )
        case "IMG":
            match model_type:
                case "hvae":
                    raise NotImplementedError(
                        f" {model_type} not implemented for IMG data"
                    )
                case "vae":
                    raise NotImplementedError(
                        f" {model_type} not implemented for IMG data"
                    )
                case "varix":
                    raise NotImplementedError(
                        f" {model_type} not implemented for IMG data"
                    )
                case "ae":
                    raise NotImplementedError(
                        f" {model_type} not implemented for IMG data"
                    )
                case "x-modalix":
                    return ImageVAETune(
                        trial=trial,
                        img_shape=input_dim,
                        latent_dim=cfg["LATENT_DIM_FIXED"],
                        cfg=cfg,
                    )
                # case "translate2":
                #     return ImageVAETune(
                #         trial=trial,
                #         img_shape=input_dim,
                #         latent_dim=cfg["LATENT_DIM_FIXED"],
                #         cfg=cfg,
                #     )
                case _:
                    raise NotImplementedError(
                        f"model_type {model_type} has to be in 'vanillix', 'varix', 'stackix', 'ontix', 'x-modalix'"
                    )
        case "COMBINED":
            match model_type:
                case "varix":
                    return VarixTune(
                        trial, input_dim=input_dim, cfg=cfg, latent_dim=latent_dim
                    )
                case "vanillix":
                    return VanillixTune(trial, input_dim=input_dim, cfg=cfg)
                case "stackix":
                    return StackixTune(trial, input_dim=input_dim, cfg=cfg)
                case "ontix":
                    model = OntixTune(
                        trial=trial,
                        input_dim=input_dim,
                        latent_dim=cfg["LATENT_DIM_FIXED"],
                        mask_1=mask_1,
                        mask_2=mask_2,
                        cfg=cfg,
                    )
                    return model
                case _:
                    raise NotImplementedError(
                        "model_type not known use 'vanillix', 'varix', 'stackix', 'ontix', 'x-modalix'"
                    )

        case _:
            raise ValueError(
                "input type not supported: use NUMERIC, MIXED, IMG or COMBINED"
            )


def get_model(
    cfg, input_dim, input_type, model_type, latent_dim=None, mask_1=None, mask_2=None
):
    """returns pytorch model depending on input dataset
    ARGS:
        cfg (dict): config dictionary
        input_dim (int): number of features
        input_type (str): as defined in src/config.yaml
        model_type (str): as defined in src/config.yaml MODEL_TYPE 'stackix', 'varix, 'vanillix', 'ontix', translate, translate2
    RETURNS:
        model (pytorch model): model to be trained
    """

    logger = getlogger(cfg)
    logger.debug(f"input_dim: {input_dim}")

    if cfg["FIX_RANDOMNESS"] == "all" or cfg["FIX_RANDOMNESS"] == "training":
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(cfg["GLOBAL_SEED"])
        torch.cuda.manual_seed(cfg["GLOBAL_SEED"])
        torch.cuda.manual_seed_all(cfg["GLOBAL_SEED"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # Use latent dim from cfg if not stated otherwise
    if latent_dim == None:
        latent_dim = cfg["LATENT_DIM_FIXED"]

    # check if input_type is in dict, if not raise error
    input_type = cfg["DATA_TYPE"][input_type]["TYPE"]
    if not type(cfg["LATENT_DIM_FIXED"]) == int:
        raise ValueError("LATENT_DIM_FIXED has to be an integer")

    if "DROP_P" in cfg:
        global_p = cfg["DROP_P"]
    else:
        global_p = 0.1

    match input_type:
        case "NUMERIC":
            match model_type:
                case "stackix":
                    return Stackix(
                        input_dim=input_dim, latent_dim=latent_dim, global_p=global_p
                    )
                case "varix":
                    return Varix(
                        input_dim=input_dim, latent_dim=latent_dim, global_p=global_p
                    )
                case "vanillix":
                    return Vanillix(
                        input_dim=input_dim, latent_dim=latent_dim, global_p=global_p
                    )
                case "ontix":
                    model = Ontix(
                        input_dim=input_dim,
                        latent_dim=latent_dim,
                        dec_fc_layer=cfg["NON_ONT_LAYER"],
                        mask_1=mask_1,
                        mask_2=mask_2,
                        global_p=global_p,
                    )
                    return model

                case "x-modalix":
                    return Varix(
                        input_dim=input_dim, latent_dim=latent_dim, global_p=global_p
                    )
                case _:
                    raise NotImplementedError(
                        f"model_type: {model_type} has to be 'stackix', 'varix, 'vanillix', 'ontix', translate, translate2"
                    )
        case "MIXED":
            match model_type:
                case "stackix":
                    return Stackix(
                        input_dim=input_dim, latent_dim=latent_dim, global_p=global_p
                    )
                case "varix":
                    return Varix(
                        input_dim=input_dim, latent_dim=latent_dim, global_p=global_p
                    )
                case "vanillix":
                    return Vanillix(
                        input_dim=input_dim, latent_dim=latent_dim, global_p=global_p
                    )
                case "ontix":
                    model = Ontix(
                        input_dim=input_dim,
                        latent_dim=latent_dim,
                        dec_fc_layer=cfg["NON_ONT_LAYER"],
                        mask_1=mask_1,
                        mask_2=mask_2,
                        global_p=global_p,
                    )
                    return model

                case "x-modalix":
                    return Varix(
                        input_dim=input_dim, latent_dim=latent_dim, global_p=global_p
                    )
                case _:
                    raise NotImplementedError(
                        "model_type has to be 'stackix', 'varix, 'vanillix', 'ontix', translate, translate2"
                    )

        case "IMG":
            match model_type:
                case "stackix":
                    raise NotImplementedError(
                        f" {model_type} not implemented for IMG data"
                    )
                case "varix":
                    raise NotImplementedError(
                        f" {model_type} not implemented for IMG data"
                    )
                case "vanillix":
                    raise NotImplementedError(
                        f" {model_type} not implemented for IMG data"
                    )
                case "x-modalix":
                    return ImageVAE(
                        img_shape=input_dim,
                        latent_dim=latent_dim,
                        hidden_dim=int(input_dim[1] / 4),
                    )
                # case "translate2":
                #     return ImageVAE(
                #         img_shape=input_dim,
                #         latent_dim=latent_dim,
                #         hidden_dim=int(input_dim[1] / 4),
                #     )
                case _:
                    raise NotImplementedError(
                        f"model_type {model_type} has to be 'stackix', 'varix, 'vanillix', 'ontix', translate, translate2"
                    )

        case "COMBINED":
            match model_type:
                case "stackix":
                    return Stackix(
                        input_dim=input_dim, latent_dim=latent_dim, global_p=global_p
                    )
                case "varix":
                    return Varix(
                        input_dim=input_dim, latent_dim=latent_dim, global_p=global_p
                    )
                case "vanillix":
                    return Vanillix(
                        input_dim=input_dim, latent_dim=latent_dim, global_p=global_p
                    )
                case "ontix":
                    model = Ontix(
                        input_dim=input_dim,
                        latent_dim=latent_dim,
                        dec_fc_layer=cfg["NON_ONT_LAYER"],
                        mask_1=mask_1,
                        mask_2=mask_2,
                        global_p=global_p,
                    )
                    return model
                case _:
                    raise NotImplementedError(
                        "model_type hast to be 'stackix', 'varix, 'vanillix', 'ontix', translate, translate2"
                    )

        case _:
            raise ValueError(
                "input type not supported: use NUMERIC, MIXED, IMG or COMBINED"
            )


def get_loader(cfg, path_key, split_type="train", tunetranslate=False):
    """returns a custom pytorch dataloader object for training and testing
    ARGS:
        cfg - (dict): config dictionary
        path_key - (str): as defined in src/config.yaml
        split_type - (str): indicator if train, valid or test dataset
    RETURNS:
        loader - (DataLoader): pytorch DataLoader for training data

    """
    logger = getlogger(cfg=cfg)
    if split_type == "train" and (
        cfg["FIX_RANDOMNESS"] == "random" or cfg["FIX_RANDOMNESS"] == "data_split"
    ):
        shuffle_active = True
    else:
        shuffle_active = False

    if cfg["MODEL_TYPE"] == "x-modalix" and not tunetranslate:
        logger.info("Cross Modalix Case")
        return DataLoader(
            dataset=CrossModaleDataset(cfg=cfg, split_type=split_type),
            batch_size=cfg["BATCH_SIZE"],
            shuffle=shuffle_active,
            worker_init_fn=seed_worker,
            generator=g,
        )
    elif cfg["DATA_TYPE"][path_key]["TYPE"] == "IMG":
        logger.info("img case, img dataloader")
        return DataLoader(
            dataset=ImageDataset(
                cfg, split_type=split_type, data_modality="IMG", path_key=path_key
            ),
            batch_size=cfg["BATCH_SIZE"],
            shuffle=shuffle_active,
            worker_init_fn=seed_worker,
            generator=g,
        )

    dataloader = DataLoader(
        UnlabelledNumericDataset(
            cfg=cfg,
            split_type=split_type,
            data_modality=cfg["DATA_TYPE"][path_key]["TYPE"],
            path_key=path_key,
        ),
        batch_size=cfg["BATCH_SIZE"],
        shuffle=shuffle_active,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return dataloader


def compute_kernel(cfg, x, y):
    """computes kernel for maximum mean discrepancy (MMD) calculation
    ARGS:
        x - (torch.tensor): input data
        y - (torch.tensor): input data
    RETURNS:
        kernel - (torch.tensor): kernel matrix (x.shape[0]  y.shape[0])

    """
    device = get_device(verbose=False)
    x_size = x.size(0)
    y_size = y.size(0)
    dim = torch.tensor(x.size(1)).to(device)
    x = x.unsqueeze(1).to(device)  # (x_size, 1, dim)
    y = y.unsqueeze(0).to(device)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def compute_mmd(true_samples, z, cfg):
    """computes maximum mean discrepancy (MMD) between two input datasets
    ARGS:
        true_samples - (torch.tensor): sampled from normal distribution N(0,1)
        z - (torch.tensor): sampled from normal distribution with N(mu, logvar)
    RETURNS:
        mmd - (torch.tensor): mmd value 1d tensor

    """
    logger = getlogger(cfg=cfg)
    true_samples_kernel = compute_kernel(cfg, true_samples, true_samples)
    z_kernel = compute_kernel(z, z)
    ztr_kernel = compute_kernel(true_samples, z)
    if cfg["LOSS_REDUCTION"] == "mean":
        return true_samples_kernel.mean() + z_kernel.mean() - 2 * ztr_kernel.mean()
    elif cfg["LOSS_REDUCTION"] == "sum":
        logger.warning(
            "Sum reduction for MMD works as loss function, but is strictly mathematically not MMD anymore"
        )
        return true_samples_kernel.sum() + z_kernel.sum() - 2 * ztr_kernel.sum()
    else:
        raise ValueError(
            f"LOSS_REDUCTION has to be sum or mean, not {cfg['LOSS_REDUCTION']}"
        )


def compute_kl(mu, logvar, cfg):
    """computes kl divergence between fitted normal distribution and standard
    normal  distribution
    ARGS:
        mu - (torch.tensor): mean of fitted normal distribution
        logvar - (torch.tensor): log variance of fitted normal distribution
    RETURNS:
        kl - (torch.tensor): kl divergence value 1d tensor

    """
    if cfg["LOSS_REDUCTION"] == "sum":
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    elif cfg["LOSS_REDUCTION"] == "mean":
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        raise ValueError(
            f"LOSS_REDUCTION has to be sum or mean, not {cfg['LOSS_REDUCTION']}"
        )


def loss_function(
    cfg, recon_x, x, mu, logvar, dim, model_type, epoch=None, pretrain=False
):
    """returns the loss function for the vae model
    ARGS:
        recon_x - (torch.tensor): reconstructed data
        x - (torch.tensor): original data
        mu - (torch.tensor): mean of latent space
        logvar - (torch.tensor): log variance of latent space
        dim - (int): dimension of data
    RETURNS:
        loss - (torch.tensor): weighted sum of rloss and vae_loss

    """
    total_epochs = cfg["EPOCHS"]
    if pretrain:
        total_epochs = cfg["PRETRAIN_EPOCHS"]
    logger = getlogger(cfg=cfg)
    if not type(dim) == int:
        dim = dim[1]

    x = x.view(-1, dim)
    recon_x = recon_x.view(-1, dim)

    if cfg["VAE_LOSS"] == "KL":
        vae_loss = compute_kl(mu=mu, logvar=logvar, cfg=cfg)
    elif cfg["VAE_LOSS"] == "MMD":
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu, dtype=float)
        z = mu + eps * std
        true_samples = torch.randn(
            cfg["BATCH_SIZE"], cfg["LATENT_DIM_FIXED"], requires_grad=False
        )
        vae_loss = (
            torch.tensor(0)
            if model_type == "vanillix"
            else compute_mmd(true_samples=true_samples, z=z, cfg=cfg)
        )
    else:
        raise NotImplementedError("VAE_LOSS has to be KL or MMD")

    if cfg["RECONSTR_LOSS"] == "BCE":
        if torch.min(x) < 0.0 or torch.max(x) > 1.0:
            logger.warning(
                "Input data not in [0,1]. If you use BCE loss, use MinMaxScaling. \n Maybe the error is elsewhere. We will apply a sigmoid to your input data"
            )
            logger.warning(f"min: {torch.min(x)}")
            logger.warning(f"max: {torch.max(x)}")
            x = torch.sigmoid(x)

        # we use BCE with logits, to apply a Sigmoid layer to the model outputs in the BCE case
        # We don't do this at model level, because depending on scaling and loss (MSE) this is wrong
        r_loss = F.binary_cross_entropy_with_logits(
            recon_x, x, reduction=cfg["LOSS_REDUCTION"]
        )
        m = torch.nn.Sigmoid()
        r2 = r2_score(
            torch.transpose(m(recon_x), 0, 1),
            torch.transpose(x, 0, 1),
            multioutput="uniform_average",
        )
    elif cfg["RECONSTR_LOSS"] == "MSE":
        r_loss = F.mse_loss(recon_x, x, reduction=cfg["LOSS_REDUCTION"])
        r2 = r2_score(
            torch.transpose(recon_x, 0, 1),
            torch.transpose(x, 0, 1),
            multioutput="uniform_average",
        )
    else:
        raise NotImplementedError("RECONSTR_LOSS has to be BCE or MSE")

    ## Calculate r2 explained variance for AE precition quality comparisons
    annealing = annealer(
        epoch_current=epoch, total_epoch=total_epochs, func=cfg["ANNEAL"]
    )
    return r_loss, vae_loss * cfg["BETA"] * annealing, r2


def get_latent_space(cfg, model, data_loader, recon_calc=False, save_file=True):
    """saves latent_space of current run in a csv
    ARGS:
        model - (torch.model)
        data_loader - (torch.utils.data.DataLoader)
    RETURNS:
        latent_space - (pd.DataFrame)

    """
    logger = getlogger(cfg=cfg)
    latent_space = []
    recon_x = []
    id_list = []
    device = get_device(verbose=False)
    model.to(device)
    model.eval()
    for batch, index in data_loader:
        batch.to(device)

        mu, logvar = model.encode(batch)
        id_list.append(index)
        if cfg["MODEL_TYPE"] == "vanillix":
            # for vanilla case mu is already the latent space
            latent_space.append(mu)
        else:
            z = model.reparameterize(mu, logvar)
            latent_space.append(z)

        if recon_calc:
            recon_x_batch = model(batch)
            recon_x.append(recon_x_batch[0])

    latent_space = torch.cat(latent_space, dim=0)

    latent_space = latent_space.detach().cpu().numpy()

    id_list = np.concatenate(id_list, axis=0)

    latent_space = pd.DataFrame(latent_space, index=data_loader.dataset.data.index)
    latent_space.index = id_list

    if recon_calc:
        recon_x = torch.cat(recon_x, dim=0)
        recon_x = recon_x.detach().cpu().numpy()
        recon_x = pd.DataFrame(recon_x, index=data_loader.dataset.data.index)

        recon_x.index = id_list

    check = all(latent_space.index == (data_loader.dataset.sample_ids))
    path_key = data_loader.dataset.path_key
    if save_file:
        latent_space.columns = [f"L_{path_key}_" + str(x) for x in latent_space.columns]

        if cfg["RECON_SAVE"]:
            recon_x.columns = data_loader.dataset.get_cols()
            recon_x.to_parquet(
                os.path.join(
                    "reports",
                    f"{cfg['RUN_ID']}",
                    f"recon_x_{path_key}_{cfg['RUN_ID']}.parquet",
                )
            )
            logger.info(
                f"Predictions saved as reports/{cfg['RUN_ID']}/recon_x_{path_key}_{cfg['RUN_ID']}.parquet"
            )
    return latent_space, recon_x


def get_checkpoint_interval(epochs):
    """returns the interval for saving checkpoints
    ARGS:
        epochs - (int): number of epochs
    RETURNS:
        interval - (int): interval for saving checkpoints

    """
    if type(epochs) != int:
        raise TypeError("epochs has to be an int")
    if epochs < 0:
        raise ValueError("epochs has to be positive")

    if epochs < 100:
        return 5
    elif epochs < 500:
        return 25
    elif epochs < 3000:
        return 50
    elif epochs < 10000:
        return 100
    else:
        return 10000


def get_last_checkpoint(file_list, model_name):
    """returns the last checkpoint of a model
    ARGS:
        model_name - (str): name of model
    RETURNS:
        checkpoint - (dict): dict with file name and epoch of last checkpoint

    """
    highest_epoch = -1
    highest_epoch_file = ""
    for f in file_list:
        basename = os.path.basename(f)
        checkpoint_info = basename.split("_")[0]

        if not "checkpoint" in basename:
            continue
        if not model_name in basename:
            continue
        checkpoint_epoch = int(checkpoint_info.split("checkpoint")[1])
        if checkpoint_epoch > highest_epoch:
            highest_epoch = checkpoint_epoch
            highest_epoch_file = f
    return {"file": highest_epoch_file, "epoch": highest_epoch}


def has_nans(model, cfg):
    logger = getlogger(cfg=cfg)
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            logger.error(f"Model has nan: {name}")
            return True
    return has_nan


def check_losses(
    r_loss,
    vae_loss,
    model_name,
    epoch,
    batch,
    cfg,
    recon_x,
    mu,
    logvar,
    mode,
    model,
    debug=False,
):
    logger = getlogger(cfg=cfg)
    if r_loss < 0 or torch.isnan(r_loss):
        logger.warning(
            f"Reconstruction Loss is negative or nan: {r_loss} in epch {epoch} for model {model_name}"
        )
        error_dict = {
            "r_loss": r_loss,
            "epoch": epoch,
            "batch": batch,
            "recon_x": recon_x,
            "model_name": model_name,
            "model": model,
        }
        file_name = f"error_dict_{model_name}_{epoch}_{mode}.pkl"
        if debug:
            with open(
                os.path.join(f'data/interim/{cfg["RUN_ID"]}', file_name), "wb"
            ) as f:
                pickle.dump(error_dict, f)
            raise ValueError("Reconstruction Loss is negative or nan")
        return False
    if vae_loss < -0 or torch.isnan(vae_loss):
        logger.warning(
            f"VAE Loss is negative or nan: {vae_loss} in epch {epoch} for model {model_name}"
        )
        error_dict = {
            "vaeloss": vae_loss,
            "epoch": epoch,
            "batch": batch,
            "mu": mu,
            "logvar": logvar,
            "model_name": model_name,
            "recon_x": recon_x,
            "model": model,
        }
        file_name = f"error_dict_{model_name}_{epoch}_{mode}.pkl"
        return False
    # if no error
    return True


def train_ae_model(
    cfg,
    train_loader,
    model,
    model_name,
    model_type="varix",
    valid_loader=None,
    optimizer=None,
    mask_1=None,
    mask_2=None,
):
    """trains a model and saves the model and losses
    ARGS:
        train_loader - (DataLoader): pytorch DataLoader for training data
        valid_loader - (DataLoaser): pytorch DataLoader for valid data
        model - (nn.Module): pytorch model
        model_name - (str): name of model
        loss_type - (str): type of loss function (optional, default='loss')
    RETURNS:
        model - (nn.Module): trained pytorch model
        losses - (dict): dictionary of losses per epoch

    """
    logger = getlogger(cfg=cfg)
    if optimizer == None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["LR_FIXED"], weight_decay=cfg["WEIGHT_DECAY"]
        )

    writer = SummaryWriter(f'reports/{cfg["RUN_ID"]}/tensorboards')

    if valid_loader == None:
        logger.warning("Valid loader is None, non, using train loader for validation")
        valid_loader = train_loader

    data_mode = train_loader.dataset.data_modality.strip()
    loss_type = f"{data_mode}_loss"
    layout = {
        f"{loss_type}": {
            "Loss train/val": ["Multiline", ["loss/train", "loss/valid"]],
        },
    }

    if cfg["CHECKPT_PLOT"] and model.latent_dim < 20:
        anno_name = [
            data_type
            for data_type in cfg["DATA_TYPE"]
            if cfg["DATA_TYPE"][data_type]["TYPE"] == "ANNOTATION"
        ]
        lat_coverage_epoch = pd.DataFrame(columns=["epoch", "coverage", "total_correlation"])
        if len(anno_name) == 1:

            clin_data = pd.read_parquet(
                os.path.join(
                    "data/processed",
                    cfg["RUN_ID"],
                    anno_name[0] + "_data.parquet",
                )
            )
        else:
            logger.warning(
                "No ANNOTATION data type found. Plotting latent without param."
            )

    device = get_device()
    if cfg["FIX_RANDOMNESS"] == "all" or cfg["FIX_RANDOMNESS"] == "training":
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(cfg["GLOBAL_SEED"])
        torch.cuda.manual_seed(cfg["GLOBAL_SEED"])
        torch.cuda.manual_seed_all(cfg["GLOBAL_SEED"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model.to(device)
    writer.add_custom_scalars(layout)
    # check if model is already trainedo
    last_epoch = {"file": "", "epoch": 0}
    if cfg["START_FROM_LAST_CHECKPOINT"]:
        files = glob.glob(os.path.join(f'models/{cfg["RUN_ID"]}', "*.pt"))
        # we don't know if the model was trained or tuned last, so we check both when we get no match,
        # for train, we know, tune has the highest epoch and vice versa
        # both cases should not occur within one RUN_ID
        last_epoch = get_last_checkpoint(files, model_name)
        logger.info(f'Loading checkpoint from Last epoch: {last_epoch["epoch"]}')
        logger.info(f'Loading model from last file: {last_epoch["file"]}')
        logger.info(f"model name: {model_name}")
        if last_epoch["epoch"] == -1:
            raise ValueError(f"No checkpoint found for model {model_name}")
        model.load_state_dict(torch.load(last_epoch["file"]))
    losses = []
    vae_losses = []
    r2_train = []
    losses_val = []
    vae_losses_val = []
    r2_val = []

    logger.info(f'total epochs:{cfg["EPOCHS"]}')
    checkpoint_interval = get_checkpoint_interval(cfg["EPOCHS"])
    for epoch in range(last_epoch["epoch"], cfg["EPOCHS"]):
        loss = 0
        r_loss = 0
        vae_loss = 0
        r2_list = []
        model.train()
        batch_counter = 0
        for batch, _ in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch)
            r_loss_batch, vae_loss_batch, r2_batch = loss_function(
                cfg,
                recon_x,
                batch,
                mu,
                logvar,
                train_loader.dataset.input_size(),
                model_type,
                epoch=epoch,
            )
            _ = check_losses(
                r_loss_batch,
                vae_loss_batch,
                model_name,
                f"{epoch}_{batch_counter}",
                batch,
                cfg,
                recon_x,
                mu,
                logvar,
                mode="train",
                model=model,
            )
            batch_loss = r_loss_batch + vae_loss_batch
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            ## Mask gradients for ontology VAE
            if model_type == "ontix":
                model.decoder.apply(model._positive_dec)
                with torch.no_grad():
                    mask_1 = mask_1.to(device)
                    model.decoder[-1].weight.mul_(mask_1)  ## Sparse Decoder Level
                    if not mask_2 == None:
                        mask_2 = mask_2.to(device)
                        model.decoder[-2].weight.mul_(mask_2)

            optimizer.step()
            loss += batch_loss.detach()
            r_loss += r_loss_batch.detach()
            vae_loss += vae_loss_batch.detach()
            r2_list.append(r2_batch.cpu())
            batch_counter += 1

        r_loss = r_loss / len(train_loader.dataset)
        vae_loss = vae_loss / len(train_loader.dataset)
        loss = loss / len(train_loader.dataset)
        r2 = np.mean(r2_list)

        if cfg["CHECKPT_PLOT"] and model.latent_dim < 20:
            latent_space, recon_x = get_latent_space(
                cfg, model, train_loader, recon_calc=False, save_file=False
            )
            bins_per_dim = int(
                np.power(len(latent_space.index), 1 / len(latent_space.columns))
            )
            if bins_per_dim < 2:
                logger.warning(
                    "Coverage calculation is meaningless since combination of sample size and latent dimension results in less than 2 bins."
                )
            latent_bins = latent_space.apply(lambda x: pd.cut(x, bins=bins_per_dim))
            latent_bins = pd.Series(zip(*[latent_bins[col] for col in latent_bins]))

            lat_coverage = dict()
            lat_coverage["coverage"] = len(latent_bins.unique()) / np.power(
                bins_per_dim, len(latent_space.columns)
            )
            lat_coverage["epoch"] = epoch
            lat_coverage["total_correlation"] = total_correlation(latent_space=latent_space)
            lat_coverage_epoch = pd.concat(
                [
                    lat_coverage_epoch,
                    pd.DataFrame(lat_coverage, index=[lat_coverage["epoch"]]),
                ]
            )

        if epoch < 50 and cfg["CHECKPT_PLOT"]:
            checkpoint_interval = 10
        else:
            checkpoint_interval = get_checkpoint_interval(epochs=cfg["EPOCHS"])

        if epoch % checkpoint_interval == 0:
            writer.add_scalar("loss/train", loss, epoch)
            writer.add_scalar(f"{loss_type}/loss/train", loss, epoch)

            logger.info(
                f"Train Set: Epoch: {epoch}, Loss: {'{:.2f}'.format(loss)}, r_Loss: {'{:.2f}'.format(r_loss)}, vae_loss: {'{:.2f}'.format(vae_loss)}, r2: {'{:.2f}'.format(r2)}"
            )
            if cfg["CHECKPT_PLOT"] and model.latent_dim < 20:
                logger.info("Plot latent space at checkpoint")
                latent_space, recon_x = get_latent_space(
                    cfg, model, train_loader, recon_calc=False, save_file=False
                )
                if len(anno_name) == 1:
                    rel_index = clin_data.index.intersection(latent_space.index)
                    embedding = dim_red(
                        latent_space=latent_space.loc[rel_index, :],
                        cfg=cfg,
                        method=cfg["DIM_RED_METH"],
                        seed=cfg["GLOBAL_SEED"],
                    )
                    if "FIX_XY_LIM" in cfg:
                        xlim = tuple(cfg["FIX_XY_LIM"][0])
                        ylim = tuple(cfg["FIX_XY_LIM"][1])
                    else:
                        xlim = None
                        ylim = None
                    plot_latent_2D(
                        cfg=cfg,
                        embedding=embedding,
                        labels=list(clin_data.loc[rel_index, cfg["CLINIC_PARAM"][0]]),
                        save_fig=os.path.join(
                            "reports",
                            cfg["RUN_ID"],
                            f"figures/latent2D_epoch{epoch}.png",
                        ),
                        param=cfg["CLINIC_PARAM"][0],
                        figsize=(15, 10),
                        xlim=xlim,
                        ylim=ylim,
                        # scale="symlog",
                        scale=None,
                        no_leg=True,
                    )
                else:
                    embedding = dim_red(
                        latent_space=latent_space,
                        cfg=cfg,
                        method=cfg["DIM_RED_METH"],
                        seed=cfg["GLOBAL_SEED"],
                    )
                    plot_latent_simple(
                        cfg=cfg,
                        embedding=embedding,
                        save_fig=os.path.join(
                            "reports",
                            cfg["RUN_ID"],
                            f"figures/latent2D_epoch{epoch}.png",
                        ),
                    )

            if cfg["CHECKPT_SAVE"]:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        "models",
                        f"{cfg['RUN_ID']}",
                        f"checkpoint{epoch}_{model_name}",
                    ),
                )

        losses.append(r_loss.detach())
        vae_losses.append(vae_loss.detach())
        r2_train.append(r2)

        # validation loss
        loss = 0
        r_loss = 0
        vae_loss = 0
        r2_list = []

        model.eval()
        # valid_loader = train_loader
        batch_counter = 0
        for batch, _ in valid_loader:
            batch.to(device)
            with torch.no_grad():
                recon_x, mu, logvar = model(batch)
            r_loss_batch, vae_loss_batch, r2_batch = loss_function(
                cfg,
                recon_x,
                batch,
                mu,
                logvar,
                valid_loader.dataset.input_size(),
                model_type,
                # epoch=epoch,
                epoch=None,  # Switch off beta anneal weight for valid loss
            )
            # batch_loss = r_loss_batch + r_loss_batch
            batch_loss = r_loss_batch + vae_loss_batch
            loss += batch_loss.detach()
            r_loss += r_loss_batch.detach()
            vae_loss += vae_loss_batch.detach()
            r2_list.append(r2_batch.cpu())
            check_losses(
                r_loss,
                vae_loss,
                model_name,
                f"{epoch}_{batch_counter}_val",
                batch,
                cfg,
                recon_x,
                mu,
                logvar,
                mode="train",
                model=model,
            )
            batch_counter += 1

        r_loss = r_loss / len(valid_loader.dataset)
        vae_loss = vae_loss / len(valid_loader.dataset)
        loss = loss / len(valid_loader.dataset)
        r2 = np.mean(r2_list)

        if epoch % checkpoint_interval == 0:
            writer.add_scalar("loss/valid", loss, epoch)
            writer.add_scalar(f"{loss_type}/loss/valid", loss, epoch)
            logger.info(
                f" Valid Set: Epoch: {epoch} Loss: {'{:.2f}'.format(loss)}, r2: {'{:.2f}'.format(r2)}"
            )
        losses_val.append(r_loss.detach())
        vae_losses_val.append(vae_loss.detach())
        r2_val.append(r2)
    writer.close()
    loss_dict = {
        "train_vae_loss": [float(x) for x in vae_losses],
        "train_recon_loss": [float(x) for x in losses],
        "train_r2": [float(x) for x in r2_train],
        "train_total_loss": [float(x + y) for x, y in zip(losses, vae_losses)],
        "valid_vae_loss": [float(x) for x in vae_losses_val],
        "valid_recon_loss": [float(x) for x in losses_val],
        "valid_total_loss": [float(x + y) for x, y in zip(losses_val, vae_losses_val)],
        "valid_r2": [float(x) for x in r2_val],
    }

    if cfg["CHECKPT_PLOT"] and model.latent_dim < 20:
        lat_coverage_epoch.to_parquet(
            os.path.join(
                "reports",
                f"{cfg['RUN_ID']}",
                f'latent_cov_per_epoch_{cfg["RUN_ID"]}.parquet',
            )
        )

    loss_df = pd.DataFrame(loss_dict)
    # loss_df["loss_type"] = loss_type  ## not used
    if cfg["MODEL_TYPE"] == "vanillix":
        loss_df.drop(
            ["train_vae_loss", "valid_vae_loss"], axis=1, inplace=True
        )  # Drop VAE loss for vanillix loss curves

    model_name_short = (
        model_name.removesuffix(".pt").removesuffix(cfg["RUN_ID"]).removesuffix("_")
    )
    loss_df.to_parquet(
        os.path.join(
            "reports",
            f"{cfg['RUN_ID']}",
            # f'losses_{model_name}_{cfg["RUN_ID"]}.parquet',
            f"losses_{model_name_short}.parquet",
        )
    )

    torch.save(
        model.state_dict(),
        os.path.join("models", f"{cfg['RUN_ID']}", model_name),
    )
    logger.info(f"Model saved as models/{cfg['RUN_ID']}/{model_name}")
    return model, losses


def make_mask_decoder(prev_lay_dim, next_lay_dim, ont_dic, node_list, cfg):
    """Calculates mask as tensor matrix with zeros or ones (weights) for a sparse decoder layer based on the provided ontology.
    ARGS:
        prev_lay_dim - (int): input dimension of sparse decoder layer
        next_lay_dim - (int): output dimension of sparse decoder layer
        ont_dic - (dict): Dictionary containing the relationship of ontologies
        node_list - (list): List of ontology names of output nodes. Used to infer correct order of nodes and weight matrix
    RETURNS:
        mask - (torch.tensor): Mask as tensor matrix with zeros or ones

    """
    logger = getlogger(cfg=cfg)
    mask = torch.zeros(next_lay_dim, prev_lay_dim)
    p_int = 0
    if len(node_list) == next_lay_dim:
        if len(ont_dic.keys()) == prev_lay_dim:
            for p_id in ont_dic:
                feature_list = ont_dic[p_id]
                for f_id in feature_list:
                    if f_id in node_list:
                        f_int = node_list.index(f_id)
                        mask[f_int, p_int] = 1

                p_int += 1
        else:
            logger.warning(
                "Mask layer cannot be calculated. Ontology key list does not match previous layer dimension"
            )
            logger.warning("Returning zero mask")
    else:
        logger.warning(
            "Mask layer cannot be calculated. Output layer list does not match next layer dimension"
        )
        logger.warning("Returning zero mask")

    if torch.max(mask) < 1:
        logger.warning(
            "You provided an ontology with no connections between layers in the decoder. Please check your ontology definition."
        )

    return mask


def distance_latent(
    cfg,
    latent_a,
    latent_b,
    a_idx_paired=None,
    b_idx_paired=None,
):
    if (a_idx_paired == None) and (b_idx_paired == None):
        if cfg["LOSS_REDUCTION"] == "sum":
            distance_ab = torch.sum(  # Sum along samples
                torch.mean(
                    torch.abs(latent_a - latent_b), dim=1
                )  # Mean along Latent Dimensions
            )
        elif cfg["LOSS_REDUCTION"] == "mean":
            distance_ab = torch.mean(
                torch.mean(
                    torch.abs(latent_a - latent_b), dim=1
                )  # Mean along Latent Dimensions
            )
        else:
            raise ValueError(
                f"LOSS_REDUCTION has to be 'sum' or 'mean', not {cfg['LOSS_REDUCTION']}"
            )

        return distance_ab
    else:
        raise ValueError("Wrong function call, always call with a and b indices = None")
        #### INFO TO TOGGLED CODE, MAYBE NEED FOR LATER IF WE ALLOW PAARTIALL PAIRED SAMPLES IN  LATENT DISTANCE
        # else:
        #     assert len(a_idx_paired) == len(b_idx_paired)
        #     if cfg["LOSS_REDUCTION"] == "sum":
        #         distance_ab = distance_ab = torch.sum(  # Sum along samples
        #             torch.mean(
        #                 torch.abs(
        #                     latent_a[a_idx_paired, :]
        #                     - latent_b[b_idx_paired, :]  # Select for paired
        #                 ),
        #                 dim=1,
        #             )  # Mean along Latent Dimensions
        #         )
        #     elif cfg["LOSS_REDUCTION"] == "mean":
        #         distance_ab = torch.mean(
        #             torch.mean(
        #                 torch.abs(
        #                     latent_a[a_idx_paired, :]
        #                     - latent_b[b_idx_paired, :]  # Select for paired
        #                 ),
        #                 dim=1,
        #             )  # Mean along Latent Dimensions
        #         )

        #     if cfg["WEIGHTED_DISTANCE_LATENT"]:
        #         distance_ab = distance_ab * (
        #             latent_a.shape[0] / len(a_idx_paired)
        #         )  # Correct for missing overlap to mimick full overlap
