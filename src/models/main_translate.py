import copy
import os
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.models import LatentSpaceClassifier
from src.models.tuning.tuning import find_best_model
from src.utils.utils import distance_latent  # annealer,
from src.utils.utils import (
    get_checkpoint_interval,
    get_device,
    get_loader,
    get_model,
    loss_function,
)
from src.utils.utils_basic import annealer, get_annealing_epoch, get_device

global g
g = torch.Generator()


def pretrain_img(loaders, model, cfg, img_direction):
    """Pretrain Image VAE for CrossModalix
    ARGS:
        loaders - (dict): contains all dataloader
        model - (nn.model)
        cfg - (dict)
    RETURNS:
        dict: with model, train and valid losses as keys

    """
    device = get_device(cfg)
    losses = []
    valid_losses = []
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["LR_FIXED"],
        weight_decay=cfg["WEIGHT_DECAY"],
    )

    for epoch in range(cfg["PRETRAIN_EPOCHS"]):
        # print(epoch)
        running_loss = 0.0
        valid_running_loss = 0.0
        # for the annealing we need to passs the current epoch to the loss function
        # in order to find the annealing weight for beta, if we don't want
        # to use annealing we can pass None
        loss_anneal_epoch = epoch if cfg["ANNEAL_PRETRAINING"] else None

        for _, from_batch, to_batch in loaders["train"]:
            model.train()
            batch = from_batch if img_direction == "from" else to_batch
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(batch)
            r_loss_batch, vae_loss_batch, *_ = loss_function(
                cfg,
                reconstruction,
                batch,
                mu,
                logvar,
                loaders["train"].dataset.input_size(
                    direction=img_direction
                ),  # return n_features or image shape, see build_features
                "varix",
                epoch=loss_anneal_epoch,
                pretrain=True,
            )
            loss = r_loss_batch + vae_loss_batch
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        for _, from_vbatch, to_vbatch in loaders["valid"]:
            model.eval()
            with torch.no_grad():
                vbatch = from_vbatch if img_direction == "from" else to_vbatch
                vbatch = vbatch.to(device)
                recons_valid, vmu, vlogvar = model(vbatch)
                r_loss_batchv, vae_loss_batchv, *_ = loss_function(
                    cfg,
                    recons_valid,
                    vbatch,
                    vmu,
                    vlogvar,
                    loaders["valid"].dataset.input_size(
                        direction=img_direction
                    ),  # return n_features or image shape, see build_features
                    "varix",
                    epoch=loss_anneal_epoch,
                )
                vloss = r_loss_batchv + vae_loss_batchv
                valid_running_loss += vloss.item()
        valid_losses.append(valid_running_loss / len(loaders["valid"]))
        losses.append(running_loss / len(loaders["train"]))
    return {"model": model, "train_loss": losses, "valid_loss": valid_losses}


def train_latent_clf(from_latent, to_latent, clf_optimizer, latent_clf, device, cfg):
    """
    Train the latent space classifier.

    Args:
        from_latent (torch.Tensor): latent space of the "from" samples.
        to_latent (torch.Tensor): latent space of the "to" samples.
        clf_optimizer (torch.optim.Optimizer): used to update the parameters of the latent classifier
        latent_clf (torch.nn.Module): The latent space classifier model.
        device (torch.device): The device on which the computations are performed.

    Returns:
        torch.Tensor: The loss value of the classifier after the training step.
    """
    clf_optimizer.zero_grad()
    latent_clf.train()
    from_latent = from_latent.to(device)
    to_latent = to_latent.to(device)

    from_scores = latent_clf(from_latent)
    to_scores = latent_clf(to_latent)
    from_labels = torch.zeros_like(from_scores, dtype=torch.float, device=device)
    to_labels = torch.ones_like(to_scores, dtype=torch.float, device=device)

    from_clf_loss = F.cross_entropy(
        from_scores, from_labels, reduction=cfg["LOSS_REDUCTION"]
    )
    to_clf_loss = F.cross_entropy(to_scores, to_labels, reduction=cfg["LOSS_REDUCTION"])
    loss = 0.5 * (from_clf_loss + to_clf_loss)

    loss.backward()
    clf_optimizer.step()
    loss.detach()
    return loss


def train_one_epoch(
    epoch,
    cfg,
    loaders,
    models,
    from_optim,
    to_optim,
    clf_optim,
    latent_clf,
    device,
    logger,
    mean_latent_group,
    sample_to_class,
):
    """
    Train the models for one epoch.

    Args:
        epoch (int): The current epoch number.
        cfg (dict): Configuration parameters for training.
        loaders (dict): Dictionary containing the data loaders.
        models (dict): Dictionary containing the models.
        from_optim (Optimizer): Optimizer for the 'from' model.
        to_optim (Optimizer): Optimizer for the 'to' model.
        clf_optim (Optimizer): Optimizer for the classifier model.
        latent_clf (nn.Module): The latent classifier model.
        device (str): Device to perform the training on.

    Returns:
        dict: report of different losses
    """
    (
        vae_epoch_loss,
        recon_epoch_loss,
        clf_epoch_loss,
        adversarial_epoch_loss,
        paired_epoch_loss,
        class_epoch_loss,
        total_epoch_loss,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    gamma_annealer_weight = 1

    if cfg["PRETRAIN_TARGET_MODALITY"] == "gamma_anneal":
        beta_anneal_epoch = get_annealing_epoch(cfg=cfg, current_epoch=epoch)
    else:
        beta_anneal_epoch = epoch
    in_pretraining = False
    if cfg["PRETRAIN_TARGET_MODALITY"] == "gamma_anneal":
        beta_anneal_epoch = get_annealing_epoch(cfg=cfg, current_epoch=epoch)
        logger.info("Pretraining with gamma annealing")
        in_pretraining = epoch < cfg["PRETRAIN_EPOCHS"]
        total_epochs = cfg["PRETRAIN_EPOCHS"] + cfg["EPOCHS"]
        gamma_annealer_weight = annealer(
            epoch_current=epoch, total_epoch=total_epochs, func=cfg["ANNEAL"]
        )
    from_input_key = cfg["TRANSLATE"].split("_to_")[0]
    to_input_key = cfg["TRANSLATE"].split("_to_")[1]
    crossmodale = True
    if from_input_key == to_input_key:
        crossmodale = False
    models["to"].train()
    models["from"].train()
    to_latents = []
    from_latents = []
    epoch_ids = []
    train_size = len(loaders["train"].dataset)
    to_number_of_features = loaders["train"].dataset.get_total_number_of_features(
        direction="to"
    )
    logger.debug(f"to_number_of_features: {to_number_of_features}")
    from_number_of_features = loaders["train"].dataset.get_total_number_of_features(
        direction="from"
    )
    logger.debug(f"from_number_of_features: {from_number_of_features}")
    logger.debug(f"number of samples in train: {train_size}")
    for batch_ids, from_batch, to_batch in loaders["train"]:
        from_optim.zero_grad(), to_optim.zero_grad()
        # Make per class
        if not mean_latent_group is None:
            epoch_ids += batch_ids
            from_mean_latent = mean_latent_group.loc[
                sample_to_class[list(batch_ids)], :
            ]
            from_mean_latent.index = list(batch_ids)

            to_mean_latent = mean_latent_group.loc[sample_to_class[list(batch_ids)], :]
            to_mean_latent.index = list(batch_ids)

        from_batch, to_batch = from_batch.to(device), to_batch.to(device)
        from_recon, from_mu, from_logvar = models["from"](from_batch)
        from_latent = models["from"].reparameterize(from_mu, from_logvar)
        from_latents.append(from_latent.detach())

        to_recon, to_mu, to_logvar = models["to"](to_batch)
        to_latent = models["to"].reparameterize(to_mu, to_logvar)
        to_latents.append(to_latent.detach())
        from_recon_loss, from_vae_loss, _ = loss_function(
            cfg,
            from_recon,
            from_batch,
            from_mu,
            from_logvar,
            loaders["train"].dataset.input_size(
                direction="from"
            ),  # return n_features (or image shape, see build_features
            "VAE",
            beta_anneal_epoch,
            pretrain=in_pretraining,
        )
        to_recon_loss, to_vae_loss, _ = loss_function(
            cfg,
            to_recon,
            to_batch,
            to_mu,
            to_logvar,
            loaders["train"].dataset.input_size(
                direction="to"
            ),  # return n_features or image shape, see build_features
            "VAE",
            beta_anneal_epoch,
            in_pretraining,
        )

        vae_batch_loss = 0.5 * (from_vae_loss + to_vae_loss)
        # scale to account for different number of features

        logger.debug(f"from_loss_scaler: {loaders['train'].dataset.from_loss_scaler}")
        logger.debug(f"to_loss_scaler: {loaders['train'].dataset.to_loss_scaler}")
        logger.debug(f" from_recon_loss before scaling: {from_recon_loss}")
        logger.debug(f" to_recon_loss before scaling: {to_recon_loss}")

        recon_batch_loss = (
            loaders["train"].dataset.from_loss_scaler * from_recon_loss
            + loaders["train"].dataset.to_loss_scaler * to_recon_loss
        )
        # force latent spaces of to and from to be aligned
        adversarial_batch_loss = torch.tensor(0.0)
        paired_loss_batch = torch.tensor(0.0)
        class_loss_batch = torch.tensor(0.0)
        if crossmodale:

            paired_loss_batch = distance_latent(
                cfg=cfg,
                latent_a=from_latent,
                latent_b=to_latent,
                a_idx_paired=None,
                b_idx_paired=None,
            )
            if not mean_latent_group is None:
                class_loss_batch = distance_latent(
                    cfg=cfg,
                    latent_a=from_latent,
                    latent_b=torch.tensor(from_mean_latent.values).to(device),
                    a_idx_paired=None,
                    b_idx_paired=None,
                ) + distance_latent(
                    cfg=cfg,
                    latent_a=to_latent,
                    latent_b=torch.tensor(to_mean_latent.values).to(device),
                    a_idx_paired=None,
                    b_idx_paired=None,
                )
            else:
                class_loss_batch = torch.tensor(0)

            from_scores = latent_clf(from_latent)
            to_scores = latent_clf(to_latent)
            adversarial_batch_loss = 0.5 * (
                F.cross_entropy(
                    from_scores,
                    torch.ones_like(
                        from_scores,
                        dtype=torch.float,
                        device=device,
                    ),
                    reduction=cfg["LOSS_REDUCTION"],
                )
                + F.cross_entropy(
                    to_scores,
                    torch.zeros_like(
                        # from_scores,
                        to_scores,
                        dtype=torch.float,
                        device=device,
                    ),
                    reduction=cfg["LOSS_REDUCTION"],
                )
            )
        total_batch_loss = (
            recon_batch_loss
            + vae_batch_loss
            + cfg["GAMMA"] * adversarial_batch_loss * gamma_annealer_weight
            + cfg["DELTA_PAIR"] * paired_loss_batch
            + cfg["DELTA_CLASS"] * class_loss_batch
        )
        total_batch_loss.backward()
        to_optim.step()
        from_optim.step()
        vae_epoch_loss += vae_batch_loss.item()
        recon_epoch_loss += recon_batch_loss.item()
        adversarial_epoch_loss += (
            cfg["GAMMA"] * adversarial_batch_loss.item() * gamma_annealer_weight
        )
        paired_epoch_loss += cfg["DELTA_PAIR"] * paired_loss_batch.item()
        class_epoch_loss += cfg["DELTA_CLASS"] * class_loss_batch.item()
        total_epoch_loss += total_batch_loss.item()
    clf_epoch_loss = torch.tensor(0.0)

    # Update latent group mean
    # print(mean_latent_group)
    if not mean_latent_group is None:
        both_latent = pd.DataFrame(
            torch.cat(
                [torch.cat(from_latents, dim=0), torch.cat(to_latents, dim=0)], dim=0
            ).cpu()
        )
        both_latent.index = epoch_ids + epoch_ids  ## Each sample is in FROM and TO
        mean_latent_group = (
            both_latent.join(sample_to_class, how="left")
            .groupby(cfg["CLASS_PARAM"])
            .mean()
        )
    # print(mean_latent_group)

    if crossmodale:
        clf_epoch_loss = train_latent_clf(
            from_latent=torch.cat(from_latents, dim=0),
            to_latent=torch.cat(to_latents, dim=0),
            clf_optimizer=clf_optim,
            latent_clf=latent_clf,
            device=device,
            cfg=cfg,
        )
    return {
        "train_vae_loss": vae_epoch_loss / train_size,
        "train_recon_loss": recon_epoch_loss / train_size,
        "train_clf_loss": clf_epoch_loss.detach().cpu().numpy() / train_size,
        "train_adversarial_loss": adversarial_epoch_loss / train_size,
        "train_paired_loss": paired_epoch_loss / train_size,
        "train_class_loss": class_epoch_loss / train_size,
        "train_total_loss": total_epoch_loss / train_size,
    }, mean_latent_group


def get_model_and_loaders(cfg, logger):
    logger.info("Getting model and loaders")
    from_input_key, to_input_key = cfg["TRANSLATE"].split("_to_")
    loaders = {
        "train": get_loader(cfg, to_input_key, split_type="train"),
        "valid": get_loader(cfg, to_input_key, split_type="valid"),
    }
    models = {
        "from": get_model(
            cfg,
            loaders["train"].dataset.input_size(direction="from"),
            from_input_key,
            cfg["MODEL_TYPE"],
        ),
        "to": get_model(
            cfg,
            loaders["train"].dataset.input_size(direction="to"),
            to_input_key,
            cfg["MODEL_TYPE"],
        ),
    }
    logger.debug(f"from model: {models['from']}")
    logger.debug(f"to model: {models['to']}")
    if hasattr(models["to"], "_get_spatial_dim"):
        logger.debug(f"to spatialdim:{models['to']._get_spatial_dim()}")
    if hasattr(models["from"], "_get_spatial_dim"):
        logger.debug(f"from spatialdim:{models['from']._get_spatial_dim()}")
    if cfg["TRAIN_TYPE"] == "tune":
        # setting MODEL_TYPE to tune temporarily to get the normal data loader
        from_tune_train_loader = get_loader(
            cfg=cfg, path_key=from_input_key, split_type="train", tunetranslate=True
        )
        from_tune_valid_loader = get_loader(
            cfg=cfg, path_key=from_input_key, split_type="valid", tunetranslate=True
        )

        to_tune_train_loader = get_loader(
            cfg=cfg, path_key=to_input_key, split_type="train", tunetranslate=True
        )
        to_tune_valid_loader = get_loader(
            cfg=cfg, path_key=to_input_key, split_type="valid", tunetranslate=True
        )

        models["from"] = find_best_model(
            cfg=cfg,
            model_name=f'{from_input_key}VAE{cfg["RUN_ID"]}.pt',
            path_key=from_input_key,
            dataloader=from_tune_train_loader,
            valloader=from_tune_valid_loader,
            model_type=cfg["MODEL_TYPE"],
            translate=True,
        )
        models["to"] = find_best_model(
            cfg=cfg,
            model_name=f'{to_input_key}VAE{cfg["RUN_ID"]}.pt',
            path_key=to_input_key,
            dataloader=to_tune_train_loader,
            valloader=to_tune_valid_loader,
            model_type=cfg["MODEL_TYPE"],
            translate=True,
        )

    return models, loaders


def log_training_stats(epoch, train_loss_dict, valid_loss_dict, logger):
    """Logs losses and epoch if checkpoint_interval is reached
    ARGS:
        epoch - (int) - current epoch
        train_loss_dict - (dict) - dictionary with training losses
        valid_loss_dict - (dict) - dictionary with validation losses
        logger - (logging.logger) - logger
    RETURNS:
        None

    """
    for (train_key, train_value), (valid_key, valid_value) in zip(
        train_loss_dict.items(), valid_loss_dict.items()
    ):
        logger.info(
            f"Epoch: {epoch}, {train_key}: {train_value[-1]}, {valid_key}: {valid_value[-1]}"
        )


def train_translate(cfg, logger):
    """Train two autoencoder models and translate between them."""
    device = get_device(cfg)
    logger.info("Training translate model")
    models, loaders = get_model_and_loaders(cfg, logger)
    from_input_key, to_input_key = cfg["TRANSLATE"].split("_to_")
    pretrained = False
    if cfg["PRETRAIN_TARGET_MODALITY"] == "pretrain_image":
        logger.info("Pretraining image model")
        if cfg["DATA_TYPE"][from_input_key]["TYPE"] == "IMG":
            logger.info("Pretraining 'from' IMG model")
            img_direction = "from"  # to select correct loader
            img_model = models["from"]
            pretrain_res = pretrain_img(loaders, img_model, cfg, img_direction)
            logger.info("Pretrained from IMG model")
            models["from"] = pretrain_res["model"]
            pretrained = True  # hinders pretraining twice for IMG_to_IMG case
        elif cfg["DATA_TYPE"][to_input_key]["TYPE"] == "IMG":
            logger.info("Pretraining 'to' IMG model")
            img_model = models["to"]
            img_direction = "to"
            if (
                not pretrained
            ):  # if we already pretrained the from model, we don't need to do it again
                pretrain_res = pretrain_img(loaders, img_model, cfg, img_direction)
                logger.info("Pretrained to IMG model")
                models["to"] = pretrain_res["model"]
    for model in models.values():
        model.to(device)
    if cfg["FIX_RANDOMNESS"] == "all" or cfg["FIX_RANDOMNESS"] == "training":
        logger.debug("fixing randomness")
        torch.manual_seed(cfg["GLOBAL_SEED"])  ## -> effects also model weight init
        torch.use_deterministic_algorithms(True)
        g.manual_seed(cfg["GLOBAL_SEED"])


    latent_clf = LatentSpaceClassifier(
        n_hidden=cfg["LATENT_DIM_FIXED"],
        input_dim=cfg["LATENT_DIM_FIXED"],
    ).to(device)

    to_optim = torch.optim.AdamW(
        models["to"].parameters(), lr=cfg["LR_FIXED"], weight_decay=cfg["WEIGHT_DECAY"]
    )
    from_optim = torch.optim.AdamW(
        models["from"].parameters(),
        lr=cfg["LR_FIXED"],
        weight_decay=cfg["WEIGHT_DECAY"],
    )
    clf_optim = torch.optim.AdamW(
        latent_clf.parameters(), lr=cfg["LR_FIXED"], weight_decay=cfg["WEIGHT_DECAY"]
    )
    best_to_model, best_from_model = None, None
    train_loss_stats = {
        "train_vae_loss": [],
        "train_recon_loss": [],
        "train_clf_loss": [],
        "train_adversarial_loss": [],
        "train_paired_loss": [],
        "train_class_loss": [],
        "train_total_loss": [],
    }
    valid_loss_stats = {
        "valid_vae_loss": [],
        "valid_recon_loss": [],
        "valid_clf_loss": [],
        "valid_adversarial_loss": [],
        "valid_paired_loss": [],
        "valid_class_loss": [],
        "valid_total_loss": [],
    }
    best_total_lossv = float("inf")
    best_last_cpt = float("inf")

    total_epochs = cfg["EPOCHS"]
    if cfg["PRETRAIN_TARGET_MODALITY"] == "gamma_anneal":
        assert (
            type(cfg["PRETRAIN_EPOCHS"]) == int
        ), "PRETRAIN_EPOCHS must be an integer Change in config file"

        total_epochs += cfg["PRETRAIN_EPOCHS"]

    ## Clin data for class-based loss
    if not cfg["CLASS_PARAM"] is None:
        anno_name = [
            data_type
            for data_type in cfg["DATA_TYPE"]
            if cfg["DATA_TYPE"][data_type]["TYPE"] == "ANNOTATION"
        ]
        if len(anno_name) == 1:
            clin_data = pd.read_parquet(
                os.path.join(
                    "data/processed",
                    cfg["RUN_ID"],
                    anno_name[0] + "_data.parquet",
                )
            )
            if type(clin_data[cfg["CLASS_PARAM"]].iloc[0]) is str:
                nb_classes = len(clin_data[cfg["CLASS_PARAM"]].unique())
                ## Initial class means for class-based loss as zeros
                mean_latent_group = pd.DataFrame(
                    np.zeros((nb_classes, cfg["LATENT_DIM_FIXED"]))
                )
                mean_latent_group.index = clin_data[cfg["CLASS_PARAM"]].unique()

                sample_to_class = clin_data[cfg["CLASS_PARAM"]]
            else:
                logger.warning(
                    f"Provided class for class loss in CLASS_PARAM{cfg['CLASS_PARAM']} is not a string. Please provide class labels as strings"
                )
                logger.warning("Class loss will be disabled.")
                mean_latent_group = None
                sample_to_class = None
        else:
            logger.warning(
                "No ANNOTATION data type found. Class loss will be disabled."
            )
            mean_latent_group = None
            sample_to_class = None

    else:
        mean_latent_group = None
        sample_to_class = None

    for epoch in range(total_epochs):
        checkpoint_intervall = get_checkpoint_interval(total_epochs)
        (
            vae_epoch_lossv,
            recon_epoch_lossv,
            clf_epoch_lossv,
            adversarial_epoch_lossv,
            paired_epoch_lossv,
            class_epoch_lossv,
            total_epoch_lossv,
        ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        epoch_loss_stats, mean_latent_group = train_one_epoch(
            epoch=epoch,
            cfg=cfg,
            loaders=loaders,
            models=models,
            from_optim=from_optim,
            to_optim=to_optim,
            clf_optim=clf_optim,
            latent_clf=latent_clf,
            device=device,
            logger=logger,
            mean_latent_group=mean_latent_group,
            sample_to_class=sample_to_class,
        )
        in_pretraining = False
        if cfg["PRETRAIN_TARGET_MODALITY"] == "gamma_anneal":
            in_pretraining = epoch < cfg["PRETRAIN_EPOCHS"]
        logger.info(f" in_pretraining: {in_pretraining}")
        for k, _ in train_loss_stats.items():
            train_loss_stats[k].append(epoch_loss_stats[k])
        # VALIDATION ROUND --------------------------------------------------------
        with torch.no_grad():
            to_latentsv = []
            from_latentsv = []
            # valid_size = len(loaders["valid"].dataset) + len(loaders["valid"].dataset)
            valid_size = len(loaders["valid"].dataset)
            for i, ((batchv_ids, from_batchv, to_batchv)) in enumerate(
                loaders["valid"]
            ):
                models["to"].eval(), models["from"].eval()
                latent_clf.eval()
                i += 1

                from_batchv, to_batchv = from_batchv.to(device), to_batchv.to(device)
                from_reconv, from_muv, from_logvarv = models["from"](from_batchv)
                from_latentv = models["from"].reparameterize(from_muv, from_logvarv)

                to_reconv, to_muv, to_logvarv = models["to"](to_batchv)
                to_latentv = models["to"].reparameterize(to_muv, to_logvarv)
                to_latentsv.append(to_latentv)
                from_latentsv.append(from_latentv)

                from_recon_lossv, from_vae_lossv, _ = loss_function(
                    cfg=cfg,
                    recon_x=from_reconv,
                    x=from_batchv,
                    mu=from_muv,
                    logvar=from_logvarv,
                    dim=loaders["valid"].dataset.input_size(
                        direction="from"
                    ),  # return n_features or image shape, see build_features
                    model_type="VAE",
                    epoch=None,  # Switch off beta anneal weight for valid loss
                    # epoch,
                    pretrain=in_pretraining,
                )
                to_recon_lossv, to_vae_lossv, _ = loss_function(
                    cfg=cfg,
                    recon_x=to_reconv,
                    x=to_batchv,
                    mu=to_muv,
                    logvar=to_logvarv,
                    dim=loaders["valid"].dataset.input_size(
                        direction="to"
                    ),  # return n_features or image shape, see build_features
                    model_type="VAE",
                    epoch=None,  # Switch off beta anneal weight for valid loss
                    pretrain=in_pretraining,
                )
                recon_batch_lossv = loaders[
                    "valid"
                ].dataset.from_loss_scaler * from_recon_lossv + (
                    loaders["valid"].dataset.to_loss_scaler * to_recon_lossv
                )
                vae_batch_lossv = 0.5 * (from_vae_lossv + to_vae_lossv)

                # Make per class
                if not mean_latent_group is None:
                    from_mean_latent = mean_latent_group.loc[
                        sample_to_class[list(batchv_ids)], :
                    ]
                    from_mean_latent.index = list(batchv_ids)

                    to_mean_latent = mean_latent_group.loc[
                        sample_to_class[list(batchv_ids)], :
                    ]
                    to_mean_latent.index = list(batchv_ids)

                    class_loss_batchv = distance_latent(
                        cfg=cfg,
                        latent_a=from_latentv,
                        latent_b=torch.tensor(from_mean_latent.values).to(device),
                        a_idx_paired=None,
                        b_idx_paired=None,
                    ) + distance_latent(
                        cfg=cfg,
                        latent_a=to_latentv,
                        latent_b=torch.tensor(to_mean_latent.values).to(device),
                        a_idx_paired=None,
                        b_idx_paired=None,
                    )
                else:
                    class_loss_batchv = torch.tensor(0)
                paired_loss_batchv = distance_latent(
                    cfg=cfg,
                    latent_a=from_latentv,
                    latent_b=to_latentv,
                    a_idx_paired=None,
                    b_idx_paired=None,
                )
                from_scores = latent_clf(from_latentv)
                from_labels = torch.zeros_like(
                    from_scores, dtype=torch.float, device=device
                )
                to_scores = latent_clf(to_latentv)
                to_labels = torch.ones_like(to_scores, dtype=torch.float, device=device)
                clf_batch_lossv = 0.5 * (
                    F.cross_entropy(
                        from_scores, from_labels, reduction=cfg["LOSS_REDUCTION"]
                    )
                    + F.cross_entropy(
                        to_scores, to_labels, reduction=cfg["LOSS_REDUCTION"]
                    )
                )
                logger.debug(f"size of to_batchv: {to_batchv.size()}")
                logger.debug(f"size of from_latentv: {from_latentv.size()}")
                logger.debug(f"size of from_batchv: {from_batchv.size()}")
                logger.debug(f"size of to_latentv: {to_latentv.size()}")
                logger.debug(f"len from_valid loader: {len(loaders['valid'])}")
                logger.debug(f"len to_valid loader: {len(loaders['valid'])}")

                logger.debug(f"size of classes: {from_scores.size()}")
                logger.debug(f"size of classes: {to_scores.size()}")

                adversarial_batch_lossv = 0.5 * (
                    F.cross_entropy(
                        from_scores, to_labels, reduction=cfg["LOSS_REDUCTION"]
                    )
                    + F.cross_entropy(
                        to_scores,
                        from_labels,
                        reduction=cfg["LOSS_REDUCTION"],
                    )
                )
                total_batch_lossv = (
                    recon_batch_lossv
                    + vae_batch_lossv
                    + cfg["GAMMA"] * adversarial_batch_lossv
                    + cfg["DELTA_PAIR"] * paired_loss_batchv
                    + cfg["DELTA_CLASS"] * class_loss_batchv
                )
                vae_epoch_lossv += vae_batch_lossv.item()
                recon_epoch_lossv += recon_batch_lossv.item()
                clf_epoch_lossv += clf_batch_lossv.item()
                adversarial_epoch_lossv += cfg["GAMMA"] * adversarial_batch_lossv.item()
                paired_epoch_lossv += cfg["DELTA_PAIR"] * paired_loss_batchv.item()
                class_epoch_lossv += cfg["DELTA_CLASS"] * class_loss_batchv.item()
                total_epoch_lossv += total_batch_lossv.item()
        valid_epoch_stats = {
            "valid_vae_loss": vae_epoch_lossv / valid_size,
            "valid_recon_loss": recon_epoch_lossv / valid_size,
            "valid_clf_loss": clf_epoch_lossv / valid_size,
            "valid_adversarial_loss": adversarial_epoch_lossv / valid_size,
            "valid_paired_loss": paired_epoch_lossv / valid_size,
            "valid_class_loss": class_epoch_lossv / valid_size,
            "valid_total_loss": total_epoch_lossv / valid_size,
        }
        logger.info(f"Epoch: {epoch}")
        logger.info(f"Valid losses: {valid_epoch_stats}")
        total_epoch_lossv = total_epoch_lossv / valid_size
        logger.info(f"Total valid loss: {total_epoch_lossv}")

        for k, _ in valid_loss_stats.items():
            valid_loss_stats[k].append(valid_epoch_stats[k])
        if (total_epoch_lossv < best_total_lossv) and not in_pretraining:
            logger.info(f"New best model found at epoch {epoch}. Saving at checkpoint")
            best_total_lossv = total_epoch_lossv
            logger.info(f"Best total loss: {best_total_lossv}")
            # best_paths = save_best_model(cfg=cfg, models=models, epoch=i, logger=logger)
            best_to_model, best_from_model = (
                copy.deepcopy(models["to"]),
                copy.deepcopy(models["from"]),
            )
        if (epoch % checkpoint_intervall == 0) and (best_total_lossv < best_last_cpt):
            logger.info(f"Checkpointing best model at epoch {epoch}")
            log_training_stats(
                epoch=epoch,
                train_loss_dict=train_loss_stats,
                valid_loss_dict=valid_loss_stats,
                logger=logger,
            )
            best_paths = save_best_model(
                cfg=cfg,
                models={"from": best_from_model, "to": best_to_model},
                epoch=i,
                logger=logger,
            )
            best_last_cpt = best_total_lossv
    save_final_model(
        cfg=cfg, best_paths=best_paths, best_models=(best_from_model, best_to_model)
    )
    save_loss_dicts(cfg=cfg, train_loss=train_loss_stats, valid_loss=valid_loss_stats)


def load_prediction_models(cfg, loader):
    from_input_key, to_input_key = cfg["TRANSLATE"].split("_to_")
    if cfg["TRAIN_TYPE"] == "train":
        from_model = get_model(
            cfg=cfg,
            input_dim=loader.dataset.input_size(direction="from"),
            input_type=from_input_key,
            model_type=cfg["MODEL_TYPE"],
        )

        to_model = get_model(
            cfg=cfg,
            input_dim=loader.dataset.input_size(direction="to"),
            input_type=to_input_key,
            model_type=cfg["MODEL_TYPE"],
        )

        from_model.load_state_dict(
            torch.load(
                os.path.join(
                    "models", cfg["RUN_ID"], f"TRANSLATE_FROM_{from_input_key}_VAE.pt"
                )
            )
        )

        to_model.load_state_dict(
            torch.load(
                os.path.join(
                    "models", cfg["RUN_ID"], f"TRANSLATE_TO_{to_input_key}_VAE.pt"
                )
            )
        )

    elif cfg["TRAIN_TYPE"] == "tune":
        to_model = torch.load(
            os.path.join(
                "models", f'tuned/{cfg["RUN_ID"]}/TRANSLATE_TO_{to_input_key}.pt'
            )
        )
        from_model = torch.load(
            os.path.join(
                "models", f'tuned/{cfg["RUN_ID"]}/TRANSLATE_FROM_{from_input_key}.pt'
            )
        )
    return from_model, to_model


def predict_translate(cfg, logger):
    """Uses the trained autoencoder models to translate between the data modalities on the test set
    ARGS:
        cfg - (dict) - configuration dictionary
        logger - (logging.logger) - logger
        RETURNS:
                from_latent_space - (pd.DataFrame) - latent space of the from data modality
                translation - (pd.DataFrame) - translated data of the to data modality

    """
    from_input_key = cfg["TRANSLATE"].split("_to_")[0]
    to_input_key = cfg["TRANSLATE"].split("_to_")[1]
    # path key is in this get_loader call not relevant, because the CrossModale Dataloader is returned in the 'x-modalix' case
    predict_loader = get_loader(
        cfg=cfg, path_key=to_input_key, split_type=cfg["PREDICT_SPLIT"]
    )
    from_model, to_model = load_prediction_models(cfg=cfg, loader=predict_loader)
    to_model.eval(), from_model.eval()
    device = get_device(cfg)
    to_model.to(device), from_model.to(device)
    from_latent = []
    to_latent = []
    translation = []
    img_img_translation = []
    from_id_list = []
    to_id_list = []

    ## Latent Space From
    for batch_ids, from_batch, to_batch in predict_loader:
        from_batch = from_batch.to(device)
        to_batch = to_batch.to(device)
        _, from_mu, from_logvar = from_model(from_batch)
        _, to_mu, to_logvar = to_model(to_batch)
        from_z = from_model.reparameterize(from_mu, from_logvar)
        to_z = to_model.reparameterize(to_mu, to_logvar)
        translated_to = to_model.decode(from_z)
        translated = to_model.decode(to_z)
        from_id_list.append(batch_ids)
        from_latent.append(from_z.cpu().detach())
        translation.append(translated.cpu().detach())
        img_img_translation.append(translated_to.cpu().detach())

        to_batch = to_batch.to(device)
        _, to_mu, to_logvar = to_model(to_batch)
        to_z = to_model.reparameterize(to_mu, to_logvar)
        # translated = to_model.decode(from_z) ## Not for "to" data set
        to_id_list.append(batch_ids)
        to_latent.append(to_z.cpu().detach())

    from_latent_space = torch.cat(from_latent, dim=0).numpy()
    to_latent_space = torch.cat(to_latent, dim=0).numpy()

    translation = np.concatenate(translation, axis=0)
    img_img_translation = np.concatenate(img_img_translation, axis=0)
    logger.info(f"concat translation shape: {translation.shape}")
    from_id_list = np.concatenate(from_id_list, axis=0)
    to_id_list = np.concatenate(to_id_list, axis=0)

    from_latent_space = pd.DataFrame(from_latent_space)
    to_latent_space = pd.DataFrame(to_latent_space)

    if not len(from_id_list) == from_latent_space.shape[0]:
        logger.warning("id_list and latent space have different lengths")
        logger.warning(f"len id_list: {len(from_id_list)}")
        logger.warning(f"len from_latent_space: {from_latent_space.shape[0]}")
        logger.warning("repeating ids")
        from_id_list = [
            sid
            for sid in from_id_list
            for _ in range(int(from_latent_space.shape[0] / len(from_id_list)))
        ]
    from_latent_space.index = from_id_list
    to_latent_space.index = to_id_list
    logger.debug(f"columns: {predict_loader.dataset.get_cols(direction='from')}")

    if cfg["DATA_TYPE"][to_input_key]["TYPE"] == "IMG":
        logger.info(f" len of loop: {translation.shape[0]}")
        sample_file = os.path.join(
            cfg["ROOT_RAW"], cfg["DATA_TYPE"][to_input_key]["FILE_RAW"]
        )
        samples = pd.read_csv(sample_file, index_col=0, sep=cfg["DELIM"])
        file_ext = samples["img_paths"][0].split(".")[-1]

        if not os.path.exists(os.path.join("reports", cfg["RUN_ID"], "IMGS")):
            os.makedirs(os.path.join("reports", cfg["RUN_ID"], "IMGS"), exist_ok=True)

            ## Create Images for Clinic Param Centers
        anno_name = [
            data_type
            for data_type in cfg["DATA_TYPE"]
            if cfg["DATA_TYPE"][data_type]["TYPE"] == "ANNOTATION"
        ]

        if (len(anno_name) == 1) and (not cfg["CLINIC_PARAM"] is None):
            clin_data = pd.read_parquet(
                os.path.join(
                    "data/processed",
                    cfg["RUN_ID"],
                    anno_name[0] + "_data.parquet",
                )
            )
            from_rel_index = clin_data.index.intersection(from_latent_space.index)
            from_latent_space = from_latent_space.loc[from_rel_index, :]

            to_rel_index = clin_data.index.intersection(
                to_latent_space.index.str.removeprefix("TO_")
            )
            to_latent_space = to_latent_space.loc[to_rel_index, :]
            for param in cfg["CLINIC_PARAM"]:
                # param = cfg['CLINIC_PARAM'][0] ## Use the first
                if not (type(clin_data[param].iloc[0]) is str):
                    logger.info(
                        f"The provided label column is numeric and converted to categories."
                    )
                    if len(np.unique(clin_data[param])) > 3:
                        labels = pd.qcut(
                            clin_data[param],
                            q=4,
                            labels=["1stQ", "2ndQ", f"3rdQ", f"4thQ"],
                        ).astype(str)
                    else:
                        labels = [str(x) for x in clin_data[param]]

                    clin_data[param] = labels

                from_mean_latent_group = (
                    from_latent_space.join(clin_data.loc[:, param], how="inner")
                    .groupby([param])
                    .median()
                )  ## TODO check if median or mean?
                to_mean_latent_group = (
                    to_latent_space.join(clin_data.loc[:, param], how="inner")
                    .groupby([param])
                    .median()
                )

                from_translated_group = to_model.decode(
                    torch.tensor(from_mean_latent_group.values).to(device)
                )
                from_translated_group = np.concatenate(
                    [from_translated_group.cpu().detach()], axis=0
                )

                to_translated_group = to_model.decode(
                    torch.tensor(to_mean_latent_group.values).to(device)
                )
                to_translated_group = np.concatenate(
                    [to_translated_group.cpu().detach()], axis=0
                )

                for center in range(from_translated_group.shape[0]):
                    from_filename = f"from_center_{from_mean_latent_group.index[center]}-{param}.png"
                    to_filename = (
                        f"to_center_{to_mean_latent_group.index[center]}-{param}.png"
                    )
                    logger.info(f"writing image {from_filename}")
                    filepath = os.path.join(
                        "reports", cfg["RUN_ID"], "IMGS", from_filename
                    )

                    cur_img = from_translated_group[center, :, :, :]
                    cur_img = cur_img.transpose(1, 2, 0)
                    cur_img = (cur_img * 255).astype(np.uint8)
                    cv2.imwrite(filepath, cur_img)

                    logger.info(f"writing image {to_filename}")
                    filepath = os.path.join(
                        "reports", cfg["RUN_ID"], "IMGS", to_filename
                    )
                    cur_img = to_translated_group[center, :, :, :]
                    cur_img = cur_img.transpose(1, 2, 0)
                    cur_img = (cur_img * 255).astype(np.uint8)
                    cv2.imwrite(filepath, cur_img)

        ## Make Image reconstruction
        logger.info(f"writing image reconstruction")
        if not os.path.exists(os.path.join("reports", cfg["RUN_ID"], "IMGS")):
            os.makedirs(os.path.join("reports", cfg["RUN_ID"], "IMGS"), exist_ok=True)
        if not os.path.exists(os.path.join("reports", cfg["RUN_ID"], "IMGS_IMG")):
            os.makedirs(os.path.join("reports", cfg["RUN_ID"], "IMGS_IMG"), exist_ok=True)
        for i in range(translation.shape[0]):

            filename = f"{from_id_list[i]}.{file_ext}"
            # logger.info(f"writing image {filename}")
            filepath = os.path.join("reports", cfg["RUN_ID"], "IMGS", filename)
            filepath_img = os.path.join("reports", cfg["RUN_ID"], "IMGS_IMG", filename)

            cur_img = translation[i, :, :, :]
            cur_img_img = img_img_translation[i, :, :, :]
            # print(f"shape cur_image{cur_img.shape}")
            cur_img = cur_img.transpose(1, 2, 0)
            cur_img_img = cur_img_img.transpose(1, 2, 0)
            # if file_ext in "tiff":
            #     cv2.imwrite(filepath, cur_img)
            # else:
            cur_img = (cur_img * 255).astype(np.uint8)
            cur_img_img = (cur_img_img * 255).astype(np.uint8)
            cv2.imwrite(filepath, cur_img)
            cv2.imwrite(filepath_img, cur_img_img)
        return (
            from_latent_space,
            to_latent_space,
            pd.DataFrame(translation.reshape(translation.shape[0], -1)),
        )
    translation = pd.DataFrame(
        translation,
        index=to_id_list,
        columns=predict_loader.dataset.get_cols(direction="to"),
    )
    translation["shape"] = str(translation.shape)
    return from_latent_space, to_latent_space, translation


def save_loss_dicts(cfg, train_loss, valid_loss):

    loss_df = pd.DataFrame(train_loss)
    valid_df = pd.DataFrame(valid_loss)
    save_df = pd.concat([loss_df, valid_df], axis=1)

    save_df.to_parquet(
        os.path.join(
            "reports",
            f'{cfg["RUN_ID"]}',
            f'losses_{cfg["TRANSLATE"]}_{cfg["MODEL_TYPE"]}.parquet',
        )
    )


def save_best_model(cfg, models, epoch, logger):
    from_input_key, to_input_key = cfg["TRANSLATE"].split("_to_")
    best_to_modelpath = os.path.join(
        "models",
        cfg["RUN_ID"],
        f"checkpoint_{epoch}_TO_{to_input_key}.pt",
    )
    torch.save(models["to"].state_dict(), best_to_modelpath)
    best_from_modelpath = os.path.join(
        "models",
        cfg["RUN_ID"],
        f"checkpoint_{epoch}_FROM_{from_input_key}.pt",
    )
    torch.save(models["from"].state_dict(), best_from_modelpath)
    logger.info(
        f"saved new best models to: to_model: {best_to_modelpath} and from_model: {best_from_modelpath} "
    )
    return (best_from_modelpath, best_to_modelpath)


def save_final_model(cfg, best_paths, best_models):
    from_input_key, to_input_key = cfg["TRANSLATE"].split("_to_")
    if cfg["TRAIN_TYPE"] == "train":
        # rename best_models to modelname on disk

        shutil.move(
            best_paths[1],
            os.path.join(
                "models", cfg["RUN_ID"], f"TRANSLATE_TO_{to_input_key}_VAE.pt"
            ),
        )
        shutil.move(
            best_paths[0],
            os.path.join(
                "models", cfg["RUN_ID"], f"TRANSLATE_FROM_{from_input_key}_VAE.pt"
            ),
        )
    # for tuning we need to save the model directly, because we don't know the architecture later in predict step
    if cfg["TRAIN_TYPE"] == "tune":
        torch.save(
            best_models[1],
            os.path.join(
                f'models/tuned/{cfg["RUN_ID"]}/TRANSLATE_TO_{to_input_key}.pt'
            ),
        )
        torch.save(
            best_models[0],
            os.path.join(
                f'models/tuned/{cfg["RUN_ID"]}/TRANSLATE_FROM_{from_input_key}.pt'
            ),
        )
