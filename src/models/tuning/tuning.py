import os
import sys
from contextlib import redirect_stderr

import optuna
import pandas as pd
import torch
import torch.optim as optim

from src.utils.utils import (
    get_model_for_tuning,
    has_nans,
    loss_function,
    train_ae_model,
)
from src.utils.utils_basic import get_device, getlogger
from src.visualization.visualize import make_optuna_plots


def objective(
    cfg,
    trial,
    path_key,
    model_type,
    trainloader,
    valloader,
    mask_1=None,
    mask_2=None,
    latent_dim=None,
):
    """objective function for optuna hyperparameter tuning
    ARGS:
        cfg - (dict): config dictionary
        trial - (optuna.trial): optuna trial object
        path_key - (str): path key for data
        model_type - (str): model type, options are: 'vanillix', 'varix', 'stackix', 'ontix'
        valloader - (torch.utils.data.DataLoader): validation data loader
        trainloader - (torch.utils.data.DataLoader): train data loader
    RETURNS:
        loss - (float): loss value

    """
    optuna.logging.disable_default_handler()
    logger = getlogger(cfg)
    logger.debug(f"pathkey in objective: {path_key}")
    device = get_device()
    loss_type = model_type + valloader.dataset.data_modality
    model = get_model_for_tuning(
        cfg,
        input_type=path_key,
        input_dim=trainloader.dataset.input_size(),
        trial=trial,
        model_type=model_type,
        mask_1=mask_1,
        mask_2=mask_2,
        latent_dim=latent_dim,
    )
    # init weights so they are normally distributed and close to 0
    model.to(device)
    optimizer_name = "AdamW"
    lr = trial.suggest_float(
        "lr", cfg["LR_LOWER_LIMIT"], cfg["LR_UPPER_LIMIT"], log=True
    )
    weight_decay = trial.suggest_float(
        "weight_decay",
        cfg["WEIGHT_DECAY_LOWER_LIMIT"],
        cfg["WEIGHT_DECAY_UPPER_LIMIT"],
        log=True,
    )
    # logger.info(
    #     f"params:\
    # path_key: {path_key},\
    # input_dim:{ trainloader.dataset.input_size()},\
    # trial: {trial.number}\
    # model_type: {model_type}\
    # optimizer: {optimizer_name}\
    # learing rate: {lr}"
    # )

    logger.info(
        f"params:\
    trial: {trial.number}\
    model_type: {model_type}\
    optimizer: {optimizer_name}\
    nb_layers: {model.n_layers} \
    weight decay: {weight_decay}\
    learning rate: {lr}"
    )
    optimizer = getattr(optim, optimizer_name)(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    losses = dict()
    vae_losses = dict()
    model.to(device)
    for epoch in range(cfg["EPOCHS"]):
        # for epoch in range(int(cfg["EPOCHS"] / 4)):
        total_train_loss = 0.0
        total_val_loss = 0.0
        model.train(True)
        for batch, _ in trainloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstr_x, mu, logvar = model(batch)
            # check if reconstr_x has nan
            if torch.isnan(reconstr_x).sum() != 0:
                logger.warning("reconstr_x has nan")
                logger.info(f"batch has finite values: {torch.isfinite(batch).all()}")
                raise optuna.exceptions.TrialPruned()
            logger.debug(f"reconstr: {reconstr_x.shape}, batch shape: {batch.shape}")
            rloss, vae_loss, _ = loss_function(
                cfg,
                reconstr_x,
                batch,
                mu,
                logvar,
                trainloader.dataset.input_size(),
                model_type,
                epoch=epoch,
            )

            batch_loss = rloss + vae_loss
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            ## Mask gradients for ontology AE and VAE
            if model_type == "ontix":
                with torch.no_grad():
                    mask_1 = mask_1.to(device)
                    model.decoder[-1].weight.mul_(mask_1)  ## Sparse Decoder Level
                    if not mask_2 == None:
                        mask_2 = mask_2.to(device)
                        model.decoder[-2].weight.mul_(mask_2)  ## Sparse Decoder Level
            optimizer.step()
            if has_nans(model, cfg):
                logger.warning("model has nans")
                raise optuna.exceptions.TrialPruned()

            total_train_loss += batch_loss.detach()

        with torch.no_grad():
            model.eval()
            for vbatch, _ in valloader:
                vbatch = vbatch.to(device)
                vreconstr_x, vmu, vlogvar = model(vbatch)
                vrloss, vvae_loss, _ = loss_function(
                    cfg,
                    vreconstr_x,
                    vbatch,
                    vmu,
                    vlogvar,
                    valloader.dataset.input_size(),
                    model_type,
                    epoch=0,  # Switch off beta anneal weight for valid loss
                )
                vbatch_loss = vrloss + vvae_loss
                total_val_loss += vbatch_loss.detach()

        losses[epoch] = vrloss.detach()
        vae_losses[epoch] = vvae_loss.detach()

        trial.report(total_val_loss / len(valloader.dataset), epoch)

        # Handle pruning based on the intermediate value.
        if (
            cfg["MODEL_TYPE"] == "vanillix"
            or cfg["BETA"] == 0
            or cfg["ANNEAL"] == "no-annealing"
        ):  # Exclude pruning for VAE with annealing
            if trial.should_prune():
                logger.info(f"Trial {trial.number} is pruned.")
                raise optuna.exceptions.TrialPruned()

    losses = {k: v.detach().cpu().numpy() for k, v in losses.items()}
    vae_losses = {k: v.detach().cpu().numpy() for k, v in vae_losses.items()}
    loss_df = pd.DataFrame.from_dict(
        losses, orient="index", columns=["recon_lossOptuna_" + loss_type]
    )

    vae_loss_df = pd.DataFrame.from_dict(
        vae_losses, orient="index", columns=["vae_lossOptuna_" + loss_type]
    )

    RUN_ID = cfg["RUN_ID"]
    # loss_df.to_csv(
    loss_df.to_parquet(
        os.path.join(
            "data",
            "interim",
            RUN_ID,
            f"losses_reconOptuna_TRIAL{trial.number}_{loss_type}_{RUN_ID}.parquet",
        )
    )
    # vae_loss_df.to_csv(
    vae_loss_df.to_parquet(
        os.path.join(
            "data",
            "interim",
            RUN_ID,
            f"lossses_vaelossOptuna_TRIAL{trial.number}_{loss_type}_{RUN_ID}.parquet",
        ),
    )
    logger.debug(
        f"Reconstruction loss saved as losses_recon_{loss_type}_{RUN_ID}.parquet in data/interim/{RUN_ID}"
    )
    logger.debug(
        f"VAE Loss loss saved as losses_vaeloss{loss_type}_{RUN_ID}.parquet in data/interim/{RUN_ID}"
    )
    logger.info(
        f"Total validation loss of Trial {trial.number} is: {total_val_loss / len(valloader.dataset)}"
    )

    return total_val_loss


def find_best_model(
    cfg,
    model_name,
    path_key,
    dataloader,
    valloader,
    model_type="vae",
    translate=False,
    mask_1=None,
    mask_2=None,
    latent_dim=None,
):
    """Find the best model for a given dataset and model type by tuning
    hyperparameters and architecture with optuna.
    ARGS:
        cfg - (dict) config file
        model_name - (str) name of the model
        path_key - (str) key to the data in the dataloader
        dataloader - (torch.utils.data.DataLoader) dataloader for training
        valloader - (torch.utils.data.DataLoader) dataloader for validation
        model_type - (str) type of model to use, default is vae

    RETURNS:
        best_model - (torch.nn.Module) best model found by optuna

    """
    # print("[2] Dense size: " + str(latent_dim))
    best_model = None
    logger = getlogger(cfg)
    # check if model already exists with tuned architecture and if user wants to start from last checkpoint
    if cfg["START_FROM_LAST_CHECKPOINT"] and os.path.exists(
        os.path.join(f"models/tuned/{cfg['RUN_ID']}/'archonly_{model_name}")
    ):
        logger.info(f"in start from lastcheckpoint loop")

        logger.info("starting from last checkpoint, using default hyperparameters")
        best_model = torch.load(
            os.path.join(f"models/tuned/{cfg['RUN_ID']}/'archonly_{model_name}")
        )

        best_model, _ = train_ae_model(
            cfg,
            dataloader,
            best_model,
            model_name,
            valid_loader=valloader,
            model_type=model_type,
            mask_1=mask_1,
            mask_2=mask_2,
        )
        torch.save(best_model, f"models/tuned/{cfg['RUN_ID']}/{model_name}")
        logger.info(
            f"saved best model to: \n models/tuned/{cfg['RUN_ID']}/{model_name}"
        )
        return best_model

    logger.info(f"MODEL_TYPE: {model_type}")
    logger.info(f"Starting tuning with optuna {model_type}")

    # standard optuna procedure

    optuna.logging.disable_default_handler()
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.PatientPruner(
            optuna.pruners.MedianPruner(),
            patience=int(cfg["EPOCHS"] * cfg["PRUN_PATIENCE"]),
        ),
    )
    # in case optuna cannot finish any trial
    # try:
    logger.info(f"Trials total: {cfg['OPTUNA_TRIALS']}")
    study.optimize(
        lambda trial: objective(
            cfg,
            trial,
            path_key,
            model_type,
            dataloader,
            valloader,
            mask_1=mask_1,
            mask_2=mask_2,
            latent_dim=latent_dim,
        ),
        n_trials=cfg["OPTUNA_TRIALS"],
        n_jobs=1,
        # show_progress_bar = True
    )

    logger.info("Best trial:")
    # in case Optuna finishes with no trial we want to give a better error message
    best_trial = study.best_trial
    logger.info(f"  Value: {best_trial.number}")
    logger.info(
        f" params that go in get_model_for_tuning: shape: {dataloader.dataset.input_size()}, input_type: {path_key}, model_type: {model_type}"
    )

    best_model = get_model_for_tuning(
        cfg,
        best_trial,
        dataloader.dataset.input_size(),
        input_type=path_key,
        model_type=model_type,
        mask_1=mask_1,
        mask_2=mask_2,
        latent_dim=latent_dim,
    )

    # save best model architecture (without tuning weights)

    if not os.path.exists(os.path.join("models", "tuned")):
        os.mkdir(os.path.join("models", "tuned"))
    if not os.path.exists(os.path.join("models", "tuned", cfg["RUN_ID"])):
        os.mkdir(os.path.join("models", "tuned", cfg["RUN_ID"]))
    outpath_arch = os.path.join(f'models/tuned/{cfg["RUN_ID"]}/archonly_{model_name}')
    logger.info(f"outpath arch in tuning: {outpath_arch}")
    torch.save(best_model, outpath_arch)
    logger.info(
        f"Found best model architecture with validation set, now training weights with train set"
    )
    logger.info(f"  Value: {best_trial.value}")

    logger.info("  Params: ")
    for key, value in best_trial.params.items():
        value = str(value)
        logger.info(f"{key}: {value}")
    lr = best_trial.params["lr"]
    weight_decay = best_trial.params["weight_decay"]
    # opimizer_name = best_trial.params["optimizer"]
    opimizer_name = "AdamW"
    optimizer = getattr(torch.optim, opimizer_name)(
        best_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    logger.info("Save best parameter to txt")
    pd.Series(best_trial.params).to_csv(
        os.path.join("reports", cfg["RUN_ID"], path_key + "_best_model_params.txt")
    )

    ## Plotting
    logger.info("Plot optuna hyperparameter visualizations")
    make_optuna_plots(
        study=study,
        savefig=os.path.join(
            "reports", cfg["RUN_ID"], "figures", path_key + "_optuna_vis_"
        ),
    )
    if translate:
        logger.info(
            f" for translation, we use the best model found for {model_type} on the validation set"
        )
        logger.info(
            f"we only tune the architecture for optimal reconstructions and vae_loss, then we use this architecture for training the weights for translation"
        )
        return best_model

    best_model, _ = train_ae_model(
        cfg,
        dataloader,
        best_model,
        model_name,
        valid_loader=valloader,
        model_type=model_type,
        optimizer=optimizer,
        mask_1=mask_1,
        mask_2=mask_2,
    )
    torch.save(best_model, f"models/tuned/{cfg['RUN_ID']}/{model_name}")
    logger.info(f"saved best model to: \n models/tuned/{cfg['RUN_ID']}/{model_name}")
    return best_model
