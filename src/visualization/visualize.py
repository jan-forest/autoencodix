import warnings


import glob

import click
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

from src.utils.config import get_cfg
from src.utils.utils_basic import getlogger, read_ont_file

import os
import pathlib

import numpy as np
import optuna
import pandas as pd
# from matplotlib.backends.backend_pdf import PdfPages

# The above code is importing the Image module from the Python Imaging Library
# (PIL) package. This module allows you to work with images in Python, such as
# opening, manipulating, and saving images.
# from PIL import Image
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from umap import UMAP

from src.utils.utils_basic import annealer, get_annealing_epoch
from src.visualization.vis_crossmodalix import (
    plot_translate_latent,
    translate_grid,
    plot_translate_latent_simple,
)
from seaborn import axes_style

from numba.core.errors import NumbaDeprecationWarning
plt.set_loglevel("WARNING")

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

sns.set_theme(font_scale=3)
sns.set_style("white")
sns_out_type = "png"
so.Plot.config.theme.update(axes_style("whitegrid"))


def make_optuna_plots(study, savefig=""):
    """
    Creates visualizations of optuna tuning results. Four types of viualizations are created.
    ARGS:
        study (optuna.study): An optuna study class with tuning results
        savefig (str): Path to save plots
    RETURNS:
        fig_list (list): List of figure handles containing visualizations

    """
    fig_imp = optuna.visualization.plot_param_importances(study)
    fig_trials = optuna.visualization.plot_intermediate_values(study)
    fig_par_coord = optuna.visualization.plot_parallel_coordinate(study)
    fig_contour = optuna.visualization.plot_contour(study)

    if len(savefig) > 0:
        fig_imp.write_image(savefig + "imp.png", width=1200, height=800)
        fig_trials.write_image(savefig + "trials.png", width=1200, height=800)
        fig_par_coord.write_image(savefig + "par_coord.png", width=1200, height=800)
        fig_contour.write_image(savefig + "contour.png", width=1200, height=800)

    return [fig_imp, fig_trials, fig_par_coord, fig_contour]


def dim_red(latent_space, cfg, dim=2, method="UMAP", seed=None):
    """
    Performs dimension reduction on the latent space (rows samples, columns embedding)
    for visualization/clustering (dim=2 default) via the specified method ("UMAP" default).
    ARGS:
        latent_space (pd.DataFrame): latent space (rows samples, columns embedding)
        method (str): method for dimensions reduction. Options are "TSNE" and "UMAP".
    RETURNS:
        embedding (pd.DataFrame): data frame of latent space embeddings
        (rows samples, colums dim (default 2) embeddings)

    """
    logger = getlogger(cfg)
    logger.debug(f"random seed for dim reduction: {seed}")
    match method:
        case "UMAP":
            reducer = UMAP(n_components=dim, random_state=seed)
        case "TSNE":
            reducer = TSNE(n_components=dim, random_state=seed)
        case "PCA":
            reducer = PCA(n_components=dim)

    if latent_space.shape[1] > dim:
        embedding = pd.DataFrame(reducer.fit_transform(latent_space))
    else:
        logger.info(
            f"Latent dimension is {latent_space.shape[1]}. No dimension reduction by {method} required and Latent space is directly used for plotting."
        )
        embedding = latent_space
    return embedding


def cluster(embedding, method="KMeans", n_clusters=5, min_cluster_size=5):
    """
    Performs clustering on the embedding (or latent space) via the specified method.
    Depending on the cluster method, additional options may be specified.
    ARGS:
        embedding (pd.DataFrame): embedding on which clustering is performed (rows samples, columns embedding)
        method (str): clustering algorithm. Options "KMeans" and "DBScan" via sklearn.
        n_clusters (int): number of clusters if KMeans is specified.
        eps (float): Maximum distance between two samples within one cluster if DBScan is specified.
    RETURNS:
        labels_cluster (list): List of cluster labels for each sample (row) in embedding.

    """
    match method:
        case "KMeans":
            c_alg = KMeans(n_clusters=n_clusters, n_init="auto")

        case "HDBScan":
            c_alg = HDBSCAN(
                min_cluster_size=min_cluster_size, allow_single_cluster=True, n_jobs=-1
            )

    c_res = c_alg.fit(embedding)

    #
    # Silhouette score (Separierung)
    sil_score = silhouette_score(embedding, c_res.labels_)

    labels_cluster = ["Cluster_" + str(x) for x in c_res.labels_]

    return labels_cluster, sil_score


def plot_latent_simple(
    cfg,
    embedding,
    figsize=(24, 15),
    save_fig="",
):
    fig, ax2 = plt.subplots(figsize=figsize)

    ax2 = sns.scatterplot(
        x=embedding.iloc[:, 0],
        y=embedding.iloc[:, 1],
        s=40,
        alpha=0.2,
        ec="black",
    )
    ax2.set_xlabel("Dim 1")
    ax2.set_ylabel("Dim 2")
    if len(save_fig) > 0:
        fig.savefig(save_fig, bbox_inches="tight")
    return fig


def plot_latent_2D(
    cfg,
    embedding,
    labels,
    param=None,
    layer="latent space",
    figsize=(24, 15),
    center=True,
    save_fig="",
    xlim=None,
    ylim=None,
    scale=None,
    no_leg=False,
):
    """
    Creates a 2D visualization of the 2D embedding of the latent space.
    ARGS:
        cfg (dict): config dictionary
        embedding (pd.DataFrame): embedding on which is visualized. Assumes prior 2D dimension reduction.
        labels (list): Clinical parameters or cluster labels to colorize samples (points)
        layer (str): Label for plot title to indicate which network layer is represented by UMAP/TSNE
        figsize (tuple): Figure size specification.
        center (boolean): If True (default) centers of clusters/groups are visualized as stars.
        save_fig (str): File path for saving the plot. Use appropriate file
                        endings to specify image type (e.g. '*.png')
    RETURNS:
        fig (matplotlib.figure): Figure handle

    """
    logger = getlogger(cfg)
    numeric = False
    if not (type(labels[0]) is str):
        if len(np.unique(labels)) > 3:
            if not cfg["PLOT_NUMERIC"]:
                logger.info(
                    f"The provided label column is numeric and converted to categories."
                )
                labels = pd.qcut(
                    labels, q=4, labels=["1stQ", "2ndQ", f"3rdQ", f"4thQ"]
                ).astype(str)
            else:
                center = False  ## Disable centering for numeric params
                numeric = True
        else:
            labels = [str(x) for x in labels]

    fig, ax1 = plt.subplots(figsize=figsize)

    # check if label or embedding is longerm and duplicate the shorter one
    if len(labels) < embedding.shape[0]:
        logger.warning(
            "Given labels do not have the same length as given sample size. Labels will be duplicated."
        )
        labels = [
            label for label in labels for _ in range(embedding.shape[0] // len(labels))
        ]
    elif len(labels) > embedding.shape[0]:
        labels = list(set(labels))

    if numeric:
        ax2 = sns.scatterplot(
            x=embedding.iloc[:, 0],
            y=embedding.iloc[:, 1],
            hue=labels,
            palette="bwr",
            s=40,
            alpha=0.8,
            ec="black",
        )
    else:
        ax2 = sns.scatterplot(
            x=embedding.iloc[:, 0],
            y=embedding.iloc[:, 1],
            hue=labels,
            hue_order=np.unique(labels),
            s=40,
            alpha=0.8,
            ec="black",
        )
    if center:
        means = embedding.groupby(by=labels).mean()
        # logger.info(labels)
        # logger.info(means)

        ax2 = sns.scatterplot(
            x=means.iloc[:, 0],
            y=means.iloc[:, 1],
            hue=np.unique(labels),
            hue_order=np.unique(labels),
            s=200,
            ec="black",
            alpha=0.9,
            marker="*",
            legend=False,
            ax=ax2,
        )

    if not xlim == None:
        ax2.set_xlim(xlim[0], xlim[1])

    if not ylim == None:
        ax2.set_ylim(ylim[0], ylim[1])

    if not scale == None:
        plt.yscale(scale)
        plt.xscale(scale)
    ax2.set_xlabel("Dim 1")
    ax2.set_ylabel("Dim 2")
    # ax2.set(title=f'{cfg["DIM_RED_METH"]} of {layer}')
    legend_cols = 1
    if len(np.unique(labels)) > 10:
        legend_cols = 2

    if no_leg:
        plt.legend([], [], frameon=False)
    else:
        sns.move_legend(
            ax2,
            "upper left",
            bbox_to_anchor=(1, 1),
            ncol=legend_cols,
            title=param,
            markerscale=3,
            frameon=False,
        )

    if len(save_fig) > 0:
        fig.savefig(save_fig, bbox_inches="tight")
    return fig


def plot_loss(cfg, loss_file_path, log_scale=False, figsize=(10, 8)):
    """
    Plots the loss (y-axis) over epochs (x-axis). If loss_file_path contains
    multiple columns, multiple loss types are plotted.

    ARGS:
        cfg (dict): config dictionary
        loss_file_path (str): File path to csv-file with loss over epochs.
        log_scale (boolean): If TRUE, y-axis (loss) is shown in log10-scale.
        figsize (tuple): Figure size specification.
    RETURNS:
        fig (matplotlib.figure): Figure handle

    """

    df = pd.read_parquet(loss_file_path)
    df["epoch"] = df.index
    df[df < 0] = np.NaN

    df = df.drop(index=[0, 1])  # Drop first epochs for better view on loss curves

    if "valid_vae_loss" in df.columns:
        anneal_type = cfg["ANNEAL"]
        if cfg["MODEL_TYPE"] == "x-modalix":
            df["valid_vae_loss_anneal"] = df["valid_vae_loss"]
            if cfg["PRETRAIN_TARGET_MODALITY"] == "gamma_anneal":
                df["valid_vae_loss_anneal"] = [
                    loss
                    * annealer(
                        get_annealing_epoch(cfg, current_epoch=epoch),
                        total_epoch=cfg["EPOCHS"],
                        func=anneal_type,
                    )
                    for epoch, loss in zip(df["epoch"], df["valid_vae_loss"])
                ]
                df["valid_adversarial_loss_anneal"] = [
                    loss
                    * annealer(
                        epoch,
                        total_epoch=cfg["EPOCHS"] + cfg["PRETRAIN_EPOCHS"],
                        func=anneal_type,
                    )
                    for epoch, loss in zip(df["epoch"], df["valid_adversarial_loss"])
                ]
                df["valid_total_loss_anneal"] = (
                    df["valid_vae_loss_anneal"]
                    + df["valid_recon_loss"]
                    + df["valid_adversarial_loss_anneal"]
                    + df["valid_paired_loss"]
                    + df["valid_class_loss"]
                )
            else:
                df["valid_vae_loss_anneal"] = [
                    loss * annealer(epoch, total_epoch=cfg["EPOCHS"], func=anneal_type)
                    for epoch, loss in zip(df["epoch"], df["valid_vae_loss"])
                ]

                df["valid_total_loss_anneal"] = (
                    df["valid_vae_loss_anneal"]
                    + df["valid_recon_loss"]
                    + df["valid_adversarial_loss"]
                    + df["valid_paired_loss"]
                    + df["valid_class_loss"]
                )

        else:
            df["valid_vae_loss_anneal"] = [
                loss * annealer(epoch, total_epoch=cfg["EPOCHS"], func=anneal_type)
                for epoch, loss in zip(df["epoch"], df["valid_vae_loss"])
            ]
            df["valid_total_loss_anneal"] = (
                df["valid_vae_loss_anneal"] + df["valid_recon_loss"]
            )

    train_col = [
        "train_vae_loss",
        "train_recon_loss",
        "train_adversarial_loss",
        "train_paired_loss",
        "train_class_loss",
        "train_r2",
        "train_total_loss",
    ]
    val_col = [
        "valid_vae_loss",
        "valid_recon_loss",
        "valid_adversarial_loss",
        "valid_paired_loss",
        "valid_class_loss",
        "valid_r2",
        "valid_total_loss",
        "valid_vae_loss_anneal",
        "valid_adversarial_loss_anneal",
        "valid_total_loss_anneal",
    ]

    cols = df.columns.intersection(["epoch"] + train_col + val_col)
    df_plot = df.loc[:, cols].melt(id_vars="epoch")

    df_plot["split"] = "train"
    df_plot.loc[df_plot["variable"].str.startswith("valid_"), "split"] = "valid"
    df_plot.loc[df_plot["variable"].str.endswith("_anneal"), "split"] = "valid_anneal"

    df_plot["variable"] = (
        df_plot["variable"]
        .str.removeprefix("train_")
        .str.removeprefix("valid_")
        .str.removesuffix("_anneal")
    )

    exclude = ~(
        (df_plot["variable"] == "total_loss")
        | (df_plot["variable"] == "r2")
        | (df_plot["split"] == "valid_anneal")
    )
    loss_type_name = os.path.split(loss_file_path)[1].removesuffix(".parquet")
    dm = [
        datatype
        for datatype in cfg["DATA_TYPE"]
        if datatype in loss_type_name.split("_")
    ]
    total_dm = [
        datatype
        for datatype in cfg["DATA_TYPE"]
        if not (cfg["DATA_TYPE"][datatype]["TYPE"] == "ANNOTATION")
    ]
    if len(dm) == len(total_dm):
        path_key = ""
    else:
        path_key = "_".join(dm) + "_"

    figsize1 = (figsize[0], int(figsize[1] / 2))
    p_loss1 = (
        so.Plot(df_plot[exclude], x="epoch", y="value", color="variable")
        .facet("split")
        .add(
            so.Area(alpha=0.7), so.Agg("sum"), so.Norm(func="sum", by=["x"]), so.Stack()
        )
        .label(x="Epoch", y="Rel. contr. loss", color="loss term")
        .scale(color="Set1")
        .layout(size=figsize1)
    )

    p_loss1.save(
        os.path.join(
            "reports",
            cfg["RUN_ID"],
            f"figures/{path_key}loss_plot_relative.{sns_out_type}",
        ),
        bbox_inches="tight",
        dpi=180
    )

    p_loss2 = (
        so.Plot(df_plot, x="epoch", y="value", linestyle="split", color="split")
        .facet("variable", wrap=2)
        .share(y=False)
        .add(so.Line())
        .label(x="Epoch", y="Loss", linestyle="split")
        .limit(y=(0, None))
        .layout(size=figsize)
    )

    if log_scale:
        p_loss2 = p_loss2.scale(y="symlog")

    p_loss2.save(
        os.path.join(
            "reports",
            cfg["RUN_ID"],
            f"figures/{path_key}loss_plot_absolute.{sns_out_type}",
        ),
        bbox_inches="tight",
        dpi=180
    )

    return p_loss2


def plot_model_weights(model, filepath=""):
    """
    Visualization of model weights in encoder and decoder layers as heatmap for each layer as subplot.
    ARGS:
        model (pd.DataFrame): DataFrame containing the latent space intensities for samples (rows) and latent dimensions (columns)
        filepath (str): Path specifying save name and location.
    RETURNS:
        fig (matplotlib.figure): Figure handle (of last plot)
    """
    all_weights = []
    names = []
    for name, param in model.named_parameters():
        if "weight" in name and len(param.shape) == 2:
            if not "var" in name:  ## For VAE plot only mu weights
                all_weights.append(param.detach().cpu().numpy())
                names.append(name[:-7])

    layers = int(len(all_weights) / 2)
    fig, axes = plt.subplots(2, layers, sharex=False, figsize=(20, 10))

    print(names)
    for l in range(layers):
        ## Encoder Layer
        if layers > 1:
            sns.heatmap(
                all_weights[l],
                cmap=sns.color_palette("Spectral", as_cmap=True),
                ax=axes[0, l],
            ).set(title=names[l])
            ## Decoder Layer
            sns.heatmap(
                all_weights[layers + l],
                cmap=sns.color_palette("Spectral", as_cmap=True),
                ax=axes[1, l],
            ).set(title=names[layers + l])
            axes[1, l].set_xlabel("In Node", size=12)
        else:
            sns.heatmap(
                all_weights[l],
                cmap=sns.color_palette("Spectral", as_cmap=True),
                ax=axes[l],
            ).set(title=names[l])
            ## Decoder Layer
            sns.heatmap(
                all_weights[l + 2],
                cmap=sns.color_palette("Spectral", as_cmap=True),
                ax=axes[l + 1],
            ).set(title=names[l + 2])
            axes[1].set_xlabel("In Node", size=12)

    if layers > 1:
        axes[1, 0].set_ylabel("Out Node", size=12)
        axes[0, 0].set_ylabel("Out Node", size=12)
    else:
        axes[1].set_ylabel("Out Node", size=12)
        axes[0].set_ylabel("Out Node", size=12)

    if len(filepath) > 0:
        fig.savefig(filepath)

    return fig


def plot_cov_epoch(cfg, lat_coverage_epoch, figsize=(10, 8)):

    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    ax1.plot(lat_coverage_epoch["epoch"], lat_coverage_epoch["coverage"])

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Coverage")

    fig.savefig(
        os.path.join(
            "reports", cfg["RUN_ID"], f"figures/latent_cov_per_epoch.{sns_out_type}"
        ),
        bbox_inches="tight",
    )

    return fig


def plot_latent_ridge(lat_space, clin_data, param, save_fig=""):
    """
    Creates a ridge line plot of latent space dimension where each row shows the density of a latent dimension and groups (ridges).
    ARGS:
        lat_space (pd.DataFrame): DataFrame containing the latent space intensities for samples (rows) and latent dimensions (columns)
        clin_data (pd.DataFrame): DataFrame containing the clinical parameters to create groupings and coloring of ridges
        param (str): Clinical parameter to create groupings and coloring of ridges. Must be a column name (str) of clin_data
        save_fig (str): Path specifying save name and location.
    RETURNS:
        fig (matplotlib.figure): Figure handle (of last plot)
    """
    sns.set_theme(
        style="white", rc={"axes.facecolor": (0, 0, 0, 0)}
    )  ## Necessary to enforce overplotting

    df = pd.melt(lat_space, var_name="latent dim", value_name="latent intensity")
    df["sample"] = len(lat_space.columns) * list(
        lat_space.index.str.removeprefix("FROM_").str.removeprefix("TO_")
    )
    df = df.join(clin_data[param], on="sample")

    labels = df[param]
    # print(labels[0])
    if not (type(labels[0]) is str):
        if len(np.unique(labels)) > 3:
            labels = pd.qcut(
                labels, q=4, labels=["1stQ", "2ndQ", f"3rdQ", f"4thQ"]
            ).astype(str)
        else:
            labels = [str(x) for x in labels]
    df[param] = labels

    exclude_missing_info = (df[param] == "unknown") | (df[param] == "nan")

    xmin = (
        df.loc[~exclude_missing_info, ["latent intensity", "latent dim", param]]
        .groupby([param, "latent dim"])
        .quantile(0.05)
        .min()
    )
    xmax = (
        df.loc[~exclude_missing_info, ["latent intensity", "latent dim", param]]
        .groupby([param, "latent dim"])
        .quantile(0.9)
        .max()
    )

    if len(np.unique(df[param])) > 8:
        cat_pal = sns.husl_palette(len(np.unique(df[param])))
    else:
        cat_pal = sns.color_palette(n_colors=len(np.unique(df[param])))

    g = sns.FacetGrid(
        df[~exclude_missing_info],
        row="latent dim",
        hue=param,
        aspect=12,
        height=0.4,
        xlim=(xmin[0], xmax[0]),
        palette=cat_pal,
    )

    g.map_dataframe(
        sns.kdeplot,
        "latent intensity",
        bw_adjust=0.5,
        clip_on=True,
        fill=True,
        alpha=0.5,
        warn_singular=False,
        ec="k",
        lw=1,
    )

    def label(data, color, label, text="latent dim"):
        ax = plt.gca()
        label_text = data[text].unique()[0]
        ax.text(
            0.0,
            0.2,
            label_text,
            fontweight="bold",
            ha="right",
            va="center",
            transform=ax.transAxes,
        )

    g.map_dataframe(label, text="latent dim")

    g.set(xlim=(xmin[0], xmax[0]))
    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    g.add_legend()

    if len(save_fig) > 0:
        g.savefig(save_fig)

    return g


### Main ###
@click.command()
@click.argument("run_id", type=str)
def main(run_id):
    """Performs visualization to evaluate autoencoder training.
    Loss plots and latent space representations are produced. Clustering is performed.
    """

    # Load config
    cfg = get_cfg(run_id)
    logger = getlogger(cfg)
    ## Plot Loss ##
    logger.info(f"Plotting loss per type over epochs.")
    loss_f_regex = f"losses_*.parquet"
    loss_file_path = os.path.join("reports", run_id, loss_f_regex)
    # skip this step for translate case

    for loss_file in glob.glob(loss_file_path):
        logger.info(f"Plotting loss for {loss_file}")
        plot_loss(cfg, loss_file, log_scale=True)

    ## Latent space ##
    lat_file = os.path.join("reports", run_id, "predicted_latent_space.parquet")
    ## Read In
    lat_space = pd.read_parquet(lat_file)

    ## add Latent space names from ontology
    if cfg["MODEL_TYPE"] == "ontix":
        if cfg["FILE_ONT_LVL2"] == None:
            ont_data = read_ont_file(
                os.path.join("data/processed", cfg["RUN_ID"], cfg["FILE_ONT_LVL1"]),
                sep=cfg["DELIM"],
            )
        else:
            ont_data = read_ont_file(
                os.path.join("data/processed", cfg["RUN_ID"], cfg["FILE_ONT_LVL2"]),
                sep=cfg["DELIM"],
            )

    if cfg["MODEL_TYPE"] == "ontix":
        if len(lat_space.columns) == len(list(ont_data.keys())):
            lat_space.columns = list(ont_data.keys())
    else:
        lat_space.columns = [
            "DIM_" + str(x) for x in range(1, len(lat_space.columns) + 1)
        ]

    ## Plot Coverage if available ##
    if cfg["CHECKPT_PLOT"] and (not cfg["MODEL_TYPE"] == "x-modalix"):
        lat_coverage_epoch = pd.read_parquet(
            os.path.join(
                "reports",
                f"{cfg['RUN_ID']}",
                f'latent_cov_per_epoch_{cfg["RUN_ID"]}.parquet',
            )
        )
        plot_cov_epoch(cfg=cfg, lat_coverage_epoch=lat_coverage_epoch)

    anno_name = [
        data_type
        for data_type in cfg["DATA_TYPE"]
        if cfg["DATA_TYPE"][data_type]["TYPE"] == "ANNOTATION"
    ]
    if not (len(anno_name) == 1):
        logger.warning(
            "No ANNOTATION (or multiple) data type found. All visualizations requiring an annotation file are skipped."
        )

        # Dimension reduction 2D by tSNE or UMAP (if not Latent Space 2D)
        logger.info(
            f'Performing {cfg["DIM_RED_METH"]} for 2D latent space visualization.'
        )
        embedding = dim_red(
            lat_space, cfg=cfg, method=cfg["DIM_RED_METH"], seed=cfg["GLOBAL_SEED"]
        )

        ## plot latent 2D without clin param
        if cfg["MODEL_TYPE"] == "x-modalix":
            logger.info(f"Plot translation latent space with no labels.")
            embedding = pd.DataFrame(embedding)
            embedding.columns = ["DIM1", "DIM2"]
            embedding.index = lat_space.index

            embedding["Translate"] = ["FROM"] * int(len(lat_space.index) / 2) + [
                "TO"
            ] * int(len(lat_space.index) / 2)
            plot1 = plot_translate_latent_simple(
                cfg=cfg,
                embedding=embedding,
                style_param="Translate",
                save_fig=os.path.join(
                    "reports",
                    run_id,
                    f"figures/latent2D_Aligned_noParam.{sns_out_type}",
                ),
            )
        else:
            logger.info(f'Plot {cfg["DIM_RED_METH"]} with no labels.')
            plot1 = plot_latent_simple(
                cfg=cfg,
                embedding=embedding,
                save_fig=os.path.join(
                    "reports",
                    run_id,
                    f"figures/latent2D_noParam.{sns_out_type}",
                ),
            )

        return  ## Skip other visualizations

    if cfg["CLINIC_PARAM"] is None:
        logger.warning(
            "No paramter in CLINIC_PARAM given. All visualizations requiring an annotation file are skipped."
        )
        return

    clin_data = pd.read_parquet(
        os.path.join(
            "data/processed",
            cfg["RUN_ID"],
            anno_name[0] + "_data.parquet",
        )
    )

    if pathlib.Path(cfg["SPLIT_FILE"]).suffix in [".csv", ".tsv", ".txt"]:
        if cfg["SPLIT"] == "pre-computed":
            sample_split = pd.read_csv(
                cfg["SPLIT_FILE"],
                index_col=0,
                sep=cfg["DELIM"],
            )
        else:
            sample_split = pd.read_csv(
                os.path.join(
                    os.path.dirname(cfg["SPLIT_FILE"]),
                    cfg["RUN_ID"],
                    os.path.basename(cfg["SPLIT_FILE"]),
                ),
                index_col=0,
                sep=cfg["DELIM"],
            )
    else:
        if pathlib.Path(cfg["SPLIT_FILE"]).suffix == ".parquet":
            if cfg["SPLIT"] == "pre-computed":
                sample_split = pd.read_parquet(
                    cfg["SPLIT_FILE"],
                )
            else:
                sample_split = pd.read_parquet(
                    os.path.join(
                        os.path.dirname(cfg["SPLIT_FILE"]),
                        cfg["RUN_ID"],
                        os.path.basename(cfg["SPLIT_FILE"]),
                    )
                )
        else:
            logger.warning(
                f"You provided a not supported input file type:{pathlib.Path(cfg['SPLIT_FILE']).suffix}"
            )

    # Merge clin_data with split info
    clin_data = clin_data.join(sample_split.set_index("SAMPLE_ID"), how="inner")
    # reorder lat_space to match clin_data
    # find intersection of lat_space and clin_data
    if cfg["FIX_RANDOMNESS"] == "all":
        seed = cfg["GLOBAL_SEED"]
    else:
        seed = None

    if cfg["MODEL_TYPE"] == "x-modalix":
        from_latent_space = lat_space.loc[lat_space.index.str.startswith("FROM_"), :]
        to_latent_space = lat_space.loc[lat_space.index.str.startswith("TO_"), :]

        from_rel_index = clin_data.index.intersection(
            from_latent_space.index.str.removeprefix("FROM_")
        )
        id_from = "FROM_" + from_rel_index
        from_latent_space = from_latent_space.loc[id_from, :]

        to_rel_index = clin_data.index.intersection(
            to_latent_space.index.str.removeprefix("TO_")
        )
        id_to = "TO_" + to_rel_index
        to_latent_space = to_latent_space.loc[id_to, :]

        lat_space = lat_space.loc[list(id_from) + list(id_to), :]

        logger.info(
            f'Performing {cfg["DIM_RED_METH"]} for 2D aligned latent space visualization.'
        )
        embedding_orig = dim_red(
            lat_space, cfg=cfg, method=cfg["DIM_RED_METH"], seed=seed
        )

        for param in cfg["CLINIC_PARAM"]:
            logger.info(f"Create plots for parameter {param}.")
            from_clin_list = clin_data.loc[from_rel_index, param]
            to_clin_list = clin_data.loc[to_rel_index, param]

            embedding = pd.DataFrame(embedding_orig)
            embedding.columns = ["DIM1", "DIM2"]
            embedding.index = lat_space.index
            embedding[param] = list(from_clin_list) + list(to_clin_list)
            embedding["Translate"] = ["FROM"] * len(from_clin_list) + ["TO"] * len(
                to_clin_list
            )

            plot1 = plot_translate_latent(
                cfg=cfg,
                embedding=embedding,
                color_param=param,
                style_param="Translate",
                save_fig=os.path.join(
                    "reports",
                    run_id,
                    f"figures/latent2D_Aligned_{param}.{sns_out_type}",
                ),
            )
            # list_dm = [
            #     cfg["DATA_TYPE"][dm]["TYPE"]
            #     for dm in cfg["DATA_TYPE"]
            #     if dm in cfg["TRANSLATE"]
            # ]
            to_key = cfg["TRANSLATE"].split("_")[2]
            to_data_type = cfg["DATA_TYPE"][to_key]["TYPE"]

            if to_data_type == "IMG":
                plot2 = translate_grid(
                    cfg=cfg,
                    img_root=os.path.join("reports", run_id, "IMGS", ""),
                    clin_data=clin_data,
                    param=param,
                    save_fig=os.path.join(
                        "reports",
                        run_id,
                        f"figures/translategrid_{param}.{sns_out_type}",
                    ),
                )

            plot3 = plot_latent_ridge(
                lat_space=lat_space,
                clin_data=clin_data,
                param=param,
                save_fig=os.path.join(
                    "reports", run_id, "figures", f"latent_dist_{param}.{sns_out_type}"
                ),
            )

    else:

        rel_index = clin_data.index.intersection(lat_space.index)

        lat_space = lat_space.loc[rel_index, :]
        clin_data = clin_data.loc[rel_index, :]

        # Dimension reduction 2D by tSNE or UMAP (if not Latent Space 2D)
        logger.info(
            f'Performing {cfg["DIM_RED_METH"]} for 2D latent space visualization.'
        )
        embedding = dim_red(lat_space, cfg=cfg, method=cfg["DIM_RED_METH"], seed=seed)

        # Plot latent space based on training/test/valid split
        logger.info(f'Plot {cfg["DIM_RED_METH"]} with SPLIT as labels.')
        plot_latent_2D(
            cfg=cfg,
            embedding=embedding,
            labels=list(clin_data["SPLIT"]),
            param="Split",
            save_fig=os.path.join(
                "reports",
                run_id,
                f"figures/latent2D_SPLIT.{sns_out_type}",
            ),
        )
        plt.close()

        # Plot Latent space: coloring by cluster label, by clinical feature
        for param in cfg["CLINIC_PARAM"]:
            logger.info(f'Plot {cfg["DIM_RED_METH"]} with {param} as labels.')
            plot_latent_2D(
                cfg=cfg,
                embedding=embedding,
                labels=list(clin_data[param]),
                param=param,
                save_fig=os.path.join(
                    "reports",
                    run_id,
                    f"figures/latent2D_{param}.{sns_out_type}",
                ),
            )
            plt.close()

        # # Perform clustering by kMEANS, DBSCAN and so on
        if cfg["PLOT_CLUSTLATENT"]:
            logger.info(
                f'Performing clustering on 2D latent space representation with {cfg["CLUSTER_ALG"]}.'
            )
            labels_cluster, sil_score = cluster(
                embedding=embedding,
                method=cfg["CLUSTER_ALG"],
                n_clusters=cfg["CLUSTER_N"],
                min_cluster_size=cfg["MIN_CLUSTER_N"],
            )
            logger.info("Clustering resulted in a Silhouette Score: " + str(sil_score))
            logger.info(
                f'Plot {cfg["DIM_RED_METH"]} with {cfg["CLUSTER_ALG"]} clusters as labels.'
            )
            plot_latent_2D(
                cfg=cfg,
                embedding=embedding,
                labels=labels_cluster,
                param="Cluster",
                save_fig=os.path.join(
                    "reports",
                    run_id,
                    f'figures/latent2D_{cfg["CLUSTER_ALG"]}.{sns_out_type}',
                ),
            )
            plt.close()

        if cfg["PLOT_INPUT2D"]:
            logger.info(
                f"Plot {cfg['DIM_RED_METH']} directly on input layer for comparison."
            )

            X = pd.DataFrame()
            usefiles = list(cfg["DATA_TYPE"].keys())
            # Remove ANNOTATION Types
            for u in usefiles:
                if cfg["DATA_TYPE"][u]["TYPE"] == "ANNOTATION":
                    usefiles.remove(u)
            for data_type in usefiles:
                if cfg["DATA_TYPE"][data_type]["TYPE"] == "IMG":
                    continue

                df = pd.read_parquet(
                    os.path.join("data/processed", run_id, data_type + "_data.parquet")
                ).add_prefix(f"{data_type}_")
                X = pd.concat(
                    [X, df],
                    axis=1,
                )
            # reorder input layer to match clin_data
            X = X.loc[clin_data.index, :]
            embedding_direct = dim_red(
                X, cfg=cfg, method=cfg["DIM_RED_METH"], seed=seed
            )

            plot_latent_2D(
                cfg=cfg,
                embedding=embedding_direct,
                labels=list(clin_data["SPLIT"]),
                layer="input layer",
                param="Split",
                save_fig=os.path.join(
                    "reports",
                    run_id,
                    "figures",
                    f"inputlayer2D_SPLIT.{sns_out_type}",
                ),
            )
            plt.close()

            for param in cfg["CLINIC_PARAM"]:
                logger.info(f'Plot {cfg["DIM_RED_METH"]} with {param} as labels.')
                plot_latent_2D(
                    cfg=cfg,
                    embedding=embedding_direct,
                    labels=list(clin_data[param]),
                    layer="input layer",
                    param=param,
                    save_fig=os.path.join(
                        "reports",
                        run_id,
                        "figures",
                        f"inputlayer2D_{param}.{sns_out_type}",
                    ),
                )
                plt.close()

        ### Plot ontology related figures
        ## visualize encoder/decoder weights in heatmaps
        #  -> for convenience done directly in build_models.py

        logger.info("Plot latent dim distributions")
        for param in cfg["CLINIC_PARAM"]:
            logger.info(f"Plot latent for {param}")

            clin_data = clin_data.sort_values(param)
            # reorder lat_space to match clin_data
            lat_space = lat_space.loc[clin_data.index, :]

            plot_latent_ridge(
                lat_space=lat_space,
                clin_data=clin_data,
                param=param,
                save_fig=os.path.join(
                    "reports", run_id, "figures", f"latent_dist_{param}.{sns_out_type}"
                ),
            )
            plt.close()


if __name__ == "__main__":
    main()
