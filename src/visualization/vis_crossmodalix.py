import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from src.utils.utils_basic import getlogger

sns_out_type = "png"


def translate_grid(cfg, img_root, clin_data, param, save_fig=""):
    img_sample_list = []
    from_center_list = []
    to_center_list = []
    nrandom = 3

    supported_extensions = [
        "jpg",
        "jpeg",
        "JPEG",
        "JPG",
        "png",
        "PNG",
        "tif",
        "TIF",
        "tiff",
        "TIFF",
    ]

    logger = getlogger(cfg)
    if not (type(clin_data[param].iloc[0]) is str):
        logger.info(
            f"The provided label column is numeric and converted to categories."
        )
        if len(np.unique(clin_data[param])) > 3:
            labels = pd.qcut(
                clin_data[param], q=4, labels=["1stQ", "2ndQ", f"3rdQ", f"4thQ"]
            ).astype(str)
        else:
            labels = [str(x) for x in clin_data[param]]

        clin_data[param] = labels

    for ext in supported_extensions:
        for img_file in sorted(glob.glob(img_root + "*." + ext)):
            img_sample_list.append(
                img_file.removeprefix(img_root).removesuffix("." + ext)
            )
            if "from_center_" in img_file:
                # print(img_file.removeprefix(img_root+"from_center_").removesuffix(".png"))
                if (
                    img_file.removeprefix(img_root + "from_center_").removesuffix(
                        "-" + param + ".png"
                    )
                    in clin_data[param].unique()
                ):
                    from_center_list.append(img_file.removeprefix(img_root))
            if "to_center_" in img_file:
                if (
                    img_file.removeprefix(img_root + "to_center_").removesuffix(
                        "-" + param + ".png"
                    )
                    in clin_data[param].unique()
                ):
                    to_center_list.append(img_file.removeprefix(img_root))
            # print(img_file.removeprefix(img_root).removesuffix(".png"))

    img_sample_list = clin_data.index.intersection(img_sample_list)
    try:
        rand_sample_param = (
            clin_data.loc[img_sample_list, [param, "SPLIT"]]
            .loc[clin_data["SPLIT"] == "test", :]
            .groupby(param)
            .sample(n=nrandom, replace=False, random_state=cfg["GLOBAL_SEED"])
        )
    except ValueError:
        logger.warning(
            "Note enough test samples per class to create translate grid with examples. Revert to whole data set and plot only one example per class."
        )
        nrandom = 1
        rand_sample_param = (
            clin_data.loc[img_sample_list, [param, "SPLIT"]]
            .groupby(param)
            .sample(n=nrandom, replace=False, random_state=cfg["GLOBAL_SEED"])
        )
    # Define labels
    row_labels = ["FROM Center", "TO Center"] + [
        "FROM_TO test pick " + str(x) for x in range(1, nrandom + 1)
    ]
    if nrandom == 1:
        row_labels = ["FROM Center", "TO Center"] + [
            "FROM_TO random pick " + str(x) for x in range(1, nrandom + 1)
        ]
    col_labels = [
        file.removeprefix("from_center_").removesuffix("-" + param + ".png")
        for file in from_center_list
    ]
    # Create figure and axes
    single_size = 2
    fig, axes = plt.subplots(
        nrandom + 2,
        len(from_center_list),
        figsize=(len(from_center_list) * single_size, (nrandom + 2) * single_size),
    )
    fig.subplots_adjust(
        left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4
    )

    # Plot images and add labels
    for i, ax in enumerate(axes.flat):
        row = int(i / len(from_center_list))
        if row == 0:  ## First row
            ax.imshow(np.asarray(Image.open(img_root + from_center_list[i])))
            ax.axis("off")
        else:
            if row == 1:  ## Second row
                ax.imshow(
                    np.asarray(
                        Image.open(img_root + to_center_list[i - len(from_center_list)])
                    )
                )
                ax.axis("off")
            else:
                test_sample = rand_sample_param[
                    rand_sample_param[param] == col_labels[i % len(col_labels)]
                ].index[row - 2]
                file_img = glob.glob(img_root + test_sample + ".*")
                ax.imshow(np.asarray(Image.open(file_img[0])))
                ax.axis("off")
        # Add row labels
        if i % len(from_center_list) == 0:
            ax.text(
                -0.5,
                0.5,
                row_labels[i // len(from_center_list)],
                va="center",
                ha="right",
                transform=ax.transAxes,
            )
        # Add column labels
        if i < len(from_center_list):
            ax.text(
                0.5,
                1.1,
                col_labels[i],
                va="bottom",
                ha="center",
                rotation='vertical',
                transform=ax.transAxes,
            )
    # plt.tight_layout()
    # fig.suptitle(param, horizontalalignment="left")

    if len(save_fig) > 0:
        plt.savefig(save_fig, bbox_inches="tight")
    return fig


def plot_translate_latent(
    cfg,
    embedding,
    color_param,
    style_param=None,
    save_fig="",
):
    """
    Creates a 2D visualization of the 2D embedding of the latent space.
    ARGS:
        cfg (dict): config dictionary
        embedding (pd.DataFrame): embedding on which is visualized. Assumes prior 2D dimension reduction.
        color_param (str): Clinical parameter to color scatter plot
        style_param (str): Parameter e.g. "Translate" to facet scatter plot
        save_fig (str): File path for saving the plot. Use appropriate file
                        endings to specify image type (e.g. '*.png')
    RETURNS:
        fig (seaborn.FacetGrid): Figure handle

    """
    labels = embedding[color_param]
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
                numeric = True
        else:
            labels = [str(x) for x in labels]

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

    if not style_param == None:
        embedding[color_param] = labels
        if numeric:
            palette = "bwr"
        else:
            palette = None
        plot = sns.relplot(
            data=embedding,
            x="DIM1",
            y="DIM2",
            hue=color_param,
            palette=palette,
            col=style_param,
            style=style_param,
            markers=["o", "v"],
            alpha=0.4,
            ec="black",
            height=10,
            aspect=1,
        )

    if len(save_fig) > 0:
        plot.savefig(save_fig, bbox_inches="tight")

    return plot


def plot_translate_latent_simple(
    cfg,
    embedding,
    style_param=None,
    save_fig="",
):
    """
    Creates a 2D visualization of the 2D embedding of the latent space.
    ARGS:
        cfg (dict): config dictionary
        embedding (pd.DataFrame): embedding on which is visualized. Assumes prior 2D dimension reduction.
        color_param (str): Clinical parameter to color scatter plot
        style_param (str): Parameter e.g. "Translate" to facet scatter plot
        save_fig (str): File path for saving the plot. Use appropriate file
                        endings to specify image type (e.g. '*.png')
    RETURNS:
        fig (seaborn.FacetGrid): Figure handle

    """

    if not style_param == None:

        plot = sns.relplot(
            data=embedding,
            x="DIM1",
            y="DIM2",
            col=style_param,
            style=style_param,
            markers=["o", "v"],
            alpha=0.4,
            ec="black",
            height=10,
            aspect=1,
        )

    if len(save_fig) > 0:
        plot.savefig(save_fig, bbox_inches="tight")

    return plot


# if __name__ == "__main__":
#     run_id = sys.argv[1]
#     main(run_id)
