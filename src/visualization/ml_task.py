import os
import sys
import warnings

from numba.core.errors import NumbaDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
from sklearn.exceptions import ConvergenceWarning

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning,ignore::RuntimeWarning"
import pathlib

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model, svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import cross_validate
from umap import UMAP

from src.utils.config import get_cfg
from src.utils.utils_basic import getlogger

sns.set(rc={"figure.figsize": (20, 15)})
# sns.set(font_scale=1.0)
sns.set_style("white")


def single_ml(df, clin_data, task_param, sklearn_ml, metrics, cfg):
    """
    Function learns on the given data frame df and label data the provided sklearn model.
    Cross validation is performed according to the config and scores are returned as output as specified by metrics
    ARGS:
        df (pd.DataFrame): Dataframe with input data
        clin_data (pd.DataFrame): Dataframe with label data
        task_param (str): Column name with label data
        sklearn_ml (sklearn.module): Sklearn ML module specifying the ML algorithm
        metrics (list): list of metrics (scores) to be calculated by cross validation
        cfg (dict): dictionary of the yaml config
    RETURNS:
        score_df (pd.DataFrame): data frame containing metrics (scores) for all CV runs (long format)

    """

    # X -> df
    # Y -> task_param
    y = clin_data.loc[df.index, task_param]
    score_df = dict()

    ## Cross Validation
    if len(y.unique()) > 1:
        scores = cross_validate(
            sklearn_ml, df, y, cv=cfg["CV"], scoring=metrics, return_train_score=True
        )

        # Output

        # Output Format
        # CV_RUN | SCORE_SPLIT | TASK_PARAM | METRIC | VALUE

        score_df["cv_run"] = list()
        score_df["score_split"] = list()
        score_df["CLINIC_PARAM"] = list()
        score_df["metric"] = list()
        score_df["value"] = list()

        cv_runs = ["CV_" + str(x) for x in range(1, cfg["CV"] + 1)]
        task_param_cv = [task_param for x in range(1, cfg["CV"] + 1)]

        for metric in scores:
            if metric.split("_")[0] == "test" or metric.split("_")[0] == "train":
                split_cv = [metric.split("_")[0] for x in range(1, cfg["CV"] + 1)]
                metric_cv = [
                    "_".join(metric.split("_")[1:]) for x in range(1, cfg["CV"] + 1)
                ]

                score_df["cv_run"].extend(cv_runs)
                score_df["score_split"].extend(split_cv)
                score_df["CLINIC_PARAM"].extend(task_param_cv)
                score_df["metric"].extend(metric_cv)
                score_df["value"].extend(scores[metric])

    return pd.DataFrame(score_df)


def single_ml_presplit(
    sample_split, df, clin_data, task_param, sklearn_ml, metrics, cfg
):
    """
    Function learns on the given data frame df and label data the provided sklearn model.
    Split infomation from autoencoder for samples are used and scores are returned as output as specified by metrics for each split
    ARGS:
        df (pd.DataFrame): Dataframe with input data
        clin_data (pd.DataFrame): Dataframe with label data
        task_param (str): Column name with label data
        sklearn_ml (sklearn.module): Sklearn ML module specifying the ML algorithm
        metrics (list): list of metrics (scores) to be calculated by cross validation
        cfg (dict): dictionary of the yaml config
    RETURNS:
        score_df (pd.DataFrame): data frame containing metrics (scores) for all CV runs (long format)

    """
    split_list = ["train", "valid", "test"]

    score_df = dict()
    score_df["score_split"] = list()
    score_df["CLINIC_PARAM"] = list()
    score_df["metric"] = list()
    score_df["value"] = list()

    logger = getlogger(cfg)

    for split in split_list:
        X = df.loc[sample_split.loc[sample_split.SPLIT == split, "SAMPLE_ID"], :]
        samples = [s for s in X.index]
        Y = clin_data.loc[samples, task_param]

        # Train on train data
        if len(Y.unique()) > 1:
            sklearn_ml.fit(X, Y)

            # Performace on train, valid and test data split

            for m in metrics:
                score_df["score_split"].append(split)
                score_df["CLINIC_PARAM"].append(task_param)
                score_df["metric"].append(m)
                match m:
                    case "roc_auc_ovo":
                        # print(f'number of unique values in Y: {len(pd.unique(Y))}')
                        y_proba = sklearn_ml.predict_proba(X)
                        if len(pd.unique(Y)) == 2:
                            y_proba = y_proba[:, 1]

                        roc_temp = roc_auc_score(
                            Y, y_proba, multi_class="ovo", average="macro"
                        )
                        score_df["value"].append(roc_temp)

                    case "r2":
                        r2_temp = r2_score(Y, sklearn_ml.predict(X))
                        score_df["value"].append(r2_temp)

                    case _:
                        logger.info(
                            f"Your metric: {m} is not yet supported or valid for this ML type"
                        )

    return pd.DataFrame(score_df)


def load_input_for_ml(task, run_id, cfg):
    logger = getlogger(cfg)
    usefiles = list(cfg["DATA_TYPE"].keys())
    # print(f'usefiles: {usefiles}')
    # Remove ANNOTATION Types
    for u in usefiles[:]:
        matchtype = cfg["DATA_TYPE"][u]["TYPE"]
        # print(f' matchtype: {matchtype}')
        if (
            cfg["DATA_TYPE"][u]["TYPE"] == "ANNOTATION"
            or cfg["DATA_TYPE"][u]["TYPE"] == "IMG"
        ):
            # print(f' remove {u}')
            usefiles.remove(u)
    if (len(usefiles) == 0) and (task in ["PCA", "UMAP", "RandomFeature"]):
        logger.error("You provided only ANNO and IMG data types")
        logger.error(
            "On IMG data type the tasks PCA, UMAP and RandomFeature cannot be performed as ML_TASK."
        )
        logger.error("Remove those tasks from the config and select only Latent.")
        raise NotImplementedError(f"No suitable data type for task {task}")
    match task:
        case "Latent":
            lat_file = os.path.join("reports", run_id, "predicted_latent_space.parquet")

            df = pd.read_parquet(lat_file)

        case "Latent_FROM":
            lat_file = os.path.join("reports", run_id, "predicted_latent_space.parquet")

            df = pd.read_parquet(lat_file)
            df = df.loc[df.index.str.startswith("FROM_"), :]
            df.index = df.index.str.removeprefix("FROM_")

        case "Latent_TO":
            lat_file = os.path.join("reports", run_id, "predicted_latent_space.parquet")

            df = pd.read_parquet(lat_file)
            df = df.loc[df.index.str.startswith("TO_"), :]
            df.index = df.index.str.removeprefix("TO_")

        case "Latent_BOTH":
            lat_file = os.path.join("reports", run_id, "predicted_latent_space.parquet")

            df = pd.read_parquet(lat_file)
            df.index = df.index.str.removeprefix("TO_").str.removeprefix("FROM_")

        case "RandomFeature":
            df = pd.DataFrame()
            for data_type in usefiles:

                if data_type == "IMG":
                    continue
                df_sub = pd.read_parquet(
                    os.path.join("data/processed", run_id, data_type + "_data.parquet")
                ).add_prefix(f"{data_type}_")

                df = pd.concat(
                    [df, df_sub],
                    axis=1,
                )

            df = df.sample(n=cfg["LATENT_DIM_FIXED"], axis=1)

        case "UMAP":
            df = pd.DataFrame()
            for data_type in usefiles:
                # df_sub = pd.read_csv(

                if data_type == "IMG":
                    continue
                df_sub = pd.read_parquet(
                    os.path.join("data/processed", run_id, data_type + "_data.parquet")
                    # index_col=0,
                    # sep=cfg['DELIM']
                ).add_prefix(f"{data_type}_")

                df = pd.concat(
                    [df, df_sub],
                    axis=1,
                )
            reducer = UMAP(n_components=cfg["LATENT_DIM_FIXED"])
            df = pd.DataFrame(reducer.fit_transform(df), index=df.index)

        case "PCA":
            df = pd.DataFrame()
            for data_type in usefiles:
                if data_type == "IMG":
                    logger.info(f"PCA not supported for {data_type} data type")
                    continue
                df_sub = pd.read_parquet(
                    os.path.join("data/processed", run_id, data_type + "_data.parquet")
                    # index_col=0,
                    # sep=cfg['DELIM']
                ).add_prefix(f"{data_type}_")

                if data_type == "IMG":
                    continue

                df = pd.concat(
                    [df, df_sub],
                    axis=1,
                )
            reducer = PCA(n_components=cfg["LATENT_DIM_FIXED"])
            df = pd.DataFrame(reducer.fit_transform(df), index=df.index)

        case _:
            logger.info(f"Your ML task {task} is not supported.")

    return df


def get_ml_type(clin_data, task_param, cfg):
    logger = getlogger(cfg)
    if cfg["ML_TYPE"] == "Auto-detect":
        if type(list(clin_data[task_param])[0]) is str:
            ml_type = "classification"
        else:
            ml_type = "regression"
    else:
        if task_param in clin_data.columns:
            if task_param in cfg["ML_TYPE"]:
                ml_type = cfg["ML_TYPE"][task_param]
            else:
                logger.error(
                    f"Your clinical parameter and ML task {task_param} is not specified as classification or regression in config parameter ML-TYPE. Use keyword Auto-detect instead."
                )
                logger.info("Reverting to auto-detect ML-TYPE")
                if type(list(clin_data[task_param])[0]) is str:
                    ml_type = "classification"
                else:
                    ml_type = "regression"
        else:
            logger.error(
                f"Your provided ML task {task_param} in config parameter ML-TYPE is not present in ANNOTATION data type. Please provide a correct column name."
            )
    return ml_type


@click.command()
@click.argument("run_id", type=str)
def main(run_id):
    """Performance downstream machine learning and evaluation of predictive power of learned latent space representation"""

    cfg = get_cfg(run_id)
    logger = getlogger(cfg)
    already_warned = False
    logger.info("Performing predictive power evaluation")

    df_results = pd.DataFrame()
    if cfg["MODEL_TYPE"] == "x-modalix":
        if "Latent" in cfg["ML_TASKS"]:
            cfg["ML_TASKS"].remove("Latent")
            cfg["ML_TASKS"].append("Latent_FROM")
            cfg["ML_TASKS"].append("Latent_TO")
            cfg["ML_TASKS"].append("Latent_BOTH")

    for task in cfg["ML_TASKS"]:
        logger.info(f"Perform ML task with feature df: {task}")

        anno_name = [
            data_type
            for data_type in cfg["DATA_TYPE"]
            if cfg["DATA_TYPE"][data_type]["TYPE"] == "ANNOTATION"
        ]

        if not (len(anno_name) == 1):
            logger.warning(
                "No ANNOTATION data type found. ML tasks cannot be performed."
            )
            return
        clin_data = pd.read_parquet(
            os.path.join(
                "data/processed",
                cfg["RUN_ID"],
                anno_name[0] + "_data.parquet",
            )
        )

        if cfg["ML_SPLIT"] == "use-split":
            if cfg["PREDICT_SPLIT"] == "all":
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
            else:
                logger.warning(
                    f"When ML_SPLIT is set to 'use-split', PREDICT_SPLIT must be set to 'all'. Currently set to: {cfg['PREDICT_SPLIT']}"
                )
                logger.warning("ml_task is stopped.")
                return

        ## df -> task
        subtask = [task]
        if task == "RandomFeature":
            subtask = [
                task + str(x) for x in range(1, 6)
            ]  
        for sub in subtask:
            # logger.info(f"Perform ML task with subtask: {sub}")
            df = load_input_for_ml(task, run_id, cfg)

            for task_param in cfg["CLINIC_PARAM"]:
                if "Latent" in task:
                    logger.info(f"Perform ML task for target parameter: {task_param}")
                ## Check if classification or regression task
                ml_type = get_ml_type(clin_data, task_param, cfg)

                if pd.isna(clin_data[task_param]).sum() > 0:
                    
                    if not already_warned:
                        logger.warning(
                            "There are NA values in the annotation file. Samples with missing data will be removed for ML task evaluation."
                        )
                    already_warned = True
                    # logger.warning(clin_data.loc[pd.isna(clin_data[task_param]), task_param])

                    samples_nonna = clin_data.loc[
                        pd.notna(clin_data[task_param]), task_param
                    ].index
                    # print(df)
                    df = df.loc[samples_nonna.intersection(df.index), :]
                    if cfg["ML_SPLIT"] == "use-split":
                        sample_split = sample_split.loc[
                            samples_nonna.intersection(sample_split.index), :
                        ]
                    # print(sample_split)

                if ml_type == "classification":
                    metrics = ["roc_auc_ovo"]

                if ml_type == "regression":
                    # metrics = ["r2", "neg_root_mean_squared_error"]
                    metrics = ["r2"]

                for ml_alg in cfg["ML_ALG"]:
                    # ml_alg -> sklearn_alg
                    # logger.info(f"Perform ML task with algorithm: {ml_alg}")
                    match ml_alg:
                        case "Linear":
                            if ml_type == "classification":
                                sklearn_ml = linear_model.LogisticRegression(
                                    solver="sag",
                                    n_jobs=-1,
                                    class_weight="balanced",
                                    max_iter=200,
                                )
                            if ml_type == "regression":
                                sklearn_ml = linear_model.LinearRegression(n_jobs=-1)

                        case "RF":
                            if ml_type == "classification":
                                sklearn_ml = RandomForestClassifier(
                                    n_jobs=-1,
                                    max_depth=10,
                                    min_samples_leaf=2,
                                    class_weight="balanced",
                                )
                            if ml_type == "regression":
                                sklearn_ml = RandomForestRegressor(
                                    n_jobs=-1, max_depth=10, min_samples_leaf=2
                                )

                        case "SVM":
                            if ml_type == "classification":
                                sklearn_ml = svm.SVC(
                                    probability=True, class_weight="balanced"
                                )
                            if ml_type == "regression":
                                sklearn_ml = svm.SVR()
                        case _:
                            logger.info(
                                f"Your ML task algorithm: {ml_alg} is not yet supported"
                            )

                    match cfg["ML_SPLIT"]:
                        case "CV-on-all-data":
                            results = single_ml(
                                df, clin_data, task_param, sklearn_ml, metrics, cfg
                            )

                        case "use-split":
                            results = single_ml_presplit(
                                sample_split,
                                df,
                                clin_data,
                                task_param,
                                sklearn_ml,
                                metrics,
                                cfg,
                            )

                        case _:
                            logger.info(
                                f"Your ML split option: {cfg['ML_SPLIT']} is not correct."
                            )

                    res_ml_alg = [ml_alg for x in range(0, results.shape[0])]
                    res_ml_type = [ml_type for x in range(0, results.shape[0])]
                    res_ml_task = [task for x in range(0, results.shape[0])]
                    res_ml_subtask = [sub for x in range(0, results.shape[0])]

                    results["ML_ALG"] = res_ml_alg
                    results["ML_TYPE"] = res_ml_type
                    results["ML_TASK"] = res_ml_task
                    results["ML_SUBTASK"] = res_ml_subtask

                    df_results = pd.concat([df_results, results])

    ## Save all results
    df_results.to_csv(
        os.path.join("reports", run_id, "ml_task_performance.txt"),
        index=False,
        sep=cfg["DELIM"],
    )

    ## Plot all results

    for c in pd.unique(df_results.CLINIC_PARAM):
        for m in pd.unique(df_results.loc[df_results.CLINIC_PARAM == c, "metric"]):

            sns_plot = sns.catplot(
                data=df_results[
                    (df_results.metric == m) & (df_results.CLINIC_PARAM == c)
                ],
                x="score_split",
                y="value",
                row="ML_ALG",
                col="ML_TASK",
                hue="score_split",
                kind="bar",
            )

            min_y = df_results[
                (df_results.metric == m) & (df_results.CLINIC_PARAM == c)
            ].value.min()
            if min_y > 0:
                min_y = 0
            sns_plot.set(ylim=(min_y, None))
            plt.savefig(
                os.path.join(
                    "reports", run_id, "figures", f"ml_task_performance_{c}_{m}.png"
                )
            )


if __name__ == "__main__":
    main()
