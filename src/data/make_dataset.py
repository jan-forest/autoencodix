import logging
import os
import pathlib
import shutil
from math import fsum

import click
from sklearn_extra.cluster import CLARA
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from src.utils.config import get_cfg
from src.utils.utils_basic import getlogger


def unified_sample_list(cfg):
    """
    Function creates an uniformed list of sample ID across the provided data
    sets defined in config.yaml. Only the intersection of samples across all
    data types is kept. If not pre-computed a training, validation and test
    split of sample ID is calculated as defined in config.yaml
    ARGS:
        cfg (dict): Dictionary of configuration parameters
    RETURNS:
        sample_list (pd.Index): List of unified sample IDs across all data types.

    """
    sample_list = pd.Index([])
    SPLIT = cfg["SPLIT"]
    logger = getlogger(cfg)
    if not SPLIT == "pre-computed":

        for data_type in cfg["DATA_TYPE"]:
            logger.debug(f"current dtype: {data_type}")
            raw_path = os.path.join(
                cfg["ROOT_RAW"], cfg["DATA_TYPE"][data_type]["FILE_RAW"]
            )
            if "REL_CLIN_N" in cfg["DATA_TYPE"][data_type]:
                if cfg["DATA_TYPE"][data_type]["FILE_RAW"].split(".")[1] in [
                    "csv",
                    "tsv",
                    "txt",
                ]:
                    df = pd.read_csv(
                        raw_path, sep=cfg["DELIM"], skiprows=0, index_col=[0]
                    )
                else:
                    if (
                        cfg["DATA_TYPE"][data_type]["FILE_RAW"].split(".")[1]
                        == "parquet"
                    ):
                        df = pd.read_parquet(raw_path)
                    else:
                        logger.warning(
                            f"You provided a not supported input file type:{cfg['DATA_TYPE'][data_type]['FILE_RAW'].split('.')[1]}"
                        )
                logger.info(
                    f"Removing samples if nan in relevant numeric and categorical features"
                )
                logger.info(f"Shape before {df.shape}")
                rel_col = (
                    cfg["DATA_TYPE"][data_type]["REL_CLIN_N"]
                    + cfg["DATA_TYPE"][data_type]["REL_CLIN_C"]
                )
                df = df.loc[:, rel_col].dropna(axis=0)
                logger.info(f"Shape after {df.shape}")
                s_id = df

            else:
                if cfg["DATA_TYPE"][data_type]["FILE_RAW"].split(".")[1] in [
                    "csv",
                    "tsv",
                    "txt",
                ]:
                    s_id = pd.read_csv(
                        raw_path,
                        sep=cfg["DELIM"],
                        skiprows=0,
                        index_col=[0],
                        usecols=[0],
                    )
                else:
                    if (
                        cfg["DATA_TYPE"][data_type]["FILE_RAW"].split(".")[1]
                        == "parquet"
                    ):
                        s_id = pd.read_parquet(raw_path)
                    else:
                        logger.warning(
                            f"You provided a not supported input file type:{cfg['DATA_TYPE'][data_type]['FILE_RAW'].split('.')[1]}"
                        )

            if len(sample_list) > 0:
                sample_list = sample_list.intersection(s_id.index)
            else:
                sample_list = s_id.index

        if "REDUCED_TEST_MODE" in cfg:  ### Hidden config option for fast testing
            sample_list = pd.Index(
                pd.DataFrame(sample_list)
                .sample(frac=cfg["REDUCED_TEST_MODE"], random_state=1)
                .iloc[:, 0]
            )
            logger.info(f"After sampling sample ID list has length {len(sample_list)}")
        # Splitting
        if fsum(SPLIT) == 1:
            if min(SPLIT) == 0:
                split_out = pd.DataFrame()
                # Supported edge cases
                if SPLIT[0] == 1:
                    logger.info(f"You selected a split with purely training samples")

                    split_out = pd.concat(
                        [
                            pd.DataFrame(sample_list),
                            pd.DataFrame(["train"] * len(sample_list)),
                        ],
                        axis=1,
                    )
                if SPLIT[2] == 1:
                    logger.info(f"You selected a split with purely test samples")

                    split_out = pd.concat(
                        [
                            pd.DataFrame(sample_list),
                            pd.DataFrame(["test"] * len(sample_list)),
                        ],
                        axis=1,
                    )

                if len(split_out) == 0:
                    logger.warning(f"You provided a not supported split: {SPLIT}")
                    logger.info(f"All samples will be assigned as training.")
                    split_out = pd.concat(
                        [
                            pd.DataFrame(sample_list),
                            pd.DataFrame(["train"] * len(sample_list)),
                        ],
                        axis=1,
                    )

                split_out.columns = ["SAMPLE_ID", "SPLIT"]
                if len(cfg["SPLIT_FILE"]) > 0:
                    logger.info(f"Save sample split.")
                    split_out.index = split_out["SAMPLE_ID"]
                    # split_out.to_csv(
                    split_out.to_parquet(
                        os.path.join(
                            os.path.dirname(cfg["SPLIT_FILE"]),
                            cfg["RUN_ID"],
                            os.path.basename(cfg["SPLIT_FILE"]),
                        )
                        # index=False,
                        # sep=cfg['DELIM']
                    )
                else:
                    logger.warning("No output file provided for data split.")

            else:
                if (
                    cfg["FIX_RANDOMNESS"] == "all"
                    or cfg["FIX_RANDOMNESS"] == "data_split"
                ):
                    random_state = cfg["GLOBAL_SEED"]
                else:
                    random_state = None

                sample_list_train, sample_list_test = train_test_split(
                    sample_list,
                    train_size=SPLIT[0],
                    random_state=random_state,
                )
                sample_list_validate, sample_list_test = train_test_split(
                    sample_list_test,
                    train_size=SPLIT[1] / (SPLIT[1] + SPLIT[2]),
                    random_state=random_state,
                )
                split = (
                    ["train"] * len(sample_list_train)
                    + ["test"] * len(sample_list_test)
                    + ["valid"] * len(sample_list_validate)
                )

                split_out = pd.concat(
                    [
                        pd.concat(
                            [
                                pd.DataFrame(sample_list_train),
                                pd.DataFrame(sample_list_test),
                                pd.DataFrame(sample_list_validate),
                            ]
                        ).reset_index(drop=True),
                        pd.DataFrame(split),
                    ],
                    axis=1,
                )
                split_out.columns = ["SAMPLE_ID", "SPLIT"]
                if len(cfg["SPLIT_FILE"]) > 0:
                    logger.info(f"Save sample split.")
                    split_out.index = split_out["SAMPLE_ID"]
                    # split_out.to_csv(
                    split_out.to_parquet(
                        os.path.join(
                            os.path.dirname(cfg["SPLIT_FILE"]),
                            cfg["RUN_ID"],
                            os.path.basename(cfg["SPLIT_FILE"]),
                        )
                    )

                else:
                    logger.warning("No output file provided for data split.")

        else:
            logger.warning(f"Your split {SPLIT} does not add up to 1.0")
            logger.info(f"All samples will be assigned as training.")
            split_out = pd.concat(
                [pd.DataFrame(sample_list), pd.DataFrame(["train"] * len(sample_list))],
                axis=1,
            )

            split_out.columns = ["SAMPLE_ID", "SPLIT"]
            if len(cfg["SPLIT_FILE"]) > 0:
                logger.info(f"Save sample split.")
                split_out.index = split_out["SAMPLE_ID"]
                split_out.to_parquet(
                    os.path.join(
                        os.path.dirname(cfg["SPLIT_FILE"]),
                        cfg["RUN_ID"],
                        os.path.basename(cfg["SPLIT_FILE"]),
                    )
                )

            else:
                logger.warning("No output file provided for data split.")
    else:
        logger.warning("Unified sample list is provided by pre-computed split")
        if len(cfg["SPLIT_FILE"]) > 0:
            logger.info(f"Load sample split.")
            if pathlib.Path(cfg["SPLIT_FILE"]).suffix in [".csv", ".tsv", ".txt"]:
                sample = pd.read_csv(
                    cfg["SPLIT_FILE"],
                    index_col=0,
                    sep=cfg["DELIM"],
                )
            else:
                if pathlib.Path(cfg["SPLIT_FILE"]).suffix == ".parquet":
                    sample = pd.read_parquet(
                        cfg["SPLIT_FILE"],
                    )
                else:
                    logger.warning(
                        f"You provided a not supported input file type:{pathlib.Path(cfg['SPLIT_FILE']).suffix}"
                    )
            sample_list = sample.index

        else:
            logger.warning("No file provided for pre-computed split.")

    return sample_list


def make_data_single(cfg, data_type, sample_list):
    """
    Function to create processed data for the provided data type and list of sample IDs.
    Processing includes filtering, scaling and encoding according definition in config.yaml.
    ARGS:
        cfg (dict): configuration dictionary
        data_type (str): name of data type as defined in config.yaml
        sample_list (pd.Index): List of sample IDs to be considered
    RETURNS:
        df (pd.DataFrame): data frame with processed data for samples in sample_list

    """

    logger = getlogger(cfg)

    raw_path = os.path.join(cfg["ROOT_RAW"], cfg["DATA_TYPE"][data_type]["FILE_RAW"])
    if cfg["DATA_TYPE"][data_type]["FILE_RAW"].split(".")[1] in ["csv", "tsv", "txt"]:
        df = pd.read_csv(raw_path, sep=cfg["DELIM"], index_col=[0], header=0)
    else:
        if cfg["DATA_TYPE"][data_type]["FILE_RAW"].split(".")[1] == "parquet":
            df = pd.read_parquet(raw_path)
        else:
            logger.warning(
                f"You provided a not supported input file type:{cfg['DATA_TYPE'][data_type]['FILE_RAW'].split('.')[1]}"
            )

    if cfg["DATA_TYPE"][data_type]["TYPE"] == "ANNOTATION":
        # Save Label File before encoding
        logger.info("Save ANNOTATION without preprocessing")
        # if "FILE_LABEL" in cfg["DATA_TYPE"][data_type]:
        # df.to_csv(
        df.to_parquet(
            os.path.join(
                "data/processed",
                cfg["RUN_ID"],
                data_type + "_data.parquet",
            )
        )
    else:

        logger.info(f"Select for samples and drop features with NA")
        if df.index.has_duplicates:
            logger.info(f"Your data has duplicate rows for sample IDs:")
            logger.info(df[df.index.duplicated()])

        df = df.loc[
            sample_list,
        ]
        df = df.dropna(axis=1)

        #### Synthetic signal ###
        if cfg["APPLY_SIGNAL"]:
            logger.info(f"Inject signal on given features and samples")
            feature_synth = pd.read_csv(cfg["FEATURE_SIGNAL"], header=None, index_col=0)
            sample_synth = pd.read_csv(cfg["SAMPLE_SIGNAL"], header=None, index_col=0)
            feature_synth = feature_synth.index.intersection(df.columns).to_list()
            logger.info("Feature length " + str(len(feature_synth)))
            sample_synth = sample_synth.index.intersection(df.index).to_list()
            logger.info("Sample length " + str(len(sample_synth)))

            df = synth_signal(df, feature_list=feature_synth, sample_list=sample_synth)
        ###################################

        ### build ontology dictionary
        ## if ontAE
        if cfg["MODEL_TYPE"] == "ontix":
            logger.info(f"Preprocessing of ontology files")

            lvl1_file_path = os.path.join(cfg["ROOT_RAW"], cfg["FILE_ONT_LVL1"])
            lvl1_proc_path = os.path.join(
                "data/processed", cfg["RUN_ID"], cfg["FILE_ONT_LVL1"]
            )

            lvl1_proc = open(lvl1_proc_path, "a")

            feature_list = list(df.columns)

            if os.path.isfile(lvl1_file_path):
                # read in given ontology
                with open(lvl1_file_path, "r") as lvl1_file:
                    for line in lvl1_file:
                        feature_id_child = line.split("\t")[0]
                        ont_id_parent = line.strip().split("\t")[1]

                        # check if feature from ontology in df after filtering
                        if feature_id_child in df.columns:
                            # add prefix for data type e.g. Pathway_1 -> [rna_geneID1, cna_geneID1]
                            out = (
                                data_type
                                + "_"
                                + feature_id_child
                                + cfg["DELIM"]
                                + ont_id_parent
                                + "\n"
                            )
                            lvl1_proc.write(out)
                            if feature_id_child in feature_list:
                                feature_list.remove(feature_id_child)

                    ## Create a bucket for unconnected features per data type
                    logger.warning(
                        f"Number of features not in ontology file: {len(feature_list)}"
                    )
                    if len(feature_list) > 0 and cfg["KEEP_NOT_ONT"]:
                        logger.warning(
                            f"Features not in ontology file are connected combined in the additional bucket: not_in_ontology_{data_type}"
                        )

                        for f_id in feature_list:
                            ont_id_parent = "not_in_ontology_" + data_type
                            out = (
                                data_type
                                + "_"
                                + f_id
                                + cfg["DELIM"]
                                + ont_id_parent
                                + "\n"
                            )
                            lvl1_proc.write(out)
            else:
                logger.warning(
                    f"You selected autoencoder type oAE or oVAE but there is no ontology file under: {cfg['FILE_ONT_LVL1']}"
                )

            lvl1_proc.close()

            ## leave lvl2 as provided
            ## add not_in_ontology buckets to lvl2
            if not cfg["FILE_ONT_LVL2"] == None:
                lvl2_file_path = os.path.join(cfg["ROOT_RAW"], cfg["FILE_ONT_LVL2"])
                lvl2_proc_path = os.path.join(
                    "data/processed", cfg["RUN_ID"], cfg["FILE_ONT_LVL2"]
                )
                if os.path.isfile(lvl2_file_path):
                    # if not os.path.isfile(lvl2_proc_path):
                    shutil.copyfile(lvl2_file_path, lvl2_proc_path)

                    ## Add not_ontology bucket to lvl2
                    if len(feature_list) > 0 and cfg["KEEP_NOT_ONT"]:
                        with open(lvl2_proc_path, "a") as lvl2_file:
                            lvl2_file.write(
                                "not_in_ontology_"
                                + data_type
                                + cfg["DELIM"]
                                + "not_in_ontology"
                                + "\n"
                            )
                        lvl2_file.close()
                else:
                    logger.warning(
                        f"You specified a second ontology layer under FILE_ONT_LVL2, but there is no ontology file under: {cfg['FILE_ONT_LVL2']}"
                    )
                    logger.warning(
                        "A second ontology layer is optional. Set FILE_ONT_LVL2 to null to unselect the second layer."
                    )

            logger.warning("Dropping features not in ontology.")
            if not cfg["KEEP_NOT_ONT"]:
                df = df.drop(columns=feature_list)

        # Feature Filtering
        logger.info(f"Filter data: {data_type}")
        match cfg["DATA_TYPE"][data_type]["FILTERING"]:
            case "Var+Corr":
                logger.info(f"Filter by variance and correlation")
                df = filter_data(df, cfg["K_FILTER"], method="NonZeroVar")
                df = filter_data(df, cfg["K_FILTER"] * 10, method="NormalVar")
                df = filter_data(df, cfg["K_FILTER"], method="CorrKMedoids", seed=cfg["GLOBAL_SEED"])

            case "NoFilt":
                logger.info("No filtering of features selected")
            case "Var":
                logger.info(f"Filter by variance and exclude with no variance")
                df = filter_data(df, cfg["K_FILTER"], method="NonZeroVar")
                df = filter_data(df, cfg["K_FILTER"], method="NormalVar")
            case "MAD":
                logger.info(
                    f"Filter features by median absolute deviation and exclude with no variance"
                )
                df = filter_data(df, cfg["K_FILTER"], method="NonZeroVar")
                df = filter_data(df, cfg["K_FILTER"], method="MedAbsDev")
            case "NonZeroVar":
                logger.info(f"Filter features with no variance")
                df = filter_data(df, cfg["K_FILTER"], method="NonZeroVar")
            case "Corr":
                logger.info(f"Filter features with no variance before filtering by correlation")
                df = filter_data(df, cfg["K_FILTER"], method="NonZeroVar")
                logger.info(f"Filter by correlation")
                df = filter_data(df, cfg["K_FILTER"], method="CorrKMedoids", seed=cfg["GLOBAL_SEED"])
            case _:
                raise ValueError(
                    "A filtering method was selected which is not supported: "
                    + cfg["DATA_TYPE"][data_type]["FILTERING"]
                )

        # Normalization
        logger.info(f"Normalize data: {data_type}")

        if cfg["DATA_TYPE"][data_type]["TYPE"] == "NUMERIC":
            logger.info(
                "Normalize features by " + cfg["DATA_TYPE"][data_type]["SCALING"]
            )
            df = normalize_data(
                df, cfg=cfg, method=cfg["DATA_TYPE"][data_type]["SCALING"]
            )

        # Encoding
        if cfg["DATA_TYPE"][data_type]["TYPE"] == "MIXED":
            logger.info(f"Encode and normalize clinical data: {data_type}")
            df = encode_clin(
                df,
                cfg=cfg,
                relevant_clin_num=cfg["DATA_TYPE"][data_type]["REL_CLIN_N"],
                relevant_clin_cat=cfg["DATA_TYPE"][data_type]["REL_CLIN_C"],
                method="OneHot",
                scaler_numeric=cfg["DATA_TYPE"][data_type]["SCALING"],
            )
            logger.info(
                f"Number of clinical features after encoding: {len(df.columns)}"
            )

        # Saving
        logger.info(f"Save data: {data_type}")
        logger.info(f"Shape of {data_type} data after cleaning: {df.shape}")
        df.add_prefix(data_type + "_").to_parquet(
            os.path.join(
                "data/processed",
                cfg["RUN_ID"],
                # data_type+"_data.txt"
                data_type + "_data.parquet",
            )
        )

    return df


def filter_data(df, k, method="MedAbsDev", seed=None):
    """
    Function to filter features data frame with n samples (rows) and p features
    (columns) such that most relevant k<=p features are kept.
    ARGS:
        df (pd.DataFrame): data of any type samples (rows), features (columns)
        k (int): number of features to select
        method (str): method to filter features. "MedAbsDev" does a variance
        based selection by selecting features with the highest median absolute
        deviance. "CorrKMedoids" performs correlation based feature selection
        using KMedoids and correlation as distance metric to obtain
        uncorrelated features (cluster centroids).
    RETURNS:
        df_filt (pd.DataFrame): data frame with n samples and k filtered features.

    """
    if k < len(df.columns):
        if method == "NonZeroVar":
            var = pd.Series(np.var(df, axis=0), index=df.columns, name="Var")
            df_filt = df[var[var > 0].index]

        if method == "NormalVar":
            var = pd.Series(np.var(df, axis=0), index=df.columns, name="Var")

            df_filt = df[var.sort_values(ascending=False).index[0:k]]
        if method == "MedAbsDev":
            mads = pd.Series(
                median_abs_deviation(df, axis=0), index=df.columns, name="MAD"
            )

            df_filt = df[mads.sort_values(ascending=False).index[0:k]]

        if method == "CorrKMedoids":
            clara_filt = CLARA(n_clusters=k,metric="correlation" , random_state=seed).fit(df.transpose())
            df_filt = df.iloc[:, clara_filt.medoid_indices_]
            # kmedoids_filt = kmedoids.KMedoids(
            #     n_clusters=k, metric="correlation", metric_params=[], method="fasterpam"
            # ).fit(df.transpose())
            # df_filt = df.iloc[:, kmedoids_filt.medoid_indices_]
    else:
        if method == "NonZeroVar":
            var = pd.Series(np.var(df, axis=0), index=df.columns, name="Var")
            df_filt = df[var[var > 0].index]
        else:
            df_filt = df

    return df_filt


def normalize_data(df, cfg, method="MinMax"):
    """
    Function to normalize features data frame with n samples (rows) and
    p features (columns) via a specified method (default MinMax Scaler).
    ARGS:
        df (pd.DataFrame): data of any type samples (rows), features (columns)
        method (str): method to normalize features. Options are Scaler from
        sklearn "MinMax", "Standard", "Robust" and "MaxAbs".
    RETURNS:
        df_scaled (pd.DataFrame): data frame with n samples and p normalized
        features.

    """
    logger = getlogger(cfg)
    match method:
        case "MinMax":
            scaler = MinMaxScaler(clip=True)
        case "Standard":
            scaler = StandardScaler()
        case "Robust":
            scaler = RobustScaler()
        case "MaxAbs":
            scaler = MaxAbsScaler()
        case "NoScaler":
            logger.info("No Scaler used, returning df as it is")
            return df
        case _:
            scaler = MinMaxScaler(clip=True)
            logger.warning("The given Scaler is not implemented yet: " + method)
            logger.info("Used MinMax instead")

    df_scaled = pd.DataFrame(
        scaler.fit_transform(df), columns=df.columns, index=df.index
    )

    return df_scaled


def encode_clin(
    df,
    cfg,
    relevant_clin_num=[],
    relevant_clin_cat=[],
    method="OneHot",
    scaler_numeric="MinMax",
):
    """
    Function to select and encode clinical features with n samples (rows) and p
    features (columns) via a specified method (default OneHot) on indicated
    feature list (relevant_clin_cat).
    Via relevant_clin_num numeric clinical features are selected and scaled by
    the indicated method (scaler_numeric).
    ARGS:
        df (pd.DataFrame): data of any type samples (rows), features (columns)
        relevant_clin_num (list): list of features (columns) with numerical
                                  features, which should be kept and scaled.
        relevant_clin_cat (list): list of features (columns) with categorical
                                  features, which should be kept and encoded
        method (str): method to encode categorical features. Default is "OneHot".
        scaler_numeric (str): method to scaler numeric features. See options
                              as in normalize_data().

    RETURNS:
        df_enc_num (pd.DataFrame): data frame with selected numeric features
        and selected + encoded categorical features.

    """
    logger = getlogger(cfg)

    if len(relevant_clin_num) > 0:
        df_num = normalize_data(
            df.loc[:, relevant_clin_num], cfg=cfg, method=scaler_numeric
        )
    else:
        df_num = pd.DataFrame()

    if len(relevant_clin_cat) > 0:
        if method == "OneHot":
            df_enc = pd.get_dummies(
                df.loc[:, relevant_clin_cat], columns=relevant_clin_cat
            )
        else:
            logger.warning("Sorry, currently only OneHot encoding available.")
            df_enc = pd.DataFrame()
    else:
        df_enc = pd.DataFrame()

    df_enc_num = pd.concat([df_num, df_enc], axis=1)
    return df_enc_num


def synth_signal(df, feature_list, sample_list):
    max_q90 = df.max().max() * 0.9
    df.loc[sample_list, feature_list] = max_q90 + np.random.normal(
        loc=0, scale=max_q90 * 0.05
    )

    return df


@click.command()
@click.argument("run_id", type=str)
def main(run_id):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed). Includes
    filtering, normalizing and encoding features.

    """
    cfg = get_cfg(run_id)
    logger = getlogger(cfg)
    logger.info("Unify sample ID list from data files")
    sample_list = unified_sample_list(cfg)
    logger.info(f"Unified sample ID list has length {len(sample_list)}")

    if cfg["MODEL_TYPE"] == "ontix":
        lvl1_proc_path = os.path.join(
            "data/processed", cfg["RUN_ID"], cfg["FILE_ONT_LVL1"]
        )

        if os.path.isfile(lvl1_proc_path):
            os.remove(
                lvl1_proc_path
            )  ## Remove possible file from previous run to avoid further appending

    for data_type in cfg["DATA_TYPE"]:
        if cfg["DATA_TYPE"][data_type]["TYPE"] == "IMG":
            raw_path = os.path.join(
                cfg["ROOT_RAW"], cfg["DATA_TYPE"][data_type]["FILE_RAW"]
            )
            logger.debug(f"root raw image: {raw_path}")
            image_mappings = pd.read_csv(raw_path, index_col=0, sep="\t")
            parquet_file = os.path.join(
                "data/processed",
                cfg["RUN_ID"],
                data_type + "_data.parquet",
            )
            image_mappings.to_parquet(parquet_file)

            logger.debug(f"skipping make_data single {data_type}")
            continue
        logger.info(f"Make data set {data_type}")

        make_data_single(cfg, data_type, sample_list)


if __name__ == "__main__":
    main()
