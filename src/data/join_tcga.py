import click
import os

# import ray
# ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}})
# import modin.pandas as mpd
import pandas as pd
import glob
from src.utils.utils_basic import getlogger
from src.utils.config import get_cfg


def single_type(filename, cfg):
    logger = getlogger(cfg)
    df = pd.DataFrame()
    if cfg["SUBTYPES"] == "ALL":
        file_regex = cfg["ROOT_RAW"] + "/*" + cfg["TCGA_FOLDER"] + filename
        file_list = glob.glob(file_regex)
    else:
        file_list = [
            cfg["ROOT_RAW"] + "/" + subtype + cfg["TCGA_FOLDER"] + filename
            for subtype in cfg["SUBTYPES"]
        ]

    match filename:
        case "data_mrna_seq_v2_rsem.txt":
            index_cols = ["Entrez_Gene_Id"]
            dytpes = {"Entrez_Gene_Id": str}
            drop_cols = ["Hugo_Symbol"]
            usecols = None
            skips = 0
            head_row = 0
            ax = 1

        case "data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt":
            index_cols = ["Entrez_Gene_Id"]
            dytpes = {"Entrez_Gene_Id": str}
            drop_cols = ["Hugo_Symbol"]
            usecols = None
            skips = 0
            head_row = 0
            ax = 1

        case "data_mrna_seq_v2_rsem_zscores_ref_normal_samples.txt":
            index_cols = ["Entrez_Gene_Id"]
            dytpes = {"Entrez_Gene_Id": str}
            drop_cols = ["Hugo_Symbol"]
            usecols = None
            skips = 0
            head_row = 0
            ax = 1

        case "data_log2_cna.txt":
            index_cols = ["Entrez_Gene_Id"]
            dytpes = {"Entrez_Gene_Id": str}
            drop_cols = ["Cytoband", "Hugo_Symbol"]
            usecols = None
            skips = 0
            head_row = 0
            ax = 1

        case "data_methylation_hm27_hm450_merged.txt":
            index_cols = ["ENTITY_STABLE_ID"]
            dytpes = {"ENTITY_STABLE_ID": str}
            drop_cols = ["NAME", "DESCRIPTION", "TRANSCRIPT_ID"]
            usecols = None
            skips = 0
            head_row = 0
            ax = 1

        case "data_methylation_per_gene.txt":
            index_cols = ["Entrez_Gene_Id"]
            dytpes = {"Entrez_Gene_Id": str}
            drop_cols = []
            usecols = None
            skips = 0
            head_row = 0
            ax = 1

        case "data_clinical_sample.txt":
            index_cols = ["PATIENT_ID"]
            dytpes = {"PATIENT_ID": str}
            drop_cols = []
            usecols = None
            skips = 3
            head_row = 1
            ax = 0

        case "data_clinical_patient.txt":
            index_cols = ["PATIENT_ID"]
            dytpes = {"PATIENT_ID": str}
            drop_cols = []
            usecols = None
            skips = 3
            head_row = 1
            ax = 0

        case "data_mutations.txt":
            index_cols = []
            dytpes = {"Entrez_Gene_Id": str}
            drop_cols = []
            usecols = [
                "Hugo_Symbol",
                "Entrez_Gene_Id",
                "PolyPhen",
                "Tumor_Sample_Barcode",
            ]
            skips = 0
            head_row = 0
            ax = 0

        case _:
            raise ValueError(f"Data type {filename} not supported.")

    for file_temp in file_list:
        logger.info(file_temp)
        df_temp = pd.read_csv(
            file_temp,
            delimiter="\t",
            index_col=index_cols,
            usecols=usecols,
            header=head_row,
            skiprows=skips,
            dtype=dytpes,
        ).drop(columns=drop_cols, errors="ignore")

        df_temp = df_temp.loc[
            ~df_temp.index.duplicated(),
        ]

        # df = mpd.concat([df, df_temp], axis=ax)
        df = pd.concat([df, df_temp], axis=ax)

    logger.info(f"Save joined {filename}")
    df.to_csv(os.path.join(cfg["ROOT_RAW"], filename), index=True, sep=cfg["DELIM"])

    return df


def make_meth_per_gene(cfg):
    filename = "data_methylation_hm27_hm450_merged.txt"

    if cfg["SUBTYPES"] == "ALL":
        file_regex = cfg["ROOT_RAW"] + "/*" + cfg["TCGA_FOLDER"] + filename
        file_list = glob.glob(file_regex)
    else:
        file_list = [
            cfg["ROOT_RAW"] + "/" + subtype + cfg["TCGA_FOLDER"] + filename
            for subtype in cfg["SUBTYPES"]
        ]

    for file_temp in file_list:

        file_tcga_unformatted_rna = (
            file_temp[: -len(filename)] + "data_mrna_seq_v2_rsem.txt"
        )

        df_meth_tcga = pd.read_csv(file_temp, delimiter="\t")
        df_rna_tcga = pd.read_csv(
            file_tcga_unformatted_rna, delimiter="\t", dtype={"Entrez_Gene_Id": str}
        )

        df_meth_tcga = df_meth_tcga.merge(
            df_rna_tcga[["Hugo_Symbol", "Entrez_Gene_Id"]],
            left_on="NAME",
            right_on="Hugo_Symbol",
        ).drop(["ENTITY_STABLE_ID", "NAME", "DESCRIPTION", "TRANSCRIPT_ID"], axis=1)

        df_meth_tcga = df_meth_tcga.groupby(["Entrez_Gene_Id"]).mean(numeric_only=True)

        df_meth_tcga.to_csv(
            file_temp[: -len(filename)] + "data_methylation_per_gene.txt",
            sep="\t",
            header=True,
            index=True,
        )

    return df_meth_tcga


@click.command()
# @click.argument("input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())
@click.argument("run_id", type=str)
def main(run_id):
    """Joins individual TCGA studies for each tumour class to a single data set"""

    # cfg_tcga = get_tcga_cfg(run_id)
    cfg = get_cfg(run_id)
    logger = getlogger(cfg)

    if "data_methylation_per_gene.txt" in cfg["DATAFILES"]:
        logger.info(
            "Calculate and create data_methylation_per_gene.txt for all cancer types."
        )
        make_meth_per_gene(cfg)

    logger.info("Join TCGA Files")
    for filename in cfg["DATAFILES"]:
        logger.info(f"Joining: {filename}")
        single_type(filename, cfg)


if __name__ == "__main__":
    main()
