import click
from pathlib import Path
import os
import pandas as pd
from src.utils.utils_basic import getlogger
from src.utils.config import get_cfg


def single_format(filename, cfg):
    """
    Function formats TCGA data files (see cases for supported files) to a unified data format with rows (samples) and columns (features)
    For special case of clinical and sample data there is another function format_clin which uses two files and joins them.
    ARGS:
        filename (str): String of the TCGA file
        cfg (dict): Dictionary of configuration parameters
    RETURNS:
        df (pd.DataFrame): Output dataframe after formatting

    """
    logger = getlogger(cfg)
    match filename:
        case "data_mrna_seq_v2_rsem.txt":
            index_cols = ["Entrez_Gene_Id"]
            dytpes = {"Entrez_Gene_Id": str}
            drop_cols = ["Hugo_Symbol"]
            skips = 0
            head_row = 0
            delimiter = "\t"

        case "data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt":
            index_cols = ["Entrez_Gene_Id"]
            dytpes = {"Entrez_Gene_Id": str}
            drop_cols = ["Hugo_Symbol"]
            skips = 0
            head_row = 0
            delimiter = "\t"

        case "data_mrna_seq_v2_rsem_zscores_ref_normal_samples.txt":
            index_cols = ["Entrez_Gene_Id"]
            dytpes = {"Entrez_Gene_Id": str}
            drop_cols = ["Hugo_Symbol"]
            skips = 0
            head_row = 0
            delimiter = "\t"

        case "data_log2_cna.txt":
            index_cols = ["Entrez_Gene_Id"]
            dytpes = {"Entrez_Gene_Id": str}
            drop_cols = ["Hugo_Symbol", "Cytoband"]
            skips = 0
            head_row = 0
            delimiter = "\t"

        case "data_methylation_hm27_hm450_merged.txt":
            index_cols = ["ENTITY_STABLE_ID"]
            dytpes = {"ENTITY_STABLE_ID": str}
            drop_cols = ["NAME", "DESCRIPTION", "TRANSCRIPT_ID"]
            skips = 0
            head_row = 0
            delimiter = "\t"

        case "data_methylation_per_gene.txt":
            index_cols = ["Entrez_Gene_Id"]
            dytpes = {"Entrez_Gene_Id": str}
            drop_cols = []
            skips = 0
            head_row = 0
            delimiter = "\t"

        case "data_clinical_sample.txt":
            index_cols = ["PATIENT_ID"]
            dytpes = {"PATIENT_ID": str}
            drop_cols = []
            skips = 3
            head_row = 1
            delimiter = "\t"

        case "data_clinical_patient.txt":
            index_cols = ["PATIENT_ID"]
            dytpes = {"PATIENT_ID": str}
            drop_cols = []
            skips = 3
            head_row = 1
            delimiter = "\t"

        case "data_mutations.txt":
            index_cols = []
            dytpes = {
                "Hugo_Symbol": str,
                "Entrez_Gene_Id": str,
                "PolyPhen": str,
                "Tumor_Sample_Barcode": str,
            }
            drop_cols = []
            skips = 0
            head_row = 1
            delimiter = "\t"

        case _:
            raise ValueError(f"Data type {filename} not supported.")

    df = pd.read_csv(
        os.path.join(cfg["ROOT_RAW"], filename),
        delimiter=delimiter,
        index_col=index_cols,
        dtype=dytpes,
        skiprows=skips,
        header=head_row,
    ).drop(columns=drop_cols, errors="ignore")
    df = df.loc[~df.index.isnull(), :]  # Filter Genes without Entrez ID
    df = df.T.dropna(axis=1)

    logger.info(f"Save formatted {filename}")
    # df.to_csv(os.path.join(cfg['ROOT_RAW'], f"{filename[0:-4]}{cfg['FORM_SUFFIX']}"),index=True, sep=cfg['DELIM'])
    df.to_parquet(
        os.path.join(cfg["ROOT_RAW"], f"{filename[0:-4]}{cfg['FORM_SUFFIX']}"),
        index=True,
    )

    return df


def format_clin(file_patient, file_sample, cfg):
    """
    Function formats TCGA data files with clinical data related to patients (file_patient) and samples (file_sample). It joins both datasets and formats in common format.
    ARGS:
        file_patient (str): String of the TCGA file for patient data
        file_sample (str): String of the TCGA file for sample data
        cfg (dict): Dictionary of configuration parameters
    RETURNS:
        df (pd.DataFrame): Output dataframe after formatting

    """
    logger = getlogger(cfg)

    df_clin_sample = pd.read_csv(cfg["ROOT_RAW"] + "/" + file_sample, delimiter="\t")
    df_clin_patient = pd.read_csv(cfg["ROOT_RAW"] + "/" + file_patient, delimiter="\t")

    df = df_clin_sample.merge(
        df_clin_patient, left_on="PATIENT_ID", right_on="PATIENT_ID"
    )
    df = df.set_index("SAMPLE_ID")

    # df = df.loc[sample_list,]

    logger.info('Fill in missing entries as "unknown" in str columns')
    for col in df:
        dt = df[col].dtype

        if dt == object or dt == str:
            logger.info("Column:" + col)
            df[col].fillna("unknown", inplace=True)

    ## Manual adjustments of Staging
    df.replace(
        to_replace=["STAGE X", "STAGE IS", "STAGE I/II (NOS)"],
        value="unknown",
        inplace=True,
    )
    ## Reduce Staging
    df["AJCC_PATHOLOGIC_TUMOR_STAGE_SHORT"] = df[
        "AJCC_PATHOLOGIC_TUMOR_STAGE"
    ].str.replace("[ABC]$", "", regex=True)

    # df.to_csv(os.path.join(cfg['ROOT_RAW'], f"data_clinical{cfg['FORM_SUFFIX']}"),
    #           index=True, sep=cfg['DELIM'])
    df.to_parquet(
        os.path.join(cfg["ROOT_RAW"], f"data_clinical{cfg['FORM_SUFFIX']}"), index=True
    )

    return df


def format_mut(file_mutation, cfg):
    """
    Turns MAF style mutation data into a matrix of patients in rows and mutated genes in columns.
    ARGS:
        file_mutation (str): String of the TCGA data_mutations.txt file (in MAF format)
        cfg (dict): Dictionary of configuration parameters
    RETURNS:
        df (pd.DataFrame): Output dataframe after formatting
    """
    logger = getlogger(cfg)
    # read the joined mutation file
    df_maf = pd.read_table(cfg["ROOT_RAW"] + "/" + file_mutation, delimiter="\t")

    df_maf["Tumor_Sample_Barcode"] = df_maf["Tumor_Sample_Barcode"].astype(str)
    df_maf["Entrez_Gene_Id"] = df_maf["Entrez_Gene_Id"].astype(str)

    logger.info(
        "Check for missing Entrez Gene Id's and discard them (including Entrez_Gene_Id=0)"
    )
    # TODO: Maybe remove this because it might not play nice with downstream analysis (e.g. PI Score)
    # if df_maf["Entrez_Gene_Id"].isna().any():
    #     # PALM2 and AKAP2 are missing Entrez IDs so add it manually
    #     df_maf["Entrez_Gene_Id"] = df_maf.apply(
    #         lambda row: "445815" if (row["Hugo_Symbol"] == "AKAP2" or row["Hugo_Symbol"] == "PALM2") else row["Entrez_Gene_Id"], axis=1)

    # drop rows with missing Entrez_Gene_Id (the above step assigned a gene ID for AKAP2 and PALM2)
    df_maf.dropna(subset=["Entrez_Gene_Id"], inplace=True)

    # this groups MAF file by patients and counts the number of times each gene appears for a patient (number of
    # times gene has been mutated for the patient). This "stacked" pandas.Series is then unpacked to a DataFrame
    df = (
        df_maf.groupby("Tumor_Sample_Barcode")["Entrez_Gene_Id"]
        .value_counts()
        .unstack(fill_value=0)
    )

    # drop the column containing 0.0 Entrez_Gene_Id and "nan"
    df.drop(columns=["0.0", "nan"], inplace=True)

    # make sure that columns are strings for parquet format
    df.columns = df.columns.astype(float).astype(int).astype(str)

    # write the file
    # df.to_csv(os.path.join(cfg['ROOT_RAW'], f"data_mutations{cfg['FORM_SUFFIX']}"),
    #           index=True, sep=cfg['DELIM'])
    df.to_parquet(
        os.path.join(cfg["ROOT_RAW"], f"data_mutations{cfg['FORM_SUFFIX']}"), index=True
    )

    return df


@click.command()
# @click.argument("input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())
@click.argument("run_id", type=str)
def main(run_id):
    """Formats TCGA data to the format rows: samples and columns: features; first row with unique feature labels, first column unique sample ID"""

    # cfg_tcga = get_tcga_cfg(run_id)
    cfg = get_cfg(run_id)

    logger = getlogger(cfg)
    logger.info("Format TCGA Files")

    file_sample = ""
    file_patient = ""
    for filename in cfg["DATAFILES"]:
        logger.info(f"Formatting: {filename}")
        if not "data_clinical" in filename:

            if "data_mutations" in filename:  # add this if for mutation file
                logger.info(f"Formatting mutation file: {filename}")
                format_mut(file_mutation=filename, cfg=cfg)

            else:
                single_format(filename, cfg)

        else:
            if "patient" in filename:
                file_patient = filename
            if "sample" in filename:
                file_sample = filename

    if len(file_patient) > 0 and len(file_sample) > 0:
        logger.info(f"Formatting: {file_patient} and {file_sample}")
        format_clin(file_patient=file_patient, file_sample=file_sample, cfg=cfg)


if __name__ == "__main__":
    main()
