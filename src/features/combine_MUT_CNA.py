import click
import os
import pandas as pd
from src.utils.utils_basic import getlogger
from src.utils.config import get_cfg


def fetch_and_format_cds_lengths(cfg):
    """
    Loads CDS data from Ensembl and matches Hugo_Symbols to Entrez_IDs to allow querying the CDS length by EntrezID

    ARGS:
        cfg (dict): Dictionary of configuration parameters
        discard_benign (bool): Discard benign mutations, default True
    RETURNS:
        data_mutations (pd.DataFrame): Output dataframe containing a mutation matrix with colnames as gene names
    """
    # Load the data_mutations.txt and keep only unique Hugo_Symbol-Entrez_Gene_Id pairs.
    data_mutations_unformatted = pd.read_csv(
        os.path.join(cfg["ROOT_RAW"], "data_mutations.txt"), delimiter="\t"
    )

    # keep only two relevant columns
    hugo_to_entrez = data_mutations_unformatted[["Hugo_Symbol", "Entrez_Gene_Id"]]

    # drop NA's, EntrezGeneID=0 entries and duplicates
    hugo_to_entrez = hugo_to_entrez.dropna()
    hugo_to_entrez = hugo_to_entrez.drop(
        hugo_to_entrez[hugo_to_entrez["Entrez_Gene_Id"] == 0].index
    )
    hugo_to_entrez.drop_duplicates(inplace=True)

    # change to string to match the columns
    hugo_to_entrez["Entrez_Gene_Id"] = (
        hugo_to_entrez["Entrez_Gene_Id"].astype(int).astype(str)
    )

    # Create a dictionary to map gene_ID to gene_name
    hugo_to_entrez_dict = pd.Series(
        hugo_to_entrez.Entrez_Gene_Id.values, index=hugo_to_entrez.Hugo_Symbol
    ).to_dict()

    gene_lengths = get_all_gene_lengths(cds_lengths="avg")

    # these are the genes for which we are missing the CDS length data
    genes_with_noCDS = hugo_to_entrez["Entrez_Gene_Id"][
        ~hugo_to_entrez["Hugo_Symbol"].isin(gene_lengths.index)
    ]

    miss_cds = len(genes_with_noCDS)
    total_genes = len(hugo_to_entrez["Hugo_Symbol"])
    percentage = round((miss_cds / total_genes) * 100, 2)

    print(f"Missing CDS data for {miss_cds}/{total_genes} ({percentage}%) genes.")
    print("These genes will be discarded from the mutation matrix.")

    gene_lengths_Entrez = gene_lengths[
        gene_lengths.index.isin(hugo_to_entrez["Hugo_Symbol"])
    ]
    gene_lengths_Entrez.rename(index=hugo_to_entrez_dict, inplace=True)

    # Check if duplicate entries present in the final gene list and discard if yes:
    if gene_lengths_Entrez.index.duplicated().any():
        # gene_lengths_Entrez = gene_lengths_Entrez.drop_duplicates() ## TODO drops duplicated gene lengths not indexes!
        gene_lengths_Entrez = gene_lengths_Entrez.loc[
            ~gene_lengths_Entrez.index.duplicated(keep="first")
        ]

    return gene_lengths_Entrez, genes_with_noCDS


def get_all_gene_lengths(cds_lengths="avg"):
    """
    Loads CDS data which was previously manually queried from Ensembl/Biomart.

    ARGS:
        cds_lengths (str): Specifies whether to take the average (of all available transcripts)
        or maximum CDS length available for a given gene name in Ensembl database.
    RETURNS:
        gene_lengths (pd.DataFrame): Output dataframe containing CDS lengths for calculation of PI scores.
    """
    # Load the CDS data obtained from Ensembl
    # user can specify whether to take an average or max cds length

    if cds_lengths == "avg":
        cds_lengths = pd.read_csv(
            "data/external/cds_length/all_cds_lengths_averaged.csv", index_col=0
        )
    elif cds_lengths == "max":
        cds_lengths = pd.read_csv(
            "data/external/cds_length/all_cds_lengths_max.csv", index_col=0
        )
    else:
        raise ValueError("Invalid argument. Choose 'max' or 'avg'.")

    gene_lengths = cds_lengths.squeeze()

    return gene_lengths


def process_cna(cfg):
    """
    Process CNA data according to the approach used for PI score calculation and discard those
    Entrez_Gene_Id's not present in the mutation file.
    log2 CNA values are transformed to raw values and for CNA < 1, inverse (1/CNA) is used

    ARGS:
        cfg (dict): Dictionary of configuration parameters.
    RETURNS:
        cna (pd.DataFrame): df of raw CNA values with patients per row and
        Entrez_Gene_Id in columns, containing only those Gene_Id's present in data_mutations
    """
    # load and process cna data
    cna_log2 = pd.read_parquet(
        os.path.join(cfg["ROOT_RAW"], "data_log2_cna_formatted.parquet")
    )

    # keep only the columns which are found in the mutation file
    data_mutation_formatted = pd.read_parquet(
        os.path.join(cfg["ROOT_RAW"], "data_mutations_formatted.parquet")
    )

    matching_cols = cna_log2.columns.intersection(data_mutation_formatted.columns)

    cna_log2 = cna_log2[matching_cols]

    # transform the CNA data from log2 values to the raw values by raising them 2^(log2CNA)
    # cna = cna_log2.rpow(2)

    cna = cna_log2.abs()

    # if there are multiple entries for the same gene, take mean value (since now linear)
    columns_with_duplicate_entries = cna.columns[cna.columns.duplicated()]

    if not columns_with_duplicate_entries.empty:
        for duplicate_gene in columns_with_duplicate_entries:
            mean_cna_values = (
                cna[duplicate_gene].groupby(cna[duplicate_gene].columns, axis=1).mean()
            )
            cna.drop(columns=duplicate_gene, inplace=True)
            cna = pd.merge(cna, mean_cna_values, left_index=True, right_index=True)

    # where fold change is < 1 -> make it into a positive value, as a decreased copy number also should increase pathway instability
    ## TODO no keep cna as is
    # cna[cna < 1] = 1 / cna

    return cna


# def calculate_combiscore_MUT_CNA(cfg, only_patients_with_both=True):
def calculate_combiscore_MUT_CNA(
    cfg, only_patients_with_both=False
):  ## TODO Default should be FALSE
    """
    Calculate the combined score for MUT and CNA data, similar to the PI score approach.
    This score is the product of normalized mutation rate per gene per patient
    (no. of mutations divided by CDS gene length) and the raw CNA values for that gene and patient.

    ARGS:
        cfg (dict): Dictionary of configuration parameters
        only_patients_with_both (bool): Choose whether to only consider patients which have CNA data
        in addition to MUT data. Alternatively, we consider also patients with MUT data but no CNA data
        in which case their normalized mutation rates are unchanged; default True
    RETURNS:
        combiscore_MUT_CNA (pd.DataFrame): df with combined MUT_CNA scores in the standard "_formatted"
        format.
    """

    logger = getlogger(cfg)

    logger.info("Fetching CDS length data.")
    gene_lengths, genes_with_noCDS = fetch_and_format_cds_lengths(cfg)

    # load the formatted mutation file
    data_mutations = pd.read_parquet(
        os.path.join(cfg["ROOT_RAW"], "data_mutations_formatted.parquet")
    )

    # remove the genes from the data_mutations_formatted which don't have cds data
    data_mutations.drop(columns=genes_with_noCDS, inplace=True)

    # load the CNA data
    data_CNA = process_cna(cfg)

    # consider only patients who have both mutation data and cna data
    if only_patients_with_both:
        common_indices = data_mutations.index.intersection(data_CNA.index)

        data_mutations = data_mutations.loc[common_indices]
        data_CNA = data_CNA.loc[common_indices]
        logger.info("Patients without CNA data are discarded.")
    logger.info("Dividing by CDS length.")
    data_mutations = data_mutations.div(gene_lengths.div(gene_lengths.mean()))

    logger.info("Add CNA.")

    data_mutations = data_mutations.add(data_CNA)

    logger.info("Combined score calculated, writing to file.")

    data_mutations.to_parquet(
        os.path.join(cfg["ROOT_RAW"], "data_combi_MUT_CNA_formatted.parquet")
    )

    return data_mutations


@click.command()
@click.argument("run_id", type=str)
def main(run_id):
    """
    Processes the mutation and CNA data and returns a matrix of MUT+CIN combined
    TODO: explain what is the feature being calculated
    """
    cfg = get_cfg(run_id)
    logger = getlogger(cfg)

    logger.info("Calculating the combined score for MUT and CNA data.")
    # calculate_combiscore_MUT_CNA(cfg, only_patients_with_both=True)
    calculate_combiscore_MUT_CNA(cfg, only_patients_with_both=True)


if __name__ == "__main__":
    main()
