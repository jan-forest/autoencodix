import click
from pathlib import Path
import os
import pandas as pd
import numpy as np
import warnings
from src.utils.utils_basic import getlogger
from src.utils.config import get_cfg


def prepare_mutations(cfg, discard_benign=True):
    """
    Turns MAF style mutation data into a matrix of patients in rows and mutated genes in columns.

    ARGS:
        cfg (dict): Dictionary of configuration parameters
        discard_benign (bool): Discard benign mutations, default True
    RETURNS:
        data_mutations (pd.DataFrame): Output dataframe containing a mutation matrix with colnames as gene names

    """
    logger = getlogger(cfg)

    # We need to map the colnames of the data_mutations_formatted file from Entrez_IDs to Hugo_Symbols
    # to make it compatible with the original PI Score calculation approach
    # read in the file that contains the matching from hugo symbols to entrezId
    data_mutations_unformatted = pd.read_csv(
        os.path.join(cfg["ROOT_RAW"], "data_mutations.txt"), delimiter="\t"
    )

    # if we wish to discard benign mutations which are per default
    # included in the data_mutations_formatted.parquet, we simply start from the data_mutations.txt
    # and build a new data_mutations_formatted dataframe for this purpose
    # (since the user might want to keep all the mutation data)
    if discard_benign:

        data_mutations_unformatted["Tumor_Sample_Barcode"] = data_mutations_unformatted[
            "Tumor_Sample_Barcode"
        ].astype(str)

        logger.info(
            "Discarding benign mutations and making a new mutation table for PI scores calculation."
        )

        # drop rows with missing Hugo_Symbol
        data_mutations_unformatted.dropna(subset=["Hugo_Symbol"], inplace=True)

        # PolyPhen classification contains probability next to the verdict, so I use regex to remove probability numbers for filtering
        data_mutations_unformatted["PolyPhen"] = data_mutations_unformatted[
            "PolyPhen"
        ].str.replace(r"\(.*?\)", "", regex=True)
        # remove benign variants
        data_mutations_unformatted = data_mutations_unformatted[
            data_mutations_unformatted["PolyPhen"] != "benign"
        ]

        # this groups MAF file by patients and counts the number of times each gene appears for a patient (number of
        # times gene has been mutated for the patient). This "stacked" pandas.Series is then unpacked to a DataFrame
        data_mutations = (
            data_mutations_unformatted.groupby("Tumor_Sample_Barcode")["Hugo_Symbol"]
            .value_counts()
            .unstack(fill_value=0)
        )

        return data_mutations

    else:  # else just rename the columns

        # load the formatted mutation file
        data_mutations = pd.read_parquet(
            os.path.join(cfg["ROOT_RAW"], "data_mutations_formatted.parquet")
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

        # These are the columns which don't have hugo_symbol in the data_mutations.txt file
        columns_to_drop = data_mutations.columns[
            ~data_mutations.columns.isin(hugo_to_entrez["Entrez_Gene_Id"])
        ]
        # We need to drop these
        data_mutations.drop(columns=columns_to_drop, inplace=True)

        # Create a dictionary to map gene_ID to gene_name
        hugo_to_entrez_dict = pd.Series(
            hugo_to_entrez.Hugo_Symbol.values, index=hugo_to_entrez.Entrez_Gene_Id
        ).to_dict()

        # Rename the columns in the second dataframe
        data_mutations.rename(columns=hugo_to_entrez_dict, inplace=True)

        return data_mutations


def load_reactome_pathways_gene_set():
    """
    Loads the reactome pathways gene set (file outlining which genes make up which pathway).

    RETURNS:
        reactome (pd.DataFrame): Dataframe with pathway names in col[0] pathway_ids in col[1]
        and constituent genes in other columns.

    """

    # Downloaded from Reactome (TODO: add link)
    reactome = pd.read_csv(
        "data/external/Reactome/ReactomePathways.gmt",
        header=None,
        names=[
            i for i in range(0, 2607)
        ],  # this is how many columns there are in the file
        sep="\t|,",
        engine="python",
    )
    return reactome


def get_reactome_pathways_IDs_names():
    """
    Returns a pd.Series object containing the names of reactome pathways

    RETURNS:
        pathway_IDs_names (pd.Series): pd.Series of all reactome pathway names.

    """
    reactome = load_reactome_pathways_gene_set()
    reactome = _clean_reactome_pathway_gene_set(reactome)
    # set pathway IDs as index
    reactome.set_index([1], inplace=True)
    # create a series of the descriptive names and their associated reactome IDs as indices
    pathway_IDs_names = pd.Series(reactome.iloc[:, 0], index=reactome.index)

    return pathway_IDs_names


def get_reactome_number_of_pathway_entities(reactome=None):
    """
    Returns the number of constituent genes in a pathway. This is used for normalizing during PI score calculation.

    ARGS:
        reactome (pd.DataFrame): df with pathway_IDs as indices and gene_names as colnames, with binary
        cell values marking whether a gene is a member of the pathway (1) or not (0).
    RETURNS:
        nr_of_entities (pd.Series): Series with a list of member genes for the provided pathway gene set dataframe.
    """
    if reactome is None:
        reactome = get_transformed_reactome_pathways_gene_set()

    nr_of_entities = reactome.astype(int).sum(axis=1)

    return nr_of_entities


def get_transformed_reactome_pathways_gene_set(reactome=None):
    """
    Returns a properly formatted data frame with pathway_IDs as indices and gene_names as colnames, with binary
        cell values marking whether a gene is a member of the pathway (1) or not (0).

    ARGS:
        reactome (pd.DataFrame): df in the ame format as default Reactome pathway gene set (pathway names, pathway_ids
        and member genes of the pathway names in separate columns). Default None; if None, standard Reactome pathway gene set is loaded.
    RETURNS:
        cleaned_transformed_reactome (pd.DataFrame): df with pathway_IDs as indices and gene_names as colnames, with binary
        cell values marking whether a gene is a member of the pathway (1) or not (0)

    """
    if reactome is None:
        reactome = load_reactome_pathways_gene_set()

    cleaned_reactome = _clean_reactome_pathway_gene_set(reactome)
    cleaned_transformed_reactome = _transform_reactome_pathway_gene_set(
        cleaned_reactome
    )

    return cleaned_transformed_reactome


def _clean_reactome_pathway_gene_set(df):
    """
    Makes sure that the first column contains the pathway descriptions, the second the pathway IDs and
    the rest gene names that participate in that pathway

    ARGS:
        df (pd.DataFrame): unprocessed standard reactome pathway gene set table
    RETURNS:
        df (pd:DataFrame): properly formatted DataFrame (witn NAs, etc excluded),
        first column contains the pathway descriptions, the second the pathway IDs and the rest participating gene names
    """

    # deletes a specified cell in a row from a DataFrame, shifts cells to the left and adds a None to the end
    def shifter(row, column_to_delete):
        # deletes the specified column of the inputted row, shifts values to the left, adds None to the end of the row
        return np.hstack((np.delete(np.array(row), [column_to_delete]), [None]))

    # two rows have unexplained NaN values in column 2 -> need to fix this before doing the next steps
    mask = (df.iloc[:, 2].str.contains("R-HSA")).isna()
    df.loc[mask, :] = df.loc[mask, :].apply(
        shifter, column_to_delete=2, axis=1, result_type="expand"
    )

    # --------  Some of the pathway descriptions are fragmented into multiple cells
    # some pathway IDs are therefore shifted from column 2 more to the right - fixing it here ---------

    # selecting the rows where the pathway ID is not in the second column (where it is supposed to be)
    df_ID_is_shifted = df[~(df.iloc[:, 1].str.contains("R-HSA"))].copy()

    # concatenating the pathway descriptions, and assigning these to column 0
    # collecting the pathway IDs and assigning them to column 1
    for index in df_ID_is_shifted.index:
        row = df_ID_is_shifted.loc[index, :].dropna()
        # find the column in which the "R-HSA-..." ID is in
        columnID = row[row.str.contains("R-HSA-")].index[0]
        # concatenate all the strings in the columns to the left of columnID and assign it into column0
        description = ""
        for i, column in enumerate(row[0:columnID]):
            if i != 0:
                description += ", "  # in file-read-in, commas were used as separator
            description += column

        # put concatenated pathway description to column 0
        df_ID_is_shifted.loc[index, 0] = description
        # put pathway ID into column 1
        df_ID_is_shifted.loc[index, 1] = row[columnID]

    # -------- tidying up, by removing from columns 2 and 3 the leftover description framgenets/pathway IDs

    # delete the column[2] of the dataframe
    df_ID_is_shifted = df_ID_is_shifted.apply(
        shifter, column_to_delete=2, axis=1, result_type="expand"
    )
    # find rows of dataframe which have the pathway ID in their 2nd column
    # (because of the previous step everything is shifted 1 to the left, so column[3] became column[2])
    mask = df_ID_is_shifted.loc[:, 2].str.contains("R-HSA-")
    # where mask is None, set it to False instead, no need to do there any changes
    mask[mask.isna()] = False
    df_ID_is_shifted.loc[mask, :] = df_ID_is_shifted.loc[mask, :].apply(
        shifter, column_to_delete=2, axis=1, result_type="expand"
    )
    # assigning back to the original dataframe the cleaned up one
    df.loc[df_ID_is_shifted.index, :] = df_ID_is_shifted.values

    return df


def _drop_meaningless_columns(df):
    """
    Drops not useful columns of the transformed reactome pathway gene set dataframe.

    ARGS:
        df (pd.DataFrame): df with pathway IDs in rows, gene symbols in columns
    RETURNS:
        df (pd.DataFrame): df with meaningless columns (defined in functions body) dropped
    """

    columns_to_drop = [
        "FKBP14 gene",
        "CSF2 gene",
        "45S pre-rRNA gene",
        "MAPK6 gene",
        # entries for all these genes are present in the dataframe, but without the " gene" appendix -> can be safely removed
        " 28S rRNA",
        " 5.8S rRNA",
        " complete genome",
        "18S rRNA",
        "1a",
        "28S rRNA",
        "3a",
        "5.8S rRNA",
        "5S rRNA",
        "7SL RNA (ENSG00000222619)",
        "7SL RNA (ENSG00000222639)",
        "7a",
        "8b",
        "9b",
    ]

    return df.drop(columns_to_drop, axis=1)


def _transform_reactome_pathway_gene_set(gene_set_matrix):
    """
    Transforms the original reactome pathway gene set to a dataframe with pathway_IDs as indices and
    gene_names as colnames and cells are binary coded marking whether the gene is a member of
    the given pathway (value 1) or not (value 0)

    ARGS:
        gene_set_matrix (pd.DataFrame): Reactome pathway gene set in the original format
        (first two cols are pathway_names and pathway_IDs; and the remaining are genes belonging to that pathway).
        make sure that it's properly formatted, see func _clean_reactome_pathway_gene_set()
    RETURNS:
        gene_set_matrix (pd.DataFrame): Dataframe with pathway_IDs as indices and gene_names as colnames, with binary
        cell values marking whether a gene is a member of the pathway (1) or not (0).
    """
    # set pathway IDs as index
    gene_set_matrix.set_index([1], inplace=True)

    # dropping the pathway name column from the matrix
    gene_set_matrix.drop([0], axis=1, inplace=True)
    # resetting the column indexing for the matrix
    gene_set_matrix.columns = range(gene_set_matrix.shape[1])

    # transforming this into a binary matrix of pathways in the rows and gene IDs in the columns
    gene_list = pd.Series({c: gene_set_matrix[c].unique() for c in gene_set_matrix})
    gene_list = pd.Series(np.concatenate([genes for genes in gene_list]))
    gene_list.dropna(inplace=True)
    gene_list = gene_list.unique()
    gene_list.sort()  # series of all the unique genes in the dataframe, will be assigned as columns in the binary matrix

    # defining the binary matrix, indices: pathwayIDs, columns: gene identifiers
    a = pd.DataFrame(0, index=gene_set_matrix.index, columns=gene_list)

    # for each row (pathway) in the dataframe
    for i, index in enumerate(gene_set_matrix.index):
        a.loc[index, gene_set_matrix.loc[index, :].dropna().values] = 1
        """
        if (i != 0) and (i % 500 == 0):
            print(f"processed: {i}/{len(gene_set_matrix.index)} rows ")
        """
    gene_set_matrix = _drop_meaningless_columns(a)

    return gene_set_matrix


def load_gene_aliases():
    """
    Loads datafile containing the alternate symbols to 65000 genes and gene products
    from: https://ncbiinsights.ncbi.nlm.nih.gov/2016/11/07/clearing-up-confusion-with-human-gene-symbols-names-using-ncbi-gene-data/

    RETURNS:
        df(pd.DataFrame): dataframe, each row contains all the aliases of a gene.
        Columns which have no values in them are set to None.
    """
    df = pd.read_csv(
        "data/external/gencode19_gene_annotation_file/gene_info_alisases.tsv", sep="\t"
    )
    temp = df.Synonyms.str.split("|", expand=True)
    temp.iloc[:, 0] = df.Symbol
    df = temp.copy()

    return df


def fetch_gene_aliases(gene_name, gene_aliases=None, printout=True):
    """
    Returns the aliases of a given gene. Usually intended to be used iteratively over a list of genes.

    ARGS:
        gene_name (str): gene symbol (Hugo_Symbol)
        gene_aliases (pd.DataFrame): dataframe with gene aliases in the rows, if None passed func load_gene_aliases() runs.
        printout (bool): Bool switch whether to print the results of the search for the alias.

    RETURNS:
        aliases (pd.Series): series with gene names, if entity was not found, returning -1 <int>, if no aliases found, returning <0>.
    """

    if gene_aliases is None:
        gene_aliases = load_gene_aliases()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        # extracting the row from the dataframe where the searched gene is present
        aliases = gene_aliases[gene_aliases.isin([gene_name]).any(1)]

    # if not found:
    if aliases.empty:
        if printout:
            print(f"No entities found called: {gene_name}, returning -1")
        return -1

    else:
        aliases = aliases.squeeze().dropna()
        # dropping the gene name that was queries for
        aliases = aliases[~aliases.isin([gene_name])]

        # if no gene is left in the Series, it means there are no aliases in the dataset
        if aliases.empty:
            if printout:
                print(f"No aliases found for gene: {gene_name}, returning 0")
            return 0

        else:
            return aliases


def look_for_matching_aliases(gene_set_variable, gene_set_fixed):
    """
    Given 2 lists of genes, aliases will be searched for the genes from "gene_set_variable",
    that match up with one of the symbols from "gene_set_fixed".

    ARGS:
        gene_set_variable (pd.Series): a pandas series/index of gene Hugo Symbols.
        Whichever of these symbols do not appear among the "gene_set_fixed" list of symbols,
        an alias will be searched for that does appear there.

        gene_set_fixed (pd.Series): a pandas series or index, containing gene Hugo Symbols.
        The aliases searched for the gene_set_variable parameter, will have to match one of the symbols in this list.

    RETURNS:
        found_aliases (dict): dict with keys being old gene names, values being the new aliases found.
        If for a gene no matching alias was found it will not be included in the dictionary.

    """

    filter = ~gene_set_variable.isin(gene_set_fixed)
    missing_genes = gene_set_variable[filter]

    filter = ~gene_set_fixed.isin(gene_set_variable)
    missing_genes_inv = gene_set_fixed[filter]

    # depending on the input matrices, searching for the aliases in one order is much more efficient than the other way
    # around -> need to look in which order there are fewer missing genes -> do alias searching this way -> optionally
    # then invert the keys/values of the resulting mapping

    if len(missing_genes) < len(missing_genes_inv):
        # print(f"{len(missing_genes)} genes are missing, searching for aliases..")
        inv = False
    else:
        # print(f"Inverting the alias search for efficiency..")
        missing_genes = missing_genes_inv
        inv = True
        # print(f"{len(missing_genes)} genes are missing, searching for aliases..")

    gene_aliases = load_gene_aliases()

    n = 0  # number of genes for which a valid alias was found
    found_aliases = {}  # old Hugo : new Hugo dictionary

    for i, gene in enumerate(missing_genes):

        # still need to import the gene_aliases dataframe
        aliases = fetch_gene_aliases(gene, gene_aliases, printout=False)

        # if a list of aliases has been returned (instead of -1 or 0 for not having found anything)
        if not isinstance(aliases, int):
            # figures out which alias is also a part of the given list of genes "gene_set_fixed"
            for alias in aliases:

                if not inv:
                    if gene_set_fixed.isin([alias]).any():
                        found_aliases[gene] = alias
                        n += 1
                else:
                    if gene_set_variable.isin([alias]).any():
                        found_aliases[gene] = alias
                        n += 1

    #     if (i % 100 == 0) and (i != 0):
    #         print(f"{n}/{i} genes were successfully salvaged")

    # if len(missing_genes) != 0:
    #     print(f"{n}/{i + 1} genes were successfully salvaged")

    return found_aliases


def harmonise_pathways_gene_sets_mut_table(pathways_gene_sets, mut_table):
    """
    "Harmonizes" the pathway gene set to only include the genes which are present in the mutation table.

    ARGS:
        pathways_gene_sets (pd.DataFrame): transformed dataframe with pathway_id as indices and gene_names as colnames
        mut_table (pd.DataFrame): mutation table containing patient_ids as indices and gene_names as colnames, with
        the number of mutation per gene per patient in the cells.
    RETURNS:
        pathway_gene_sets (pd.DataFrame): df in the same format as pathway_gene_sets but only containing the genes present
        in the mut_table.
    """
    # find aliases for genes in "pathways_gene_sets" that do not appear in mut_table
    mapping = look_for_matching_aliases(pathways_gene_sets.columns, mut_table.columns)

    mapping_not_already_present_in_pathways = {
        key: value
        for key, value in mapping.items()
        if value not in pathways_gene_sets.columns
    }

    # rename columns in "pathways_gene_sets" to match "mut_table"
    pathways_gene_sets = pathways_gene_sets.rename(
        columns=mapping_not_already_present_in_pathways
    )
    # selecting the genes to keep from pathwats_gene_sets (that also appear in mut_table)
    filter = pathways_gene_sets.columns.isin(mut_table.columns)
    # dropping the rest
    pathways_gene_sets = pathways_gene_sets.loc[:, filter].copy()

    return pathways_gene_sets


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


def harmonise_pathways_gene_sets_gene_lengths(pathways_gene_sets, gene_lengths):
    """
    "Harmonizes" the pathway gene set to only include the genes for which the gene length data
    is also available.

    ARGS:
        pathways_gene_sets (pd.DataFrame): transformed dataframe with pathway_id as
        indices and gene_names as colnames
        gene_lengths (pd.DataFrame): dataframe where one column lists gene names and
          the second column lists their corresponding CDS length
    RETURNS:
        pathways_gene_sets (pd.DataFrame): df in the same format as pathway_gene_sets but only containing
        genes for which CDS gene length data is available (which are present in the gene_lengths).
    """
    # find aliases for genes in "pathways_gene_sets" that do not appear in gene_lengths
    mapping = look_for_matching_aliases(pathways_gene_sets.columns, gene_lengths.index)

    gene_lengths = pd.DataFrame({"genes": gene_lengths.index, "lengths": gene_lengths})
    gene_lengths.genes.replace(mapping, inplace=True)
    # back to Series
    gene_lengths = gene_lengths.set_index("genes", drop=True).squeeze()

    # dropping the rest of the genes, which still don't have length information available in gencode
    filter = pathways_gene_sets.columns.isin(gene_lengths.index)
    pathways_gene_sets = pathways_gene_sets.loc[:, filter].copy()

    return pathways_gene_sets


def make_pathway_binary(pathways, pathways_gene_sets):
    """
    Creates a dataframe with pathway_IDs as indices and gene_names as column names, and subsets a selection of pathways.

    ARGS:
        pathways (pd.DataFrame): input a DataFrame of pathways with the pathway IDs being in the "Pathway identifier" column.
        This is probably meant to only accept output of reactome enrichment analysis (e.g. done in browser for a list of genes)

        pathways_gene_sets (pd.DataFrame): df with the reactome pathway gene set which should contain all reactome pathways
        and their member genes (binary representation).

    RETURNS:
        pathways_binary (pd.DataFrame): df with pathway_IDs as indices and gene_names as colnames and
        binary representation of a gene's membership in a pathway (1 if member, 0 if not).
    """

    pathways_binary = pathways_gene_sets.loc[pathways["Pathway_ID"].values, :]

    return pathways_binary


def make_pathway_dictionary(pathways_binary):
    """
    Creates a dictionary, with keys as pathway identifiers, and values as arrays of associated genes.

    ARGS:
        pathway_binary (pd.DataFrame): df with pathway_IDs as indices and gene_names as colnames and
        binary representation of a gene's membership in a pathway (1 if member, 0 if not)

    RETURNS:
        genes_relapsed_nonrelapsed_pathways (dict): dict with keys as pathway_IDs and values as arrays of member genes
    """

    genes_relapsed_nonrelapsed_pathways = {}

    for pathway in pathways_binary.T:
        # select the columns (genes) which have a value of 1 associated with them for the given pathway
        genes_in_pathway = pathways_binary.columns[
            (pathways_binary.loc[pathway, :] == 1)
        ].values
        # add ot dictionary
        genes_relapsed_nonrelapsed_pathways[pathway] = genes_in_pathway

    return genes_relapsed_nonrelapsed_pathways


def get_PI_matrix(cfg, pathway_gene_sets, gene_lengths, mut_table, include_cna=True):
    """
    Calculates pathway instability scores based on the mutation rates of the genes participating in each inputted pathway.

    The three input dataframes should all have been harmonized, as in: pathway_gene_sets should only contain gene entries
    for which there is associated data both in gene_lengths and mut_table.

    ARGS:
        cfg (dict): Dictionary of configuration parameters
        pathway_gene_sets (dict): pathway IDs as keys, participating genes as an array in values || or binary DataFrame
            with columns as genes, rows as pathway IDs
        gene_lengths (pd.Series): HugoSymbols as indices and CDS Lengths as values
        mut_table (pd.DataFrame): mutation table with patients in rows and gene in columns.
            Values depict number of mutations per given gene.
        include_cna (bool): choose whether to include the CNA data in the calculation of the PI score, default True.
            If True, normalized mutation rate of the gene is multiplied by the CNA value (log2)^2

    RETURNS:
        PIscore_matrix (pd.DataFrame): PI score matrix with patients in rows and pathways in columns;
        values are calculated pathway instability scores (not normalized by the number of participating entities yet.)
    """

    if isinstance(pathway_gene_sets, pd.DataFrame):
        # Transforming df to a dictionary, with keys as pathway identifiers, and values as arrays of associated genes.
        pathway_gene_sets = make_pathway_dictionary(pathway_gene_sets)

    if include_cna:
        # load and process cna data
        cna_log2 = pd.read_parquet(
            os.path.join(cfg["ROOT_RAW"], "data_log2_cna_formatted.parquet")
        )

        cna = cna_log2.rpow(2)

        # read in the file that contains the matching from hugo symbols to entrezId
        hugo_to_entrez = pd.read_csv(
            os.path.join(cfg["ROOT_RAW"], "data_mutations.txt"), sep="\t"
        )

        # keep only two relevant columns
        hugo_to_entrez = hugo_to_entrez[["Hugo_Symbol", "Entrez_Gene_Id"]]

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

        # These are the columns which don't have hugo_symbol in the data_mutations.txt file
        columns_to_drop = cna.columns[
            ~cna.columns.isin(hugo_to_entrez["Entrez_Gene_Id"])
        ]
        # We need to drop these
        cna.drop(columns=columns_to_drop, inplace=True)

        # Create a dictionary to map gene_ID to gene_name
        hugo_to_entrez_dict = pd.Series(
            hugo_to_entrez.Hugo_Symbol.values, index=hugo_to_entrez.Entrez_Gene_Id
        ).to_dict()

        # Rename the columns in the second dataframe
        cna.rename(columns=hugo_to_entrez_dict, inplace=True)

        # only keep patients which also appear in the mut_table
        cna = cna.loc[cna.index.isin(mut_table.index), :].copy()

        # if there are multiple entries for the same gene, take mean value (since now linear)
        columns_with_duplicate_entries = cna.columns[cna.columns.duplicated()]

        if not columns_with_duplicate_entries.empty:
            for duplicate_gene in columns_with_duplicate_entries:
                mean_cna_values = (
                    cna[duplicate_gene]
                    .groupby(cna[duplicate_gene].columns, axis=1)
                    .mean()
                )
                cna.drop(columns=duplicate_gene, inplace=True)
                cna = pd.merge(cna, mean_cna_values, left_index=True, right_index=True)

        # where fold change is < 1 -> make it into a positive value, as a decreased copy number also should increase pathway instability
        cna[cna < 1] = 1 / cna

    PIscore_matrix = pd.DataFrame(
        0, index=mut_table.index, columns=list(pathway_gene_sets.keys())
    )

    for pathway in PIscore_matrix.columns:

        # retrieve genes belonging to the current pathway
        pathway_genes = pathway_gene_sets[pathway]
        # there are potential duplicate genes in the produced list, removing these
        pathway_genes = np.unique(pathway_genes)
        # retrieveing length of all genes particiapting in this pathway
        pathway_gene_lengths = gene_lengths[pathway_genes]
        # filter mut_table columns using these genes -> got mutation rate for each gene in pathway for each patient
        pathway_filtered_mut_table = mut_table[pathway_genes]
        # # normalize mutation rates -> dividing columns by gene lengths
        norm_pathway_filtered_mut_table = (
            pathway_filtered_mut_table * 1000 / pathway_gene_lengths.values
        )

        # ==== incorporating cna fold change in PI matrix calculation ====
        if include_cna:
            # in norm_pathway_filtered_mut_table there are always a changing set of genes
            # (which participate in the given pathway)
            # see if any of them appear in the cna table and select their symbols if so)
            genes_of_interest = norm_pathway_filtered_mut_table.columns[
                norm_pathway_filtered_mut_table.columns.isin(cna.columns)
            ]

            # if there are any intersecting genes then:
            if not genes_of_interest.empty:
                # multiplying normalised mutation rates with cna disruption values
                # (i.e. with the fold change; or 1/fc if fc was <1)
                norm_pathway_filtered_mut_table.loc[:, genes_of_interest] = (
                    norm_pathway_filtered_mut_table.loc[:, genes_of_interest]
                    * cna.loc[:, genes_of_interest]
                )
        # ==================================================================

        # sum table in axis=1 so every patient has the sum of mutations -> PI score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            PIscore_matrix.loc[:, pathway] = norm_pathway_filtered_mut_table.sum(axis=1)

    return PIscore_matrix


def normalise_PI_matrix(PI_matrix):
    """
     Normalizes the calculated PI score matrix along the number of participating entities in each pathway.

     If patients did not receive a WES, only from a limited panel of genes, it is probably worth normalising these
     values by the number of pathway entities that are also present in that panel of genes.

    ARGS:
         PI_matrix(pd.DataFrame): not-normalized "pre-PI score matrix", obtained by func. get_PI_matrix()

     RETURNS:
         PI_matrix(pd.DataFrame): PI score matrix normalized by the number of entities in each pathway.
    """

    reactome_pathways_full = get_transformed_reactome_pathways_gene_set()
    reactome_pathways_selected = reactome_pathways_full.loc[PI_matrix.columns, :]
    # using the total number of entities in each pathway for the normalisation
    nr_of_entities = reactome_pathways_selected.astype(int).sum(axis=1)
    PI_matrix = PI_matrix.copy() / nr_of_entities

    return PI_matrix


@click.command()
@click.argument("run_id", type=str)
def main(run_id):
    """
    Processes the mutation and CNA data and writes a matrix of PI scores for all Reactome pathways in data/raw.

    ARGS:
        run_id (str): run_ID passed via CLI corresponding to the .yaml config file.
    """
    cfg = get_cfg(run_id)
    logger = getlogger(cfg)

    logger.info("Loaded and processed the mutation file.")
    mutation_table = prepare_mutations(cfg)

    logger.info("Obtaining the CDS length data.")
    gene_lengths = get_all_gene_lengths(cds_lengths="avg")

    logger.info("Loading and processing pathway-gene sets for Reactome pathways.")
    pathways_gene_sets = get_transformed_reactome_pathways_gene_set()
    pathways_gene_sets = harmonise_pathways_gene_sets_mut_table(
        pathways_gene_sets, mutation_table
    )
    pathways_gene_sets = harmonise_pathways_gene_sets_gene_lengths(
        pathways_gene_sets, gene_lengths
    )

    # ==========> Consider all pathways
    all_pathways_list = pd.DataFrame(pathways_gene_sets.index)
    all_pathways_list = all_pathways_list.rename(columns={1: "Pathway_ID"})

    # turning into a binary matrix, indices: pathwayIDs, columns: geneIDs
    pathways_binary = make_pathway_binary(all_pathways_list, pathways_gene_sets)

    logger.info("Calculating the PI score with the CNA data included.")
    PIscore_matrix_notNormalized = get_PI_matrix(
        cfg, pathways_binary, gene_lengths, mutation_table, include_cna=True
    )
    # normalising PI scores by the number of entities in each pathway
    PIscore_matrix = normalise_PI_matrix(PIscore_matrix_notNormalized)

    logger.info("Exporting PI scores matrix.")
    # # Export the PI matrix
    # PIscore_matrix.to_csv(
    #     os.path.join(cfg['ROOT_RAW'], "PIscore_matrix_withCNA.txt"), index=True, sep="\t")

    PIscore_matrix.to_parquet(
        os.path.join(cfg["ROOT_RAW"], "PIscore_matrix_withCNA.parquet"), index=True
    )


if __name__ == "__main__":
    main()
