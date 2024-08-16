import click

from src.utils.utils_basic import getlogger
from src.utils.config import get_cfg
import networkx as nx


def get_reactome_ont_dict(reactome_rel_file_path, depth_parent=0, max_child_lvl=1):

    G = nx.DiGraph()

    reactome_dict = dict()

    with open(reactome_rel_file_path, "r") as relation_file:
        for line in relation_file:
            if "R-HSA" in line:  ## Select only human pathways
                r_id_parent = line.split("\t")[0]
                r_id_child = line.strip().split("\t")[1]

                if not G.has_node(r_id_parent):
                    G.add_node(r_id_parent)

                if not G.has_node(r_id_child):
                    G.add_node(r_id_child)

                G.add_edge(r_id_parent, r_id_child)

    lvl_count = 0
    toplvl = [n for n in G.nodes if G.in_degree(n) == 0]

    while lvl_count < depth_parent:
        next_lvl = []
        for t in toplvl:
            #
            succ_dict = nx.dfs_successors(G, source=t, depth_limit=1)
            if len(succ_dict) > 0:
                next_lvl.extend(succ_dict[t])

        toplvl = next_lvl
        lvl_count += 1

    for t in toplvl:
        child_lvl = []
        lvl_count = 1

        succ_dict = nx.dfs_successors(G, source=t, depth_limit=1)

        if len(succ_dict) > 0:
            child_lvl.extend(succ_dict[t])

        while lvl_count < max_child_lvl:

            for t2 in child_lvl:

                succ_dict = nx.dfs_successors(G, source=t2, depth_limit=1)

                if len(succ_dict) > 0:
                    child_lvl = list(set(child_lvl + succ_dict[t2]))

            lvl_count += 1

        reactome_dict[t] = child_lvl

    return reactome_dict


def save_ont_dict(ont_dic, file_path, delim="\t"):
    ## format
    # member | ontology
    with open(file_path, "w") as dict_file:
        for ont in ont_dic:
            for member in ont_dic[ont]:
                dict_file.write(member + delim + ont + "\n")

    return ont_dic


def reactome_to_ncbi(reactome_ncbi_file_path, sub_pwy, singleton=False):
    lvl2_dic = dict()

    ncbi_pwy = dict()
    with open(reactome_ncbi_file_path, "r") as ncbi_file:
        for line in ncbi_file:
            ncbi_id = line.split("\t")[0]
            r_id = line.split("\t")[1]

            if r_id in sub_pwy:
                if singleton and (not ncbi_id in ncbi_pwy):
                    ncbi_pwy[ncbi_id] = r_id

                    if r_id in lvl2_dic:
                        lvl2_dic[r_id].append(ncbi_id)
                    else:
                        lvl2_dic[r_id] = []
                        lvl2_dic[r_id].append(ncbi_id)
                else:
                    if not singleton:
                        if r_id in lvl2_dic:
                            lvl2_dic[r_id].append(ncbi_id)
                        else:
                            lvl2_dic[r_id] = []
                            lvl2_dic[r_id].append(ncbi_id)

    return lvl2_dic


@click.command()
@click.argument("run_id", type=str)
def main(run_id):
    """Creates reactome ontology"""

    cfg = get_cfg(run_id)

    logger = getlogger(cfg)
    logger.info("Make Ontology Files")

    reactome_rel_file_path = "./data/raw/ReactomePathwaysRelation.txt"  

    logger.info("Make Dictionaries")
    reactome_dic = get_reactome_ont_dict(reactome_rel_file_path,depth_parent=0, max_child_lvl=2) 


    all_sub_pwy = []
    for r in reactome_dic:
        all_sub_pwy = list(set(reactome_dic[r] + all_sub_pwy))

    reactome_ncbi_file_path = "./data/raw/NCBI2Reactome_All_Levels.txt"
    reactome_ensembl_file_path = "./data/raw/Ensembl2Reactome_All_Levels.txt"

    reactome_ncbi_dic = reactome_to_ncbi(reactome_ncbi_file_path=reactome_ncbi_file_path,sub_pwy=all_sub_pwy)
    reactome_ensembl_dic = reactome_to_ncbi(reactome_ncbi_file_path=reactome_ensembl_file_path,sub_pwy=all_sub_pwy)


    logger.info("Save Dictionaries")
    save_ont_dict(reactome_dic,file_path="./data/raw/full_ont_lvl2_reactome.txt", delim=cfg['DELIM'])
    save_ont_dict(reactome_ncbi_dic,file_path="./data/raw/full_ont_lvl1_reactome.txt", delim=cfg['DELIM'])

    save_ont_dict(reactome_ensembl_dic,file_path="./data/raw/full_ont_lvl1_ensembl_reactome.txt", delim=cfg['DELIM'])




if __name__ == "__main__":
    main()
