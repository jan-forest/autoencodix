import glob
import yaml
import pandas as pd
import numpy as np
from src.utils.utils_basic import read_ont_file
from itertools import combinations
import seaborn as sns
import seaborn.objects as so
from matplotlib import pyplot as plt


def calculate_corr(dataset, ontology_type, randomness, param):

    if ontology_type == "Rea":
        latent_names = read_ont_file("data/raw/full_ont_lvl2_reactome.txt", sep="\t")
        latent_names = latent_names.keys()

    if ontology_type == "Chr":
        latent_names = read_ont_file("data/raw/chromosome_ont_lvl2.txt", sep="\t")
        latent_names = latent_names.keys()

    file_regex = (
        "./reports/"
        + "Exp3_"
        + dataset
        + "_"
        + ontology_type
        + "_"
        + randomness
        + "*_"
        + param
        + "/"
        + "predicted_latent_space.parquet"
    )
    file_list = glob.glob(file_regex)

    corr_to_repetition = pd.DataFrame(columns=["corr_to_other_repetition", "std_dev"])

    latent_repetitions = dict()
    for f in file_list:
        rep = f.split("/")[-2].split("_")[-2][-1]
        df_latent = pd.read_parquet(f)
        df_latent.columns = latent_names
        df_latent = df_latent.loc[sample_list, :]  # unify order
        latent_repetitions[rep] = df_latent

    for dim in latent_names:
        # print(dim)
        r_list = []
        for rep_comb in combinations(list(latent_repetitions.keys()), 2):
            r = np.corrcoef(
                x=latent_repetitions[rep_comb[0]].loc[:, dim],
                y=latent_repetitions[rep_comb[1]].loc[:, dim],
            )
            r_list.append(np.abs(r[0][1]))
            # print(rep_comb)

        r_list = np.array(r_list)
        for rep in list(latent_repetitions.keys()):
            boolean_slice = [
                rep in rep_comb
                for rep_comb in combinations(list(latent_repetitions.keys()), 2)
            ]
            run_id = (
                config_prefix
                + dataset
                + "_"
                + ontology_type
                + "_"
                + randomness
                + rep
                + "_"
                + param
            )

            row = {
                "corr_to_other_repetition": np.mean(r_list[np.array(boolean_slice)]),
                "std_dev": np.std(r_list[np.array(boolean_slice)]),
            }

            corr_to_repetition.loc[run_id + "_" + dim, :] = row

    return corr_to_repetition


def single_heatmap(df_robust, param, ont_type, ont_order, ax, randomness="random"):
    df_heatmap = (
        df_robust.loc[
            (df_robust["param_to_test"] == param)
            & (df_robust["ontology_type"] == ont_type)
            & (df_robust["randomness"] == randomness),
            ["ontology_dim", param, "corr_to_other_repetition"],
        ]
        .groupby(["ontology_dim", param], as_index=False)
        .mean()
        .pivot(index="ontology_dim", columns=param, values="corr_to_other_repetition")
    )

    match param:
        case "beta":
            color = "Blues"
        case "drop_out":
            color = "Reds"
        case "learn_rate":
            color = "Greens"
        case _:
            color = "Grays"
    # print(df_heatmap.loc[ont_order.index,:])
    df_heatmap = df_heatmap.astype(float)
    heatmap = sns.heatmap(
        df_heatmap.loc[ont_order.index, :],
        cmap=color,
        cbar_kws={
            "label": "Absolute correlation to other repetitions",
            "orientation": "horizontal",
        },
        vmin=0,
        vmax=1,
        ax=ax,
        yticklabels=True,
    )

    return heatmap


datasets = ["TCGA", "SC"]
ontologies = ["Rea", "Chr"]

params = dict()
params["B"] = "beta"
params["D"] = "drop_out"
params["L"] = "learn_rate"

root = "./"

output_type = ".png"

config_prefix = "Exp3_"
for dataset in datasets:
    print(f"Making ontix robustness plots for data set: {dataset}")
    ### Get the sample list from one run to match order with other runs
    if dataset == "TCGA":
        sample_list = pd.read_parquet(
            "./reports/Exp3_TCGA_Chr_rand1_B2/predicted_latent_space.parquet"
        ).index
        name_toplvl = read_ont_file(
            "data/raw/Reactome_TopLvl_hsa.txt", sep="\t"
        )  ## For renaming of latent dim
        name_toplvl = {v[0].rstrip(): k for k, v in name_toplvl.items()}
    if dataset == "SC":
        sample_list = pd.read_parquet(
            "./reports/Exp3_SC_Chr_rand1_B2/predicted_latent_space.parquet"
        ).index

    ## df_robust for plotting
    # run_id | dataset | ontology_type | param_to_test | ontology_dim | randomness | repetition | beta | drop_out | corr_to_other_repetition | std_dev
    df_robust = pd.DataFrame(
        columns=[
            "run_id",
            "dataset",
            "ontology_type",
            "param_to_test",
            "ontology_dim",
            "randomness",
            "repetition",
            "beta",
            "drop_out",
            "learn_rate",
            "corr_to_other_repetition",
            "std_dev",
            "R2_valid",
            "VAE_loss_valid",
        ]
    )

    file_regex = root + "reports/" + "*/" + config_prefix + dataset + "*_config.yaml"
    file_list = glob.glob(file_regex)

    for config_path in file_list:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # print(config_path)
        dm = list(config["DATA_TYPE"].keys())
        dm.remove("ANNO")

        if config["RUN_ID"].split("_")[2] == "Rea":
            latent_names = read_ont_file(
                "data/raw/full_ont_lvl2_reactome.txt", sep="\t"
            )
            latent_names = latent_names.keys()

        if config["RUN_ID"].split("_")[2] == "Chr":
            latent_names = read_ont_file("data/raw/chromosome_ont_lvl2.txt", sep="\t")
            latent_names = latent_names.keys()

        for dim in latent_names:
            if config["RUN_ID"].split("_")[2] == "Rea":
                dim_name = name_toplvl[dim]
            else:
                dim_name = dim
            results_row = {
                "run_id": config["RUN_ID"],
                "dataset": dataset,
                "ontology_type": config["RUN_ID"].split("_")[2],
                "param_to_test": params[config["RUN_ID"].split("_")[4][0]],
                "ontology_dim": dim_name,
                "randomness": config["FIX_RANDOMNESS"],
                "repetition": config["RUN_ID"].split("_")[3][-1],
                "beta": config["BETA"],
                "drop_out": config["DROP_P"],
                "learn_rate": config["LR_FIXED"],
                "corr_to_other_repetition": np.NAN,
                "std_dev": np.NAN,
                "R2_valid": np.NAN,
                "VAE_loss_valid": np.NAN,
            }
            df_robust.loc[config["RUN_ID"] + "_" + str(dim), :] = results_row

    ## df_recon for plotting
    # run_id | dataset | ontology_type | param_to_test | randomness | repetition | beta | drop_out | r2_valid | r2_train

    for run_id in df_robust.run_id.unique():

        file_regex = root + "reports/" + run_id + "/losses_*.parquet"
        file_list = glob.glob(file_regex)

        if len(file_list) > 0:
            loss_df = pd.read_parquet(file_list[0])
            if "train_r2" in loss_df.columns:
                beta_temp = df_robust.loc[df_robust.run_id == run_id, "beta"].unique()
                df_robust.loc[df_robust.run_id == run_id, "VAE_loss_valid"] = (
                    loss_df["valid_vae_loss"].iloc[-1] / beta_temp[0]
                )
                df_robust.loc[df_robust.run_id == run_id, "R2_valid"] = loss_df[
                    "valid_r2"
                ].iloc[-1]

    ## Calculate correlation of repetitions
    for ontology_type in df_robust.ontology_type.unique():
        rand_all = (
            df_robust.run_id.str.split("_", expand=True)[3]
            .str.rstrip("0123456789")
            .unique()
        )
        for randomness in rand_all:
            param_all = df_robust.run_id.str.split("_", expand=True)[4].unique()
            for param in param_all:
                try:
                    df_out = calculate_corr(dataset, ontology_type, randomness, param)
                    df_robust.loc[df_out.index, df_out.columns] = df_out
                except ValueError:
                    pass

    df_robust = df_robust.astype(
        {
            "beta": float,
            "drop_out": float,
            "learn_rate": float,
        }
    )

    ### Create Heatmaps Robustness
    count_ont_type = 0
    fig, axes = plt.subplots(
        len(ontologies),
        4,
        figsize=(16, len(ontologies) * 6),
        gridspec_kw={"width_ratios": [0.5, 2, 2, 2]},
    )
    for ont_type in ontologies:
        order = pd.read_csv(
            "data/raw/top_lvl_order_" + dataset + "_" + ont_type + ".txt",
            sep="\t",
            index_col=0,
        )

        sns.heatmap(
            order * 100,
            cmap="cividis",
            cbar_kws={"label": "Percentage of all features", "location": "left"},
            vmin=0,
            ax=axes[count_ont_type, 0],
            yticklabels=False,
            xticklabels=False,
            annot=True,
        )

        single_heatmap(
            df_robust,
            param="beta",
            ont_type=ont_type,
            ont_order=order,
            ax=axes[count_ont_type, 1],
        )
        single_heatmap(
            df_robust,
            param="drop_out",
            ont_type=ont_type,
            ont_order=order,
            ax=axes[count_ont_type, 2],
        )
        single_heatmap(
            df_robust,
            param="learn_rate",
            ont_type=ont_type,
            ont_order=order,
            ax=axes[count_ont_type, 3],
        )

        axes[count_ont_type, 0].set_ylabel("")
        axes[count_ont_type, 1].set_ylabel("")
        axes[count_ont_type, 2].set_ylabel("")

        count_ont_type += 1

    plt.tight_layout()

    fig.savefig(
        root
        + "reports/paper-visualizations/Exp3/Ontix_robustness_"
        + dataset
        + output_type,
        bbox_inches="tight",
    )

    ### ML task performance
    ml_results = pd.DataFrame(
        columns=[
            "run_id",
            "Architecture",
            "ML_Algorithm",
            "Parameter",
            "Metric",
            "Performance",
            "Split",
            "ontology_type",
            "param_to_test",
            "randomness",
            "repetition",
            "beta",
            "drop_out",
            "learn_rate",
        ]
    )

    for run_id in df_robust.run_id.unique():
        file_ml = root + "reports/" + run_id + "/ml_task_performance.txt"
        ml_df = pd.read_csv(file_ml, sep="\t")
        ml_df = ml_df.groupby(
            ["metric", "ML_TASK", "ML_ALG", "score_split", "CLINIC_PARAM"],
            as_index=False,
        ).mean(numeric_only=True)

        ml_df = ml_df.rename(
            columns={
                "CLINIC_PARAM": "Parameter",
                "metric": "Metric",
                "value": "Performance",
                "ML_ALG": "ML_Algorithm",
                "ML_TASK": "Architecture",
                "score_split": "Split",
            }
        )

        ml_df.loc[ml_df.loc[:, "Architecture"] == "Latent", "Architecture"] = "ontix"
        ml_df.loc[:, "run_id"] = run_id
        add_data = df_robust.loc[
            df_robust["run_id"] == run_id,
            [
                "ontology_type",
                "param_to_test",
                "randomness",
                "repetition",
                "beta",
                "drop_out",
                "learn_rate",
            ],
        ].drop_duplicates()
        ml_df.loc[
            :,
            [
                "ontology_type",
                "param_to_test",
                "randomness",
                "repetition",
                "beta",
                "drop_out",
                "learn_rate",
            ],
        ] = add_data.loc[add_data.index.repeat(len(ml_df.index))].reset_index(drop=True)

        if len(ml_results.index) > 0:
            ml_results = pd.concat([ml_results, ml_df])
        else:
            ml_results = ml_df.copy()

    # Influence of beta on performance
    p_beta = (
        so.Plot(
            ml_results[
                (ml_results["Architecture"] == "ontix")
                & (ml_results["Split"] == "test")
                & (ml_results["ML_Algorithm"] == "RF")
                & (ml_results["param_to_test"] == "beta")
            ],
            x="beta",
            y="Performance",
            color="Parameter",
        )
        .add(so.Line(), so.Agg())
        .add(so.Range(), so.Est(errorbar="sd"))
        .facet("ontology_type", wrap=2)
        # .limit(y=(0, 1))
        .layout(size=(12, 6))
        .scale(x=so.Continuous(trans="log"))
    )
    p_beta.save(
        root + "reports/paper-visualizations/Exp3/Ontix_beta_" + dataset + output_type,
        bbox_inches="tight",
    )

    # Influence of drop_out on performance
    p_drop = (
        so.Plot(
            ml_results[
                (ml_results["Architecture"] == "ontix")
                & (ml_results["Split"] == "test")
                & (ml_results["ML_Algorithm"] == "RF")
                & (ml_results["param_to_test"] == "drop_out")
            ],
            x="drop_out",
            y="Performance",
            color="Parameter",
        )
        .add(so.Line(), so.Agg())
        .add(so.Range(), so.Est(errorbar="sd"))
        .facet("ontology_type", wrap=2)
        # .limit(y=(0, 1))
        .layout(size=(12, 6))
    )
    p_drop.save(
        root
        + "reports/paper-visualizations/Exp3/Ontix_dropout_"
        + dataset
        + output_type,
        bbox_inches="tight",
    )

    # Influence of learn_rate on performance
    p_learn = (
        so.Plot(
            ml_results[
                (ml_results["Architecture"] == "ontix")
                & (ml_results["Split"] == "test")
                & (ml_results["ML_Algorithm"] == "RF")
                & (ml_results["param_to_test"] == "learn_rate")
            ],
            x="learn_rate",
            y="Performance",
            color="Parameter",
        )
        .add(so.Line(), so.Agg())
        .add(so.Range(), so.Est(errorbar="sd"))
        .facet("ontology_type", wrap=2)
        # .limit(y=(0, 1))
        .layout(size=(12, 6))
        .scale(x=so.Continuous(trans="log"))
    )
    p_drop.save(
        root
        + "reports/paper-visualizations/Exp3/Ontix_learnrate_"
        + dataset
        + output_type,
        bbox_inches="tight",
    )

    ### Save df's

    df_robust.to_csv(
        root + "reports/paper-visualizations/Exp3/df_robust_" + dataset + ".txt",
        sep="\t",
    )
    ml_results.to_csv(
        root + "reports/paper-visualizations/Exp3/ml_results_" + dataset + ".txt",
        sep="\t",
    )
