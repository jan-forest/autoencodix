import glob
import yaml
import pandas as pd
import numpy as np

import seaborn.objects as so
import seaborn as sns
from matplotlib.ticker import LinearLocator, LogLocator

from seaborn import axes_style
so.Plot.config.theme.update(axes_style("whitegrid"))

config_prefix_list = ["Exp2_TCGA_tune", "Exp2_SC_tune"]

rootdir = "/mnt/c/Users/ewald/Nextcloud/eigene_shares/AutoEncoderOmics/SaveResults/SC-UL-Connection/"
# rootdir = "./"
rootsave = "./reports/paper-visualizations/Exp2/"
output_type = ".png"
# output_type = ".svg"

print("read in")

df_results = pd.DataFrame(
    columns=[
        "Config_ID",
        "Architecture",
        "Data_Modalities",
        "Latent_Dim",
        "Latent_Coverage",
        "R2_valid",
        "R2_train",
        "Rec. loss",
        "Total loss",
        "Data_set",
    ]
)

## Browse configs in reports and get infos
for config_prefix in config_prefix_list:
    file_regex = "reports/" + "*/" + config_prefix + "*_config.yaml"
    file_list = glob.glob(rootdir + file_regex)

    for config_path in file_list:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        dm = list(config["DATA_TYPE"].keys())
        dm.remove("ANNO")

        results_row = {
            "Config_ID": config["RUN_ID"],
            "Architecture": config["MODEL_TYPE"] + "_B" + str(config["BETA"]),
            "Data_Modalities": "+".join(dm),
            "Latent_Dim": config["LATENT_DIM_FIXED"],
            "Latent_Coverage": 0.0,
            "R2_valid": 0.0,
            "R2_train": 0.0,
            "Rec. loss": 0.0,
            "Total loss": 0.0,
            "Data_set": config_prefix.split("_")[1],
            "weight_decay": 0.0,
            "dropout_all": 0.0,
            "encoding_factor": 0,
            "lr": 0.0,
        }
        df_results.loc[config["RUN_ID"], :] = results_row
        # print(results_row)

        param_path_regex = (
            rootdir
            + "reports/"
            + config["RUN_ID"]
            # + "/COMBINED-*"
            + "/CO*"    # match CONCAT or COMBINED
            + "best_model_params.txt"
        )
        if len(glob.glob(param_path_regex)) > 0:
            param_path = glob.glob(param_path_regex)[0]  # Should only be one matching file
            best_params_series = pd.read_csv(
                param_path, header=0, names=["parameter", "values"], index_col=0
            )["values"]
            run_id = config["RUN_ID"]
            df_results.loc[run_id, "weight_decay"] = best_params_series["weight_decay"]
            df_results.loc[run_id, "dropout_all"] = best_params_series["dropout_all"]
            df_results.loc[run_id, "encoding_factor"] = best_params_series[
                "encoding_factor"
            ]
            df_results.loc[run_id, "lr"] = best_params_series["lr"]

# print(df_results)

for run_id in df_results.index:
	# file_regex = "reports/" + run_id + "/losses_*.parquet"
	# file_list = glob.glob(rootdir + file_regex)
	dm = df_results.loc[run_id, "Data_Modalities"].split("+")

	# loss_file = max(file_list, key=len)  # combined loss always has the longest name
	if "stackix" in run_id:
		df_results.loc[run_id, "Rec. loss"] = 0
		df_results.loc[run_id, "Total loss"] = 0
		## single dm
		for d in dm:
			loss_file = rootdir + "reports/" + run_id + "/losses_tuned_" + df_results.loc[run_id,"Architecture"].split("_")[0] + "_base_" + d + ".parquet"
			loss_df = pd.read_parquet(loss_file)
			if "train_r2" in loss_df.columns:
				df_results.loc[run_id, "Rec. loss"] +=  loss_df["valid_recon_loss"].iloc[-1]
				df_results.loc[run_id, "Total loss"] += loss_df["valid_total_loss"].iloc[-1]

		## combined
		loss_file = rootdir + "reports/" + run_id + "/losses_tuned_" + df_results.loc[run_id,"Architecture"].split("_")[0] + "_concat_" + "_".join(dm) + ".parquet"
		loss_df = pd.read_parquet(loss_file)
		if "train_r2" in loss_df.columns:
			df_results.loc[run_id, "Rec. loss"] +=  loss_df["valid_recon_loss"].iloc[-1]
			df_results.loc[run_id, "Total loss"] += loss_df["valid_total_loss"].iloc[-1]

	else:
		loss_file = rootdir + "reports/" + run_id + "/losses_tuned_" + "_".join(dm) + "_" + df_results.loc[run_id,"Architecture"].split("_")[0] + ".parquet"

		loss_df = pd.read_parquet(loss_file)
		if "train_r2" in loss_df.columns:
			df_results.loc[run_id, "R2_train"] = loss_df["train_r2"].iloc[-1]
			df_results.loc[run_id, "R2_valid"] = loss_df["valid_r2"].iloc[-1]
			df_results.loc[run_id, "Rec. loss"] = loss_df["valid_recon_loss"].iloc[-1]
			df_results.loc[run_id, "Total loss"] = loss_df["valid_total_loss"].iloc[-1]


print(df_results)

arch_order = [
    "vanillix_B1",
    "varix_B1",
    "varix_B0.1",
    "varix_B0.01",
    "ontix_B1",
    "ontix_B0.1",
    "ontix_B0.01",
    "stackix_B1",
    "stackix_B0.1",
    "stackix_B0.01",
]
dm_order = ["RNA", "METH", "MUT", "METH+RNA", "MUT+RNA", "METH+MUT", "METH+MUT+RNA"]

print("make plot reconstruction")
# p_recon = (
#     so.Plot(df_results, y="Architecture", x="R2_valid", color="Latent_Dim")
#     .facet(col="Data_Modalities", wrap=7, order=dm_order)
#     .add(so.Dot(), so.Dodge(), fill="Data_set", marker="Latent_Dim")
#     .scale(color=so.Nominal(), y=so.Nominal(order=arch_order), marker=["o", "^", "*"])
#     .layout(size=(20, 5))
#     .label(legend=None)
#     .limit(x=(-0.4, 1.1))
# )

p_recon2 = (
    so.Plot(df_results, y="Architecture", x="Rec. loss", color="Latent_Dim")
    .facet(col="Data_Modalities", wrap=7, order=dm_order)
    .add(so.Dot(), so.Dodge(), fill="Data_set", marker="Latent_Dim")
    .scale(color=so.Nominal(), y=so.Nominal(order=arch_order), marker=["o", "^", "*"])
    .layout(size=(20, 5))
    .label(legend=None)
    # .limit(x=(-0.4,1.1))
)

# p_recon.save(rootsave + "Exp2_Supp_reconcap-R2" + output_type, bbox_inches="tight")
p_recon2.save(rootsave + "Exp2_Fig3A_reconcap" + output_type, bbox_inches="tight")


print("read in ml results")
## get ml results
ml_results = pd.DataFrame(
    columns=[
        "Config_ID",
        "Architecture",
        "Data_Modalities",
        "Latent_Dim",
        "ML_Algorithm",
        "Parameter",
        "Metric",
        "Performance",
        "Perf_std",
        "Split",
        "Data_set",
    ]
)

for config_prefix in config_prefix_list:
    file_regex = "reports/" + config_prefix + "*/" + "*ml_task_performance.txt"
    file_list = glob.glob(rootdir + file_regex)

    sep = "\t"

    for ml_perf_file in file_list:
        ml_df = pd.read_csv(ml_perf_file, sep=sep)
        if not (
            "RandomFeature" in ml_df["ML_TASK"].unique()
        ):  ### Only for fast testing ###
            fake_random = ml_df[ml_df["ML_TASK"] == "Latent"].replace(
                "Latent", "RandomFeature"
            )
            fake_random2 = fake_random.copy()
            fake_random["value"] = fake_random["value"] * np.random.uniform(0, 1, 1)[0]
            fake_random2["value"] = (
                fake_random2["value"] * np.random.uniform(0, 1, 1)[0]
            )
            fake_random["ML_SUBTASK"] = fake_random["ML_SUBTASK"].replace(
                "RandonFeature", "RandonFeature1"
            )
            fake_random2["ML_SUBTASK"] = fake_random2["ML_SUBTASK"].replace(
                "RandonFeature", "RandonFeature2"
            )
            ml_df = pd.concat([ml_df, fake_random, fake_random2]).reset_index(drop=True)

        # run_id = ml_perf_file.split("/")[2]
        run_id = ml_perf_file.lstrip(rootdir).split("/")[1]
        arch = df_results.loc[run_id, "Architecture"]
        dm = df_results.loc[run_id, "Data_Modalities"]
        latent_dim = df_results.loc[run_id, "Latent_Dim"]

        ml_std = (
            ml_df.groupby(
                ["metric", "ML_TASK", "ML_ALG", "score_split", "CLINIC_PARAM"],
                as_index=False,
            )
            .std(numeric_only=True)
            .loc[:, "value"]
        )
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

        ml_df.loc[:, "Perf_std"] = ml_std
        ml_df.loc[ml_df.loc[:, "Architecture"] == "Latent", "Architecture"] = arch
        ml_df.loc[:, "Config_ID"] = run_id
        ml_df.loc[:, "Data_Modalities"] = dm
        ml_df.loc[:, "Latent_Dim"] = float(
            latent_dim
        )  ## Avoiding error in seaborn PlotSpecError
        ml_df.loc[:, "Data_set"] = config_prefix.split("_")[1]

        if len(ml_results.index) > 0:
            ml_results = pd.concat([ml_results, ml_df])
        else:
            ml_results = ml_df

ml_results.index = ml_results["Config_ID"]
ml_results.index.name = "Index"

print("make plot ml tasks")

arch_plus_order = [
    "vanillix_B1",
    "varix_B1",
    "varix_B0.1",
    "varix_B0.01",
    "ontix_B1",
    "ontix_B0.1",
    "ontix_B0.01",
    "stackix_B1",
    "stackix_B0.1",
    "stackix_B0.01",
    "PCA",
    "UMAP",
]

sel_ml_alg = "RF"
ml_results.reset_index(inplace=True)
# print(ml_results.shape)

ml_results_normed = pd.DataFrame(
    columns=[
        "Config_ID",
        "Architecture",
        "Data_Modalities",
        "Latent_Dim",
        "ML_Algorithm",
        "Parameter",
        "Metric",
        "Performance",
        "Perf_std",
        "Split",
        "Data_set",
    ]
)

for config_id in ml_results["Config_ID"].unique():
    # print(config_id)
    for arch in ml_results.loc[
        ml_results["Config_ID"] == config_id, "Architecture"
    ].unique():
        # print(arch)
        if not arch == "RandomFeature":
            ml_results_normed_new = ml_results.loc[
                (ml_results["Architecture"] == arch)
                & (ml_results["Config_ID"] == config_id),
                :,
            ].reset_index(drop=True)
            ml_random_perf = ml_results.loc[
                (ml_results["Architecture"] == "RandomFeature")
                & (ml_results["Config_ID"] == config_id),
                "Performance",
            ].reset_index(drop=True)
            ml_random_perf_std = ml_results.loc[
                (ml_results["Architecture"] == "RandomFeature")
                & (ml_results["Config_ID"] == config_id),
                "Perf_std",
            ].reset_index(drop=True)
            ml_results_normed_new.loc[:, "Performance"] = (
                ml_results_normed_new.loc[:, "Performance"] - ml_random_perf
            ) / ml_random_perf_std
            if len(ml_results_normed.index) > 0:
                ml_results_normed = pd.concat(
                    [ml_results_normed, ml_results_normed_new]
                )
            else:
                ml_results_normed = ml_results_normed_new

ml_results_normed.rename(columns={"Performance": "Perf. over random"}, inplace=True)
ml_results_normed.reset_index(inplace=True)

p_ml_task = (
    # so.Plot(ml_results.loc[ml_results['ML_Algorithm'] == sel_ml_alg,:],y="Architecture",x="Performance",color="Latent_Dim")
    so.Plot(
        ml_results_normed.loc[ml_results_normed["ML_Algorithm"] == sel_ml_alg, :],
        y="Architecture",
        x="Perf. over random",
        color="Latent_Dim",
    )
    .facet(row="Metric", col="Data_Modalities", order={"col": dm_order})
    .add(
        so.Dash(color="0.3"),
        so.Agg(),
        so.Dodge(),
    )
    .add(so.Range(color="0.3"), so.Est(errorbar="sd"), so.Dodge())
    .add(so.Dots(alpha=0.01), so.Dodge(), so.Jitter())
    .scale(
        x=so.Continuous(trans="sqrt"),
        color=so.Nominal(),
        y=so.Nominal(order=arch_plus_order),
    )
    .limit(x=(ml_results_normed["Perf. over random"].quantile(q=0.01), ml_results_normed["Perf. over random"].quantile(q=0.99)))
    .layout(size=(20, 10))
    .label(legend=None)
)
p_ml_task.save(
    rootsave + "Exp2_Supp_embeddingcap-detailed" + output_type, bbox_inches="tight"
)

p_ml_task_dense = (
    # so.Plot(ml_results.loc[ml_results['ML_Algorithm'] == sel_ml_alg,:],y="Architecture",x="Performance",color="Latent_Dim")
    so.Plot(
        ml_results_normed,
        y="Architecture",
        x="Perf. over random",
        color="Latent_Dim",
        marker="Latent_Dim",
    )
    .facet(row="Metric", col="ML_Algorithm")
    .add(so.Dots(), so.Agg(), so.Dodge())
    .add(so.Range(), so.Est(errorbar="sd"), so.Dodge())
    .scale(
        x=so.Continuous(trans="sqrt"),
        color=so.Nominal(),
        y=so.Nominal(order=arch_plus_order),
        marker=["o", "^", "*"],
    )
    .layout(size=(10, 12))
    .label(legend=None)
)
p_ml_task_dense.save(
    rootsave + "Exp2_Fig3BC_embeddingcap" + output_type, bbox_inches="tight"
)


#####################
### Tuning impact ###
#####################

not_tuned_list = ["Exp2_TCGA_train", "Exp2_SC_train"]

## read in reconstruction performance

df_results_untuned = pd.DataFrame(
    columns=[
        "Config_ID",
        "Architecture",
        "Data_Modalities",
        "Latent_Dim",
        "Latent_Coverage",
        "Rec. loss",
        "R2_valid",
        "R2_train",
        "Total loss",
        "Data_set",
    ]
)

## Browse configs in reports and get infos
for config_prefix in not_tuned_list:
    file_regex = "reports/" + "*/" + config_prefix + "*_config.yaml"
    file_list = glob.glob(rootdir + file_regex)

    for config_path in file_list:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        dm = list(config["DATA_TYPE"].keys())
        dm.remove("ANNO")

        results_row = {
            "Config_ID": config["RUN_ID"],
            "Architecture": config["MODEL_TYPE"] + "_B" + str(config["BETA"]),
            "Data_Modalities": "+".join(dm),
            "Latent_Dim": config["LATENT_DIM_FIXED"],
            "Latent_Coverage": 0.0,
            "R2_valid": 0.0,
            "R2_train": 0.0,
            "Rec. loss": 0.0,
            "Total loss": 0.0,
            "Data_set": config_prefix.split("_")[1],
            "weight_decay": 0.0,
            "dropout_all": 0.0,
            "encoding_factor": 0,
            "lr": 0.0,
        }
        df_results_untuned.loc[config["RUN_ID"], :] = results_row

for run_id in df_results_untuned.index:

    file_regex = "reports/" + run_id + "/losses_*.parquet"
    file_list = glob.glob(rootdir + file_regex)
    dm = df_results_untuned.loc[run_id, "Data_Modalities"].split("+")
    # file_list = [ f for f in file_list if all(c in f for c in dm) ]
    loss_file = max(file_list, key=len)  # combined loss always has the longest name

    if len(file_list) > 0:
        loss_df = pd.read_parquet(loss_file)
        if "train_r2" in loss_df.columns:
            df_results_untuned.loc[run_id, "R2_train"] = loss_df["train_r2"].iloc[-1]
            df_results_untuned.loc[run_id, "R2_valid"] = loss_df["valid_r2"].iloc[-1]
            df_results_untuned.loc[run_id, "Rec. loss"] = loss_df[
                "valid_recon_loss"
            ].iloc[-1]
            df_results_untuned.loc[run_id, "Total loss"] = loss_df[
                "valid_total_loss"
            ].iloc[-1]

## read in ml task performance

print("read in ml results")
ml_results_untuned = pd.DataFrame(
    columns=[
        "Config_ID",
        "Architecture",
        "Data_Modalities",
        "Latent_Dim",
        "ML_Algorithm",
        "Parameter",
        "Metric",
        "Performance",
        "Split",
        "Data_set",
    ]
)

for config_prefix in not_tuned_list:
    file_regex = "reports/" + config_prefix + "*/" + "*ml_task_performance.txt"
    file_list = glob.glob(rootdir + file_regex)

    sep = "\t"

    for ml_perf_file in file_list:
        ml_df = pd.read_csv(ml_perf_file, sep=sep)

        run_id = ml_perf_file.lstrip(rootdir).split("/")[1]
        arch = df_results_untuned.loc[run_id, "Architecture"]
        dm = df_results_untuned.loc[run_id, "Data_Modalities"]
        latent_dim = df_results_untuned.loc[run_id, "Latent_Dim"]

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

        ml_df.loc[ml_df.loc[:, "Architecture"] == "Latent", "Architecture"] = arch
        ml_df.loc[:, "Config_ID"] = run_id
        ml_df.loc[:, "Data_Modalities"] = dm
        ml_df.loc[:, "Latent_Dim"] = latent_dim
        ml_df.loc[:, "Data_set"] = config_prefix.split("_")[1]

        if len(ml_results_untuned.index) > 0:
            ml_results_untuned = pd.concat([ml_results_untuned, ml_df])
        else:
            ml_results_untuned = ml_df


ml_results_untuned.index = ml_results_untuned["Config_ID"]
ml_results_untuned.index.name = "Index"

## calculate difference to tuned AE
tune_train_both = df_results_untuned.index.intersection(df_results["Config_ID"].str.replace("_tune_", "_train_"))
# print(df_results_untuned)
r2_untuned = df_results_untuned.loc[
    # df_results["Config_ID"].str.replace("_tune_", "_train_"), "R2_valid"
    tune_train_both, "R2_valid"
]

r2_untuned.index = r2_untuned.index.str.replace("_train_", "_tune_")

df_results.loc[:, "R2_valid_untuned"] = r2_untuned
df_results.loc[:, "R2_valid_diff"] = (
    df_results.loc[:, "R2_valid"] - df_results.loc[:, "R2_valid_untuned"]
)
df_results["R2_valid_diff"] = df_results["R2_valid_diff"].astype(float)

#
recon_untuned = df_results_untuned.loc[
    # df_results["Config_ID"].str.replace("_tune_", "_train_"), "Rec. loss"
    tune_train_both, "Rec. loss"
]

recon_untuned.index = recon_untuned.index.str.replace("_train_", "_tune_")

df_results.loc[:, "Rec. loss_untuned"] = recon_untuned
df_results.loc[:, "Rec. loss improvement"] = -1 * (
    df_results.loc[:, "Rec. loss"] - df_results.loc[:, "Rec. loss_untuned"]
)  # change sign direction
df_results["Rec. loss improvement"] = df_results["Rec. loss improvement"].astype(float)

#
total_untuned = df_results_untuned.loc[
    # df_results["Config_ID"].str.replace("_tune_", "_train_"), "Total loss"
    tune_train_both, "Total loss"
]

total_untuned.index = total_untuned.index.str.replace("_train_", "_tune_")

df_results.loc[:, "Total loss_untuned"] = total_untuned
df_results.loc[:, "Total loss improvement"] = -1 * (
    df_results.loc[:, "Total loss"] - df_results.loc[:, "Total loss_untuned"]
)  # change sign direction
df_results["Total loss improvement"] = df_results["Total loss improvement"].astype(
    float
)


# print(df_results)

ml_results_untuned.rename(columns={"Performance": "Performance_untuned"}, inplace=True)
ml_results_untuned["Config_ID"] = ml_results_untuned["Config_ID"].str.replace(
    "_train_", "_tune_"
)


ml_results = ml_results.merge(
    ml_results_untuned,
    how="left",
    on=[
        "Config_ID",
        "Architecture",
        "Data_Modalities",
        "Latent_Dim",
        "ML_Algorithm",
        "Parameter",
        "Metric",
        "Split",
        "Data_set",
    ],
)
ml_results["Perf. improvement"] = (
    ml_results["Performance"] - ml_results["Performance_untuned"]
)

## plot improvement
print("make plot tuning R2 improvement")
p_recon_diff = (
    so.Plot(df_results, y="Architecture", x="R2_valid_diff", color="Latent_Dim")
    .facet(col="Data_Modalities", wrap=7, order=dm_order)
    .add(so.Dot(), so.Dodge(), fill="Data_set", marker="Latent_Dim")
    .scale(
        x=so.Continuous(trans="sqrt").tick(at=[-1.0, -0.5, -0.1, 0, 0.1, 0.5, 1.0]),
        color=so.Nominal(),
        y=so.Nominal(order=arch_order),
        marker=["o", "^", "*"],
    )
    .layout(size=(20, 5))
    .label(legend=None)
    .limit(x=(-1.1, 1.1))
)


p_recon_diff.save(
    rootsave + "Exp2_Supp_R2improvement-detailed" + output_type, bbox_inches="tight"
)

p_recon_diff2 = (
    so.Plot(df_results, y="Architecture", x="R2_valid_diff", color="Latent_Dim")
    .add(
        so.Dash(color="0.3"),
        so.Agg(),
        so.Dodge(),
    )
    .add(so.Range(color="0.3"), so.Est(errorbar="sd"), so.Dodge())
    .add(so.Dot(), so.Dodge(), so.Jitter(), marker="Latent_Dim")
    .scale(
        x=so.Continuous(trans="sqrt").tick(at=[-1.0, -0.5, -0.1, 0, 0.1, 0.5, 1.0]),
        color=so.Nominal(),
        y=so.Nominal(order=arch_order),
        marker=["o", "^", "*"],
    )
    .layout(size=(7, 5))
    .limit(x=(-1.1, 1.1))
)


p_recon_diff2.save(
    rootsave + "Exp2_Supp_R2improvement" + output_type, bbox_inches="tight"
)

print("make plot tuning Recon improvement")
p_recon_diff3 = (
    so.Plot(df_results, y="Architecture", x="Rec. loss improvement", color="Latent_Dim")
    .facet(col="Data_Modalities", wrap=7, order=dm_order)
    .add(so.Dot(), so.Dodge(), fill="Data_set", marker="Latent_Dim")
    .scale(
        x=so.Continuous(),
        color=so.Nominal(),
        y=so.Nominal(order=arch_order),
        marker=["o", "^", "*"],
    )
    .layout(size=(20, 5))
    .label(legend=None)
)


p_recon_diff3.save(
    rootsave + "Exp2_Supp_reconlossimprovement-detailed" + output_type,
    bbox_inches="tight",
)

p_recon_diff4 = (
    so.Plot(df_results, y="Architecture", x="Rec. loss improvement", color="Latent_Dim")
    .add(
        so.Dash(color="0.3"),
        so.Agg(),
        so.Dodge(),
    )
    .add(so.Range(color="0.3"), so.Est(errorbar="sd"), so.Dodge())
    .add(so.Dot(), so.Dodge(), so.Jitter(), marker="Latent_Dim")
    .scale(
        x=so.Continuous(),
        color=so.Nominal(),
        y=so.Nominal(order=arch_order),
        marker=["o", "^", "*"],
    )
    .layout(size=(7, 5))
)


p_recon_diff4.save(
    rootsave + "Exp2_Supp_reconlossimprovement" + output_type, bbox_inches="tight"
)

p_recon_diff5 = (
    so.Plot(
        df_results, y="Architecture", x="Total loss improvement", color="Latent_Dim"
    )
    .add(
        so.Dash(color="0.3"),
        so.Agg(),
        so.Dodge(),
    )
    .add(so.Range(color="0.3"), so.Est(errorbar="sd"), so.Dodge())
    .add(so.Dot(), so.Dodge(), so.Jitter(), marker="Latent_Dim")
    .scale(
        x=so.Continuous(),
        color=so.Nominal(),
        y=so.Nominal(order=arch_order),
        marker=["o", "^", "*"],
    )
    .layout(size=(7, 5))
)


p_recon_diff5.save(
    rootsave + "Exp2_Fig4A_totallossimprovement" + output_type, bbox_inches="tight"
)
# print(ml_results.dtypes)
# print(df_results.dtypes)
print("make plot tuning ML improvement")

p_ml_diff = (
    so.Plot(
        ml_results.loc[ml_results["ML_Algorithm"] == sel_ml_alg, :],
        y="Architecture",
        x="Perf. improvement",
        color="Latent_Dim",
    )
    .facet(col="Data_Modalities", row="Metric", order={"col": dm_order})
    .add(
        so.Dash(color="0.3"),
        so.Agg(),
        so.Dodge(),
    )
    .add(so.Range(color="0.3"), so.Est(errorbar="sd"), so.Dodge())
    .add(
        so.Dots(alpha=0.9),
        so.Dodge(),
        so.Jitter(),
        fill="Data_set",
        marker="Latent_Dim",
    )
    .scale(
        x=so.Continuous(trans="sqrt").tick(at=[-1.0, -0.5, -0.1, 0, 0.1, 0.5, 1.0]),
        color=so.Nominal(),
        y=so.Nominal(order=arch_order),
        marker=["o", "^", "*"],
    )
    .layout(size=(20, 10))
    # .label(legend=Non)
    .limit(x=(ml_results["Perf. improvement"].quantile(q=0.01), ml_results["Perf. improvement"].quantile(q=0.99)))
)


p_ml_diff.save(
    rootsave + "Exp2_Supp_MLimprovement-detailed" + output_type, bbox_inches="tight"
)

p_ml_diff2 = (
    so.Plot(
        ml_results.loc[ml_results["ML_Algorithm"] == sel_ml_alg, :],
        y="Architecture",
        x="Perf. improvement",
        color="Latent_Dim",
    )
    .facet(col="Metric")
    .add(
        so.Dash(color="0.3"),
        so.Agg(),
        so.Dodge(),
    )
    .add(so.Range(color="0.3"), so.Est(errorbar="sd"), so.Dodge())
    .add(so.Dots(alpha=0.1), so.Dodge(), so.Jitter(), marker="Latent_Dim")
    .scale(
        x=so.Continuous(trans="sqrt").tick(at=[-1.0, -0.5, -0.1, 0, 0.1, 0.5, 1.0]),
        color=so.Nominal(),
        y=so.Nominal(order=arch_order),
        marker=["o", "^", "*"],
    )
    .layout(size=(10, 5))
    .limit(x=(ml_results["Perf. improvement"].quantile(q=0.01), ml_results["Perf. improvement"].quantile(q=0.99)))
)


p_ml_diff2.save(
    rootsave + "Exp2_Fig4B_MLimprovement" + output_type, bbox_inches="tight"
)

## plot parameter distribution

df_results_tuning = df_results.melt(
    id_vars=[
        "Config_ID",
        "Architecture",
        "Data_Modalities",
        "Latent_Dim",
        "Latent_Coverage",
    ],
    value_vars=["weight_decay", "dropout_all", "encoding_factor", "lr"],
    var_name="Parameter",
    value_name="Value",
)
df_results_tuning["Value"] = df_results_tuning["Value"].apply(
    pd.to_numeric, args=("coerce",)
)

p_hyperparam = (
    so.Plot(df_results_tuning, y="Architecture", x="Value", color="Latent_Dim")
    .facet(col="Parameter")
    .share(x=False, y=True)
    .add(so.Dash(), so.Agg(), so.Dodge())
    .add(so.Range(), so.Est(errorbar="sd"), so.Dodge())
    .add(so.Dots(alpha=0.1), so.Dodge(), so.Jitter(), marker="Latent_Dim")
    .scale(
        x=so.Continuous(),
        y=so.Nominal(order=arch_order),
        color=so.Nominal(),
        marker=["o", "^", "*"],
    )
    .layout(size=(14, 5))
    .theme({"axes.formatter.limits": [-2, 3]})
)
p_hyperparam_log = (
    so.Plot(df_results_tuning, y="Architecture", x="Value", color="Latent_Dim")
    .facet(col="Parameter")
    .share(x=False, y=True)
    .add(so.Dash(), so.Agg(), so.Dodge())
    .add(so.Range(), so.Est(errorbar="sd"), so.Dodge())
    .add(so.Dots(alpha=0.1), so.Dodge(), so.Jitter(), marker="Latent_Dim")
    .scale(
        x=so.Continuous(trans="log"),
        y=so.Nominal(order=arch_order),
        color=so.Nominal(),
        marker=["o", "^", "*"],
    )
    .layout(size=(14, 5))
    .theme({"axes.formatter.limits": [-2, 3]})
)
p_hyperparam.save(
    rootsave + "Exp2_Fig4C_hyperparam-linear" + output_type, bbox_inches="tight"
)
p_hyperparam_log.save(
    rootsave + "Exp2_Fig4C_hyperparam-log" + output_type, bbox_inches="tight"
)


## save df with results used for plotting
df_results.to_csv(rootsave + "df_results.txt", sep="\t")
ml_results.to_csv(rootsave + "ml_results.txt", sep="\t")
