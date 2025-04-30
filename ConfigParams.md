# Config file parameters



## External config:

\_config.yaml

### DATA DEFINITIONS

#### <code>DATA_TYPE</code>

> **Description:** This variable specifies the details for each data type.The following details can be specified:
>
> > `FILE_RAW`: path of raw input data file
> >
> > `TYPE`: data modality of this file: MIXED, IMG, NUMERIC, ANNOTATION are possible
> >
> > `SCALING`: Method for scaling the input data. This depends on your data and loss function. Usually the data should be scaled between 0 and 1 for BCE Loss, so we recommend "MinMax" here. Other possible options are: "Standard" (good with MSE loss), "Robust", "MaxAbs" via scikit-learn.
> >
> > `FILTERING`: Method to use for filtering the input data. You can filter the data based on Variance "Var", Correlation "Corr, or both"Var+Corr". Alternatively to standard variance you can filter also by Median Absolute Deviation "MAD". You can also use no filtering "NoFilt" or only features with no variance at all "NonZeroVar". Filtering by correlation uses clustering of features based correlation as distance metric and KMedoids. This is benefecial when having many highly correlated features (with high variance). However, it drastically increases runtime and is recommended when `K_FILTER<1000`.
>
> > Example:
>
> > \`\`\`yaml
> >
> > DATA_TYPE:
> >
> > RNA:
> >
> > FILE_RAW: "data_mrna_seq_v2_rsem_formatted.txt" #path of raw input data file
> >
> > TYPE: "NUMERIC" \# MIXED, IMAGE, NUMERIC
> >
> > SCALING: "MinMax"
> >
> > FILTERING: "Var" \# NoFilt, Var Corr, Var+Corr
> >
> > \`\`\`
>
> **Default:** Not defined.
>
> **Type:** dictionary
>
> **NOTE** If you want to use clinical features, please provide the numeric and categorial columns in your clinical file. Like:
>
> > \`\`\`yaml
> >
> > CLIN:
> >
> > FILE_RAW: "data_clinical_formatted.txt"
> >
> > TYPE: "MIXED"
> >
> > REL_CLIN_C:
> >
> > -   "SUBTYPE"
> >
> > <!-- -->
> >
> > -   "ONCOTREE_CODE"
> >
> > <!-- -->
> >
> > -   "GRADE"
> >
> > REL_CLIN_N:
> >
> > -   "AGE"
> >
> > <!-- -->
> >
> > -   "WEIGHT"
> >
> > SCALING: "MinMax"
> >
> > \`\`\`

TODO: Add examples for IMG and ANNO?


#### <code>K_FILTER</code>

> **Description:** This variable specifies the number of features to use after filtering per data modality specified in DATA_TYPE.
>
> **Default:** 100
>
> **Type:** integer



### MODEL AND TRAINING:

#### <code>TRAIN_TYPE</code>

> **Description:** This variable specifies the type of training to perform. The options are `train` and `tune`. For training the weights of the given Autoencoder architecture use `train`. If you want to find the best architecture and hyperparameters use `tune`. For training we use the training data set. For tuning the valid data set (for finding hyperparameters and architecture) and for fitting the weights of the tuned architecture, we use theh training data set.
>
> **Default:** "train"
>
> **Type:** string

#### <code>MODEL_TYPE</code>

> **Description:** This variable specifies the type of model to use. The options are `stackix` (hierarchical variational autoencoder) similar to [Simidjievski 2019](https://doi.org/10.3389/fgene.2019.01205), `ontix` (ontology-based autoencoder) comparable to REF, `x-modalix` (cross-modal autoencoder) similar to REF, `varix` (variational autoencoder and `vanillix` (vanilla autoencoder)). To see our implementation go to `./src/models/models.py` or `./src/models/tuning/models_for_tuning.py`
>
> **Default:** "varix"
>
> **Type:** string

#### <code>BETA</code>

> **Description:** This variable specifies the weight to add to the KL divergence or MMD term for all variational autoencoders.
>
> **Default:** 1.0
>
> **Type:** float

#### <code>LATENT_DIM_FIXED</code>

> **Description:** This variable specifies the number of neurons in the latent dimension to use in the model.
>
> **Default:** 8
>
> **Type:** integer

#### <code>BATCH_SIZE</code>

> **Description:** This variable specifies the batch size used during training.
>
> **Default:** 64
>
> **Type:** integer

#### <code>EPOCHS</code>

> **Description:** This variable specifies the number of epochs to use during training.
>
> **Default:** 200
>
> **Type:** integer

#### <code>LR_FIXED</code>

> **Description:** This variable specifies the learning rate to use during training.
>
> **Default:** 0.0001
>
> **Type:** float

#### <code>VAE_LOSS</code>

> **Description:** This variable specifies the loss function for VAE to fit the normal distribution by KL divergence (`KL`) or VAE can be trained with Maximum Mean Discrepancy  (`MMD`) without an assumption about the distribution.
>
> **Default:** "KL"
>
> **Type:** string

#### <code>RECONSTR_LOSS</code>

> **Description:** This variable specifies the reconstruction loss function (`BCE` or `MSE`).
>
> **Default:** "BCE"
>
> **Type:** string

#### <code>PREDICT_SPLIT</code>

> **Description:** This variable specifies the split to use during prediction.The options are `all`, `train`, `test` and `valid`. These variables determines which data split is used to obtain the latent space from the trained model. 
>
> **Default:** "test"
>
> **Type:** string

### Ontix specific


#### <code>NON_ONT_LAYER</code>

> **Description:** This variable specifies the additional layers for ontix to get a latent dimension (`LATENT_DIM_FIXED`) smaller than the given ontology dimension specified in `FILE_ONT_LVL1`, respectively `FILE_ONT_LVL2`. To keep explainability of latent space as ontology layer keep default value `0`.
>
> **Default:** 0
>
> **Type:** integer

#### <code>FILE_ONT_LVL1</code>

> **Description:** This variable specifies the filename with the description of the relationship of features and first ontology level in sparse decoder (in `data/raw` or as specified unter `ROOT_RAW`). Format is that each row contains a relationship with `Feature_ID` *delimiter* `Ontology_ID`. Must be set to use `ontix`.
>
> **Default:** "full_ont_lvl1_reactome.txt"
>
> **Type:** string

#### <code>FILE_ONT_LVL2</code>

> **Description:** This variable specifies the filename with the description of the relationship of first ontology level and a second level (dim(lvl1) \> dim(lvl2)) in sparse decoder. Format is that each row contains a relationship with `OntologyLVL1_ID` *delimiter* `OntologyLVL2_ID`. To deactivate second level, set parameter to `null`.
>
> **Default:** `null`
>
> **Type:** string

### X-Modalix specific


#### <code>TRANSLATE</code>

> **Description:** When using `x-modalix`, this string specifies the translation order for prediction. Data labels as used in `DATA_TYPE` must be used in the format *FROMDATA*_to_*TODATA* 
>
> **Default:** "RNA_to_IMG"
>
> **Type:** string

#### <code>GAMMA</code>

> **Description:** When using `x-modalix` a latent space classifier is trained to discriminate between the given data modalities. To align both latent spaces of each data modalities in the `x-modalix` an adversarial loss is calculated based on the latent space classifier. Parameter `GAMMA` controls weighting of the adversarial loss term in relation to `RECONSTR_LOSS` loss and `VAE_LOSS`.
>
> **Default:** 5
>
> **Type:** float

#### <code>DELTA_PAIR</code>

> **Description:** When using `x-modalix` a supervision loss term can be applied to better align both latent spaces of each data modality for a better translation. If identical samples are measured for both data modalities, a paired loss can be calculated and the distance of the sample is minimized between each data modality in the respective latent space. The value is highly dependent on your data. Try to balance between `RECONSTR_LOSS`,  `VAE_LOSS`, adversarial loss and supervsion loss (class and/or paired). Recommendation for well aligned latent space is that total loss is composed to >50% `RECONSTR_LOSS`, 5-10% `VAE_LOSS`, 15% adversarial loss and 15% supervision loss. 
>
> **Default:** 5
>
> **Type:** float

#### <code>DELTA_CLASS</code>

> **Description:** When using `x-modalix` a supervision loss term can be applied to better align both latent spaces of each data modality for a better translation. If class information of samples is available for both data modalities, the distance of the sample to the class center is minimized in each latent space. The value is highly dependent on your data. Try to balance between `RECONSTR_LOSS`,  `VAE_LOSS`, adversarial loss and supervsion loss (class and/or paired). Recommendation for well aligned latent space is that total loss is composed to >50% `RECONSTR_LOSS`, 5-10% `VAE_LOSS`, 15% adversarial loss and 15% supervision loss. 
>
> **Default:** 5
>
> **Type:** float

#### <code>CLASS_PARAM</code>

> **Description:** Column label specifying the parameter taken from the ANNOTATION data set to calculate class loss for `x-modalix`. Can be set to `null` to deactivate class loss. 
>
> **Default:** null
>
> **Type:** string

#### <code>PRETRAIN_TARGET_MODALITY</code>

> **Description:** For `x-modalix` VAE's can be pretrained before latent spaces will be aligned, which is recommended for images and other complex data modalities. If only Image VAE should be pretrained, use option `pretrain_image` to pretrain for `PRETRAIN_EPOCHS`. Alternatively, smooth annealing of latent space alignment can be realized by option `gamma_anneal` affecting both data modalities. If no pretraining is wanted, set to `null`.
>
> **Default:** "gamma_anneal"
>
> **Type:** string



#### <code>PRETRAIN_EPOCHS</code>

> **Description:** If pretraining is performed, this parameter controls the number of epochs for pretraining. 
>
> **Default:** true
>
> **Type:** integer



#### OPTUNA TUNING VARS:

#### <code>LAYERS_LOWER_LIMIT</code>

> **Description:** This variable specifies the lower limit for the number of layers in the nerual net architecture. Optuna will try to find the best architecture with minimum <LAYERS_LOWER_LIMIT> layers. Should not be smaller than 2.
>
> **Default:** 2
>
> **Type:** integer

#### <code>LAYERS_UPPER_LIMIT</code>

> **Description:** This variable specifies the upper limit for the number of layers to use in tuning. Optuna will try to find the best architecture with maximum <LAYERS_UPPER_LIMIT> layers.
>
> **Default:** 5
>
> **Type:** integer

#### <code>LR_LOWER_LIMIT</code>

> **Description:** This variable specifies the lower limit for the learning rate to use in tuning.
>
> **Default:** 0.0001
>
> **Type:** float

#### <code>LR_UPPER_LIMIT</code>

> **Description:** This variable specifies the upper limit for the learning rate to use in tuning.
>
> **Default:** 0.01
>
> **Type:** float

#### <code>DROPOUT_LOWER_LIMIT</code>

> **Description:** This variable specifies the lower limit for the dropout rate to use in tuning.
>
> **Default:** 0.00
>
> **Type:** float

#### <code>DROPOUT_UPPER_LIMIT</code>

> **Description:** This variable specifies the upper limit for the dropout rate to use in tuning.
>
> **Default:** 0.50
>
> **Type:** float

#### <code>OPTUNA_TRIALS</code>

> **Description:** This variable specifies the number of trials to run during tuning.
>
> **Default:** 25
>
> **Type:** integer


#### EVALUATION AND VISUALIZATION

#### <code>DIM_RED_METH</code>

> **Description:** This variable specifies the dimensionality reduction method to use for visualization. The options `PCA` or `UMAP` or `TSNE` are possible. For large sample sizes `UMAP` and `TSNE` may take some time to calculate. 
>
> **Default:** UMAP
>
> **Type:** string

#### <code>CLUSTER_ALG</code>

> **Description:** This variable specifies the clustering algorithm to cluster latent space after `DIM_RED_METH` and to use for visualization. The options are `KMeans` or `HDBSCAN`.
>
> **Default:** KMeans
>
> **Type:** string

#### <code>CLUSTER_N</code>

> **Description:** This variable specifies the number of clusters k to use for KMeans clustering.
>
> **Default:** 3
>
> **Type:** integer

#### <code>MIN_CLUSTER_N</code>

> **Description:** Minimal number of samples per cluster as crucial parameter for `HDBSCAN`. This is dependent on your total number of samples and cluster sizes you expect. For more details see scikit-learn tutorials and documentation on `HDBSCAN`.
>
> **Default:** 20
>
> **Type:** float

#### <code>CLINIC_PARAM</code>

> **Description:** This variable specifies the clinical parameters to generate visualizations and to evaluate embedding performances in `ML_TASKS`. Defined as list of strings which are column names of parameters in `ANNOTATION` data type. 
>
> **Default:** \["CANCER_TYPE", "TMB_NONSYNONYMOUS"\]
>
> **Type:** list of strings

#### <code>ML_TYPE</code>

> **Description:** For the given `CLINIC_PARAM` either regression or classification ML tasks will be performed. If `auto-detect`, non-string parameters will be assumed to be regression tasks. Otherwise it is recommended to specify as dictionary each column name under `CLINIC_PARAM` if they are `classification` or `regression` task.
>
> **Default:** "auto-detect"
>
> **Type:** string or dictionary

#### <code>ML_ALG</code>

> **Description:** This variable specifies which machine learning algorithms should be used for evaluation. `Linear` (Linear Regression or Logistic Regression), `RF` (maximal depth = 10, minimal samples per leaf = 2), `SVM` (radial kernel) are possible.
>
> **Default:** `Linear` , `RF`
>
> **Type:** list of strings

#### <code>ML_SPLIT</code>

> **Description:** Ths variable specifies whether cross-validation will be performed ("CV-on-all-data") or if the original split should be used ("use-split") for model evaluation.
>
> **Default:** "CV-on-all-data"
>
> **Type:** string



#### <code>ML_TASKS</code>

> **Description:** This variable specifies which dimension reduction methods should be used for the ml task. `Latent`, `UMAP`, `PCA` and `RandomFeature` are possible. For `RandomFeature` performance evaluation will be repeated five times to estimate average and deviation of embedding performance.
>
> **Default:** `Latent`, `PCA` and `RandomFeature`
>
> **Type:** list of strings



## Internal config:

src/000_internal_config.yaml

#### <code>LOGLEVEL</code>

> **Description: ...** Controls logging detail with possible levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`.
>
> **Default:** "INFO"
>
> **Type:** string

### DATA SPECIFICATION

#### <code>ROOT_RAW</code>

> **Description:** This variable specifies the root directory path for raw data.
>
> **Default:** "data/raw"
>
> **Type:** string

#### <code>ROOT_PROCESSED</code>

> **Description:** This variable specifies the root directory path for processed data.
>
> **Default:** "./data/processed/"
>
> **Type:** string

#### <code>ROOT_IMAGE</code>

> **Description:** This variable specifies the root directory path for raw images data.
>
> **Default:** "data/raw/images"
>
> **Type:** string

#### <code>DELIM</code>

> **Description:** This variable specifies the delimiter character used in input and output data files. Input files need to have file extension csv, tsv or txt.
>
> **Default:** "\\t"
>
> **Type:** string

#### <code>SPLIT</code>

> **Description:** This variable specifies the proportions for the train, validate, and test splits.
>
> **Default:** \[0.6, 0.2, 0.2\], which means 60% for training, 20% for validation, and 20% for testing.
>
> **Type:** list of floats

#### <code>SPLIT_FILE</code>

> **Description:** This variable specifies the path to the file containing the generated (use always default path) or pre-computed data split. If you want to give a specific train/test/valid split then you need to provide a file with the sample ID and a column that indicates to which split the sample belongs (train, test, valid). To activate pre-computed split, you must set config option SPLIT to "pre-computed".
>
> **Default:** "data/processed/sample_split.txt"
>
> **Type:** string

#### <code>FORM_SUFFIX</code>

> **Description:** This variable specifies a suffix for the formatted data files when running join_format_tcga or format_singlecell.

> **Default:** "\_formatted.txt"
>
> **Type:** string

### SINGLE CELL
#### <code>MIN_PERC_CELLS</code>

> **Description:** Controls filtering of sparse samples (cells) with less then MIN_GENE percent expressed genes of total gene using scanpy filter_cells 
>
> **Default:** 0.06
>
> **Type:** float

#### <code>MIN_PERC_CELLS</code>

> **Description:**  Keep only genes in single-cell data which are present in at least x fraction of all cells using filter scanpy filter_genes. Filtering is done after sample filtering.
>
> **Default:** 0.2
>
> **Type:** float

#### <code>K_FILTER_SC</code>

> **Description:** After filtering for sparse samples and sparse genes the K_FILTER_SC genes with the highes variables are selected using scanpy highly_variable_genes. Should be higher than K_FILTER in the run config.
>
> **Default:** 4000
>
> **Type:** integer

#### <code>H5AD_FILES</code>

> **Description:** Path to single-cell files in h5ad-format.
>
> **Default:** \["scATAC_human_cortex.h5ad", "scRNA_human_cortex.h5ad"\]
>
> **Type:** list of strings

#### <code>ANNDATA_LAYER</code>

> **Description:** Indicate which data layers in single-cell h5ad files are present and should be used to create formatted files as a single data modality. 'X' will be treated as standard data layer. "obs" as annotation layer. "obs/velocity" can be used to extract RNA velocity as additional data modality.
>
> **Default:** \["X", "obs"\]
>
> **Type:** list of strings

##### TCGA

#### <code>TCGA_FOLDER</code>

> **Description:** This variable specifies the subdirectory path for TCGA data.
>
> **Default:** "\_tcga_pan_can_atlas_2018/"
>
> **Type:** string

#### <code>SUBTYPES</code>

> **Description:** This variable specifies a list of cancer subtypes to use. See list for details https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations. "All" will give all available subtypes.
>
> **Default:** "All"
>
> **Type:** list of strings

#### <code>DATAFILES</code>

> **Description:** This variable specifies a list of data file names to use. This does not need no be changed.
>
> **Default:** \["data_mrna_seq_v2_rsem.txt", "data_log2_cna.txt", "data_methylation_hm27_hm450_merged.txt", "data_clinical_patient.txt"\
> "data_clinical_sample.txt", "data_mutations.txt"\]
>
> **Type:** list of strings

### Model and Training Specification

#### <code>START_FROM_LAST_CHECKPOINT</code>

> **Description:** This variable specifies whether to start training from the last checkpoint. Useful when machine or server crashed during training. Recommended to restart training with make model_only.
>
> **Default:** False
>
> **Type:** boolean

#### <code>WEIGHT_DECAY</code>

> **Description:** This variables specifies the hyperparameter weight decay for training with solver (AdamW).
>
> **Default:** 0.001
>
> **Type:** float

#### <code>DROP_P</code>

> **Description:** Probability of drop out in drop out layer in all autoencoder architectures with drop out. 
>
> **Default:** 0.1
>
> **Type:** float

#### <code>ANNEAL</code>

> **Description:** This variables controls the beta-annealing (VAE-loss) of the training of all variational autoencoders. Options are logistic-early, logistic-mid, logistic-late, 3phase-linear,3phase-log or no-annealing
>
> **Default:** "logistic-mid"
>
> **Type:** string

#### <code>ANNEAL_PRETRAINING</code>

> **Description:** If pretraining is performed, should the VAE loss follow a beta annealing as defined in `ANNEAL` or beta annealing be disabled for pretraining. 
>
> **Default:** true
>
> **Type:** boolean

#### <code>PRUN_PATIENCE</code>

> **Description:** This variables controls the pruning of runs during hyperparameter tuning with optuna. "0" is full pruning (no patience) and stops trials early. "1" leads to no pruning at all since all epochs are considered. Pruning is disabled when beta-annealing is active.
>
> **Default:** 0.05
>
> **Type:** float

#### <code>OPTUNA_TRIAL_TIME</code>

> **Description:** This variable specifies the maximum time in seconds after one trial is cancelled for hyperparamter tuning.
>
> **Default:** 1200
>
> **Type:** integer

#### <code>FIX_RANDOMNESS</code>

> **Description:** This variables controls randomness in Autoencodix. "random" for no fixing, "all" fix all randomness, "data_split" fix train/test/valid split, "training" fix weight initialization and randomness in training; not available for tuning with optuna.
ATTENTION: Set the environment variable CUBLAS_WORKSPACE_CONFIG, when setting FIX_RANDOMNESS to "all", like so: `export CUBLAS_WORKSPACE_CONFIG=:4096:8 `
>
> **Default:** "random"
>
> **Type:** string

#### <code>GLOBAL_SEED</code>

> **Description:** Seed used to fix randomness.
>
> **Default:** 42
>
> **Type:** integer

#### <code>KEEP_NOT_ONT</code>

> **Description:** Should features kept in seperate ontology bucket, which are not part of the provided ontology-file, but are part in the provided training data. 
>
> **Default:** false
>
> **Type:** boolean

#### <code>NORMALIZE_RECONS_LOSS_XMODALIX</code>

> **Description:** Should the reconstruction loss terms for each data modality which are translated by x-modalix be normalized by their respective feature size? Recommended for unbalanced feature sizes, in particular when translating images.
>
> **Default:** true
>
> **Type:** boolean

#### <code>CHECKPT_SAVE</code>

> **Description:** Should intermediate models in training be saved in models folder at checkpoint epochs? Not available for x-modalix
>
> **Default:** false
>
> **Type:** boolean

#### <code>RECON_SAVE</code>

> **Description:** Should the reconstructed values of input features (x) be saved under reports?
>
> **Default:** false
>
> **Type:** boolean

#### <code>WRITE_H5AD</code>

> **Description:** Should the embedding be saved and stored in the given h5ad-files under H5AD_FILES? Will write the embedding into obsm of the anndata object with the name e.g. `X_varix` depending on the trained model type.
>
> **Default:** false
>
> **Type:** boolean

### Visualize control

#### <code>CHECKPT_PLOT</code>

> **Description:** Should the latent space be plotted at checkpoint epochs? Not available for x-modalix
>
> **Default:** false
>
> **Type:** boolean

#### <code>PLOT_INPUT2D</code>

> **Description:** Should a direct embedding of inputlayer performed (using DIM_RED_METH) and be plotted for comparison to autoencoder?
>
> **Default:** false
>
> **Type:** boolean

#### <code>PLOT_WEIGHTS</code>

> **Description:** Should the initial and trained weights be plotted as heatmaps?
>
> **Default:** false
>
> **Type:** boolean

#### <code>PLOT_CLUSTLATENT</code>

> **Description:** Should latent space be clustered and be visualized?
>
> **Default:** false
>
> **Type:** boolean

#### <code>PLOT_NUMERIC</code>

> **Description:** Should numeric parameters in CLIN_PARAM be plotted as continuous (true) or as categories/percentiles (false, default)
>
> **Default:** false
>
> **Type:** boolean

### ml_task control

#### <code>CV</code>

> **Description:** This variable specifies the the number of cross-validation rounds (only if ML_SPLIT: "CV-on-all-data")
>
> **Default:** 5
>
> **Type:** integer

### Test synthetic signal

#### <code>APPLY_SIGNAL</code>

> **Description:** For testing a synthetic signal can be applied to the provided features and samples (see below). Synthetic signal is defined as setting the value of the features to 90% of the maximum feature value. 
> **Default:** false
>
> **Type:** boolean

#### <code>SAMPLE_SIGNAL</code>

> **Description:** Path to file file with list of samples to which signal should be applied. One row per sample. No header.

> **Default:** "data/raw/female_scCells.txt"
>
> **Type:** string

#### <code>FEATURE_SIGNAL</code>

> **Description:** Path to file file with list of features to which signal should be applied. One row per feature. No header.

> **Default:** "data/raw/chrX_ensemblID.txt"
>
> **Type:** string

