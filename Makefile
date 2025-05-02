.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = venv-gallia
PYTHON_INTERPRETER = python3
RUN_ID = $1
OLD_RUN_ID = $2
PYV = $(shell $(PYTHON_INTERPRETER) -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
UV_EXISTS := $(shell command -v uv 2> /dev/null)
ifeq ($(UV_EXISTS),)
    @echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
endif
# convert to integer

# HELPERS TO CONTROL PYTHON VERSIONS --------------------------------------------------------
# Function to convert version numbers to a numeric representation
version = $(shell echo "$1" | awk -F. '{ printf("%d%03d%03d%03d\n", $$1,$$2,$$3,$$4); }')
PYV_FORMATED = $(call version,$(PYV))
MIN_PYV = "3.10"
MIN_PYV_FORMATED = $(call version,$(MIN_PYV))
# -------------------------------------------------------------------------------------------

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif


# create configs dir if not exists
#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
requirements: test_environment
	uv pip install -e .
	uv pip install -r requirements.txt
	touch src/utils/__init__.py
	touch src/data/__init__.py
	touch src/features/__init__.py
	touch src/models/__init__.py
	touch src/models/tuning/__init__.py
	touch src/visualization/__init__.py

config:
	mkdir -p data/processed
	mkdir -p data/raw
	mkdir -p data/raw/images
	mkdir -p reports
	mkdir -p reports/figures
	mkdir -p models
	mkdir -p models/tuning/${RUN_ID}
	mkdir -p data/interim/$(RUN_ID)
	mkdir -p data/processed/$(RUN_ID)
	mkdir -p models/$(RUN_ID)
	mkdir -p models/tuned/$(RUN_ID)
	mkdir -p reports/$(RUN_ID)
	mkdir -p reports/$(RUN_ID)/figures

	# check if run_id_config.yaml exists
	# if not, copy config.yaml to run_id_config.yaml
	# if yes copy the existing run_id_config.yaml to reports/$(RUN_ID)/$(RUN_ID)_config.yaml
	if [ ! -f $(RUN_ID)_config.yaml ]; then \
		sed -i '' '/RUN_ID/d' config.yaml; \
		echo "RUN_ID: $(RUN_ID)" >> config.yaml; \
		cp config.yaml reports/$(RUN_ID)/$(RUN_ID)_config.yaml; \
	fi
	if [ -f $(RUN_ID)_config.yaml ]; then \
		sed -i '' '/RUN_ID/d' $(RUN_ID)_config.yaml; \
		echo "RUN_ID: $(RUN_ID)" >> $(RUN_ID)_config.yaml; \
		cp $(RUN_ID)_config.yaml reports/$(RUN_ID)/$(RUN_ID)_config.yaml; \
	fi
	echo "done config"



# Join and format TCGA data sets
join_format_tcga: config
ifneq ("$(wildcard data/raw/data_clinical_formatted.parquet)","")
	echo "Formatted TCGA data detected. Join and format will be skipped. Remove data files in data/raw to avoid skipping."
else
	$(PYTHON_INTERPRETER) src/data/join_tcga.py $(RUN_ID)
	$(PYTHON_INTERPRETER) src/data/format_tcga.py $(RUN_ID)
	echo "done join and format tcga"
endif

## Calculate the PI score
pi_score:
ifneq ("$(wildcard data/raw/PIscore_matrix_withCNA.parquet)","")
	echo "PI-Score already prepared. Skipping prep."
else
	# make the folder to store the ancillary data
	mkdir -p data/external

	# download the zip archive containing CDS length data, reactome pathway-gene sets etc.
	wget -P data/external https://cloud.scadsai.uni-leipzig.de/index.php/s/i2FFoi2jojBfwc4/download/piscore_external_data.zip

	# unzip the archive
	unzip data/external/piscore_external_data.zip -d data/external

	# remove the archive upon extraction
	rm data/external/piscore_external_data.zip

	$(PYTHON_INTERPRETER) src/features/get_PIscores.py $(RUN_ID)
	echo "Done calculating PI score"

	# remove the folder and the ancillary data after completing the calculation
	rm -r data/external
endif

combiscore_MUT_CNA:
ifneq ("$(wildcard data/raw/data_combi_MUT_CNA_formatted.parquet)","")
	echo "Combi-Score already prepared. Skipping prep."
else
	# make the folder to store the ancillary data
	mkdir -p data/external

	# download the zip archive containing CDS length data, reactome pathway-gene sets etc.
	wget -P data/external https://cloud.scadsai.uni-leipzig.de/index.php/s/i2FFoi2jojBfwc4/download/piscore_external_data.zip

	# unzip the archive
	unzip data/external/piscore_external_data.zip -d data/external

	# remove the archive upon extraction
	rm data/external/piscore_external_data.zip

	$(PYTHON_INTERPRETER) src/features/combine_MUT_CNA.py $(RUN_ID)
	echo "Done calculating Combi-score score"

	# remove the folder and the ancillary data after completing the calculation
	rm -r data/external
endif

ontology_prep:
ifneq ("$(wildcard data/raw/full_ont_lvl1_reactome.txt)","")
	echo "Ontology already prepared. Skipping prep."
else
	wget -P data/raw https://reactome.org/download/current/ReactomePathwaysRelation.txt
	wget -P data/raw https://reactome.org/download/current/NCBI2Reactome_All_Levels.txt
	wget -P data/raw https://reactome.org/download/current/Ensembl2Reactome_All_Levels.txt
	$(PYTHON_INTERPRETER) src/data/make_ontology.py $(RUN_ID)
	rm data/raw/ReactomePathwaysRelation.txt
	rm data/raw/NCBI2Reactome_All_Levels.txt
	rm data/raw/Ensembl2Reactformaome_All_Levels.txt
endif
	echo "done ontology prep"

# Format single cell data sets
format_singlecell: config
	$(PYTHON_INTERPRETER) src/data/format_sc_h5ad.py $(RUN_ID)

## Make Dataset
data: config
	$(PYTHON_INTERPRETER) src/data/make_dataset.py $(RUN_ID)
	echo "done data"

data_only: config
	$(PYTHON_INTERPRETER) src/data/make_dataset.py $(RUN_ID)
	echo "done data only"

model: data
	$(PYTHON_INTERPRETER) src/models/train.py $(RUN_ID)
	echo "Done training"

model_only: config
	$(PYTHON_INTERPRETER) src/models/train.py $(RUN_ID)
	echo "Done training only"

prediction: model
	$(PYTHON_INTERPRETER) src/models/predict.py $(RUN_ID)
	echo "Done predicting"

prediction_only: config
	$(PYTHON_INTERPRETER) src/models/predict.py $(RUN_ID)
	echo "Done predicting only"

visualize: prediction
	$(PYTHON_INTERPRETER) src/visualization/visualize.py $(RUN_ID)
	echo "Done visualizing"

visualize_only: config
	$(PYTHON_INTERPRETER) src/visualization/visualize.py $(RUN_ID)
	echo "Done visualizing only"

train_n_visualize: config
	$(PYTHON_INTERPRETER) src/models/train.py $(RUN_ID)
	$(PYTHON_INTERPRETER) src/models/predict.py $(RUN_ID)
	$(PYTHON_INTERPRETER) src/visualization/visualize.py $(RUN_ID)
	echo "Done training, predicting and visualizing"


ml_task: visualize
	$(PYTHON_INTERPRETER) src/visualization/ml_task.py $(RUN_ID)
	echo "Done ml_task"
ml_task_only: config
	$(PYTHON_INTERPRETER) src/visualization/ml_task.py $(RUN_ID)
	echo "Done Ml task only"

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src


## Set up python interpreter environment
create_environment:
	# curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv $(PROJECT_NAME) --python 3.10
	@echo ">>> New virtualenv created. Activate with:\nsource $(PROJECT_NAME)/bin/activate"


## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')