#!/bin/bash
mkdir -p data/raw
if [ -f "data/raw/data_clinical_formatted.parquet" ]; then
	echo "TCGA data detected in data/raw. Download will be skipped"
else
	wget -nv -P data/raw https://cloud.scadsai.uni-leipzig.de/index.php/s/mCMCZ8ZY9LX4MQz/download/tcga_formatted.zip
	unzip data/raw/tcga_formatted.zip -d data/raw

	rm data/raw/tcga_formatted.zip
fi

if [ -f "data/raw/full_ont_lvl1_reactome.txt" ]; then
	echo "Ontology data detected in data/raw. Download will be skipped"
else
	wget -nv -P data/raw https://cloud.scadsai.uni-leipzig.de/index.php/s/ACCq8grxYT7SZw9/download/ontologies.zip
	unzip data/raw/ontologies.zip -d data/raw

	rm data/raw/ontologies.zip
fi

make ml_task RUN_ID=TCGAexample
make ml_task RUN_ID=TuningExample