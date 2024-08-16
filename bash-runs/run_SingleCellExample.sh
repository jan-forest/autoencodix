#!/bin/bash
mkdir -p data/raw
if [ -f "data/raw/scATAC_human_cortex_formatted.parquet" ]; then
	echo "Single Cell human cortex data detected in data/raw. Download will be skipped"
else
	wget -nv -P data/raw https://cloud.scadsai.uni-leipzig.de/index.php/s/qqKYA9fkxAmPkje/download/sc_human_cortex.zip
	unzip data/raw/sc_human_cortex.zip -d data/raw
	rm data/raw/sc_human_cortex.zip
	make format_singlecell RUN_ID=scExample
fi

if [ -f "data/raw/full_ont_lvl1_ensembl_reactome.txt" ]; then
	echo "Ontology data detected in data/raw. Download will be skipped"
else
	wget -nv -P data/raw https://cloud.scadsai.uni-leipzig.de/index.php/s/ACCq8grxYT7SZw9/download/ontologies.zip
	unzip data/raw/ontologies.zip -d data/raw

	rm data/raw/ontologies.zip
fi


make ml_task RUN_ID=scExample
