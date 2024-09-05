# Create directories if they don't exist
if (-not (Test-Path -Path "data\raw")) {
    New-Item -Path "data\raw" -ItemType Directory
}

# Check if the TCGA data file exists and download if not
if (Test-Path "data\raw\data_clinical_formatted.parquet") {
    Write-Output "TCGA data detected in data\raw. Starting the make command"
} else {
   Write-Output "Downloading TCGA data does not exist in data\raw. Please download the data as described in the README and place it in data\raw"
}

# Check if the ontology data file exists and download if not
if (Test-Path "data\raw\full_ont_lvl1_reactome.txt") {
    Write-Output "Ontology data detected in data\raw. Starting the make command"
} else {
    Write-Output "Downloading ontology data does not exist in data\raw. Please download the data as described in the README and place it in data\raw"
}

# Execute make commands
& make ml_task RUN_ID=TCGAexample
& make ml_task RUN_ID=TuningExample
