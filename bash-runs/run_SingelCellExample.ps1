# Create directories if they don't exist
if (-not (Test-Path -Path "data\raw")) {
    New-Item -Path "data\raw" -ItemType Directory
}

# Check if the file exists and download if not
if (Test-Path "data\raw\scATAC_human_cortex_formatted.parquet") {
    Write-Output "Single Cell human cortex data detected in data\raw. Starting the make command"
} else {
    Write-Output "Downloading Single Cell human cortex data does not exist in data\raw. Please download the data as described in the README and place it in data\raw"
}

# Check if the file exists and download if not
if (Test-Path "data\raw\full_ont_lvl1_ensembl_reactome.txt") {
    Write-Output "Ontology data detected in data\raw. Starting the make command"
} else {
    Write-Output "Downloading ontology data does not exist in data\raw. Please download the data as described in the README and place it in data\raw"
}

# Call the 'make' command
& make ml_task RUN_ID=scExample
