#!/bin/bash

# Function to display usage instructions
usage() {
    echo "Usage: $0 -r <run_ids> [-d] [-k]"
    echo "  -r  Comma-separated list of RUN_IDs (required)"
    echo "  -d  Delete files instead of archiving (optional)"
    echo "  -k  Keep report files, do not archive or delete (optional)"
    exit 1
}


run_ids=""
delete=false
keep_report=false

# Parse command-line options
while getopts ":r:dk" opt; do
    case $opt in
        r) run_ids=$OPTARG ;;
        d) delete=true ;;
        k) keep_report=true ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
    esac
done

# Check if RUN_IDs were provided
if [ -z "$run_ids" ]; then
    echo "RUN_IDs are required."
    usage
fi

# Main processing loop for each RUN_ID
for run_id in ${run_ids//,/ }
do
    processed=true
    interim=true
    reports=true
    models=true

    echo "RUN_ID: $run_id"

    # Check if RUN_ID exists in various directories
    if [ ! -d "data/processed/$run_id" ]; then
        echo "RUN_ID $run_id does not exist in processed folder"
        processed=false
    fi

    if [ ! -d "data/interim/$run_id" ]; then
        echo "RUN_ID $run_id does not exist in interim folder"
        interim=false
    fi

    if [ ! -d "reports/$run_id" ]; then
        echo "RUN_ID $run_id does not exist in reports folder"
        reports=false
    fi

    if [ ! -d "models/$run_id" ]; then
        echo "RUN_ID $run_id does not exist in models folder"
        models=false
    fi

    # Create archive directories if they don't exist
    if [ ! -d "archive" ]; then
        echo "Creating archive folder"
        mkdir -p archive/processed archive/interim archive/reports archive/models
    fi

    # Archive or delete files based on flags and existence
    if [ "$processed" = true ]; then
        if [ "$delete" = true ]; then
            echo "Deleting processed files for RUN_ID $run_id"
            rm -rf data/processed/$run_id
        else
            echo "Archiving processed files for RUN_ID $run_id"
            zip -r data/processed/$run_id.zip data/processed/$run_id
            mv data/processed/$run_id.zip archive/processed/
            rm -rf data/processed/$run_id
        fi
    fi

    if [ "$interim" = true ]; then
        if [ "$delete" = true ]; then
            echo "Deleting interim files for RUN_ID $run_id"
            rm -rf data/interim/$run_id
        else
            echo "Archiving interim files for RUN_ID $run_id"
            zip -r data/interim/$run_id.zip data/interim/$run_id
            mv data/interim/$run_id.zip archive/interim/
            rm -rf data/interim/$run_id
        fi
    fi

    if [ "$keep_report" = true ]; then 
        reports=false
    fi

    if [ "$reports" = true ]; then
        if [ "$delete" = true ]; then
            echo "Deleting reports files for RUN_ID $run_id"
            rm -rf reports/$run_id
        else
            echo "Archiving reports files for RUN_ID $run_id"
            zip -r reports/$run_id.zip reports/$run_id
            mv reports/$run_id.zip archive/reports/
            rm -rf reports/$run_id
        fi
    fi

    if [ "$models" = true ]; then
        if [ "$delete" = true ]; then
            echo "Deleting models files for RUN_ID $run_id"
            rm -rf models/$run_id
        else
            echo "Archiving models files for RUN_ID $run_id"
            zip -r models/$run_id.zip models/$run_id
            mv models/$run_id.zip archive/models/
            rm -rf models/$run_id
        fi
    fi
done
