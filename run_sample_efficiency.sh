#!/bin/bash

# Usage:
# bash run_metrics_over_subsets.sh /path/to/folder1 /path/to/folder2 "10 50 100"
# If no size list is given, it defaults to a preset list

FOLDER1=$1
FOLDER2=$2
SIZES_STR=$3

# Default list if none is provided
if [ -z "$SIZES_STR" ]; then
    SIZES=(10 50 100 500 1000 2000 3000)
else
    read -ra SIZES <<< "$SIZES_STR"  # Convert space-separated string into array
fi

for N in "${SIZES[@]}"; do
    echo "Processing subset size $N..."

    # Create temporary directories
    TMP1="tmp_folder1_$N"
    TMP2="tmp_folder2_$N"
    mkdir -p $TMP1 $TMP2

    # Sample N images randomly into the temp folders
    find "$FOLDER1" -type f | shuf -n $N | xargs -I{} cp "{}" "$TMP1/"
    find "$FOLDER2" -type f | shuf -n $N | xargs -I{} cp "{}" "$TMP2/"

    # Run your metric computation script
    bash compute_allmetrics.sh "$TMP1" "$TMP2"

    # Optional: delete the temp folders
    rm -r "$TMP1" "$TMP2"
done
