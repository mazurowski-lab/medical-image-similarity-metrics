#!/bin/bash

# Check for the required arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <IMAGE_FOLDER1> <IMAGE_FOLDER2> [metrics: all|FRD,FID,RadFID,KID,CMMD]"
    exit 1
fi

IMAGE_FOLDER1=$1
IMAGE_FOLDER2=$2
METRICS=${3:-all}

# Convert to uppercase and remove spaces
METRICS=$(echo "$METRICS" | tr '[:lower:]' '[:upper:]' | tr -d ' ')

contains_metric() {
    [[ "$METRICS" == "ALL" || "$METRICS" == *"$1"* ]]
}

if contains_metric "FRD"; then
    echo "FRD:"
    /usr/bin/time -f "Time to compute FRD: %E" python3 analyze_radiomics.py \
        --image_folder1 "$IMAGE_FOLDER1" \
        --image_folder2 "$IMAGE_FOLDER2"
fi

if contains_metric "FID" || contains_metric "RADFID" || contains_metric "KID"; then
    cd src/gan-metrics-pytorch || exit
fi

if contains_metric "FID"; then
    echo "FID:"
    /usr/bin/time -f "Time to compute FID: %E" python3 fid_score.py \
        --true "../../${IMAGE_FOLDER1}" \
        --fake "../../${IMAGE_FOLDER2}"
fi

if contains_metric "RADFID"; then
    echo "RadFID:"
    /usr/bin/time -f "Time to compute RadFID: %E" python3 fid_score.py \
        --true "../../${IMAGE_FOLDER1}" \
        --fake "../../${IMAGE_FOLDER2}" \
        --use-rad-imagenet-features
fi

if contains_metric "KID"; then
    echo "KID:"
    /usr/bin/time -f "Time to compute KID: %E" python3 kid_score.py \
        --true "../../${IMAGE_FOLDER1}" \
        --fake "../../${IMAGE_FOLDER2}" \
        --img-size 256
    cd ../..
fi

if contains_metric "CMMD"; then
    cd src/cmmd-pytorch || exit
    echo "CMMD:"
    /usr/bin/time -f "Time to compute CMMD: %E" python3 main_cmmd.py \
        "../../${IMAGE_FOLDER1}" \
        "../../${IMAGE_FOLDER2}" \
        --batch_size=32 \
        --max_count=30000
    cd ../..
fi
