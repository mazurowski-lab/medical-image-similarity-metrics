#!/bin/bash

# Check for the required arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <IMAGE_FOLDER1> <IMAGE_FOLDER2>"
    exit 1
fi

IMAGE_FOLDER1=$1
IMAGE_FOLDER2=$2

echo "FRD:"
/usr/bin/time -f "Time for FRD: %E sec" python3 analyze_radiomics.py \
    --image_folder1 "$IMAGE_FOLDER1" \
    --image_folder2 "$IMAGE_FOLDER2"

cd src/gan-metrics-pytorch || exit

echo "FID:"
/usr/bin/time -f "Time for FID: %E sec" python3 fid_score.py \
    --true "../../${IMAGE_FOLDER1}" \
    --fake "../../${IMAGE_FOLDER2}"

echo "RadFID:"
/usr/bin/time -f "Time for RadFID: %E sec" python3 fid_score.py \
    --true "../../${IMAGE_FOLDER1}" \
    --fake "../../${IMAGE_FOLDER2}" \
    --use-rad-imagenet-features

echo "KID:"
/usr/bin/time -f "Time for KID: %E sec" python3 kid_score.py \
    --true "../../${IMAGE_FOLDER1}" \
    --fake "../../${IMAGE_FOLDER2}" \
    --img-size 256

cd ../cmmd-pytorch || exit

echo "CMMD:"
/usr/bin/time -f "Time for CMMD: %E sec" python3 main_cmmd.py \
    "../../${IMAGE_FOLDER1}" \
    "../../${IMAGE_FOLDER2}" \
    --batch_size=32 \
    --max_count=30000

cd ../..
