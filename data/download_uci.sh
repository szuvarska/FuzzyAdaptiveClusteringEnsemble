#!/usr/bin/env bash
# Downloads the raw UCI files used by src/datasets.py into data/raw/.
# Run from the project root:  bash data/download_uci.sh
set -euo pipefail
BASE="https://archive.ics.uci.edu/ml/machine-learning-databases"
RAW="$(cd "$(dirname "$0")" && pwd)/raw"
get() { mkdir -p "$RAW/$1"; echo "-> $1/$(basename "$2")"; curl -fsSL --retry 3 -o "$RAW/$1/$(basename "$2")" "$BASE/$2"; }
get iris                    iris/iris.data
get wine                    wine/wine.data
get thyroid                 thyroid-disease/new-thyroid.data
get glass                   glass/glass.data
get breast_cancer_wisconsin breast-cancer-wisconsin/breast-cancer-wisconsin.data
get soybean                 soybean/soybean-small.data
get ionosphere              ionosphere/ionosphere.data
get ecoli                   ecoli/ecoli.data
get yeast                   yeast/yeast.data
for p in a b c d e f g h i; do get vehicle statlog/vehicle/xa$p.dat; done
get image_segmentation      image/segmentation.data
get image_segmentation      image/segmentation.test
get dermatology             dermatology/dermatology.data
echo "Done."
