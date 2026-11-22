#!/usr/bin/env bash
# Downloads the raw UCI and SIPU files used by src/datasets.py into data/raw/.
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

# Synthetic SIPU benchmarks (https://cs.joensuu.fi/sipu/datasets/)
BASE="https://cs.joensuu.fi/sipu/datasets"
for f in birch1.txt birch2.txt birch3.txt b1-gt.pa b2-gt.pa b3-gt.txt; do get birch "$f"; done
get unbalance unbalance.txt
get unbalance unbalance-gt-pa.zip
python -c "import zipfile; zipfile.ZipFile('$RAW/unbalance/unbalance-gt-pa.zip').extractall('$RAW/unbalance')"
for f in dim032.txt dim032.pa dim064.txt dim064.pa; do get dim "$f"; done
echo "Done."
