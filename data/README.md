# Datasets

The experiment code never downloads anything. To fetch all raw files once, run from the project root
(on Windows use Git Bash):

```bash
bash data/download_uci.sh
```

or download them manually and put the original files in `data/raw/<folder>/` exactly as named below.
Iris and Wine fall back to the copies bundled with scikit-learn if their files are absent.

Check what is available with:

```bash
python scripts/check_datasets.py
```

| # | Key | Dataset | Put these files in | n | d | k | Source |
|---|---|---|---|---|---|---|---|
| 1 | `iris` | Iris | `data/raw/iris/` `iris.data` | 150 | 4 | 3 | [UCI](https://archive.ics.uci.edu/dataset/53/iris) |
| 2 | `wine` | Wine | `data/raw/wine/` `wine.data` | 178 | 13 | 3 | [UCI](https://archive.ics.uci.edu/dataset/109/wine) |
| 3 | `new_thyroid` | New-Thyroid | `data/raw/thyroid/` `new-thyroid.data` | 215 | 5 | 3 | [UCI](https://archive.ics.uci.edu/dataset/102/thyroid+disease) |
| 4 | `glass` | Glass | `data/raw/glass/` `glass.data` | 214 | 9 | 6 | [UCI](https://archive.ics.uci.edu/dataset/42/glass+identification) |
| 5 | `breast_cancer` | Breast Cancer (Original) | `data/raw/breast_cancer_wisconsin/` `breast-cancer-wisconsin.data` | 699 | 9 | 2 | [UCI](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) |
| 6 | `soybean_small` | Soybean (Small) | `data/raw/soybean/` `soybean-small.data` | 47 | 35 | 4 | [UCI](https://archive.ics.uci.edu/dataset/91/soybean+small) |
| 7 | `ionosphere` | Ionosphere | `data/raw/ionosphere/` `ionosphere.data` | 351 | 34 | 2 | [UCI](https://archive.ics.uci.edu/dataset/52/ionosphere) |
| 8 | `ecoli` | Ecoli | `data/raw/ecoli/` `ecoli.data` | 336 | 7 | 8 | [UCI](https://archive.ics.uci.edu/dataset/39/ecoli) |
| 9 | `yeast` | Yeast | `data/raw/yeast/` `yeast.data` | 1484 | 8 | 10 | [UCI](https://archive.ics.uci.edu/dataset/110/yeast) |
| 10 | `vehicle` | Statlog (Vehicle) | `data/raw/vehicle/` `xa?.dat` | 846 | 18 | 4 | [UCI](https://archive.ics.uci.edu/dataset/149/statlog+vehicle+silhouettes) |
| 11 | `image_segmentation` | Image Segmentation | `data/raw/image_segmentation/` `segmentation.data`, `segmentation.test` | 2310 | 19 | 7 | [UCI](https://archive.ics.uci.edu/dataset/50/image+segmentation) |
| 12 | `dermatology` | Dermatology | `data/raw/dermatology/` `dermatology.data` | 366 | 34 | 6 | [UCI](https://archive.ics.uci.edu/dataset/33/dermatology) |

Notes

- **New-Thyroid** is `new-thyroid.data` inside the *Thyroid Disease* archive.
- **Statlog (Vehicle)** comes as nine files `xaa.dat` … `xai.dat`; put all of them in the folder.
- **Image Segmentation** uses both `segmentation.data` (210 rows) and `segmentation.test` (2100 rows); the text header is skipped automatically.
- ID / sequence-name columns (Glass, Breast Cancer, Ecoli, Yeast) are dropped, `?` is read as missing.
- Preprocessing (`DataPreprocessor`): median imputation → removal of constant columns → z-score. Labels are used only for ARI/NMI.
- The loader checks n, d and k against the table above and prints a warning if a file looks different.
