# middle-axis-estimations

This repository contains an algorithm for finding the most symmetric axis (w.r.t. the distance transform) between two points in 3D image volumes.

## Installation
- Clone this repo:
```bash
git clone git@github.com:uhlmanngroup/middle-axis-estimations.git
```
- Install dependencies via [[conda]](https://www.anaconda.com/): `conda env create -f environment.yml`

## Usage

Activate conda environment

```bash
conda activate ma_estimations
```

### Processing

For processing the data make sure that all the files/image-volumes are in one folder
```bash
- data_directory
 - file1
 - file2
 - file3
 ...
```

Make sure that the endpoints-csv file contains the following fields and corresponding data
entry for each file you want to process
```bash
File_name,Point,Position X,Position Y,Position Z,X_px_size_um,Y_px_size_um,Z_px_size_um,M,num_pole_pts
```
Example:
```bash
File_name,Point,Position X,Position Y,Position Z,X_px_size_um,Y_px_size_um,Z_px_size_um,M,num_pole_pts
C2-2020_02_07_E1_BM_thr_fill_iso,1,308.50,508.96,261.52,0.2306294,0.2306294,0.2306294,15,10
C2-2020_02_07_E1_BM_thr_fill_iso,2,757.60,594.40,293.50,0.2306294,0.2306294,0.2306294,15,10
```
The same applies for the midpoint-csv file which should have the following fields and data entries
```bash
File_name,Point,Position X,Position Y,Position Z,X_px_size_um,Y_px_size_um,Z_px_size_um,
```
Example:
```bash
File_name,Point,Position X,Position Y,Position Z,X_px_size_um,Y_px_size_um,Z_px_size_um,
C2-2020_02_07_E1_BM_thr_fill_iso,1,531,308,243,0.2306294,0.2306294,0.2306294,
C2-2020_02_07_E1_BM_thr_fill_iso,2,519,683,240,0.2306294,0.2306294,0.2306294,
```

For processing the data run:
```bash
python main.py --vol_dir path_to_directory_with_tif_files --endpoints_file path_to_csv_file_with_endpoints --midpoints_file path_to_csv_file_with_midpoints --zoom float --save_to path_to_save_file+name
```

Example:
```bash
python main.py --vol_dir 'example_data' --endpoints_file 'example_data/D3SPEG-RGD_endpoints.csv' --midpoints_file 'example_data/D3SPEG-RGD_midpoints.csv' --zoom 0.3 --save_to 'example_data/D3SPEG-RGD_descriptors.csv'
```
