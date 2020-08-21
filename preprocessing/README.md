# Data preprocessing
Data preprocessing typically includes converting DICOM images and DICOM RTstructures to NIFTI files.


##  Requirements

The following setup has been tested on Ubuntu 20.04.

* 3D Slicer (https://download.slicer.org/)
* SlicerRT (http://slicerrt.github.io/Download.html)

* Download the raw data in <DATA_FOLDER>

        
## DICOM Images + RTStructure conversion to NIFTI files

To preprocess the data, run the following command from a terminal in the VS_Seg repository:

``` <SLICER_DIR>/Slicer --no-main-window --python-script preprocessing/data_conversion.py --input-folder --results_folder_name <DATA_FOLDER> --results_folder_name <OUTPUT_DATA_FOLDER> ```



