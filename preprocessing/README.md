# Data preprocessing
Data preprocessing typically includes converting DICOM images and DICOM RTstructures to NIFTI files.


##  Requirements

The following setup has been tested on Ubuntu 20.04.

* 3D Slicer (https://download.slicer.org/). Unpack the archive in <SLICER_DIR>.
* SlicerRT (http://slicerrt.github.io/Download.html)

* Download the raw data in <DATA_FOLDER> which contains the patient-specific subfolders (<DATA_FOLDER>/<PATIENT_ID>/*.dcm).

        
## DICOM Images + RTStructure conversion to NIFTI files

To preprocess the data, run the following command from a terminal in the VS_Seg repository:

``` <SLICER_DIR>/Slicer --no-main-window --python-script preprocessing/data_conversion.py --input-folder <DATA_FOLDER> --results_folder_name <OUTPUT_DATA_FOLDER> ```

The script iterates among the list of <PATIENT_ID>.

The <OUTPUT_DATA_FOLDER> has the same structure as <DATA_FOLDER>, i.e for each <PATIENT_ID>:
* NIFTI scan will be located at: <OUTPUT_DATA_FOLDER>/<PATIENT_ID>/<PATIENT_ID>.nii.gz
* Structures will be located at: <OUTPUT_DATA_FOLDER>/<PATIENT_ID>/<PATIENT_ID>_<STRUCTURE>.nii.gz


