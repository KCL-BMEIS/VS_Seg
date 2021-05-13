# Data preprocessing
This readme explains how to convert the TCIA data set into a data set with a more convenient folder structure and file 
names and how to subsequently convert the DICOM images and JSON contour lines into NIFTI format.

##  Requirements

The following setup has been tested on Ubuntu 20.04.

* 3D Slicer 4.13 or later (https://download.slicer.org/). 
* SlicerRT (http://slicerrt.github.io/Download.html)
  

* Download the following raw data from https://doi.org/10.7937/TCIA.9YTJ-5Q73:
  * Images and Radiation Therapy Structures (DICOM, 26 GB) in "Descriptive Directory Name" format
  * Contours (JSON, zip, 16.7 MB)
  * Registration Matrices (.tfm, zip, 257 KB)

## Create data set with convenient folder structure:
The folder structure and file names of the "Images and Radiation Therapy Structures" are not intuitive. 
Therefore, run the following command to convert the TCIA data set in "Descriptive Directory Name" format into a new folder
structure: 

```python3 TCIA_data_convert_into_convenient_folder_structure.py  --input <input_folder> --output <output_folder>```

The new folder structure will have sub-folders called `vs_gk_<subject_number>_<modality>`.

The script has the following dependencies:

* pydicom (`pip install pydicom`)
* natsort (`pip install natsort`)

This folder structure corresponds to the folder structure of the "Contours" and "Registration Matrices" downloaded from 
TCIA. 

Next, manually merge the files from "Contours" and "Registration Matrices" into the new folder structure. The merged dataset will look like this:

<img src="figures/TCIA_convenient_folder_structure.png" width="600" height="200">

This folder structure is required as the input for the conversion script, which will produce (registered) NIFTI images and segmentations. 

## Conversion of DICOM images and contours.json files to NIFTI and (optional) registration 

To convert the DICOM images into NIFTI images and the planar contour lines of the tumour segmentation from the 
contours.json files into binary segmentations in NIFTI format, run the following command in the VS_Seg repository:

``` <SLICER_DIR>/Slicer --python-script preprocessing/data_conversion.py --input-folder <INPUT_DATA_FOLDER> --output-folder <OUTPUT_DATA_FOLDER> ```

where:
* The 3DSlicer archive has been unpacked at <SLICER_DIR>.

description:

`````--input-folder <INPUT_DATA_FOLDER>  `````
* <INPUT_DATA_FOLDER> is a path to a folder containing sub-folders named `vs_gk_<case_number>_t1`,
                                and `vs_gk_<case_number>_t2`, which contain image files in DICOM format, and the
                                contours.json file and registration matrices in .tfm format. 
                                This corresponds to the output of the script 
                                "TCIA_data_convert_into_convenient_folder_structure.py" described in the previous 
                                section.  
  
`````--output-folder <OUTPUT_DATA_FOLDER>`````
* <OUTPUT_DATA_FOLDER> is the path to the folder where the NIFTI files are supposed to be saved

`--register` ... optional keyword:
* options:
    * if not used, no registration will be performed. The T1 and T2 image will be exported as
vs_gk_t1_refT1.nii.gz and vs_gk_t2_refT2.nii.gz . The tumour segmentations will be exported as
vs_gk_seg_refT1.nii.gz with the dimensions of the T1 image and vs_gk_seg_refT2.nii.gz with the
dimensions of the T2 image.
   
    * `--register T1`: The T2 image will be registered to the T1 image. The exported image files will be named
                vs_gk_t1_refT1.nii.gz and vs_gk_t1_refT1.nii.gz. Only one segmentation with the dimensions of the T1
                image will be exported, named vs_gk_seg_refT1.nii.gz
      
    * `--register T2`: The T1 image will be registered to the T2 image. The exported image files will be named
                vs_gk_t1_refT2.nii.gz and vs_gk_t1_refT2.nii.gz. Only one segmentation with the dimensions of the T2
                image will be exported, named vs_gk_seg_refT2.nii.gz
      
`--export_all_structures` ... optional keyword:
* if used, all structures in the contours.json file will be exported, not
                            only the tumour. The exported structures will be named
                            vs\_gk\_struc<structure_index>\_<structure_name>\_refT1.nii.gz and
                            vs\_gk\_struc<structure_index>\_<structure_name>\_refT2.nii.gz where <structure_index> refers to
                            the order of the structures in the contours.json file (starting from 1) and <structure_name>
                            is the name of the structure as specified in the contours.json file. If `--register T1` or `--register T2` is used, only one of the two files is exported.
