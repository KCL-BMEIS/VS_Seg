# Data preprocessing
Data preprocessing typically includes converting DICOM images and DICOM RTstructures to NIFTI files.


##  Requirements

The following setup has been tested on Ubuntu 20.04.

* 3D Slicer 4.13 or later (https://download.slicer.org/). 
* SlicerRT (http://slicerrt.github.io/Download.html)
* Download the raw data. (Dicom images and contours.json)

        
## DICOM Images + contours.json conversion to NIFTI files

To preprocess the data, run the following command in the VS_Seg repository:

``` <SLICER_DIR>/Slicer --python-script preprocessing/data_conversion.py --input-folder <DATA_FOLDER> --results_folder_name <OUTPUT_DATA_FOLDER> ```
where:
* The 3DSlicer archive has been unpacked at <SLICER_DIR>.
* <DATA_FOLDER> denotes the folder that contains all the patient-specific subfolders (i.e. <DATA_FOLDER>/<PATIENT_ID>/*.dcm).

description:

--input-folder <DATA_FOLDER>  
* <DATA_FOLDER> is a path to a folder containing sub-folders named `vs_gk_<case_number>_t1`,
                                and `vs_gk_<case_number>_t2`, which contain image files in DICOM format and the
                                contours.json file 

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
                            vs_gk_struc<structure_index>_<structure_name>_refT1.nii.gz and
                            vs_gk_struc<structure_index>_<structure_name>_refT2.nii.gz where <structure_index> refers to
                            the order of the structures in the contours.json file (starting from 1) and <structure_name>
                            is the name of the structure as specified in the contours.json file. If `--register T1` or `--register T2` is used, only one of the two files is exported.
