# How to create a Brain Imaging Data Structure (BIDS) data set

The Brain Imaging Data Structure (BIDS) is a standard specifying the description of neuroimaging data in a filesystem 
hierarchy and of the metadata associated with the imaging data. 
More information can be found on https://bids.neuroimaging.io/index.html

We provide a [python script](data_conversion_BIDS.py) that creates a valid BIDS dataset from the source data found on 
TCIA using 3D Slicer.

The following command can be used to create the dataset:
        
         <SLICER_DIR>/Slicer  --python-script ./data_conversion_BIDS.py --input-folder <INPUT_DATA_FOLDER> --output-folder <OUTPUT_DATA_FOLDER>

where:
* The 3DSlicer archive has been unpacked at <SLICER_DIR>.

description:

`````--input-folder <INPUT_DATA_FOLDER>  `````
* <INPUT_DATA_FOLDER> is a path to a folder containing sub-folders named `vs_gk_<case_number>_t1`,
                                and `vs_gk_<case_number>_t2`, which contain image files in DICOM format, and the
                                contours.json file and registration matrices in .tfm format. 
                                This corresponds to the output of the script 
                                "TCIA_data_convert_into_convenient_folder_structure.py" described 
                                [here](../README.md).
  
`````--output-folder <OUTPUT_DATA_FOLDER>`````
* <OUTPUT_DATA_FOLDER> is the path to the new BIDS data set root folder

The folder [VS-SEG-BIDS-nonifti](VS-SEG-BIDS-nonifti) contains the output of this script without the nifti files, since
they are too large to be stored in this repository. 
