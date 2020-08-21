"""
@author: reubendo
Adapted from: https://github.com/SlicerRt/SlicerRT/edit/master/BatchProcessing/BatchStructureSetConversion.py
[path_slicer] --no-main-window --python-script ./VSSegmentation/VSScripts/convert_label.py --input-folder [path_input] --output-folder [path_output]
Remark:
    For Mac [path_slicer] = /Applications/Slicer.app/Contents/MacOS/Slicer
"""


from __future__ import absolute_import # Makes moving python2 to python3 much easier and ensures that nasty bugs involving integer division don't creep in
import os
import vtk, slicer
from slicer.ScriptedLoadableModule import *
import argparse
import sys
import logging
from DICOMLib import DICOMUtils
import SimpleITK
import traceback


charsRoRemove = ['!', '?', ':', ';', '*']


def check_DICOM_folder(folder):
    if os.path.isdir(folder) and len([k for k in os.listdir(folder) if 'dcm' in k])>0:
        return True
    else:
        return False

def convert_image_and_segmentation(output_dir, pat=False):
    """ Convert Dicom volume to nifti
        Convert RT structures to nifti
    """
    #RT structure node
    ns = getNodesByClass('vtkMRMLSegmentationNode')
    segmentation = ns[0]

    #Volume node
    ns = getNodesByClass('vtkMRMLScalarVolumeNode')
    for k in ns:
        if not 'RT' in k.GetName():
            vol = k
    

    # Saving Volume to nii.gz
    if not pat:
        fileName_vol = vol.GetName() + '.nii.gz'
    else:
        fileName_vol = pat + '.nii.gz'
    fileName_vol = output_dir + '/' + fileName_vol
    fileName_vol = fileName_vol.translate(None, ''.join(charsRoRemove))
    logging.info('[WRITING] Saving volume to file ' + fileName_vol)
    slicer.util.saveNode(vol, fileName_vol)  



    segmentIDs = vtk.vtkStringArray()
    segmentation.GetSegmentation().GetSegmentIDs(segmentIDs)
    for segmentIndex in xrange(0, segmentIDs.GetNumberOfValues()):
        # Selecting a RT structure
        segmentID = segmentIDs.GetValue(segmentIndex)
        segmentID_a = vtk.vtkStringArray()
        segmentID_a.SetNumberOfValues(1)
        segmentID_a.SetValue(0, segmentID)

        # Creating a Label Map node with the RT structure
        sl = slicer.modules.segmentations.logic()
        rt_lm = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
        sl.ExportSegmentsToLabelmapNode(segmentation,segmentID_a, rt_lm, vol)

        # Saving the LabelMap to nii.gz
        if not pat:
            fileName_rt = rt_lm.GetName() + "_" +segmentID + '.nii.gz'
        else:
            fileName_rt = pat + "_" +segmentID + '.nii.gz'
        fileName_rt = output_dir + '/' + fileName_rt
        fileName_rt = fileName_rt.translate(None, ''.join(charsRoRemove))
        logging.info('[WRITING]  Saving structure to file ' + fileName_rt)
        slicer.util.saveNode(rt_lm, fileName_rt)


def main(argv):

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Batch Structure Set Conversion")
    parser.add_argument("-i", "--input-folder", dest="input_folder", metavar="PATH",
                        default="-", required=True, help="Folder of input DICOM study (or database path to use existing)")
    parser.add_argument("-o", "--output-folder", dest="output_folder", metavar="PATH",
                        default=".", help="Folder for output labelmaps")

    args = parser.parse_args(argv)

    # Check required arguments
    if args.input_folder == "-":
        logging.warning('Please specify input DICOM study folder!')
    if args.output_folder == ".":
        logging.info('Current directory is selected as output folder (default). To change it, please specify --output-folder')

    # Convert to python path style
    input_folder = args.input_folder.replace('\\', '/')
    output_folder = args.output_folder.replace('\\', '/')

    if not os.access(output_folder, os.F_OK):
        os.mkdir(output_folder)


    DICOMUtils.openTemporaryDatabase()

    patient_dirs = os.listdir(input_folder)
    patient_dirs  = [patient_dir for patient_dir in patient_dirs if os.path.isdir(os.path.join(input_folder, patient_dir))]

    list_fails = []
    for patient_dir in patient_dirs:
        try: 
            slicer.dicomDatabase.initializeDatabase()
            DICOMUtils.importDicom(os.path.join(input_folder,patient_dir))

            logging.info("Import DICOM data from " + os.path.join(input_folder,patient_dir))
            slicer.mrmlScene.Clear(0)
            patient = slicer.dicomDatabase.patients()[0]
            DICOMUtils.loadPatientByUID(patient)
            output_dir = os.path.join(output_folder,patient_dir)
            if not os.access(output_dir, os.F_OK):
                os.mkdir(output_dir)
            patient_dir = patient_dir.replace(' ', '_')
            convert_image_and_segmentation(output_dir,patient_dir)
        except Exception:
            list_fails.append('Error with ' + str(patient_dir) + "\n" + traceback.format_exc())

    logging.info("\n".join(list_fails))
    logging.info("End")
    sys.exit(0)




if __name__ == "__main__":
    main(sys.argv[1:])