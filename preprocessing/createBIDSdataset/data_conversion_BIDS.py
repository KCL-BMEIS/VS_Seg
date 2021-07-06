"""
@authors: reubendo, aaronkujawa
Adapted from: https://github.com/SlicerRt/SlicerRT/edit/master/BatchProcessing/BatchStructureSetConversion.py

usage from command line:
[path_slicer] --python-script [path_to_this_python_file] --input-folder [path_input] --output-folder [path_output]
[optional] --register [T1 or T2] --export_all_structures

description:
--input-folder [path_input] ... path_input is a path to a folder containing sub-folders named vs_gk_<case_number>_t1
                                and vs_gk_<case_number>_t2, which contain image files in DICOM format and the
                                contours.json file
--register ... optional keyword:
                if not used, no registration will be performed. The T1 and T2 image will be exported as
                vs_gk_t1_refT1.nii.gz and vs_gk_t2_refT2.nii.gz . The tumour segmentations will be exported as
                vs_gk_seg_refT1.nii.gz with the dimensions of the T1 image and vs_gk_seg_refT2.nii.gz with the
                dimensions of the T2 image.

                --register T1: The T2 image will be registered to the T1 image. The exported image files will be named
                vs_gk_t1_refT1.nii.gz and vs_gk_t1_refT1.nii.gz. Only one segmentation with the dimensions of the T1
                image will be exported, named vs_gk_seg_refT1.nii.gz

                --register T2: The T1 image will be registered to the T2 image. The exported image files will be named
                vs_gk_t1_refT2.nii.gz and vs_gk_t1_refT2.nii.gz. Only one segmentation with the dimensions of the T2
                image will be exported, named vs_gk_seg_refT2.nii.gz

--export_all_structures ... optional keyword: if used, all structures in the contours.json file will be exported, not
                            only the tumour. The exported structures will be named
                            vs_gk_struc<structure_index>_<structure_name>_refT1.nii.gz and
                            vs_gk_struc<structure_index>_<structure_name>_refT2.nii.gz where structure_index refers to
                            the order of the structures in the contours.json file (starting at 1) and structure_name
                            is the name of the structure as specified in the contours.json file.
                            If --register T1 or --register T2 is used, only one of the two files is exported.


Remarks:
    3D Slicer version has to be 4.13 or newer
    For Mac [path_slicer] = /Applications/Slicer.app/Contents/MacOS/Slicer
"""

from __future__ import absolute_import
import os
import vtk, slicer
from slicer.ScriptedLoadableModule import *
import argparse
import sys
import logging
from DICOMLib import DICOMUtils
import re
import glob
import json
import numpy as np
import shutil
import pydicom


def loadCheckedLoadables(self):
    # method copied from https://github.com/Slicer/Slicer/blob/b3a78b1cf7cbe6e832ffe6b149bec39d9539f4c6/Modules/
    # Scripted/DICOMLib/DICOMBrowser.py#L495
    # to overwrite the source code method
    # only change is that the first three lines are commented out, because they reset the "selected" attribute which
    # can be set manually before calling this method to decide which series are loaded into slicer

    """Invoke the load method on each plugin for the loadable
    (DICOMLoadable or qSlicerDICOMLoadable) instances that are selected"""
    #     if self.advancedViewButton.checkState() == 0:
    #       self.examineForLoading()

    #     self.loadableTable.updateSelectedFromCheckstate()

    # TODO: add check that disables all referenced stuff to be considered?
    # get all the references from the checked loadables
    referencedFileLists = []
    for plugin in self.loadablesByPlugin:
        for loadable in self.loadablesByPlugin[plugin]:
            if hasattr(loadable, "referencedInstanceUIDs"):
                instanceFileList = []
                for instance in loadable.referencedInstanceUIDs:
                    instanceFile = slicer.dicomDatabase.fileForInstance(instance)
                    if instanceFile != "":
                        instanceFileList.append(instanceFile)
                if len(instanceFileList) and not self.isFileListInCheckedLoadables(instanceFileList):
                    referencedFileLists.append(instanceFileList)

    # if applicable, find all loadables from the file lists
    loadEnabled = False
    if len(referencedFileLists):
        (self.referencedLoadables, loadEnabled) = self.getLoadablesFromFileLists(referencedFileLists)

    automaticallyLoadReferences = int(
        slicer.util.settingsValue("DICOM/automaticallyLoadReferences", qt.QMessageBox.InvalidRole)
    )
    if slicer.app.commandOptions().testingEnabled:
        automaticallyLoadReferences = qt.QMessageBox.No
    if loadEnabled and automaticallyLoadReferences == qt.QMessageBox.InvalidRole:
        self.showReferenceDialogAndProceed()
    elif loadEnabled and automaticallyLoadReferences == qt.QMessageBox.Yes:
        self.addReferencesAndProceed()
    else:
        self.proceedWithReferencedLoadablesSelection()


def import_T1_and_T2_data(input_folder, case_number):
    patient_dir1 = f"vs_gk_{case_number}_t1"
    patient_dir2 = f"vs_gk_{case_number}_t2"

    slicer.dicomDatabase.initializeDatabase()
    DICOMUtils.importDicom(os.path.join(input_folder, patient_dir1))  # import T1 folder files
    DICOMUtils.importDicom(os.path.join(input_folder, patient_dir2))  # import T2 folder files

    logging.info("Import DICOM data from " + os.path.join(input_folder, patient_dir1))
    logging.info("Import DICOM data from " + os.path.join(input_folder, patient_dir2))

    slicer.mrmlScene.Clear(0)  # clear the scene

    logging.info(slicer.dicomDatabase.patients())
    patient = slicer.dicomDatabase.patients()[0]  # select the patient from the database (which has only one patient)

    # get all available series for the current patient
    studies = slicer.dicomDatabase.studiesForPatient(patient)
    series = [slicer.dicomDatabase.seriesForStudy(study) for study in studies]
    seriesUIDs = [uid for uidList in series for uid in uidList]

    # activate the selection window in the dicom widget
    dicomWidget = slicer.modules.dicom.widgetRepresentation().self()
    dicomWidget.browserWidget.onSeriesSelected(seriesUIDs)
    dicomWidget.browserWidget.examineForLoading()

    # get all available series that are loadable
    loadables = dicomWidget.browserWidget.loadableTable.loadables

    # loop over loadables and select for loading
    counter = 0
    to_load = []
    for key in loadables:
        name = loadables[key].name
        if "RTSTRUCT" in name or (("t1_" in name or "t2_" in name) and not " MR " in name):
            loadables[key].selected = True
            counter += 1
            to_load.append(loadables[key].name)

        else:
            loadables[key].selected = False

    # check if exactly 4 loadables (2 images and 2 RT structures were selected)
    assert counter == 4, (
        f"Not exactly 4, but {counter} files selected for loading of case {case_number}. \n"
        f"Selected files are {to_load}"
    )

    # perform loading operation
    loadCheckedLoadables(dicomWidget.browserWidget)

    #     # to load all loadables of the patient use instead:
    #     DICOMUtils.loadPatientByUID(patient)  # load selected patient into slicer

    ### finished dicom widged operations ###
    ### now in slicer data module ###

    # get RT structure nodes
    ns = getNodesByClass("vtkMRMLSegmentationNode")  # gets the nodes that correspond to RT structures

    assert len(ns) == 2, f"Not exactly 2, but {len(ns)} node of class vtkMRMLSegmentationNode."
    RTSS1 = ns[0]  # picks the first RT structure
    RTSS2 = ns[1]

    ref1 = RTSS1.GetNodeReference("referenceImageGeometryRef")
    ref2 = RTSS2.GetNodeReference("referenceImageGeometryRef")

    ref1_name = ref1.GetName()
    ref2_name = ref2.GetName()

    print(ref1_name)
    print(ref2_name)

    # make sure that 1-variables are always related to T1 image/segmentation
    if "t1_" in ref1_name and "t2_" in ref2_name:
        print("T1 first")
    elif "t2_" in ref1_name and "t1_" in ref2_name:
        print("T2 first")
        RTSS1, RTSS2 = RTSS2, RTSS1
        ref1, ref2 = ref2, ref1
    else:
        raise Error("Series names do not contain proper t1 or t2 identifiers.")

    ref1_meta = pydicom.read_file(os.path.join(input_folder, patient_dir1, "IMG0000000000.dcm"))
    ref2_meta = pydicom.read_file(os.path.join(input_folder, patient_dir2, "IMG0000000000.dcm"))
    return ref1, ref2, ref1_meta, ref2_meta


def register_and_resample(input_node, reference_node, transform_node=None, interpolationMode="Linear"):
    # when loaded with slicer, the matrix in tfm file is multiplied with LPS_to_RAS transforms from both sides
    # furthermore the transformNode will be set to FromParent instead of ToParent, which has the same effect
    # as inverting it before application to the volume node

    if transform_node:
        # make a temporary copy of the input node on which the transform can be hardened
        copy_input_node = slicer.modules.volumes.logic().CloneVolume(slicer.mrmlScene, input_node, "translated")
        copy_input_node.SetAndObserveTransformNodeID(transform_node.GetID())
        logic = slicer.vtkSlicerTransformLogic()
        logic.hardenTransform(copy_input_node)
        print("hardened transformation")
    else:
        copy_input_node = slicer.modules.volumes.logic().CloneVolume(slicer.mrmlScene, input_node, "copy")

    # resample volume
    registered_and_resampled_node = slicer.mrmlScene.AddNewNodeByClass(copy_input_node.GetClassName())
    parameters = {
        "inputVolume": copy_input_node,
        "referenceVolume": reference_node,
        "outputVolume": registered_and_resampled_node,
        "interpolationMode": interpolationMode,
        "defaultValue": 0.0,
    }
    slicer.cli.run(slicer.modules.brainsresample, None, parameters, wait_for_completion=True)
    slicer.mrmlScene.RemoveNode(copy_input_node)  # remove temporary copy of input node
    registered_and_resampled_node.SetName(input_node.GetName() + "_registered_and_resampled")
    return registered_and_resampled_node


def createSegNodeFromContourPoints(segmentationNode, contours, name):
    # set up contour objects
    contoursPolyData = vtk.vtkPolyData()
    contourPoints = vtk.vtkPoints()
    contourLines = vtk.vtkCellArray()
    contoursPolyData.SetLines(contourLines)
    contoursPolyData.SetPoints(contourPoints)

    for contour in contours:
        startPointIndex = contourPoints.GetNumberOfPoints()
        contourLine = vtk.vtkPolyLine()
        linePointIds = contourLine.GetPointIds()
        for point in contour:
            linePointIds.InsertNextId(contourPoints.InsertNextPoint(point))
        linePointIds.InsertNextId(startPointIndex)  # make the contour line closed
        contourLines.InsertNextCell(contourLine)

    segment = slicer.vtkSegment()
    segment.SetName(name)
    # segment.SetColor(segmentColor)
    segment.AddRepresentation("Planar contour", contoursPolyData)
    segmentationNode.GetSegmentation().SetMasterRepresentationName("Planar contour")
    segmentationNode.GetSegmentation().AddSegment(segment)


def load_LPS_contour_points(json_file_path):
    with open(json_file_path, "r") as json_file:
        structure_contour_list = json.load(json_file)
    return structure_contour_list


def transform_contour_points(affine, contour_points):
    transformed_contour_points = []
    for point in contour_points:
        transformed_contour_points.append((affine @ np.append(point, 1))[:3].tolist())
    return np.array(transformed_contour_points)


def create_segmentation_node_with_reference_geometry(name, ref_geometry_image_node):
    new_segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    new_segmentation_node.CreateDefaultDisplayNodes()
    new_segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(ref_geometry_image_node)
    new_segmentation_node.SetName(name)

    return new_segmentation_node


def create_segments_from_structure_contour_list(segmentationNode, structure_contour_list):
    RAS_to_LPS = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
    for i, struc in enumerate(structure_contour_list):
        # select a structure
        contours = struc["LPS_contour_points"]

        # transform contours from LPS to RAS
        contours_RAS = []
        for region in contours:
            contours_RAS.append(transform_contour_points(RAS_to_LPS, region))

        # create segment from contours_RAS in segmentationNode
        createSegNodeFromContourPoints(segmentationNode, contours_RAS, struc["structure_name"])


def save_labelmaps_from_planar_contour(
    planar_contour_segmentation_node, ref, save_path
):
    pc_node = planar_contour_segmentation_node
    segmentIDs = vtk.vtkStringArray()  # create new array
    pc_node.GetSegmentation().GetSegmentIDs(
        segmentIDs
    )  # save IDs of all Segmentations in segmentIDs array, e.g. skull, tumor, cochlea
    lm_nodes = []

    # only export the first structure, the tumour
    nb_structures = 1

    for segmentIndex in range(0, nb_structures):
        # Selecting a structure
        segmentID = segmentIDs.GetValue(segmentIndex)

        # create new array and store only the ID of current structure segmentation in it
        segmentID_a = vtk.vtkStringArray()  # new array
        segmentID_a.SetNumberOfValues(1)  # define length
        segmentID_a.SetValue(0, segmentID)  # define first value by segmentID

        # Creating a Label Map nodes that will store the binary segmentation
        lm_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")

        # arguments: input node, Segmentation ID (skull?), new label map node, reference volume)
        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(pc_node, segmentID_a, lm_node, ref)

        # save node
        slicer.util.saveNode(lm_node, save_path)  # save planar contour points

        lm_nodes.append(lm_node)
    return lm_nodes


def createBIDSPath(BIDSRootFolderPath, case, folderID):
    root = BIDSRootFolderPath
    #subject_entity = "sub-VS-SEG-" + f"{int(case):03d}"
    subject_entity = "sub-" + f"{int(case):03d}"

    if folderID == "raw":
        # path to raw data (nifti files converted directly from DICOM)
        path = os.path.join(BIDSRootFolderPath)
    elif folderID == "raw_README":
        # path to README file
        path = os.path.join(BIDSRootFolderPath, "README")
    elif folderID == "raw_description_json":
        # path to raw dataset_description.json
        path = os.path.join(BIDSRootFolderPath, "dataset_description.json")
    elif folderID == "raw_sub_anat_T1w_nii":
        # path to nifti file in that folder
        path = os.path.join(BIDSRootFolderPath, subject_entity, "anat", subject_entity + "_T1w.nii.gz")
    elif folderID == "raw_sub_anat_T2w_nii":
        path = os.path.join(BIDSRootFolderPath, subject_entity, "anat", subject_entity + "_T2w.nii.gz")
    elif folderID == "raw_sub_anat_T1w_json":
        # path to nifti file in that folder
        path = os.path.join(BIDSRootFolderPath, subject_entity, "anat", subject_entity + "_T1w.json")
    elif folderID == "raw_sub_anat_T2w_json":
        path = os.path.join(BIDSRootFolderPath, subject_entity, "anat", subject_entity + "_T2w.json")

    elif folderID == "source":
        path = os.path.join(BIDSRootFolderPath, "sourcedata")
    elif folderID == "source_contours_T1w_json":
        path = os.path.join(BIDSRootFolderPath, "sourcedata", "contours", subject_entity, "anat", subject_entity+"_contours_space-individual_T1w.json")
    elif folderID == "source_contours_T2w_json":
        path = os.path.join(BIDSRootFolderPath, "sourcedata", "contours", subject_entity, "anat", subject_entity+"_contours_space-individual_T2w.json")
    elif folderID == "source_regmat_T1wtoT2w_tfm":
        path = os.path.join(BIDSRootFolderPath, "sourcedata", "registration_matrices", subject_entity, "anat", subject_entity+"_inv_T1_LPS_to_T2_LPS.tfm")
    elif folderID == "source_regmat_T2wtoT1w_tfm":
        path = os.path.join(BIDSRootFolderPath, "sourcedata", "registration_matrices", subject_entity, "anat", subject_entity+"_inv_T2_LPS_to_T1_LPS.tfm")

    elif folderID == "derivatives":
        path = os.path.join(BIDSRootFolderPath, "derivatives")

    elif folderID == "derivatives_T1wRegtoT2w_description_json":
        path = os.path.join(BIDSRootFolderPath, "derivatives", "T1w_registered_to_T2w", "dataset_description.json")
    elif folderID == "derivatives_T2wRegtoT1w_description_json":
        path = os.path.join(BIDSRootFolderPath, "derivatives", "T2w_registered_to_T1w", "dataset_description.json")
    elif folderID == "derivatives_T1wRegtoT2w_nii":
        path = os.path.join(BIDSRootFolderPath, "derivatives", "T1w_registered_to_T2w", subject_entity, "anat", subject_entity+"_space-individual_T1w.nii.gz")
    elif folderID == "derivatives_T2wRegtoT1w_nii":
        path = os.path.join(BIDSRootFolderPath, "derivatives", "T2w_registered_to_T1w", subject_entity, "anat", subject_entity+"_space-individual_T2w.nii.gz")
    elif folderID == "derivatives_T1wRegtoT2w_json":
        path = os.path.join(BIDSRootFolderPath, "derivatives", "T1w_registered_to_T2w", subject_entity, "anat", subject_entity+"_space-individual_T1w.json")
    elif folderID == "derivatives_T2wRegtoT1w_json":
        path = os.path.join(BIDSRootFolderPath, "derivatives", "T2w_registered_to_T1w", subject_entity, "anat", subject_entity+"_space-individual_T2w.json")

    elif folderID == "derivatives_masks_T1w_description_json":
        path = os.path.join(BIDSRootFolderPath, "derivatives", "manual_segmentation_masks_of_T1w", "dataset_description.json")
    elif folderID == "derivatives_masks_T2w_description_json":
        path = os.path.join(BIDSRootFolderPath, "derivatives", "manual_segmentation_masks_of_T2w", "dataset_description.json")
    elif folderID == "derivatives_masks_T1w_nii":
        path = os.path.join(BIDSRootFolderPath, "derivatives", "manual_segmentation_masks_of_T1w", subject_entity, "anat", subject_entity+"_space-individual_desc-tumor_mask.nii.gz")
    elif folderID == "derivatives_masks_T2w_nii":
        path = os.path.join(BIDSRootFolderPath, "derivatives", "manual_segmentation_masks_of_T2w", subject_entity, "anat", subject_entity+"_space-individual_desc-tumor_mask.nii.gz")
    elif folderID == "derivatives_masks_T1w_json":
        path = os.path.join(BIDSRootFolderPath, "derivatives", "manual_segmentation_masks_of_T1w", subject_entity, "anat", subject_entity+"_space-individual_desc-tumor_mask.json")
    elif folderID == "derivatives_masks_T2w_json":
        path = os.path.join(BIDSRootFolderPath, "derivatives", "manual_segmentation_masks_of_T2w", subject_entity, "anat", subject_entity+"_space-individual_desc-tumor_mask.json")

    else:
        raise Exception("folderID does not exist.")

    # if the paths are folder-paths, create the folders, if they are file-paths create the containing folders
    if not any(ext in path for ext in [".nii.gz", ".json", ".tfm", "README"]):
        os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    return path


def create_sidecar_dict(meta_data, tags):
    data = {}
    for tag in tags:
        if type(tag) == str:
            try:
                data[tag] = meta_data[tag].value
                if type(data[tag]) == pydicom.multival.MultiValue:
                    data[tag] = "\\".join(data[tag])
                else:
                    data[tag] = str(data[tag])
            except:
                print(tag, ": tag not found in DICOM metadata")
            if tag == "EchoTime":
                data[tag] = str(float(data[tag])/1000)
        elif type(tag) == tuple:
            try:
                data[tag[0]] = meta_data[tag[1]].value
                if type(data[tag[0]]) == pydicom.multival.MultiValue:
                    data[tag[0]] = "\\".join(data[tag[0]])
                else:
                    data[tag[0]] = str(data[tag[0]])
            except:
                print(tag[1], "not found")
        else:
            raise Exception("tag type not defined")
    return data


def main(argv):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Batch Structure Set Conversion")
    parser.add_argument(
        "-i",
        "--input-folder",
        dest="input_folder",
        metavar="PATH",
        default="-",
        required=True,
        help="Folder of input DICOM study (or database path to use existing)",
    )
    parser.add_argument(
        "-o", "--output-folder", dest="output_folder", metavar="PATH", default=".", help="Folder for output labelmaps"
    )
    parser.add_argument(
        "--no_nifti",
        dest="no_nifti",
        action="store_true",
        help="If --no_nifti is specified, all files, apart from nifti files will be saved.",
    )
    parser.set_defaults(export_all_structures=False)


    args = parser.parse_args(argv)

    # Check required arguments
    if args.input_folder == "-":
        logging.warning("Please specify input DICOM study folder!")
    if args.output_folder == ".":
        logging.info(
            "Current directory is selected as output folder (default). To change it, please specify --output-folder"
        )

    # Convert to python path style
    input_folder = args.input_folder.replace("\\", "/")
    output_folder = args.output_folder.replace("\\", "/")

    no_nifti = args.no_nifti

    if not os.access(output_folder, os.F_OK):
        os.mkdir(output_folder)

    DICOMUtils.openTemporaryDatabase()

    patient_dirs = glob.glob(os.path.join(input_folder, "vs_gk_*"))

    # create and compile a regex pattern
    pattern = re.compile(r"_([0-9]+)_t[1-2]$")

    case_numbers = []

    # first_case = 1
    # last_case = 2

    # for case_number in range(first_case, last_case + 1):
    # exclude cases
    # if case_number in [39, 97, 130, 160, 168, 208, 219, 227]:
    #     continue


    ## create README
    readme_content = "###Segmentation of Vestibular Schwannoma from Magnetic Resonance Imaging: An Open Annotated " \
                     "Dataset and Baseline Algorithm (Vestibular-Schwannoma-SEG)\n\n"\
                     "This collection contains a labeled dataset of MRI images collected on 242 consecutive patients " \
                     "with vestibular schwannoma (VS) undergoing Gamma Knife stereotactic radiosurgery (GK SRS). " \
                     "The structural images included contrast-enhanced T1-weighted (ceT1) images and high-resolution " \
                     "T2-weighted (hrT2) images. Each imaging dataset is accompanied by the patient’s radiation " \
                     "therapy (RT) dataset including the RTDose, RTStructures and RTPlan. Additionally, registration " \
                     "matrices (.tfm format) and segmentation contour lines (JSON format) are provided. " \
                     "All structures were manually segmented in consensus by the treating neurosurgeon and physicist " \
                     "using both the ceT1 and hrT2 images. The value of this collection is to provide clinical " \
                     "image data including fully annotated tumour segmentations to facilitate the development and " \
                     "validation of automated segmentation frameworks. It may also be used for research relating " \
                     "to radiation treatment. \n\n" \
                     "Registration matrices and JSON contours \n\n" \
                     "Registration Matrices: For each subject and each modality there is a text file named " \
                     "sub-<case>-inv_T1_LPS_to_T2_LPS.tfm or sub-<case>-inv_T2_LPS_to_T1_LPS.tfm. " \
                     "The files specify affine transformation matrices that can be used to co-register the T1 image " \
                     "to the T2 image and vice versa. The file format is a standard format defined by the Insight " \
                     "Toolkit (ITK) library. The matrices are the result of the co-registration of fiducials of the " \
                     "Leksell Stereotactic System MR Indicator box into which the patient’s head is fixed during " \
                     "image acquisition. The localization of fiducials and co-registration was performed " \
                     "automatically by the LeksellGammaPlan software.\n\n" \
                     "Contours: For each subject and each modality there is a text file named contours.json. " \
                     "These contour files in the T1 and T2 folder contain the contour points of the segmented " \
                     "structures in JavaScript Object Notation (JSON) format, mapped in the coordinate frames of " \
                     "the T1 image and the T2 image, respectively. There can be small differences between the " \
                     "contour points of the RTSTRUCT and the contour points of the JSON files as explained in " \
                     "the following: " \
                     "In most cases, the tumour was segmented on the T1 image while the cochlea was typically " \
                     "segmented on the T2 image. This meant that some contour lines (typically for the tumour) " \
                     "were coplanar with the slices of the T1 image while others (typically for the cochlea) were" \
                     " coplanar with T2 slices. After co-registration, the (un-resampled) slices of the T1 and T2 " \
                     "image generally did not coincide; for example, due to different image position and, " \
                     "occasionally, slice thickness. Therefore, the combined co-registered contour lines were " \
                     "neither jointly coplanar with the T1 nor with the T2 image slices. Upon export of the " \
                     "segmentations in a given target space, the LeksellGammaPlan software interpolates between " \
                     "the original contour lines to create new slice-aligned contour lines in the target image " \
                     "space (T1 or T2). This results in the interpolated slice-aligned contour lines found in " \
                     "the RTSTRUCTs. In contrast, the contours in the JSON files were not interpolated after " \
                     "co-registration, and therefore describe the original (potentially off-target-space-slice) " \
                     "manual segmentation accurately."

    with open(createBIDSPath(output_folder, case=-1, folderID="raw_README"), 'w') as f:
        f.write(readme_content)

    ## create dataset_description.json

    dataset_description_root_dict = {
        "Name": "Segmentation of Vestibular Schwannoma from Magnetic Resonance Imaging: An Open Annotated Dataset and "
                "Baseline Algorithm (Vestibular-Schwannoma-SEG)",
        "BIDSVersion": "1.6.0",
        "DatasetType": "raw",
        "License": "TCIA Data Usage Policy and the Creative Commons Attribution 4.0 International License",
        "Authors":
        ["Shapey, J.",
         "Kujawa, A.",
         "Dorent, R.",
         "Wang, G.",
         "Bisdas, S.",
         "Dimitriadis, A.",
         "Grishchuck, D.",
         "Paddick, I.",
         "Kitchen, N.",
         "Bradford, R.",
         "Saeed, S.",
         "Ourselin, S.",
         "Vercauteren, T."],
        "Acknowledgements": "This work was supported by Wellcome Trust (203145Z/16/Z, 203148/Z/16/Z, WT106882), "
                            "EPSRC (NS/A000050/1, NS/A000049/1) and MRC (MC_PC_180520) funding. Tom Vercauteren is "
                            "also supported by a Medtronic/Royal Academy of Engineering Research Chair (RCSRF1819\\7\\34).",
        "HowToAcknowledge": "Please cite the references under the ReferencesAndLinks key.",
        "ReferencesAndLinks":
        [
            "Shapey, J., Kujawa, A., Dorent, R., Wang, G., Bisdas, S., Dimitriadis, A., Grishchuck, D., Paddick, I., "
            "Kitchen, N., Bradford, R., Saeed, S., Ourselin, S., & Vercauteren, T. (2021). Segmentation of Vestibular "
            "Schwannoma from Magnetic Resonance Imaging: An Open Annotated Dataset and Baseline Algorithm [Data set]. "
            "The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.9YTJ-5Q73",
            "Shapey, J., Wang, G., Dorent, R., Dimitriadis, A., Li, W., Paddick, I., Kitchen, N., Bisdas, S., Saeed, "
            "S. R., Ourselin, S., Bradford, R., & Vercauteren, T. (2021). An artificial intelligence framework for "
            "automatic segmentation and volumetry of vestibular schwannomas from contrast-enhanced T1-weighted and "
            "high-resolution T2-weighted MRI. Journal of Neurosurgery, 134(1), "
            "171–179. https://doi.org/10.3171/2019.9.jns191949",
            "Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., Moore, S., Phillips, S., Maffitt, D., "
            "Pringle, M., Tarbox, L., & Prior, F. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a "
            "Public Information Repository. Journal of Digital Imaging, 26(6), "
            "1045–1057. https://doi.org/10.1007/s10278-013-9622-7"],
        "DatasetDOI": "https://doi.org/10.1007/s10278-013-9622-7"
    }

    with open(createBIDSPath(output_folder, case=-1, folderID="raw_description_json"), 'w') as ff:
        json.dump(dataset_description_root_dict, ff, indent=4)

    # derivatives/masks_T1w dataset_description.json
    dataset_description_dict = {
        "Name": "manual_segmentation_masks_of_T1w",
        "BIDSVersion": "1.6.0",
        "DatasetType": "derivative",
        "GeneratedBy":
            [
                {
                    "Name": "Manual",
                    "Description": "Manual segmentation of the Vestibular Schwannoma based on T1w and T2w image."
                },
                {
                    "Name": "3DSlicer with python script: https://github.com/KCL-BMEIS/VS_Seg/tree/master/preprocessing/createBIDSdataset/data_conversion_BIDS.py",
                    "Description": "Conversion from contour points in space of the T1w image to nifti."
                },
            ],
        "SourceDatasets":
            [
                {
                    "URL": "file://../../../VS_SEG_BIDS",
                },
                {
                    "URL": "file://../../../VS_SEG_BIDS/sourcedata/registration_matrices",
                }
            ]
    }

    with open(createBIDSPath(output_folder, case=-1, folderID="derivatives_masks_T1w_description_json"), 'w') as ff:
        json.dump(dataset_description_dict, ff, indent=4)

    # dataset_description.json of T1w_registered_to_T2w
    dataset_description_dict = {
        "Name": "T1w_registered_to_T2w",
        "BIDSVersion": "1.6.0",
        "DatasetType": "derivative",
        "GeneratedBy":
            [
                {
                    "Name": "3DSlicer with python script: https://github.com/KCL-BMEIS/VS_Seg/tree/master/preprocessing/createBIDSdataset/data_conversion_BIDS.py",
                    "Description": "T1w images co-registered to their corresponding T2w images and resampled at the T2w grid points."
                }
            ],
        "SourceDatasets":
            [
                {
                    "URL": "file://../../../VS_SEG_BIDS",
                },
                {
                    "URL": "file://../../../VS_SEG_BIDS/sourcedata/registration_matrices",
                }
            ]
    }

    with open(createBIDSPath(output_folder, case=-1, folderID="derivatives_T1wRegtoT2w_description_json"), 'w') as ff:
        json.dump(dataset_description_dict, ff, indent=4)

    # derivatives/masks_T2w dataset_description.json
    dataset_description_dict = {
        "Name": "manual_segmentation_masks_of_T2w",
        "BIDSVersion": "1.6.0",
        "DatasetType": "derivative",
        "GeneratedBy":
            [
                {
                    "Name": "Manual",
                    "Description": "Manual segmentation of the Vestibular Schwannoma based on T1w and T2w image."
                },
                {
                    "Name": "3DSlicer with python script: https://github.com/KCL-BMEIS/VS_Seg/tree/master/preprocessing/createBIDSdataset/data_conversion_BIDS.py",
                    "Description": "Conversion from contour points in space of the T2w image to nifti."
                },
            ],
        "SourceDatasets":
            [
                {
                    "URL": "file://../../../VS_SEG_BIDS",
                },
                {
                    "URL": "file://../../../VS_SEG_BIDS/sourcedata/registration_matrices",
                }
            ]
    }

    with open(createBIDSPath(output_folder, case=-1, folderID="derivatives_masks_T2w_description_json"), 'w') as ff:
        json.dump(dataset_description_dict, ff, indent=4)

    # dataset_description.json of T2w_registered_to_T1w
    dataset_description_dict = {
        "Name": "T2w_registered_to_T1w",
        "BIDSVersion": "1.6.0",
        "DatasetType": "derivative",
        "GeneratedBy":
            [
                {
                    "Name": "3DSlicer with python script: https://github.com/KCL-BMEIS/VS_Seg/tree/master/preprocessing/createBIDSdataset/data_conversion_BIDS.py",
                    "Description": "T2w images co-registered to their corresponding T1w images and resampled at the T1w grid points."
                }
            ],
        "SourceDatasets":
            [
                {
                    "URL": "file://../../../VS_SEG_BIDS",
                },
                {
                    "URL": "file://../../../VS_SEG_BIDS/sourcedata/registration_matrices",
                }
            ]
    }

    with open(createBIDSPath(output_folder, case=-1, folderID="derivatives_T2wRegtoT1w_description_json"), 'w') as ff:
        json.dump(dataset_description_dict, ff, indent=4)


## loop over all patients
    for i in range(len(patient_dirs)):
        # get case number from folder name
        print(pattern.findall(patient_dirs[i]))
        case_number = pattern.findall(patient_dirs[i])[0]

        # skip iteration if case has already been dealt with
        if case_number in case_numbers:
            continue
        case_numbers.append(case_number)

        print(f"case: {case_number}")

        [ref1, ref2, ref1_meta, ref2_meta] = import_T1_and_T2_data(input_folder, case_number)

        ## COPY SOURCE DATA (contours and registration matrices)
        # registration matrices
        shutil.copy(os.path.join(input_folder, f"vs_gk_{case_number}" + "_t1", "inv_T1_LPS_to_T2_LPS.tfm"),
                    createBIDSPath(output_folder, case_number, folderID="source_regmat_T1wtoT2w_tfm"))
        shutil.copy(os.path.join(input_folder, f"vs_gk_{case_number}" + "_t2", "inv_T2_LPS_to_T1_LPS.tfm"),
                    createBIDSPath(output_folder, case_number, folderID="source_regmat_T2wtoT1w_tfm"))
        # contour files
        shutil.copy(os.path.join(input_folder, f"vs_gk_{case_number}" + "_t1", "contours.json"),
                    createBIDSPath(output_folder, case_number, folderID="source_contours_T1w_json"))
        shutil.copy(os.path.join(input_folder, f"vs_gk_{case_number}" + "_t2", "contours.json"),
                    createBIDSPath(output_folder, case_number, folderID="source_contours_T2w_json"))

        ## REGISTRATION

        if not no_nifti:
            # register T1 image to T2
            transform_path = createBIDSPath(output_folder, case_number, folderID="source_regmat_T1wtoT2w_tfm")
            transformNode = slicer.util.loadNodeFromFile(transform_path, filetype="TransformFile")
            ref1_regtoref2 = register_and_resample(
                input_node=ref1, reference_node=ref2, transform_node=transformNode, interpolationMode="Linear")

            # register T2 image to T1
            transform_path = createBIDSPath(output_folder, case_number, folderID="source_regmat_T2wtoT1w_tfm")
            transformNode = slicer.util.loadNodeFromFile(transform_path, filetype="TransformFile")
            ref2_regtoref1 = register_and_resample(
                input_node=ref2, reference_node=ref1, transform_node=transformNode, interpolationMode="Linear")

        ## Create segments from contour files

        # Create segmentation node where we will store segments
        segmentationNode_T1 = create_segmentation_node_with_reference_geometry(
            "SegmentationFromContourPoints_refT1", ref_geometry_image_node=ref1)
        segmentationNode_T2 = create_segmentation_node_with_reference_geometry(
            "SegmentationFromContourPoints_refT2", ref_geometry_image_node=ref2)

        # load structure contour list from json file
        # these are registered to original T1 and T2 images (they are not affected by registrations of the ref_geometry
        # node, which only defines extent and the IJK to RAS transformation)
        structure_contour_list_T1 = load_LPS_contour_points(
            createBIDSPath(output_folder, case_number, folderID="source_contours_T1w_json"))
        structure_contour_list_T2 = load_LPS_contour_points(
            createBIDSPath(output_folder, case_number, folderID="source_contours_T2w_json"))

        # create segments for all structures
        create_segments_from_structure_contour_list(segmentationNode_T1, structure_contour_list_T1)
        create_segments_from_structure_contour_list(segmentationNode_T2, structure_contour_list_T2)

        ## export all files

        if not no_nifti:
            # save contour points registered to T1 image
            lm_node_in_T1_list = save_labelmaps_from_planar_contour(
                segmentationNode_T1, ref1, createBIDSPath(output_folder, case_number, folderID="derivatives_masks_T1w_nii"))
            # save contour points registered to T2 image
            lm_node_in_T2_list = save_labelmaps_from_planar_contour(
                segmentationNode_T2, ref2, createBIDSPath(output_folder, case_number, folderID="derivatives_masks_T2w_nii"))

            # save images as nifti files
            # T1 and T2 in their original spaces
            slicer.util.saveNode(ref1, createBIDSPath(output_folder, case_number, folderID="raw_sub_anat_T1w_nii"))
            slicer.util.saveNode(ref2, createBIDSPath(output_folder, case_number, folderID="raw_sub_anat_T2w_nii"))
            # T1 registered to T2
            slicer.util.saveNode(ref1_regtoref2, createBIDSPath(output_folder, case_number, folderID="derivatives_T1wRegtoT2w_nii"))
            # T2 registered to T1
            slicer.util.saveNode(ref2_regtoref1, createBIDSPath(output_folder, case_number, folderID="derivatives_T2wRegtoT1w_nii"))

        ## create metadata sidecars

        # original T1w sidecar

        tags = [
            "Manufacturer",
            ("ManufacturersModelName", "ManufacturerModelName"),
            "DeviceSerialNumber",
            "StationName",
            "SoftwareVersions",
            "MagneticFieldStrength",
            "TransmitCoilName",
            "ReceiveCoilName",
            "ReceiveCoilActiveElements",
            "GradientSetType",
            "MRTransmitCoilSequence",
            "MatrixCoilMode",
            "CoilCombinationMethod",
            "PulseSequenceType",
            "ScanningSequence",
            "SequenceVariant",
            "ScanOptions",
            "SequenceName",
            "PulseSequenceDetails",
            "NonlinearGradientCorrection",
            "MRAcquisitionType",
            "MTState",
            "SpoilingState",
            "SpoilingType",
            "SpoilingRFPhaseIncrement",
            "SpoilingGradientMoment",
            "SpoilingGradientDuration",
            "NumberShots",
            "ParallelReductionFactorInPlane",
            "ParallelAcquisitionTechnique",
            "PartialFourier",
            "PartialFourierDirection",
            "PhaseEncodingDirection",
            "EffectiveEchoSpacing",
            "TotalReadoutTime",
            "MixingTime",
            "EchoTime",
            "InversionTime",
            "SliceTiming",
            "SliceEncodingDirection",
            "DwellTime",
            "FlipAngle",
            "NegativeContrast",
            "MultibandAccelerationFactor",
            "AnatomicalLandmarkCoordinates",
            "InstitutionName",
            "InstitutionAddress",
            "InstitutionalDepartmentName",
            "ContrastBolusIngredient",
            "RepetitionTime",
            "RepetitionTimeExcitation",
            "RepetitionTimePreparation",
            "Modality",
            "ImagingFrequency",
            "PatientPosition",
            ("ProcedureStepDescription", (0x0040, 0x0254)),
            "SeriesDescription",
            "ProtocolName",
            "ImageType",
            "SeriesNumber",
            "AcquisitionTime",
            "AcquisitionNumber",
            "SliceThickness",
            "SAR",
            ("CoilString", (0x0051, 0x100f)),
            ("PercentPhaseFOV", ("PercentPhaseFieldOfView")),
            "PercentSampling",
            ("PhaseEncodingSteps", ("NumberOfPhaseEncodingSteps")),
            "AcquisitionMatrixPE",
            "ReconMatrixPE",
            "PixelBandwidth",
            "DwellTime",
            ("InPlanePhaseEncodingDirectionDICOM", (0x0018, 0x1312)),
        ]

        # original T1
        data_dict = create_sidecar_dict(ref1_meta, tags)
        with open(createBIDSPath(output_folder, case_number, folderID="raw_sub_anat_T1w_json"), 'w') as ff:
            json.dump(data_dict, ff, indent=4)

        # T1 registered to T2
        data_dict = {}
        data_dict["Description"] = "T1w image after after affine transformation to space corresponding T2w image. The " \
                                  "affine transformation matrix was obtained from the Gamma Knife treatment planning " \
                                  "system, which itself uses the fiducial cage into which the patients head is fixed."
        data_dict["Sources"] = os.path.relpath(createBIDSPath(output_folder, case_number,
                                                              folderID="source_regmat_T1wtoT2w_tfm"), output_folder)
        data_dict["RawSources"] = [os.path.relpath(createBIDSPath(output_folder, case_number,
                                                              folderID="raw_sub_anat_T1w_nii"), output_folder),
                                  os.path.relpath(createBIDSPath(output_folder, case_number,
                                                              folderID="raw_sub_anat_T2w_nii"), output_folder)]
        data_dict["SpatialReference"] = os.path.relpath(createBIDSPath(output_folder, case_number,
                                                              folderID="raw_sub_anat_T2w_nii"), output_folder)

        with open(createBIDSPath(output_folder, case_number, folderID="derivatives_T1wRegtoT2w_json"), 'w') as ff:
            json.dump(data_dict, ff, indent=4)

        # tumour mask in T1 space
        data_dict = {}
        data_dict["Description"] = "Manually created mask of the Vestibular Schwannoma based on both T1w and T2w " \
                                   "image. The binary mask was derived from contour points and discretized by 3DSlicer."
        data_dict["Manual"] = True
        data_dict["Sources"] = os.path.relpath(createBIDSPath(output_folder, case_number,
                                                              folderID="source_contours_T1w_json"), output_folder)
        data_dict["RawSources"] = [os.path.relpath(createBIDSPath(output_folder, case_number,
                                                              folderID="raw_sub_anat_T1w_nii"), output_folder),
                                  os.path.relpath(createBIDSPath(output_folder, case_number,
                                                              folderID="raw_sub_anat_T2w_nii"), output_folder)]
        data_dict["SpatialReference"] = os.path.relpath(createBIDSPath(output_folder, case_number,
                                                              folderID="raw_sub_anat_T1w_nii"), output_folder)

        with open(createBIDSPath(output_folder, case_number, folderID="derivatives_masks_T1w_json"), 'w') as ff:
            json.dump(data_dict, ff, indent=4)

        # original T2
        data_dict = create_sidecar_dict(ref2_meta, tags)
        with open(createBIDSPath(output_folder, case_number, folderID="raw_sub_anat_T2w_json"), 'w') as ff:
            json.dump(data_dict, ff, indent=4)

        # T2 registered to T1
        data_dict = {}
        data_dict["Description"] = "T2w image after after affine transformation to space corresponding T1w image. The " \
                                  "affine transformation matrix was obtained from the Gamma Knife treatment planning " \
                                  "system, which itself uses the fiducial cage into which the patients head is fixed."
        data_dict["Sources"] = os.path.relpath(createBIDSPath(output_folder, case_number,
                                                              folderID="source_regmat_T2wtoT1w_tfm"), output_folder)
        data_dict["RawSources"] = [os.path.relpath(createBIDSPath(output_folder, case_number,
                                                              folderID="raw_sub_anat_T1w_nii"), output_folder),
                                  os.path.relpath(createBIDSPath(output_folder, case_number,
                                                              folderID="raw_sub_anat_T2w_nii"), output_folder)]
        data_dict["SpatialReference"] = os.path.relpath(createBIDSPath(output_folder, case_number,
                                                              folderID="raw_sub_anat_T1w_nii"), output_folder)

        with open(createBIDSPath(output_folder, case_number, folderID="derivatives_T2wRegtoT1w_json"), 'w') as ff:
            json.dump(data_dict, ff, indent=4)

        # tumour mask in T2 space
        data_dict = {}
        data_dict["Description"] = "Manually created mask of the Vestibular Schwannoma based on both T1w and T2w " \
                                   "image. The binary mask was derived from contour points and discretized by 3DSlicer."
        data_dict["Manual"] = True
        data_dict["Sources"] = os.path.relpath(createBIDSPath(output_folder, case_number,
                                                              folderID="source_contours_T2w_json"), output_folder)
        data_dict["RawSources"] = [os.path.relpath(createBIDSPath(output_folder, case_number,
                                                                  folderID="raw_sub_anat_T1w_nii"), output_folder),
                                   os.path.relpath(createBIDSPath(output_folder, case_number,
                                                                  folderID="raw_sub_anat_T2w_nii"), output_folder)]
        data_dict["SpatialReference"] = os.path.relpath(createBIDSPath(output_folder, case_number,
                                                                       folderID="raw_sub_anat_T2w_nii"), output_folder)

        with open(createBIDSPath(output_folder, case_number, folderID="derivatives_masks_T2w_json"), 'w') as ff:
            json.dump(data_dict, ff, indent=4)


    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1:])
