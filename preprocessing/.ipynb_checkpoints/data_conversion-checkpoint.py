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

    return ref1, ref2, RTSS1, RTSS2


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
    planar_contour_segmentation_node, ref, export_only_tumour_seg, case_number, output_folder
):
    pc_node = planar_contour_segmentation_node
    segmentIDs = vtk.vtkStringArray()  # create new array
    pc_node.GetSegmentation().GetSegmentIDs(
        segmentIDs
    )  # save IDs of all Segmentations in segmentIDs array, e.g. skull, tumor, cochlea
    lm_nodes = []

    if export_only_tumour_seg:
        nb_structures = 1
    else:
        nb_structures = segmentIDs.GetNumberOfValues()

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

        # specialcharacters to remove from output filename (they could come from name of segmented structure)
        charsRoRemove = ["!", "?", ";", "*", " "]

        # create filenames and remove special characters from output filename (they could come from name of segmented structure)
        if export_only_tumour_seg:
            if "t1_" in ref.GetName():
                fileName_rt = os.path.join(
                    output_folder, f"vs_gk_{case_number}", f"vs_gk_seg" + "_refT1.nii.gz"
                ).translate({ord(i): None for i in charsRoRemove})
            elif "t2_" in ref.GetName():
                fileName_rt = os.path.join(
                    output_folder, f"vs_gk_{case_number}", f"vs_gk_seg" + "_refT2.nii.gz"
                ).translate({ord(i): None for i in charsRoRemove})
            else:
                raise Exception("Reference volume not valid.")
        else:
            if "t1_" in ref.GetName():
                fileName_rt = os.path.join(
                    output_folder,
                    f"vs_gk_{case_number}",
                    f"vs_gk_struc{segmentIndex + 1}_" + segmentID + "_refT1.nii.gz",
                ).translate({ord(i): None for i in charsRoRemove})
            elif "t2_" in ref.GetName():
                fileName_rt = os.path.join(
                    output_folder,
                    f"vs_gk_{case_number}",
                    f"vs_gk_struc{segmentIndex + 1}_" + segmentID + "_refT2.nii.gz",
                ).translate({ord(i): None for i in charsRoRemove})
            else:
                raise Exception("Reference volume not valid.")

        # save node
        slicer.util.saveNode(lm_node, fileName_rt)  # save planar contour points)

        lm_nodes.append(lm_node)
    return lm_nodes


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
        "-r",
        "--register",
        dest="register",
        metavar="PATH",
        default="no_registration",
        help='"T1" for registration to T1 image, "T2" for registration to T2 image. ' 'Defaults to "no_registration".',
    )
    parser.add_argument(
        "--export_all_structures",
        dest="export_all_structures",
        action="store_true",
        help="All available structures will be exported. By default, only the VS is exported.",
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
    if args.register not in ["no_registration", "T1", "T2"]:
        logging.error('Invalid value for keyword "--register": choose "T1" or "T2" or "no_registration"')

    # Convert to python path style
    input_folder = args.input_folder.replace("\\", "/")
    output_folder = args.output_folder.replace("\\", "/")
    register = args.register
    export_all_structures = args.export_all_structures

    register_T1_image_and_contour_points_to_T2_image = False
    register_T2_image_and_contour_points_to_T1_image = False
    if register == "T1":
        register_T2_image_and_contour_points_to_T1_image = True
    elif register == "T2":
        register_T1_image_and_contour_points_to_T2_image = True

    if export_all_structures:
        export_only_tumour_seg = False
    else:
        export_only_tumour_seg = True

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

    for i in range(len(patient_dirs)):
        # get case number from folder name
        print(pattern.findall(patient_dirs[i]))
        case_number = pattern.findall(patient_dirs[i])[0]

        # skip iteration if case has already been dealt with
        if case_number in case_numbers:
            continue
        case_numbers.append(case_number)

        print(f"case: {case_number}")

        [ref1, ref2, RTSS1, RTSS2] = import_T1_and_T2_data(input_folder, case_number)

        ## REGISTRATION

        if register_T1_image_and_contour_points_to_T2_image:
            transform_path = os.path.join(input_folder, f"vs_gk_{case_number}" + "_t1", "inv_T1_LPS_to_T2_LPS.tfm")
            transformNode = slicer.util.loadNodeFromFile(transform_path, filetype="TransformFile")
            ref1 = register_and_resample(
                input_node=ref1, reference_node=ref2, transform_node=transformNode, interpolationMode="Linear"
            )
            # also transform contour points
            RTSS1.SetAndObserveTransformNodeID(transformNode.GetID())

        elif register_T2_image_and_contour_points_to_T1_image:
            transform_path = os.path.join(input_folder, f"vs_gk_{case_number}" + "_t2", "inv_T2_LPS_to_T1_LPS.tfm")
            transformNode = slicer.util.loadNodeFromFile(transform_path, filetype="TransformFile")
            ref2 = register_and_resample(
                input_node=ref2, reference_node=ref1, transform_node=transformNode, interpolationMode="Linear"
            )
            # also transform contour points
            RTSS2.SetAndObserveTransformNodeID(transformNode.GetID())

        ## Create segments from contour files
        # Create segmentation node where we will store segments
        segmentationNode_T1 = create_segmentation_node_with_reference_geometry(
            "SegmentationFromContourPoints_refT1", ref_geometry_image_node=ref1
        )
        segmentationNode_T2 = create_segmentation_node_with_reference_geometry(
            "SegmentationFromContourPoints_refT2", ref_geometry_image_node=ref2
        )

        # load structure contour list from json file
        # these are registered to original T1 and T2 images (they are not affected by registrations of the ref_geometry
        # node, which only defines extent and the IJK to RAS transformation)
        structure_contour_list_T1 = load_LPS_contour_points(
            os.path.join(input_folder, f"vs_gk_{case_number}" + "_t1", "contours.json")
        )
        structure_contour_list_T2 = load_LPS_contour_points(
            os.path.join(input_folder, f"vs_gk_{case_number}" + "_t2", "contours.json")
        )

        # create segments for all structures
        create_segments_from_structure_contour_list(segmentationNode_T1, structure_contour_list_T1)
        create_segments_from_structure_contour_list(segmentationNode_T2, structure_contour_list_T2)

        ## export all files

        if register_T1_image_and_contour_points_to_T2_image:
            lm_node_in_T2_list = save_labelmaps_from_planar_contour(
                segmentationNode_T2, ref2, export_only_tumour_seg, case_number, output_folder
            )  # save contour points registered to T2 image
        if register_T2_image_and_contour_points_to_T1_image:
            lm_node_in_T1_list = save_labelmaps_from_planar_contour(
                segmentationNode_T1, ref1, export_only_tumour_seg, case_number, output_folder
            )

        if not register_T1_image_and_contour_points_to_T2_image and not register_T2_image_and_contour_points_to_T1_image:
            lm_node_in_T2_list = save_labelmaps_from_planar_contour(
                segmentationNode_T2, ref2, export_only_tumour_seg, case_number, output_folder
            )  # save contour points registered to T2 image
            lm_node_in_T1_list = save_labelmaps_from_planar_contour(
                segmentationNode_T1, ref1, export_only_tumour_seg, case_number, output_folder
            )

        # save images as nifti files
        if register_T1_image_and_contour_points_to_T2_image:
            slicer.util.saveNode(
                ref1, os.path.join(output_folder, f"vs_gk_{case_number}", f"vs_gk_t1_refT2.nii.gz")
            )  # pass vol node and destination filename
            slicer.util.saveNode(
                ref2, os.path.join(output_folder, f"vs_gk_{case_number}", f"vs_gk_t2_refT2.nii.gz")
            )  # pass vol node and destination filename
        if register_T2_image_and_contour_points_to_T1_image:
            slicer.util.saveNode(
                ref1, os.path.join(output_folder, f"vs_gk_{case_number}", f"vs_gk_t1_refT1.nii.gz")
            )  # pass vol node and destination filename
            slicer.util.saveNode(
                ref2, os.path.join(output_folder, f"vs_gk_{case_number}", f"vs_gk_t2_refT1.nii.gz")
            )  # pass vol node and destination filename

        if not register_T1_image_and_contour_points_to_T2_image and not register_T2_image_and_contour_points_to_T1_image:
            slicer.util.saveNode(
                ref1, os.path.join(output_folder, f"vs_gk_{case_number}", f"vs_gk_t1_refT1.nii.gz")
            )  # pass vol node and destination filename
            slicer.util.saveNode(
                ref2, os.path.join(output_folder, f"vs_gk_{case_number}", f"vs_gk_t2_refT2.nii.gz")
            )  # pass vol node and destination filename
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
