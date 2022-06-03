#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob
from natsort import natsorted
import pydicom
import shutil
import re
import argparse

parser = argparse.ArgumentParser(description='Create a new folder that contains the whole TCIA dataset in a more convenient folder structure')
parser.add_argument('--input', type=str, help='(string) path to TCIA dataset, in "Descriptive Directory Name" format, for example /home/user/.../manifest-1614264588831/Vestibular-Schwannoma-SEG')
parser.add_argument('--output', type=str, help='(string) path to output folder')
args = parser.parse_args()

input_path = args.input 
output_path = args.output 

if not os.path.isdir(output_path):
    os.makedirs(output_path, exist_ok=True)

cases = natsorted(glob(os.path.join(input_path, 'VS-SEG-*')))

for case in cases:
    folders = glob(case+'/*/*')
    
    MRs = [] 
    MRs_paths = []
    
    RTSTRUCTs = [] 
    RTSTRUCTs_paths = []
    
    RTPLANs = [] 
    RTPLANs_paths = []
    
    RTDOSEs = [] 
    RTDOSEs_paths = []
    
    for folder in folders:
        first_file = glob(folder+"/*")[0]
        dd = pydicom.read_file(first_file)

        if dd['Modality'].value == 'MR':
            MRs.append(dd)
            MRs_paths.append(first_file)
        elif dd['Modality'].value == 'RTSTRUCT':
            RTSTRUCTs.append(dd)
            RTSTRUCTs_paths.append(first_file)
        elif dd['Modality'].value == 'RTPLAN':
            RTPLANs.append(dd)
            RTPLANs_paths.append(first_file)
        elif dd['Modality'].value == 'RTDOSE':
            RTDOSEs.append(dd)
            RTDOSEs_paths.append(first_file)
            
    assert(len(MRs) == len(RTSTRUCTs) == len(RTPLANs) == len(RTDOSEs)), f"Did not find all required files."
    
    found = [False, False, False, False, False, False, False, False]
    file_paths = [None] * 8
    # sort for T1 or T2
    for MR, path in zip(MRs, MRs_paths):
        if "t1_" in MR['SeriesDescription'].value:
            MR_T1 = MR
            found[0] = True
            file_paths[0] = path
        elif "t2_" in MR['SeriesDescription'].value:
            MR_T2 = MR
            found[1] = True
            file_paths[1] = path
        else:
            raise Exception
            
    # assign RTSTRUCTs     
    for RTSTRUCT, path in zip(RTSTRUCTs, RTSTRUCTs_paths):
        
        refUID = RTSTRUCT['ReferencedFrameOfReferenceSequence'][0]['RTReferencedStudySequence'][0]              ['RTReferencedSeriesSequence'][0]['SeriesInstanceUID'].value
        
        MR_T1_UID = MR_T1['SeriesInstanceUID'].value
        MR_T2_UID = MR_T2['SeriesInstanceUID'].value
        
        if refUID == MR_T1_UID:
            RTSTRUCT_T1 = RTSTRUCT
            found[2] = True
            file_paths[2] = path
        elif refUID == MR_T2_UID:
            RTSTRUCT_T2 = RTSTRUCT
            found[3] = True
            file_paths[3] = path
            
    # assign RTPLANs        
    for RTPLAN, path in zip(RTPLANs, RTPLANs_paths):
        
        refUID = RTPLAN['ReferencedStructureSetSequence'][0]['ReferencedSOPInstanceUID'].value

        RTSTRUCT_T1_UID = RTSTRUCT_T1['SOPInstanceUID'].value
        RTSTRUCT_T2_UID = RTSTRUCT_T2['SOPInstanceUID'].value

        if refUID == RTSTRUCT_T1_UID:
            RTPLAN_T1 = RTPLAN
            found[4] = True
            file_paths[4] = path
        elif refUID == RTSTRUCT_T2_UID:
            RTPLAN_T2 = RTPLAN
            found[5] = True
            file_paths[5] = path
            
    # assign RTDOSEs        
    for RTDOSE, path in zip(RTDOSEs, RTDOSEs_paths):
        
        refUID = RTDOSE['ReferencedRTPlanSequence'][0]['ReferencedSOPInstanceUID'].value

        RTPLAN_T1_UID = RTPLAN_T1['SOPInstanceUID'].value
        RTPLAN_T2_UID = RTPLAN_T2['SOPInstanceUID'].value
        
        if refUID == RTPLAN_T1_UID:
            RTDOSE_T1 = RTPLAN
            found[6] = True
            file_paths[6] = path
        elif refUID == RTPLAN_T2_UID:
            RTDOSE_T2 = RTPLAN
            found[7] = True
            file_paths[7] = path
            
    assert(all(found)), f"Not all required files found"
    assert(all([p != None for p in file_paths]))

    # write files into new folder structure
    
    p = re.compile(r'VS-SEG-(\d+)')
    case_idx = int(p.findall(case)[0])

    print(case_idx)
    
    new_T1_path = os.path.join(output_path, 'vs_gk_' + str(case_idx) +'_t1')
    new_T2_path = os.path.join(output_path, 'vs_gk_' + str(case_idx) +'_t2')
    
    os.mkdir(new_T1_path)
    os.mkdir(new_T2_path)
    
    old_T1_folder = os.path.dirname(file_paths[0])
    old_T2_folder = os.path.dirname(file_paths[1])
    old_T1_files = natsorted(os.listdir(old_T1_folder))
    old_T2_files = natsorted(os.listdir(old_T2_folder))
    
    for file_idx, file in enumerate(old_T1_files):
        new_file_path = os.path.join(new_T1_path, 'IMG'+ str(file_idx).zfill(10) +'.dcm')
        shutil.copy(os.path.join(old_T1_folder, file), new_file_path)
        
    for file_idx, file in enumerate(old_T2_files):
        new_file_path = os.path.join(new_T2_path, 'IMG'+ str(file_idx).zfill(10) +'.dcm')
        shutil.copy(os.path.join(old_T2_folder, file), new_file_path)
        
    # copy RT files
    shutil.copy(file_paths[2], os.path.join(new_T1_path, 'RTSS.dcm'))
    shutil.copy(file_paths[3], os.path.join(new_T2_path, 'RTSS.dcm'))
    shutil.copy(file_paths[4], os.path.join(new_T1_path, 'RTPLAN.dcm'))
    shutil.copy(file_paths[5], os.path.join(new_T2_path, 'RTPLAN.dcm'))
    shutil.copy(file_paths[6], os.path.join(new_T1_path, 'RTDOSE.dcm'))
    shutil.copy(file_paths[7], os.path.join(new_T2_path, 'RTDOSE.dcm'))
    
print("Complete")

