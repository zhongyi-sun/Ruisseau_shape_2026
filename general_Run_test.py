import yaml
import multiprocessing
import sys
import os
import time
import shutil
import pandas as pd
import numpy as np
#from itertools import chain

from prepBckTal import prepBckTal
from calcDist import calcDist
from distProcessing import distProcessing
from isomapcoord import isomapcoord
from database_projection import database_projection
from MA_tools import MA_tools
from MA_basic_tools import MA_basic_tools
from general_tools import general_tools

DEFAULT_CONFIG_NAME = 'config_test.yaml'
#########################################################  2026 update  #########################################################
import traceback                 # for cluster log
from datetime import datetime    # for cluster log
import argparse
import json # for saving the dictionary
import Icp  # for calcOneSubj task CPU core pickle 
from distProcessing_manager import distProcessing_manager # for merging master results
# ----------------------------------------------------------------------
# A. CORE WORK UNIT (calcOneSubj) - Must be at top level
# ----------------------------------------------------------------------
# 1. DECLARE GLOBALS AT THE TOP
GLOBAL_FILE_LIST = None
GLOBAL_BCK_LIST = None

# 2. THE INITIALIZER (Runs once when each core starts)
def init_worker(shared_file_list, shared_bck_list):
    global GLOBAL_FILE_LIST, GLOBAL_BCK_LIST
    GLOBAL_FILE_LIST = shared_file_list
    GLOBAL_BCK_LIST = shared_bck_list

# 3. THE WORKER (Uses the globals)
def calcOneSubj(subj_i, regionICPPath, numSubj):
    """
    Executes the inner loop (j) and saves the results for one source subject (subj_i).
    Accesses point clouds from Global memory instead of passed arguments.
    """
    subj_1 = GLOBAL_FILE_LIST[subj_i]
    #bck_1 = np.array(GLOBAL_BCK_LIST[subj_i])
    bck_1 = GLOBAL_BCK_LIST[subj_i]  # faster than converting to np.array
    curCalcU_instance = calcDist(1)
    source_results = {} # Dictionary to store results for this source subject
    print(f"Process {os.getpid()} starting for source: {subj_1}")

    for j in range(numSubj): # Inner Loop (j)
        subj_2 = GLOBAL_FILE_LIST[j]

        if (subj_i != j):
            bck_2 = GLOBAL_BCK_LIST[j]  # faster than converting to np.array
            # 1. Perform the core calculation
            # NOTE: calcOneDist returns the structured result dictionary 
            dist_rot_trans = curCalcU_instance.calcOneDist_New(
                bck_1=bck_1,
                bck_2=bck_2
            )            
            dist, rot, trans = dist_rot_trans
            source_results[subj_2] = {
                'distance': float(dist), 
                'rotation': rot.tolist(), 
                'translation': trans.tolist(),
                'status': 'calculated',
                'source_subj': subj_1 # Optional: helpful to keep track
            }       
        else: # i==j, dentity case: Subject compared to itself
            source_results[subj_2] = {
                'distance': 0.0,
                'rotation': np.eye(3).tolist(), # [[1,0,0],[0,1,0],[0,0,1]]
                'translation': [0.0, 0.0, 0.0],
                'status': 'identity'
            }
    # Write out the single dictionary for this source subject
    os.makedirs(regionICPPath, exist_ok=True)  # ENSURE FOLDER EXISTS RIGHT BEFORE SAVING
    output_filename = os.path.join(regionICPPath, f'Source_{subj_1}_results.json')
    with open(output_filename, 'w') as f:
        json.dump(source_results, f, indent=4)
    print(f"Process {os.getpid()} finished for source: {subj_1}. File saved.")
    return subj_i # Return the index to track completion


# ----------------------------------------------------------------------
# B. ORCHESTRATION FUNCTION - Must be at top level
# ----------------------------------------------------------------------
def orchestrate_local_parallel(fileList, tal_bck_list, num_cores, regionICPPath, numSubj):
    from datetime import timedelta

    #### start for time verification ####
    print(f'Orchestrating local parallel with {num_cores} cores......')
    start_time = time.time()
    completed_count = 0
    def update_progress(result):
        nonlocal completed_count
        completed_count += 1
        # Calculate timing
        elapsed = time.time() - start_time
        avg_time_per_subj = elapsed / completed_count
        remaining_subjs = numSubj - completed_count
        # Estimate remaining time (scaled by number of cores)
        # Note: Since it's parallel, we estimate based on the "throughput"
        est_remaining_seconds = (remaining_subjs * avg_time_per_subj)
        eta = timedelta(seconds=int(est_remaining_seconds))
        print(f"   [Progress] {completed_count}/{numSubj} finished | "
              f"Avg: {avg_time_per_subj/60:.1f}m per subj | "
              f"Est. Remaining: {eta}")
    #### end for time verification ####

    pool = multiprocessing.Pool(
        processes=num_cores, 
        initializer=init_worker, 
        initargs=(fileList, tal_bck_list)
    )
    for i in range(numSubj): # Use apply_async to allow the callback to fire as each job finishes
        pool.apply_async(calcOneSubj, args=(i, regionICPPath, numSubj), callback=update_progress)
    pool.close()
    pool.join()
    total_elapsed = time.time() - start_time
    print(f"--- All {numSubj} subjects processed in {total_elapsed/3600:.2f} hours ---")


def main(argv):
    ###############################  Argparse parsing, cluster listening  ##################################
    parser = argparse.ArgumentParser(description="ICP Processing Script")
    parser.add_argument('--id', type=str, help="Subject ID (e.g. 100307) or Index (e.g. 5)",default=None)    
    parser.add_argument('--mode', default=None, choices=['local','local_parallel','cluster_job'])
    parser.add_argument('--num_cores', type=int, help="Number of CPU cores to use for parallel processing")
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_NAME)
    cmd_args = parser.parse_args()

    ###############################  Configuration loading  ##################################
    #config_path = '/media/sf_C_DRIVE/test_2025_6_Joy/config_test.yaml' # Use the full path if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Find the directory where THIS script is located
    config_path = os.path.join(script_dir, cmd_args.config)  # Assume the corresponding yaml file is in the same dir
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    ###############################  YAML extraction, find the YAML values  ##################################
    # Using .get() ensures it won't crash if the section is missing
    global_cfg = config.get('global', {})
    mode_from_yaml = global_cfg.get('mode', 'local_parallel')     
    call_dist_from_yaml = global_cfg.get('callCalcDist_bySubject', False)              
    local_cfg = global_cfg.get('local_config', {})   
    num_cores_from_yaml = local_cfg.get('num_cores', 8)    # Need num_cores from the nested local_config

    ###############################  Priority Decision  ##################################
    #  Highest Priority: Command Line Arguments (from the cluster).
    #  Second Priority: YAML file settings.
    #  Third Priority: Hardcoded defaults
    callCalcDist_bySubject = call_dist_from_yaml
    # Logic: Only worry about 'mode' if callCalcDist_bySubject is True
    mode = None
    if callCalcDist_bySubject:# Priority: Command line > YAML > Fallback default ('local_parallel') 
        if cmd_args.mode:    # 1. Check Command Line first
            mode = cmd_args.mode       
        elif mode_from_yaml: # 2. If Command Line is empty, check YAML
            mode = mode_from_yaml    
        else:                # 3. Final Fallback
            mode = 'local_parallel'  
        print(f"Running in {mode} mode.")
    else:
        print("callCalcDist_bySubject is False; skipping mode selection.")
    num_cores = cmd_args.num_cores if cmd_args.num_cores is not None else num_cores_from_yaml
    id = cmd_args.id # id usually only comes from Argparse in cluster mode


    ####################################  initialize tools  ####################################    
    curTools = general_tools(1)
    curMA_basic_U = MA_basic_tools(1)

    callPrepBckTal = config['global']['callPrepBckTal']
    call_Add_PrepBckTal = config['global']['call_Add_PrepBckTal']
    callCalcDistParallel_Legacy = config['global']['callCalcDistParallel_Legacy']
    callCalcDist_Legacy = config['global']['callCalcDist_Legacy']
    constructICPres = config['global']['constructICPres']
    constructICPres_Legacy = config['global']['constructICPres_Legacy']

    removeICPres = config['global']['removeICPres']
    removeJson = config['global']['removeJson']
    call_Add_Dist = config['global']['call_Add_Dist']
    call_Add_DistRecompose = config['global']['call_Add_DistRecompose']
    symDist_Legacy = config['global']['symDist_Legacy']
    outlierRemovalGetCenter_Legacy = config['global']['outlierRemovalGetCenter_Legacy']
    outlierRemovalGetCenter = config['global']['outlierRemovalGetCenter']       
    #######################################  Isomap functionalities  ####################################
    callGetIsomap = config['global']['callGetIsomap']
    callGetCoordWeight = config['global']['callGetCoordWeight']
    callGetNormWeight = config['global']['callGetNormWeight']    
    #######################################  MA functionalities  #######################################
    callPrepICPTrm = config['global']['callPrepICPTrm']    
    #######################################   subj meshes    ###########################################
    call_Add_TargetMesh = config['global']['call_Add_TargetMesh']
    callGenICPMesh = config['global']['callGenICPMesh']    
    #############  sub-proceses IF callGenICPMesh is true  #############
    saveOriMesh = config['global']['saveOriMesh']
    saveTalMesh = config['global']['saveTalMesh']
    saveRmesh = config['global']['saveRmesh']
    saveTalLMesh = config['global']['saveTalLMesh']
    saveTargetMesh = config['global']['saveTargetMesh']    
    #################################################################### 
    callGenISOMesh = config['global']['callGenISOMesh']       
    ######################################  MA ima and mesh  ###########################################  
    callPrepMA_ima = config['global']['callPrepMA_ima']   
    call_Add_PrepMA_ima = config['global']['call_Add_PrepMA_ima']  
    callGenMA_ima = config['global']['callGenMA_ima'] 
    callGenMA_mesh = config['global']['callGenMA_mesh']    
    ###########################################
    # datastruture and filenaming convention  #
    ###########################################
    generalRoot = config['global']['generalRoot']             
    curBase = config['global']['curBase']
    curRegion = config['global']['curRegion']
    path = generalRoot + curBase                     
    regionICPPath = path + os.sep + curRegion + os.sep + "ICP" + os.sep
    regionISOPath = path + os.sep + curRegion + os.sep + "Isomap" + os.sep 

# ======================================================================
    # DATA LOADING SUMMARY (Diagnostic Print)
    # ======================================================================
    print("\n" + "="*50)
    print("🚀 PIPELINE INITIALIZATION SUMMARY")
    print("="*50)
    print(f"  > Execution Mode:    {(mode or 'not active').upper()}")
    
    if mode == 'cluster_job':
        print(f"  > Cluster Index:     {id}")
    else:
        print(f"  > Parallel Cores:    {num_cores}")
        
    print(f"  > Run Calculation:   {'✅ YES' if callCalcDist_bySubject else '❌ NO'}")
    print(f"  > Region Target:     {curRegion}")
    print(f"  > Output Path:       {regionICPPath}")
    print("="*50 + "\n")
    # ======================================================================

#############################################################################################
#############################################################################################
    # Prepare bck and tal files for ICP calculation

    if callPrepBckTal:    
        oriBckLoc = path + os.sep + curRegion + os.sep
        outBckLoc = path + os.sep + curRegion + os.sep + "bck/" 
        outTalBckLoc = path + os.sep + curRegion + os.sep + "talai_bck/" 
        oriTalLoc = path + '/tal/' 

        curPrepBckTalU = prepBckTal(1)
        curPrepBckTalU.prepTal(oriTalLoc)
        
        print("Calling prepBckTal...")
        curPrepBckTalU.prepBck(oriBckLoc=oriBckLoc,outBckLoc=outBckLoc,outTalBckLoc=outTalBckLoc,oriTalLoc=oriTalLoc)


#############################################################################################
#############################################################################################
    # Prepare bck and tal files for ICP calculation


    if call_Add_PrepBckTal:
        old_base_1 = config['functions']['call_Add_PrepBckTal']['old_base_1']
        old_base_2 = config['functions']['call_Add_PrepBckTal']['old_base_2']
        new_base = config['functions']['call_Add_PrepBckTal']['new_base']
        old_db_version = config['functions']['call_Add_PrepBckTal']['old_db_version']
        new_db_version = config['functions']['call_Add_PrepBckTal']['new_db_version']       
        pathOldRoot = generalRoot + old_base_1                   
        pathNewRoot = generalRoot + old_base_2                  
        pathCombRoot = generalRoot + new_base   

        combTalLoc = pathCombRoot + '/tal/' 
        comb_region_loc = pathCombRoot+os.sep+curRegion+os.sep   
        old_region_loc = pathOldRoot+os.sep+curRegion+os.sep
        new_region_loc = pathNewRoot+os.sep+curRegion+os.sep 
        bck_old_path_left = old_region_loc+'left'
        bck_old_path_right = old_region_loc+'right'
        tal_old_path = pathOldRoot + os.sep + 'tal' + os.sep
        bck_new_path_left = new_region_loc+'left'
        bck_new_path_right = new_region_loc+'right'
        tal_new_path = pathNewRoot + os.sep + 'tal' + os.sep         
        bck_comb_path_left = comb_region_loc+'left'
        bck_comb_path_right = comb_region_loc+'right'               
        tal_comb_path = pathCombRoot + os.sep + 'tal' + os.sep              
        if (os.path.exists(pathCombRoot)==0):
            os.mkdir(pathCombRoot)
        if (os.path.exists(comb_region_loc)==0):
            os.mkdir(comb_region_loc)
        if (os.path.exists(combTalLoc)==0):
            os.mkdir(combTalLoc)   
        if (os.path.exists(bck_comb_path_left)==0):
            os.mkdir(bck_comb_path_left)
        if (os.path.exists(bck_comb_path_right)==0):
            os.mkdir(bck_comb_path_right)

        # copy bck and tal files to the comb dir        
        extension = '.bck'
        curTools.merge_directories(bck_old_path_left, bck_new_path_left, bck_comb_path_left,extension)
        curTools.merge_directories(bck_old_path_right, bck_new_path_right, bck_comb_path_right,extension)        
        extension = '.trm'
        curTools.merge_directories(tal_old_path,tal_new_path,tal_comb_path,extension)
        # prepTal and Bck in the comb dir    
        outBckLoc = pathCombRoot + os.sep + curRegion + os.sep + "bck/" 
        outTalBckLoc = pathCombRoot + os.sep + curRegion + os.sep + "talai_bck/" 
        oriTalLoc = tal_comb_path

        curPrepBckTalU = prepBckTal(1)
        curPrepBckTalU.prepTal(combTalLoc)

        curPrepBckTalU.prepBck(oriBckLoc=comb_region_loc,outBckLoc=outBckLoc,outTalBckLoc=outTalBckLoc,oriTalLoc=tal_comb_path)

        # prepare old and new db if needed, make bck dir and write the names and json bck file
        if old_db_version == 'v1':
            # prepTal and Bck in the old dir
            old_region_bck_loc = old_region_loc + "bck/" 
            if (os.path.exists(old_region_bck_loc)==0):
                os.mkdir(old_region_bck_loc)
            old_tal_bck_loc = old_region_loc + "talai_bck/"
            if (os.path.exists(old_tal_bck_loc)==0):
                os.mkdir(old_tal_bck_loc)
            oriTalLoc = tal_old_path

            curPrepBckTalU = prepBckTal(1)
            curPrepBckTalU.prepTal(oriTalLoc)
            curPrepBckTalU.prepBck(oriBckLoc=old_region_loc,outBckLoc=old_region_bck_loc,outTalBckLoc=old_tal_bck_loc,oriTalLoc=tal_old_path,writeTalai=False)

        if new_db_version == 'v1':
            # prepTal and Bck in the new dir                
            new_region_bck_loc = new_region_loc + "bck/" 
            if (os.path.exists(new_region_bck_loc)==0):            
                os.mkdir(new_region_bck_loc)
            new_tal_bck_loc = new_region_loc + "talai_bck/"
            if (os.path.exists(new_tal_bck_loc)==0):            
                os.mkdir(new_tal_bck_loc)
            oriTalLoc = tal_new_path

            curPrepBckTalU = prepBckTal(1)
            curPrepBckTalU.prepTal(oriTalLoc)
            curPrepBckTalU.prepBck(oriBckLoc=new_region_loc,outBckLoc=new_region_bck_loc,outTalBckLoc=new_tal_bck_loc,oriTalLoc=tal_new_path,writeTalai=False)



#############################################################################################
#####################################  Calculation ICP ######################################
#############################################################################################

#############################################################################################
    if callCalcDist_Legacy:
        if (os.path.exists(regionICPPath)==0):
            os.mkdir(regionICPPath)
        namesFile = path+os.sep+curRegion+os.sep+"bck"+os.sep+"subjNames_bck.txt"
        talBckFile = path+os.sep+curRegion+os.sep+"bck"+os.sep+"tal_L_bck.json"
        fileList = curTools.retrieveNames(namesFile)
        tal_bck_list = curTools.retrieveFloatListsJson(talBckFile)
        numSubj = len(fileList)

        curCalcU = calcDist(1)
        start = time.time()
        for i in range(numSubj):
            subj_1 = fileList[i]
            print(subj_1)
            bck_1 = np.array(tal_bck_list[i])
            for j in range(numSubj):
                if (i != j):
                    subj_2 = fileList[j]
                    bck_2 = np.array(tal_bck_list[j])
                    curCalcU.calcOneDist_Legacy(outPath=regionICPPath,subj_1=subj_1,subj_2=subj_2,bck_1=bck_1,bck_2=bck_2)
        end = time.time()
        print("ICP took %u seconds" % (end - start))

##############################################################################################
    if callCalcDistParallel_Legacy:
        if (os.path.exists(regionICPPath)==0):
            os.mkdir(regionICPPath)
        namesFile = path+os.sep+curRegion+os.sep+"bck"+os.sep+"subjNames_bck.txt"
        talBckFile = path+os.sep+curRegion+os.sep+"bck"+os.sep+"tal_L_bck.json"
        fileList = curTools.retrieveNames(namesFile)
        tal_bck_list = curTools.retrieveFloatListsJson(talBckFile)
        numSubj = len(fileList)

        curCalcU = calcDist(1)
        start = time.time()
        num_processes = multiprocessing.cpu_count()
        print('Number of cpu: '+str(num_processes))
        print('Calculating...')
        i_range, j_range = range(numSubj), range(numSubj)
        tasks = [(regionICPPath,fileList[i], fileList[j],np.array(tal_bck_list[i]),np.array(tal_bck_list[j])) for i in i_range for j in j_range if i != j]
        pool = multiprocessing.Pool(processes=num_processes)
        pool.starmap(curCalcU.calcOneDist_Legacy, tasks)
        pool.close()
        pool.join()
        end = time.time()
        print("ICP took %u seconds" % (end - start))


##################################### 2026 update #####################################
    if callCalcDist_bySubject:
        try:
            if not os.path.exists(regionICPPath):
                os.makedirs(regionICPPath, exist_ok=True)
                
            if mode in ['local_parallel', 'local']:  # --- Execute based on Mode ---
                start_time = time.time()  # 1. Record the start time
                print('Number of cores asked: '+str(num_cores))     
                num_processors = multiprocessing.cpu_count() - 1 # -1 leave one core for the main process
                print('Number of cpu of the machine: '+str(num_processors))
                if num_cores > num_processors:
                    print('Number of cores asked is greater than the number of cpu of the machine. Using all the cpu of the machine.')
                    num_cores = num_processors
                elif num_cores < 1:
                    print('Number of cores asked is less than 1. Using 1 core.')
                    num_cores = 1
                else:
                    print('Number of cores asked is less than the number of cpu of the machine. Using the number of cores asked.')
                    num_cores = num_cores
                print('Number of cores used: '+str(num_cores))

                namesFile = path+os.sep+curRegion+os.sep+"bck"+os.sep+"subjNames_bck.txt"
                talBckFile = path+os.sep+curRegion+os.sep+"bck"+os.sep+"tal_L_bck.json"
                fileList = curTools.retrieveNames(namesFile)
                tal_bck_list_raw = curTools.retrieveFloatListsJson(talBckFile) # to handle ragged arrays
                tal_bck_list = [np.array(item) for item in tal_bck_list_raw]   # formatted as np.array
                numSubj = len(fileList)

                orchestrate_local_parallel(fileList, tal_bck_list, num_cores,regionICPPath, numSubj)
                # Calculate and print
                end_time = time.time()
                elapsed = end_time - start_time
                print(f"\n--- Processing Complete ---")
                if elapsed > 3600:
                    print(f"Total time: {elapsed / 3600:.2f} hours")
                else:
                    print(f"Total time: {elapsed / 60:.2f} minutes")
                log_path = os.path.join(regionICPPath, "execution_log.txt")
                with open(log_path, "a") as log:
                    log.write(f"Region: {curRegion} | Cores: {num_cores} | Subjects: {numSubj} | Time: {elapsed / 60:.2f} min\n")

            elif mode == 'cluster_job':
                if not cmd_args.id: # Check if ID was provided
                    raise ValueError("Cluster mode requires an --id (Index or Subject Name).")
                namesFile = os.path.join(path, curRegion, "bck", "subjNames_bck.txt")  # Load reference files
                fileList = curTools.retrieveNames(namesFile)   

                # set Globals MANUALLY so the worker function can find them
                global GLOBAL_FILE_LIST, GLOBAL_BCK_LIST
                GLOBAL_FILE_LIST = fileList

                ##################### --- START SMART DETECTION --- #####################
                raw_val = cmd_args.id
                if raw_val.isdigit():
                    id = int(raw_val)  # If user typed a number, it's an index
                    if id >= len(fileList):
                        raise IndexError(f"ID {id} is out of bounds. Only {len(fileList)} subjects available.")
                else:
                    if raw_val in fileList:  # If user typed a name, find where that name is in the list
                        id = fileList.index(raw_val)
                    else:
                        raise ValueError(f"Subject ID '{raw_val}' not found in {namesFile}")

                #################### --- END SMART DETECTION --- #####################

                # set tal_bck_list AFTER smart detection, in case it is very long, do it only if no user typo
                talBckFile = os.path.join(path, curRegion, "bck", "tal_L_bck.json")
                tal_bck_list = curTools.retrieveFloatListsJson(talBckFile)   
                GLOBAL_BCK_LIST = tal_bck_list  

                # Call the worker function directly for just ONE subject (the one assigned by the cluster)
                print(f"--- Cluster Job Started ---")
                print(f"Region: {curRegion}")
                print(f"Subject Index: {id}")
                print(f"Subject Name: {fileList[id]}")
                print(f"---------------------------")
                calcOneSubj(id, regionICPPath, len(fileList))

            else:
                raise ValueError(f"Unknown execution mode: {mode}")
        except Exception as e:    
            os.makedirs(regionICPPath, exist_ok=True) # Ensure the directory exists before logging the crash
            # --- The Logger Catch ---
            error_file = os.path.join(regionICPPath, f"ERROR_log_{mode}.txt")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
            with open(error_file, "a") as f:
                f.write(f"\n{'!'*60}\n")
                f.write(f"CRASH OCCURRED AT: {timestamp}\n")
                f.write(f"MODE: {mode} | INDEX: {id if mode=='cluster_job' else 'N/A'}\n") ########## WIP souce_index to id
                f.write(f"ERROR TYPE: {type(e).__name__}\n")
                f.write(f"ERROR MESSAGE: {str(e)}\n")
                f.write("FULL TRACEBACK:\n")
                f.write(traceback.format_exc())
                f.write(f"{'!'*60}\n")
                
            print(f"\n❌ CRITICAL ERROR: Saved to {error_file}")
            sys.exit(1) # Tell the cluster the job actually failed



##############################################################################################
############################  WIP  ############################
    if constructICPres:
        namesFile = path+os.sep+curRegion+os.sep+"bck"+os.sep+"subjNames_bck.txt"
        fileList = curTools.retrieveNames(namesFile)
        if (os.path.exists(regionISOPath)==0):
            os.mkdir(regionISOPath)

        manager = distProcessing_manager(regionICPPath,regionISOPath) # Initialize (The class 'memorizes' the path here)
        manager.merge_results()  # Consolidate all the small JSONs into memory
        manager.save_distance_matrix(fileList, curRegion) # Write the N x N Matrix (for math/Isomap)
        manager.save_symmetric_matrices(curRegion) # Create the Symmetric versions (Min/Max)
        manager.save_transformation_df(curRegion) # Write the Clean Table (for transformations/geometry)
        #manager.save_matrices(fileList, curRegion)  # OLD format, generates both the Distance and TransRot files

##############################################################################################
    if constructICPres_Legacy:
        numTransRot = 12
        namesFile = path+os.sep+curRegion+os.sep+"bck"+os.sep+"subjNames_bck.txt"
        fileList = curTools.retrieveNames(namesFile)
        curProcessingU = distProcessing(1)       
        curProcessingU.constructICPresult(curRegion=curRegion,regionICPPath=regionICPPath,numTransRot=numTransRot,fileList=fileList)




##################################################################################################
    if call_Add_Dist:
        """
        Pre: 
             The bck files are stored in path/region/bck/tal_L_bck.json
        """        
        old_base_1 = config['functions']['call_Add_Dist']['old_base_1']
        old_base_2 = config['functions']['call_Add_Dist']['old_base_2']
        new_base = config['functions']['call_Add_Dist']['new_base']        
        
        pathOldRoot = generalRoot + old_base_1                   
        pathNewRoot = generalRoot + old_base_2                  
        pathCombRoot = generalRoot + new_base   

        pathOldBck = pathOldRoot + os.sep + curRegion + os.sep + "bck/"
        pathNewBck = pathNewRoot + os.sep + curRegion + os.sep + "bck/"   
        pathCombRegion = pathCombRoot + os.sep + curRegion
        pathCombICP = pathCombRegion + os.sep + "ICP/"
        if not os.path.exists(pathCombICP):
            os.mkdir(pathCombICP)

        # defining old and new subj for calculation
        namesFile_old = pathOldBck+"subjNames_bck.txt"
        talBckFile_old = pathOldBck+"tal_L_bck.json"
        fileList_old = curTools.retrieveNames(namesFile_old)
        tal_bck_list_old = curTools.retrieveFloatListsJson(talBckFile_old)
        numSubj_old = len(fileList_old)
        namesFile_new = pathNewBck+"subjNames_bck.txt"
        talBckFile_new = pathNewBck+"tal_L_bck.json"
        fileList_new = curTools.retrieveNames(namesFile_new)
        tal_bck_list_new = curTools.retrieveFloatListsJson(talBckFile_new)
        numSubj_new = len(fileList_new)
            
        # calculating the rows and columns
        curCalcU = calcDist(1)
        start = time.time()
        num_processes = multiprocessing.cpu_count()
        print('numProcess: ' + str(num_processes))
        print("Start calculating columns:")
        i_range, j_range = range(numSubj_old), range(numSubj_new)
        curType = 'cols_'
        tasks = [(pathCombICP,curType,fileList_old[i], fileList_new[j],np.array(tal_bck_list_old[i]),np.array(tal_bck_list_new[j])) for i in i_range for j in j_range]
        pool = multiprocessing.Pool(processes=num_processes)
        pool.starmap(curCalcU.calcAddOneDist, tasks)
        pool.close()
        pool.join()

        print("Start calculating rows:")
        i_range, j_range = range(numSubj_new), range(numSubj_old)
        curType = 'rows_'
        tasks = [(pathCombICP,curType,fileList_new[i], fileList_old[j],np.array(tal_bck_list_new[i]),np.array(tal_bck_list_old[j])) for i in i_range for j in j_range]
        pool = multiprocessing.Pool(processes=num_processes)
        pool.starmap(curCalcU.calcAddOneDist, tasks)
        pool.close()
        pool.join()

        end = time.time()
        print("ICP took %u seconds" % (end - start))



#########################################################################################################################
    if call_Add_DistRecompose:
        """
            Pre: The calcuations for old and new are done already and stored in 
             pathOldRoot/curRegion/ICP and pathNewRoot/curRegion/ICP.
        """
        old_base_1 = config['functions']['call_Add_DistRecompose']['old_base_1']
        old_base_2 = config['functions']['call_Add_DistRecompose']['old_base_2']
        new_base = config['functions']['call_Add_DistRecompose']['new_base']
        old_db_version = config['functions']['call_Add_DistRecompose']['old_db_version']
        new_db_version = config['functions']['call_Add_DistRecompose']['new_db_version']                       
        pathOldRoot = generalRoot + old_base_1                   
        pathNewRoot = generalRoot + old_base_2                  
        pathCombRoot = generalRoot + new_base   

        numTransRot = 12
        namesFileOld = pathOldRoot+os.sep+curRegion+os.sep+"bck"+os.sep+"subjNames_bck.txt"
        fileListOld = curTools.retrieveNames(namesFileOld)
        namesFileNew = pathNewRoot+os.sep+curRegion+os.sep+"bck"+os.sep+"subjNames_bck.txt"
        fileListNew = curTools.retrieveNames(namesFileNew)
        pathRegionICP = pathCombRoot + os.sep + curRegion + os.sep + "ICP" + os.sep


        # recompose the distMat and transRotMat
        curProcessingU = distProcessing(1)
        curProcessingU.constructAddICPresult(curRegion=curRegion,pathRegionICP=pathRegionICP,numTransRot=numTransRot,fileListOld=fileListOld,fileListNew=fileListNew) 
        curProcessingU.constructAddFinalResult(curRegion=curRegion,pathOldRoot=pathOldRoot,pathNewRoot=pathNewRoot,pathRegionICP=pathRegionICP,old_db_version=old_db_version,new_db_version=new_db_version) 


#######################################################################################################################
    if removeICPres:
        namesFile = path+os.sep+curRegion+os.sep+"bck"+os.sep+"subjNames_bck.txt"
        fileList = curTools.retrieveNames(namesFile)
        numSubj = len(fileList)

        for i in range(numSubj):
            subj_1 = fileList[i]
            for j in range(numSubj):
                if (i != j):
                    subj_2 = fileList[j]
                    resName = regionICPPath + subj_1 + '_' + subj_2 + '.json'
                    print("resName: "+resName)
                    if os.path.exists(resName):
                        os.remove(resName)
                    else:
                        print(f"{resName} does not exist.")
                else:
                    sameSubjRes = regionICPPath + fileList[i] + '_' + fileList[j] + '.json'
                    if os.path.exists(sameSubjRes):
                        os.remove(sameSubjRes)


#######################################################################################################################
    if removeJson:
        dir_to_remove = path+os.sep+curRegion+os.sep+'ICP'+os.sep
        files = os.listdir(dir_to_remove)
        # Iterate over the files
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(dir_to_remove, file)
                os.remove(file_path)
                print(f"Removed file: {file_path}")        


##################################################################################################
######################################   dist processing   #######################################
##################################################################################################

##################################################################################################
    if symDist_Legacy:
        if (os.path.exists(regionISOPath)==0):
            os.mkdir(regionISOPath)

        distMatName = regionICPPath + 'distMat_' + curRegion + '.txt'
        outFileNameMin = regionISOPath + 'minDist' + curRegion + '.txt'
        outFileNameMax = regionISOPath + 'maxDist' + curRegion + '.txt' 

        curProcessingU = distProcessing(1)       
        curProcessingU.symDist(distMatName=distMatName,outFileNameMin=outFileNameMin,outFileNameMax=outFileNameMax)


##################################################################################################
    ###########################
    ##  for outlier removal  ##
    ###########################
    distMatMaxName = regionISOPath + 'maxDist' + curRegion + '.txt'
    distMatMinName = regionISOPath + 'minDist' + curRegion + '.txt'
    fileNameSubjOut = regionISOPath + 'maxDist' + curRegion + 'outlierNames.txt'
    fileNameMaxDistOut = regionISOPath + 'maxDist' + curRegion + 'outlierRemoved.txt'
    fileNameMinDistOut = regionISOPath + 'minDist' + curRegion + 'outlierRemoved.txt'
    maxCenterName = regionISOPath + 'maxcenter' + curRegion + '.txt'
    minCenterName = regionISOPath + 'mincenter' + curRegion + '.txt'
    maxCenterListName = regionISOPath + 'maxcenterList' + curRegion + '.txt'
    minCenterListName = regionISOPath + 'mincenterList' + curRegion + '.txt'
        
    curOutMethod = config['functions']['outlierRemovalGetCenter']['curOutMethod']
    sdFactor = config['functions']['outlierRemovalGetCenter']['sdFactor']  
    if outlierRemovalGetCenter_Legacy:               
        curProcessingU = distProcessing(1) 
              #curProcessingU.removeOutlierGetCenter_old(subjListFileName=subjListFileName,distMatMaxName=distMatMaxName,distMatMinName=distMatMinName,fileNameSubjKeep=fileNameSubjKeep,fileNameSubjOut=fileNameSubjOut,fileNameMaxDistOut=fileNameMaxDistOut,fileNameMinDistOut=fileNameMinDistOut,maxCenterName=maxCenterName,minCenterName=minCenterName,maxCenterListName=maxCenterListName,minCenterListName=minCenterListName,method=curOutMethod,sdFactor=sdFactor)
        curProcessingU.removeOutlierGetCenter(distMatMaxName=distMatMaxName,distMatMinName=distMatMinName,fileNameSubjOut=fileNameSubjOut,fileNameMaxDistOut=fileNameMaxDistOut,fileNameMinDistOut=fileNameMinDistOut,maxCenterName=maxCenterName,minCenterName=minCenterName,maxCenterListName=maxCenterListName,minCenterListName=minCenterListName,method=curOutMethod,sdFactor=sdFactor)
    if outlierRemovalGetCenter:               
        curProcessingU = distProcessing_manager(regionICPPath,regionISOPath) 
              #curProcessingU.removeOutlierGetCenter_old(subjListFileName=subjListFileName,distMatMaxName=distMatMaxName,distMatMinName=distMatMinName,fileNameSubjKeep=fileNameSubjKeep,fileNameSubjOut=fileNameSubjOut,fileNameMaxDistOut=fileNameMaxDistOut,fileNameMinDistOut=fileNameMinDistOut,maxCenterName=maxCenterName,minCenterName=minCenterName,maxCenterListName=maxCenterListName,minCenterListName=minCenterListName,method=curOutMethod,sdFactor=sdFactor)
        curProcessingU.removeOutlierGetCenter(distMatMaxName=distMatMaxName,distMatMinName=distMatMinName,fileNameSubjOut=fileNameSubjOut,fileNameMaxDistOut=fileNameMaxDistOut,fileNameMinDistOut=fileNameMinDistOut,maxCenterName=maxCenterName,minCenterName=minCenterName,maxCenterListName=maxCenterListName,minCenterListName=minCenterListName,method=curOutMethod,sdFactor=sdFactor)


#######################################################################################################
#####################################     calculate isomap     ########################################
#######################################################################################################
    if callGetIsomap:
        filterOut = config['functions']['callGetIsomap']['filterOut']
        typeIso = config['functions']['callGetIsomap']['typeIso']  
        numNeig = config['functions']['callGetIsomap']['numNeig']
        typeDist = config['functions']['callGetIsomap']['typeDist']
        numDim = config['functions']['callGetIsomap']['numDim']
        curIsomapU = isomapcoord(1)   

        curIsomapU.getIsomap(sulciName=curRegion,ICPfilePath=regionISOPath,ISOfilePath=regionISOPath,typeIso=typeIso,numNeig=numNeig,filterOut=filterOut,typeDist=typeDist,numDim=numDim)
        

#######################################################################################################
#################################     get coordinates and weights     #################################
#######################################################################################################        
    if callGetCoordWeight:
        filterOut = config['functions']['callGetCoordWeight']['filterOut']
        typeIso = config['functions']['callGetCoordWeight']['typeIso'] 
        numNeig = config['functions']['callGetCoordWeight']['numNeig']
        typeDist = config['functions']['callGetCoordWeight']['typeDist']
        numDim = config['functions']['callGetCoordWeight']['numDim']
        numCoord = config['functions']['callGetCoordWeight']['numCoord']
        scale = config['functions']['callGetCoordWeight']['scale']       
        coordStep = config['functions']['callGetCoordWeight']['coordStep']                                      
        extremityPercent = config['functions']['callGetCoordWeight']['extremityPercent']  
        coordPercent = config['functions']['callGetCoordWeight']['coordPercent']   
        minExtremitySubject = config['functions']['callGetCoordWeight']['minExtremitySubject']
        distThresh = config['functions']['callGetCoordWeight']['distThresh']
                            
        outFileNameMin = regionISOPath + 'minDist' + curRegion + '.txt'
        outFileNameMax = regionISOPath + 'maxDist' + curRegion + '.txt' 

        curIsomapU = isomapcoord(1)
        coordDim = 1  # the numDim_coordDim                     
        for i in range(numDim):
            coordDim = i + 1
            curDistName,isomapName,coordFileName,outDetailName,outSimpleName = "","","","",""
            if filterOut:
                curDistName = regionISOPath + typeDist + 'Dist' + curRegion + 'outlierRemoved.txt'
                isomapName = regionISOPath+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_dist'+typeDist+'.txt'

                coordFileName = regionISOPath+'coord'+str(numCoord)+typeIso+'k'+str(numNeig)+'dim'+str(numDim)+'_'+str(coordDim)+'dist'+typeDist+'filterOut.txt'
                outDetailName = regionISOPath+'nameDCoord'+str(numCoord)+typeIso+'k'+str(numNeig)+'dim'+str(numDim)+'_'+str(coordDim)+'dist'+typeDist+'filterOut.txt'
                outSimpleName = regionISOPath+'spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'SubjNames.txt'

                # test new coord alg
                coordFileNameNew = regionISOPath+'coord'+str(numCoord)+typeIso+'k'+str(numNeig)+'dim'+str(numDim)+'_'+str(coordDim)+'dist'+typeDist+'filterOut_new.txt'
                outDetailNameNew = regionISOPath+'nameDCoord'+str(numCoord)+typeIso+'k'+str(numNeig)+'dim'+str(numDim)+'_'+str(coordDim)+'dist'+typeDist+'filterOut_new.txt'
                outSimpleNameNew = regionISOPath+'spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'SubjNames_new.txt'
                

                noneNormWeightName = regionISOPath+'dimSpamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+'.txt'
                oriNormWeightName = regionISOPath+'spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+'.txt'

            else:
                curDistName = regionISOPath + typeDist + 'Dist' + curRegion + '.txt'
                isomapName = regionISOPath+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_dist'+typeDist+'_keepOut.txt'
                
                coordFileName = regionISOPath+'coord'+str(numCoord)+typeIso+'k'+str(numNeig)+'dim'+str(numDim)+'_'+str(coordDim)+'dist'+typeDist+'filterNone.txt'
                outDetailName = regionISOPath+'nameDCoord'+str(numCoord)+typeIso+'k'+str(numNeig)+'dim'+str(numDim)+'_'+str(coordDim)+'dist'+typeDist+'filterNone.txt'
                outSimpleName = regionISOPath+'spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'SubjNames_keepOut.txt'

                # test new coord alg
                coordFileNameNew = regionISOPath+'coord'+str(numCoord)+typeIso+'k'+str(numNeig)+'dim'+str(numDim)+'_'+str(coordDim)+'dist'+typeDist+'filterNone_new.txt'
                outDetailNameNew = regionISOPath+'nameDCoord'+str(numCoord)+typeIso+'k'+str(numNeig)+'dim'+str(numDim)+'_'+str(coordDim)+'dist'+typeDist+'filterNone_new.txt'
                outSimpleNameNew = regionISOPath+'spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'SubjNames_keepOut_new.txt'


                noneNormWeightName = regionISOPath+'dimSpamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+'_keepOut.txt'
                oriNormWeightName = regionISOPath+'spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+'_keepOut.txt'

            if (extremityPercent == True):
                curIsomapU.getCoordAndSubj(curDistName=curDistName,maxDistName=outFileNameMax,isomapName=isomapName,coordFileName=coordFileName,outDetailName=outDetailName,outSimpleName=outSimpleName,coordStep=coordStep,coordPercent=coordPercent,minExtremitySubject=-1,numCoord=numCoord,distThresh=distThresh)
            if (extremityPercent == False):
                curIsomapU.getCoordAndSubj(curDistName=curDistName,maxDistName=outFileNameMax,isomapName=isomapName,coordFileName=coordFileName,outDetailName=outDetailName,outSimpleName=outSimpleName,coordStep=coordStep,coordPercent=-1,minExtremitySubject=minExtremitySubject,numCoord=numCoord,distThresh=distThresh)            
                #curIsomapU.getCoordAndSubj_new(isomapName=isomapName,coordFileName=coordFileNameNew,outDetailName=outDetailNameNew,outSimpleName=outSimpleNameNew,coordPercent=coordPercent,minExtremitySubject=minExtremitySubject,numCoord=numCoord,coordDim = str(coordDim),distThresh=distThresh)
            # get callGetNoneNormWeight:
            curIsomapU.getNoneNormWeightOriWeight(curDistName=curDistName,isomapName=isomapName,coordFileName=coordFileName,noneNormWeightName=noneNormWeightName,oriNormWeightName=oriNormWeightName,coordSubjName=outSimpleName,numCoord=numCoord,scale=scale)



###################################################################################################
###############################   calculate normWeight   ##########################################
###################################################################################################
    if callGetNormWeight:
        curIsomapU = isomapcoord(1) 

        filterOut = config['functions']['callGetNormWeight']['filterOut']
        typeIso = config['functions']['callGetNormWeight']['typeIso']
        numNeig = config['functions']['callGetNormWeight']['numNeig']
        typeDist = config['functions']['callGetNormWeight']['typeDist']
        numDim = config['functions']['callGetNormWeight']['numDim']
        numCoord = config['functions']['callGetNormWeight']['numCoord']
        scale = config['functions']['callGetNormWeight']['scale']

        noneNormWeightName =  regionISOPath+'dimSpamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+'.txt'
        normWeightOutFile =  regionISOPath+'normSpamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+'_testProjection.txt'


        # Read subjNames from a names file
        #subjNames = pd.read_csv(subjNameFile,sep=' ',index_col=0,header=None)        
        #subjNames = np.concatenate(np.array(subjNames.values))

        # subjNames from an existing weight file
        dimDist = pd.read_csv(noneNormWeightName,sep=' ',index_col=0)  
        subjNames = dimDist.index.tolist()
        subjNames = subjNames[:10]

        # subjNames from a projection directory
        process = 'Anatomy'                     #### TO SPECIFY ####
        curList,pbList = [],[]
        baseLoc = path
        if (process=='Anatomy'):
            curList = curIsomapU.getAnatomyProjectionNames(baseLoc,curAct)
        else: # HCP-specific
            curList, pbList = curIsomapU.getHCPprojectionNames()
            print("Problem images: "+pbList)    

        curActCommonNames = list(set(curList).intersection(set(subjNames)))
        curIsomapU.getNormWeight(subjNames=curActCommonNames,noneNormWeightName=noneNormWeightName,normWeightOutFile=normWeightOutFile)



##############################################    WIP projection    ############################################
    ####################    
    ## for projection ##
    ####################
    projection = 'Anatomy' # Anatomy,'HCPactivation','HCPmask','HCPbundleM','HCPcurvature','HCPmyelin','HCPbundle'
    subProcess = ""  # eg: 2
    activation = ""  # eg: /neurospin/dico/zsun/3T_tfMRI_conversion/
    curAct = "preCS" # eg: language
################################################################################################################



#######################################################################################################
##########################################   prep Trms   ##############################################
#######################################################################################################
    if callPrepICPTrm: 
        target_file_name = regionISOPath+'mincenterList'+curRegion+'.txt'
        print("targetFile: "+target_file_name)
        target_name = curTools.getCenterSubj(target_file_name,rankPosition=1)
        print("target: "+target_name)

        oriTalLoc = path + '/tal/' 
        df_tal_L_name = oriTalLoc + 'df_tal_L.csv'
        df_allICP_name = regionICPPath + 'transRotMat_' + curRegion + '.txt'
        df_tal_L_targetICP_name = regionICPPath + 'transRotMat_tal_L_' + target_name + '_' + curRegion + '.txt'
        df_tal_L_targetICP_voirT_name = regionICPPath + 'transRotMat_tal_L_' + target_name + '_voirT_' + curRegion + '.txt'

        #curMA_basic_U.prep_ICP_Trm(df_tal_L_name,df_allICP_name,target_name,df_tal_L_targetICP_name,df_tal_L_targetICP_voirT_name)
        curMA_basic_U.prep_ICP_Trm_New(df_tal_L_name,df_allICP_name,target_name,df_tal_L_targetICP_name,df_tal_L_targetICP_voirT_name)



#################################################################################################
#####################################   generate Meshes   #######################################
#################################################################################################
    if callGenICPMesh:
        target_file_name = regionISOPath+'mincenterList'+curRegion+'.txt'
        target_name = curTools.getCenterSubj(target_file_name,rankPosition=1)
        print(target_file_name)
        print(target_name)
        curMA_basic_U.prep_ICP_Mesh(path,curRegion,target_name,saveOriMesh,saveTalMesh,saveRmesh,saveTalLMesh,saveTargetMesh)      
        

#################################################################################################
    if call_Add_TargetMesh:   
        old_base_1 = config['functions']['call_Add_TargetMesh']['old_base_1']
        old_base_2 = config['functions']['call_Add_TargetMesh']['old_base_2']
        new_base = config['functions']['call_Add_TargetMesh']['new_base']
        old_db_version = config['functions']['call_Add_TargetMesh']['old_db_version']
        new_db_version = config['functions']['call_Add_TargetMesh']['new_db_version']       
        pathOldRoot = generalRoot + old_base_1                   
        pathNewRoot = generalRoot + old_base_2                  
        pathCombRoot = generalRoot + new_base   
                
        pathOldMesh,pathNewMesh = '',''
        if old_db_version == "v1":
            pathOldMesh = pathOldRoot+os.sep+'dataProcessing'+os.sep+'SPAM'+os.sep+curRegion+os.sep+'mesh'+os.sep
        if old_db_version == "v2":
            pathOldMesh =  pathOldRoot + os.sep + curRegion + os.sep + "mesh_ori" + os.sep 
        if new_db_version == "v1":
            pathNewMesh = pathNewRoot+os.sep+'dataProcessing'+os.sep+'SPAM'+os.sep+curRegion+os.sep+'mesh'+os.sep
        if new_db_version == 'v2':
            pathNewMesh = pathNewRoot + os.sep + curRegion + os.sep + "mesh_ori" + os.sep

        print('pathOldMesh:')
        print(pathOldMesh)
        print('pathNewMesh:')
        print(pathNewMesh)

        target_file_name = pathCombRoot+os.sep+curRegion+os.sep+'Isomap'+os.sep+'mincenterList'+curRegion+'.txt'
        target_name = curTools.getCenterSubj(target_file_name,rankPosition=1)
        print(target_file_name)
        print('Center is: ' + target_name)
        df_tal_L_target_name = regionICPPath + 'transRotMat_tal_L_' + target_name + '_' + curRegion + '.txt'

        curMA_basic_U.add_target_Mesh(curRegion,target_name,pathOldMesh,pathNewMesh,pathCombRoot,df_tal_L_target_name,old_db_version,new_db_version)
        

#################################################################################################
    if callGenISOMesh:
        filterOut = config['functions']['callGenISOMesh']['filterOut']
        typeIso = config['functions']['callGenISOMesh']['typeIso']           
        numNeig = config['functions']['callGenISOMesh']['numNeig']
        typeDist = config['functions']['callGenISOMesh']['typeDist']
        numDim = config['functions']['callGenISOMesh']['numDim']
        numCoord = config['functions']['callGenISOMesh']['numCoord']
        scale = config['functions']['callGenISOMesh']['scale']
        coordScale = config['functions']['callGenISOMesh']['coordScale']
        
        target_file_name = regionISOPath+'mincenterList'+curRegion+'.txt'
        target_name = curTools.getCenterSubj(target_file_name,rankPosition=1)
        targetMesh_path = path + os.sep + curRegion + "/mesh_" + target_name + os.sep 

        for i in range(numDim):
            coordDim = i + 1
            outMesh_Dir = path + os.sep + curRegion + os.sep + "mesh_isomap"+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_dist'+typeDist+'_space'+str(coordScale)+"/" 
            if filterOut == False:
                outMesh_Dir = path + os.sep + curRegion + os.sep + "mesh_isomap"+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_dist'+typeDist+'_space'+str(coordScale)+"_keepOut/" 


            if not os.path.exists(outMesh_Dir):
                os.makedirs(outMesh_Dir)
            isomapName = regionISOPath+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_dist'+typeDist+'.txt'
            weightName = regionISOPath+'spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+'.txt'
            
            if filterOut == False:
                isomapName = regionISOPath+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_dist'+typeDist+'_keepOut.txt'
                weightName = regionISOPath+'spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+'_keepOut.txt'
                
            curMA_basic_U.prep_ISO_Mesh(isomapName,weightName,coordScale,numCoord,outMesh_Dir,targetMesh_path)     



##########################################################################################################
##########################################   preparing MA ima   ##########################################
##########################################################################################################
    if callPrepMA_ima:         
        filterOut = config['functions']['callPrepMA_ima']['filterOut']
        scale = config['functions']['callPrepMA_ima']['scale']       
        typeIso = config['functions']['callPrepMA_ima']['typeIso']           
        numNeig = config['functions']['callPrepMA_ima']['numNeig'] 
        typeDist = config['functions']['callPrepMA_ima']['typeDist']
        numDim = config['functions']['callPrepMA_ima']['numDim']
        numCoord = config['functions']['callPrepMA_ima']['numCoord']


        target_file_name = regionISOPath+'mincenterList'+curRegion+'.txt'
        target_name = curTools.getCenterSubj(target_file_name,rankPosition=1)
        df_tal_L_target_voirT_name = regionICPPath + 'transRotMat_tal_L_' + target_name + '_voirT_' + curRegion + '.txt'
            
        out_ima_dir = path+os.sep+curRegion+os.sep+'ima_'+curRegion+"/"  
        if not os.path.exists(out_ima_dir):
                os.makedirs(out_ima_dir)
        weightName = regionISOPath+'spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_1_'+typeDist+'scale'+str(scale)+'_keepOut.txt'
        curMA_basic_U.prep_MA_ima(path,curRegion,weightName,out_ima_dir,df_tal_L_target_voirT_name)                

############################################################################################################
    if call_Add_PrepMA_ima:   
        filterOut = config['functions']['call_Add_PrepMA_ima']['filterOut']
        scale = config['functions']['call_Add_PrepMA_ima']['scale']        
        typeIso = config['functions']['call_Add_PrepMA_ima']['typeIso']           
        numNeig = config['functions']['call_Add_PrepMA_ima']['numNeig'] 
        typeDist = config['functions']['call_Add_PrepMA_ima']['typeDist']
        numDim = config['functions']['call_Add_PrepMA_ima']['numDim']
        numCoord = config['functions']['call_Add_PrepMA_ima']['numCoord']
        
#        old_base_1 = "/Base62"                ##   pathOldRoot one: TO SPECIFY   ##
        old_base_1 = config['functions']['call_Add_PrepMA_ima']['old_base_1']
#        old_base_2 = "/testAmp"               ##   pathOldRoot two: TO SPECIFY   ##
        old_base_2 = config['functions']['call_Add_PrepMA_ima']['old_base_2']
#        new_base = "/Base62_testAmp"          ##   combPathRoot: TO SPECIFY   ##
        new_base = config['functions']['call_Add_PrepMA_ima']['new_base']                  
        pathOldRoot = generalRoot + old_base_1                   
        pathNewRoot = generalRoot + old_base_2                  
        pathCombRoot = generalRoot + new_base           

        pathOldMesh =  pathOldRoot + os.sep + curRegion + os.sep + "mesh_ori" + os.sep 
        pathNewMesh = pathNewRoot + os.sep + curRegion + os.sep + "mesh_ori" + os.sep
        regionISOPath = pathCombRoot + os.sep + curRegion + os.sep + "Isomap/" 
        target_file_name = regionISOPath+'mincenterList'+curRegion+'.txt'
        target_name = curTools.getCenterSubj(target_file_name,rankPosition=1)

        print('Target name is: '+target_name)

        df_tal_L_target_voirT_name = regionICPPath + 'transRotMat_tal_L_' + target_name + '_voirT_' + curRegion + '.txt'

        for i in range(numDim):
            coordDim = i + 1
            out_ima_dir = pathCombRoot+os.sep+curRegion+os.sep+'ima_spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+"/"
            if filterOut == False:
                out_ima_dir = path+os.sep+curRegion+os.sep+'ima_spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+"_keepOut/"
            
            if not os.path.exists(out_ima_dir):
                os.makedirs(out_ima_dir)
            print('Weight name: ')
            weightName = regionISOPath+'spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+'.txt'
            if filterOut == False:
                weightName = regionISOPath+'spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+'_keepOut.txt'

            curMA_basic_U.prep_ADD_MA_ima(pathOldRoot,pathNewRoot,pathCombRoot,curRegion,weightName,out_ima_dir,df_tal_L_target_voirT_name)
            


##########################################################################################################
#########################################   generating MA ima   ##########################################
##########################################################################################################
    if callGenMA_ima:   
        target_file_name = regionISOPath+'mincenterList'+curRegion+'.txt'
        target_name = curTools.getCenterSubj(target_file_name,rankPosition=1)

        filterOut = config['functions']['callGenMA_ima']['filterOut']
        scale = config['functions']['callGenMA_ima']['scale']        
        typeIso = config['functions']['callGenMA_ima']['typeIso']           
        numNeig = config['functions']['callGenMA_ima']['numNeig'] 
        typeDist = config['functions']['callGenMA_ima']['typeDist']
        numDim = config['functions']['callGenMA_ima']['numDim']
        numCoord = config['functions']['callGenMA_ima']['numCoord']

        
        df_tal_L_target_voirT_name = regionICPPath + 'transRotMat_tal_L_' + target_name + '_voirT_' + curRegion + '.txt'
        
        # 2/2025
        out_ima_dir = path+os.sep+curRegion+os.sep+'ima_'+curRegion+'/'

        for i in range(numDim):
            coordDim = i + 1
            #out_ima_dir = path+os.sep+curRegion+os.sep+'ima_spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+"/" 
            outMA_ima_dir = path+os.sep+curRegion+os.sep+'imaMA_spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+"/"

            if filterOut == False:
                #out_ima_dir = path+os.sep+curRegion+os.sep+'ima_spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+"_keepOut/"
                outMA_ima_dir = path+os.sep+curRegion+os.sep+'imaMA_spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+"_keepOut/"

            if not os.path.exists(outMA_ima_dir):
                os.makedirs(outMA_ima_dir)
            weightName = regionISOPath+'spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+'.txt'

            if filterOut == False:
                weightName = regionISOPath+'spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+'_keepOut.txt'

            curMA_basic_U.compose_MA_ima(weightName,out_ima_dir,outMA_ima_dir)



###################################################################################################
#########################################   MA to mesh   ##########################################
###################################################################################################
    if callGenMA_mesh:   
        target_file_name = regionISOPath+'mincenterList'+curRegion+'.txt'
        target_name = curTools.getCenterSubj(target_file_name,rankPosition=1)

        filterOut = config['functions']['callGenMA_mesh']['filterOut']
        scale = config['functions']['callGenMA_mesh']['scale']        
        typeIso = config['functions']['callGenMA_mesh']['typeIso']           
        numNeig = config['functions']['callGenMA_mesh']['numNeig'] 
        typeDist = config['functions']['callGenMA_mesh']['typeDist']
        numDim = config['functions']['callGenMA_mesh']['numDim']
        numCoord = config['functions']['callGenMA_mesh']['numCoord']
                        
        smoothingFactor = config['functions']['callGenMA_mesh']['smoothingFactor']
        coordScale = config['functions']['callGenMA_mesh']['coordScale']        
        weightScale = config['functions']['callGenMA_mesh']['weightScale']           
        aimsThreshold = config['functions']['callGenMA_mesh']['aimsThreshold']         
        
        df_tal_L_target_voirT_name = regionICPPath + 'transRotMat_tal_L_' + target_name + '_voirT_' + curRegion + '.txt'
        targetMesh_path = path + os.sep + curRegion + os.sep + "mesh_" + target_name + '/' 
        inv_file = path + os.sep + 'voirTalairachInv.trm'
        if not os.path.exists(inv_file):
            with open(inv_file, 'w') as file:
                lines = ['-110.0 -90.0 -90.0', '1.0 0.0 0.0', '0.0 1.0 0.0', '0.0 0.0 1.0']
                file.write('\n'.join(lines))
        

        for i in range(numDim):
            coordDim = i + 1  
            outMA_ima_dir = path+os.sep+curRegion+os.sep+'imaMA_spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+"/"
            outMA_mesh_dir = path+os.sep+curRegion+os.sep+'meshMA_spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+str(weightScale)+'_space'+str(coordScale)+'_smooth'+str(smoothingFactor)+'_smT'+str(aimsThreshold)+'/'

            if filterOut == False:
                outMA_ima_dir = path+os.sep+curRegion+os.sep+'imaMA_spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+"_keepOut/"
                outMA_mesh_dir = path+os.sep+curRegion+os.sep+'meshMA_spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+str(weightScale)+'_space'+str(coordScale)+'_smooth'+str(smoothingFactor)+'_smT'+str(aimsThreshold)+'_keepOut/'

            if not os.path.exists(outMA_mesh_dir):
                os.makedirs(outMA_mesh_dir)

            weightName = regionISOPath+'spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+'.txt'

            if filterOut == False:
                weightName = regionISOPath+'spamCoord'+str(numCoord)+'isomap'+typeIso+curRegion+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(coordDim)+'_'+typeDist+'scale'+str(scale)+'_keepOut.txt'

            curMA_basic_U.compose_MA_mesh(weightName,outMA_ima_dir,inv_file,smoothingFactor,coordScale,outMA_mesh_dir,targetMesh_path,aimsThreshold,target_name)


if __name__ == "__main__":
    main(sys.argv[1:])

