#! /usr/bin/env python
from soma_workflow.client import Job, Workflow, Helper
import os

# --- CONFIGURATION ---
# We use /tmp because it's local and fast for the workflow database
workflow_file_path = "/tmp/Calc_ICP_Production_Workflow"
python_executable = "/casa/install/bin/python" 
#script_path = "/media/sf_C_DRIVE/test_2025_6_Joy/general_Run_test.py" 
#script_path = "/home/sulci/JS/A_joyCODE/test_2025_6_Joy/general_Run_Champollion_sca.py"
script_path = "/home/sulci/JS/A_joyCODE/test_2025_6_Joy/Champollion_sca_general_study.py"
#subject_file = "/media/sf_C_DRIVE/A_joyCODE/DEV_LIN/testHCP10/CSSyl/bck/subjNames_bck.txt"
#subject_file = "/home/sulci/JS/A_joyCODE/DEV_LIN/AtrilBioscaCermoi_time1/FPOCalCu/bck/subjNames_bck.txt"
subject_file = "/home/sulci/JS/A_joyCODE/DEV_LIN/AtrilBioscaCermoi_time1/Calc/bck/subjNames_bck.txt"
curName = "Calc_ICP_Distance_Calculation"

# 1. Load actual subject names
with open(subject_file, 'r') as f:
    subjects = [line.strip() for line in f if line.strip()]

job_list = []

# 2. Build a Job for every subject found in the text file
for i, subj_id in enumerate(subjects):
    # Command: python script.py --mode cluster_job --id index
    cmd = [python_executable, script_path, '--mode', 'cluster_job', '--id', str(i)]
    
    # Name the job with the index AND the subject name for easy tracking
    job_name = "ICP_" + str(i) + "_" + str(subj_id)
    
    job = Job(command=cmd, name=job_name)
    job_list.append(job)

# 3. Create the Workflow object
workflow = Workflow(jobs=job_list, name=curName)

# 4. Serialize to file
# This creates the file that the GUI will "Run"
Helper.serialize(workflow_file_path, workflow)

print("--------------------------------------------------")
print(f"SUCCESS: Workflow generated with {len(job_list)} subjects.")
print(f"FILE LOCATION: {workflow_file_path}")
print("NEXT STEP: Open soma_workflow_gui -> Load this file -> Submit.")
print("--------------------------------------------------")
