import sys
import os
from pathlib import Path
import shutil
import numpy as np 
import pandas as pd
import json
from itertools import chain

from soma import aims


class general_tools:
    def __init__(self,arg1):
        self.arg1 = arg1


#########################################################################################################
    def add_prefix_to_csv_index(self,input_csv, output_csv, prefix):
        """
        Reads a CSV, adds a prefix to the index (subjID), 
        and saves the result to a new CSV.
        """ 
        # Assume the first column is the index; 
        df = pd.read_csv(input_csv, index_col=0)
	    
        # Ensure index is string and add prefix
        # We use a lambda to avoid double-prefixing if the script is re-run
        df.index = df.index.map(lambda x: f"{prefix}{x}" if not str(x).startswith(prefix) else x)
	    
        df.to_csv(output_csv)
        print(f"Success: {output_csv} created with prefix '{prefix}'.")
	    

    

#########################################################################################################
    def add_postfix_all_files_in_dir(self,directory, postfix):
        """
        Renames all files in the given path by adding a postfix 
        before the file extension.
        """
        # 1. Convert to Path object for easier manipulation
        folder = Path(directory)

        # 2. Check if the directory exists
        if not folder.is_dir():
            print(f"Error: {directory} is not a valid directory.")
            return

        # 3. Iterate through files
        for file_path in folder.iterdir():
            # Only process files, skip directories
            if file_path.is_file():
                # file_path.stem is the name without extension
                # file_path.suffix is the extension (e.g., .json)
                new_name = f"{file_path.stem}{postfix}{file_path.suffix}"
            
                # Create the full new path
                new_file_path = file_path.with_name(new_name)
                
                # Rename the file
                file_path.rename(new_file_path)
                print(f"Renamed: {file_path.name} -> {new_name}")

    # --- Example Usage ---
    # add_postfix_to_files('/home/user/my_data', '_M0')

#########################################################################################################
    def checkIfArray(self,in_list):
        """
            Check if the input is an array
        """
        for array_data in in_list:
            print(array_data)
            if isinstance(array_data, np.ndarray) and array_data.dtype.kind == 'f':
                print("The variable is a NumPy array with floats.")
            else:
                print("The variable is not a NumPy array with floats.")

#########################################################################################################
    def merge_directories(self, dir1, dir2, dir3,extension):
        # Copy files with extension from dir1 to dir3
        [shutil.copy2(os.path.join(dir1, file), os.path.join(dir3, file)) for file in os.listdir(dir1) if file.endswith(extension)]

        # Copy files with extension from dir2 to dir3
        [shutil.copy2(os.path.join(dir2, file), os.path.join(dir3, file)) for file in os.listdir(dir2) if file.endswith(extension)]

  
#########################################################################################################
    def getIndexOfName(self,oneName,nameList):
        """
          given a name and a name list, return the index of the name in the list
        """
        index_of_name = -1
        if oneName in nameList:
            index_of_name = nameList.index(oneName)
        else:
            print(f"'{oneName}' is not in the list.")
        return index_of_name
    

    def saveNames(self,file_path,name_list):
        """
            save a list of subject names
        """
        with open(file_path, 'w') as file:
            for string in name_list:
                file.write(string + '\n')


    def retrieveNames(self,file_path):
        """
            read a list of subject names
        """
        retrieved_list = []
        with open(file_path, 'r') as file:
            for line in file:
                retrieved_list.append(line.strip())  # Remove newline characters
        return(retrieved_list)


###########################################################################################################
    def getCenterSubj(self,fileName,rankPosition):
        """    
            given the centers file name, read it and return the rankPosition element
        """
        centerList_df = pd.read_csv(fileName,index_col='subjName')
        curPos = rankPosition - 1
        centerSubj = centerList_df.index[curPos]
        return centerSubj



    def saveFloatLists(self,file_path,float_list):
        """
            save a list of floats to a txt file
        """
        with open(file_path, 'w') as file:
            for sublist in float_list:
                line = " ".join(map(str, sublist))  # Convert floats to strings and join with spaces
                file.write(line + '\n')


    def retrieveFloatLists(self,file_path):
        """
            retrieve a list of list of floats from txt
        """
        retrieved_list = []
        with open(file_path, 'r') as file:
            for line in file:
                sublist = [float(val) for val in line.strip().split()]  # Split by spaces and convert to floats
                retrieved_list.append(sublist)
        return (retrieved_list)        


    def saveFloatListsJson(self,file_path,float_list):
        """
            save a list of list of floats to json format
        """
        data = float_list
        data = [item.tolist() if isinstance(item, np.ndarray) else item for item in data]        
        with open(file_path, 'w') as file:
            json.dump(data, file)


    #def retrieveFloatListsJson(self,file_path):
    #    """
    #        older version, retrieve a list of list of floats from json
    #    """
    #    with open(file_path, 'r') as file:
    #        retrieved_data = json.load(file)
    #    retrieved_data = [np.array(item) if isinstance(item, list) else item for item in retrieved_data]
    #    return(retrieved_data)   
        

    def retrieveFloatListsJson(self, file_path):
        """
        Retrieve a list of list of floats from json with a retry mechanism 
        to handle cluster file system latency.
        """
        max_retries = 5
        retry_delay = 3  # seconds

        for attempt in range(max_retries):
            try:
                with open(file_path, 'r') as file:
                    retrieved_data = json.load(file)
            
                # If successful, process and return
                retrieved_data = [np.array(item) if isinstance(item, list) else item for item in retrieved_data]
                return retrieved_data

            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    print(f"JSON Load failed (Attempt {attempt + 1}/{max_retries}). "
                          f"File might be locked or syncing. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print(f"Critical JSON Error after {max_retries} attempts at path: {file_path}")
                    raise e


    def retrieveICPresJson(self,file_path):
        """
            retrieve a float, a rotation list and a translation list
        """
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        dist = data[0]
        rot = data[1]
        trans = data[2]
        return dist,rot,trans


    def flattenTransRot(self,trans,rot):
        """
            take in nested structures of translation and rotation values (trans and rot) and return 
            flattened versions of these structures. 
            The specific flattening methods depend on whether the input translation is a list or tuple, 
            and the assumption that rotation is a list of lists.
        """
        curTrans = trans
        curRot = rot
        unpacked_rot = [value for sublist in curRot for value in sublist]
        if isinstance(curTrans, (list, tuple)):
            unpacked_trans = list(chain.from_iterable(curTrans))
        else:
            unpacked_trans = curTrans          
        return unpacked_trans, unpacked_rot


###########################################################################################################################
    def makeTranRotLine(self,trans,rot):
        """
            take np arrays trans (3,1) and rot (3,3), 
            return a single list with 12 numbers, concatenating trans and rot
        """
        oneline_trans = np.array(trans.reshape(1,3))
        oneline_rot = np.array(rot.reshape(1,9))
        transRot = np.concatenate((oneline_trans,oneline_rot),axis=1)
        return transRot


###########################################################################################################################
    def parseTransRotLine_to_array(self,transRot):
        """
            take a single list with 12 numbers, concatenating trans and rot
            return a combined array, first row trans (1,3) and the rest of the rowsas rot (3,3)
        """
        # Convert the list to a NumPy array
        numbers_array = np.array(transRot)
        # Extract the first 3 numbers into a 1x3 array
        array_trans = numbers_array[:3].reshape((1, 3))
        # Extract the rest 9 numbers into a 3x3 array
        array_rot = numbers_array[3:].reshape((3, 3))
        # horizontal stack
        array_transRot = np.vstack((array_trans,array_rot))
        return array_transRot
    

###########################################################################################################################
    def parseTransRotLine(self,transRot):
        """
            take a single list with 12 numbers, concatenating trans and rot
            return trans (3,1) and rot (3,3)
        """
        # Convert the list to a NumPy array
        numbers_array = np.array(transRot)
        # Extract the first 3 numbers into a 3x1 array
        array_trans = numbers_array[:3].reshape((3, 1))
        # Extract the rest 9 numbers into a 3x3 array
        array_rot = numbers_array[3:].reshape((3, 3))
        return array_trans, array_rot


###########################################################################################################################
    def namesInDir(self,workingDir,curExt,outPath,outFileName,write,removeFirstLetter):
        """ 
          list and return file names with a given extension
          write it to the specified output file if needed
        """
        curExt = '.' + curExt
        nameList = []
        names = os.listdir(workingDir)
        for name in names:
            fileExt = os.path.splitext(name)[-1]
            if fileExt == curExt:
                if (removeFirstLetter == 'T'):
                    name = name[1:] #remove 'l' or 'r' infront
                name = name[:-len(curExt)] 
                nameList.append(name)
        if (write == 't'):
            outFile = outPath + os.sep + outFileName
            outSubjFile = file(outFile,'w')
            outSubjFile.write(str(nameList))
            outSubjFile.close   
        return(nameList)


    def readSubjName(self,inFileName):
        """ 
          read a list of subjects from file, comma separated with single quotes
          beginning with '[', end with ']' 
        """
        curNameList = ''
        with open(inFileName) as f:
            curNameList = f.read().strip().split(', ')
            curNameList = list(map(lambda x: x[1:-1], curNameList))  #Python3: remove quotes             
            #curNameList = map(lambda x: x[1:-1], curNameList)  # Python2: remove quotes       
            curNameList[0] = curNameList[0][1:]   #remove first subj quote
            curNumSubj = len(curNameList)
            curNameList[(curNumSubj-1)] = curNameList[(curNumSubj-1)][0:-1]   #remove last subj quote
        return curNameList


    def stripSubjName(self,curName):
        """
          remove the 'L' or 'flip-R' from the name of subjects
        """
        curSide = curName[0]
        newName = ''
        if curSide == 'L':   
            newName = curName[1:]
        if curSide == 'f':   
            newName = curName[6:]
        return newName


    def sepLRnames(self,fullName):
        """
          given a names list, return left and right names
        """
        leftNames = []
        rightNames = []
        for element in fullName:
            if element[0] == 'L':
                leftNames.append(element)
            if element[0] == 'f':
                rightNames.append(element) 
        return leftNames, rightNames


    def get_LflipR_names(self,names):
        """
          given a name list, return a new list adding 'L' and 'flip-R' in front of the names
        """
        newNames, leftNames, rightNames = [], [], []
        leftNames = ['L' + name for name in names]
        rightNames = ['flip-R' + name for name in names]
        newNames = leftNames + rightNames
        return newNames

    def get_add_flip_to_Rsubjects(self,names):
        """
            given a list of names starting with L or R, add 'flip-' to all subjects starting
            with 'R', those starting with 'L' remains the same
        """
        newNames = ['flip-' + name if name.startswith('R') else name for name in names]
        return newNames

    def trmListToFile(self,oneTrm,oneTrmFileName):
        """
          given a trm line, compose and write to a file
        """
        oneTrm = np.asarray(oneTrm)
        oneTrm = oneTrm.reshape(4,3)          
        np.savetxt(oneTrmFileName,oneTrm)


#########################################################################################
# 2019, added to handle different versions of isomap (R or Python)
    def getNamesFromIsomap(self,isomapFileName,fileformat,outFileName):
        """
          read in the isomap file and write a list of subjects to output file
        """        
        isomapNames = None
        
        with open(isomapFileName) as f:
            mylist = f.read().strip().split() 
            if (fileformat == "v1"):
                mylist = mylist[1:]      #remove first line
            isomapNames = mylist[::2]
        
        
#        if (fileformat == "v2"):
#            curIso = None
#            curIso = pd.read_csv(isomapFileName,sep=' ',index_col=0,header=None)
#            print('b')
#            print(isomapFileName)
#            print(curIso)
#            isomapNames = curIso.index.values    
            
#        if (fileformat == "v1"):
#            with open(isomapFileName) as f:
#                mylist = f.read().strip().split() 
#                mylist = mylist[1:]      #remove first line
#                isomapNames = mylist[::2]    
#                isomapNames = map(lambda x: x[1:-1], isomapNames)    # remove the "" from the names
    
                
#        file = open(outFileName,'w') 
#        file.write(str(isomapNames)) 
#        file.close() 
        nameList = pd.Series(isomapNames)
        nameList.to_csv(outFileName,sep=' ')  


