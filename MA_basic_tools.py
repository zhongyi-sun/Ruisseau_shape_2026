import os
import glob
import numpy as np 
import pandas as pd
from general_tools import general_tools
from MA_tools import MA_tools
import shutil
from soma import aims


class MA_basic_tools:
    def __init__(self,arg1):
        self.arg1 = arg1
        self.curTools = general_tools(1)
        self.cur_MA_tools = MA_tools(1)


    def prep_ICP_Trm_New(self, df_tal_L_name, new_trans_rot_csv, target_name, 
                        df_tal_L_targetICP_name, df_tal_L_targetICP_voirT_name):
        df_tal_L = pd.read_csv(df_tal_L_name, index_col=0)  # load data
        df_long = pd.read_csv(new_trans_rot_csv)        
        df_tal_L.index = df_tal_L.index.astype(str).str.strip() # removes hidden spaces
        df_target_subset = df_long[df_long['Target'] == target_name].set_index('Source')  # filter for the target
        subject_list = df_target_subset.index.tolist()

        processed_names, np_tal_L_target, np_tal_L_target_voirT = [],[],[]
        colNames = list(range(1, 13))
        for name in subject_list:
            clean_name = str(name).strip() # Clean the ICP name                       
            if clean_name.startswith('R'):  # Determine the lookup name
                tal_lookup_name = f"flip-{clean_name}"
            else:
                tal_lookup_name = clean_name
            if tal_lookup_name not in df_tal_L.index:  # Check existence in cleaned index
                print(f"Warning: Lookup '{tal_lookup_name}' not in Talairach. Skipping.")
                continue
            try:                
                row = df_target_subset.loc[name]  # Get ICP data
                if isinstance(row, pd.DataFrame): row = row.iloc[0]  # Handle duplicates if they exist
                trans_ICP = np.array([[row['tx'], row['ty'], row['tz']]], dtype=float).T
                rot_ICP = np.array([
                    [row['r11'], row['r12'], row['r13']],
                    [row['r21'], row['r22'], row['r23']],
                    [row['r31'], row['r32'], row['r33']]
                ], dtype=float)               
                cur_tal_L = df_tal_L.loc[tal_lookup_name]  # Get Talairach data (using lookup name)
                trans_tal_L, rot_tal_L = self.curTools.parseTransRotLine(cur_tal_L)
                # Convert to numpy and ensure shapes (3,1) and (3,3)
                trans_tal_L = np.array(trans_tal_L, dtype=float).reshape(3, 1)
                rot_tal_L = np.array(rot_tal_L, dtype=float).reshape(3, 3)
                # Math Composition
                t_final, r_final = self.cur_MA_tools.compose_Trm_Matrix(
                    trans_tal_L, trans_ICP, rot_tal_L, rot_ICP
                )
                # VoirT Conversion
                t_voir, r_voir = self.cur_MA_tools.voirT_Trm_Matrix(t_final, r_final)
                # Flatten to 12-col lines
                line_target = self.curTools.makeTranRotLine(t_final, r_final)
                line_voirT = self.curTools.makeTranRotLine(t_voir, r_voir)
                np_tal_L_target.append(line_target.flatten())
                np_tal_L_target_voirT.append(line_voirT.flatten())
                # Record the successful subject (using tal name)
                #processed_names.append(clean_name)
                processed_names.append(tal_lookup_name)                
            except Exception as e:
                print(f"Error processing {clean_name}: {e}")
        # Save Outputs
        if len(processed_names) > 0:
            df_out = pd.DataFrame(np_tal_L_target, index=processed_names, columns=colNames)
            df_out.to_csv(df_tal_L_targetICP_name, index=True, index_label='subjName')
            
            df_out_voirT = pd.DataFrame(np_tal_L_target_voirT, index=processed_names, columns=colNames)
            df_out_voirT.to_csv(df_tal_L_targetICP_voirT_name, index=True, index_label='subjName')
            print(f"✅ Successfully processed {len(processed_names)} subjects for {target_name}")
        else:
            print("❌ No subjects were processed successfully.")
            

    
    ##########################################################################################################################
    def prep_ICP_Trm(self,df_tal_L_name,df_allICP_name,target_name,df_tal_L_targetICP_name,df_tal_L_targetICP_voirT_name):
        """
            prepare trm's: to tal_L_target, to tal_L_target_voirT
        """
        df_tal_L = pd.read_csv(df_tal_L_name,index_col=0,header=0)
        df_allICP = pd.read_csv(df_allICP_name,index_col=0,header=0)
        nameList_ICP = df_allICP.index.to_list()
        print(df_allICP_name)
        print(nameList_ICP)

        target_index = self.curTools.getIndexOfName(target_name,nameList_ICP)
        print('target_index: '+str(target_index))

        np_tal_L_target,np_tal_L_target_voirT = np.empty((0,12),dtype=float),np.empty((0,12),dtype=float)     
        start_pos, end_pos = 0, 12
        if target_index != 0:
            start_pos = target_index * 12
            end_pos = start_pos + 12
        name_index_ICP = 0
        for name in nameList_ICP:
            print(name)
            # get ICP to target trm
            cur_ICP = df_allICP.iloc[name_index_ICP,start_pos:end_pos]
            trans_ICP, rot_ICP = self.curTools.parseTransRotLine(cur_ICP)      
            # get tal_L trm
            cur_tal_L = df_tal_L.loc[name]
            trans_tal_L, rot_tal_L = self.curTools.parseTransRotLine(cur_tal_L)    
            # compose tal_L_target trm
            trans_tal_L_target,rot_tal_L_target=self.cur_MA_tools.compose_Trm_Matrix(trans_tal_L,trans_ICP,rot_tal_L,rot_ICP)
            cur_tal_L_target_line = self.curTools.makeTranRotLine(trans_tal_L_target,rot_tal_L_target) 
            # compose tal_L_target_voirT trm
            trans_tal_L_target_voirT,rot_tal_L_target_voirT=self.cur_MA_tools.voirT_Trm_Matrix(trans_tal_L_target,rot_tal_L_target)
            cur_tal_L_target_voirT_line = self.curTools.makeTranRotLine(trans_tal_L_target_voirT,rot_tal_L_target_voirT) 
            # update np's
            np_tal_L_target = np.concatenate((np_tal_L_target,cur_tal_L_target_line),axis=0)
            np_tal_L_target_voirT = np.concatenate((np_tal_L_target_voirT,cur_tal_L_target_voirT_line),axis=0)
            name_index_ICP = name_index_ICP + 1

        colNames = list(range(1,13))
        df_tal_L_target = pd.DataFrame(np_tal_L_target, index=nameList_ICP, columns=colNames)
        df_tal_L_target.to_csv(df_tal_L_targetICP_name,index=True,index_label='subjName')
        df_tal_L_target_voirT = pd.DataFrame(np_tal_L_target_voirT, index=nameList_ICP, columns=colNames)
        df_tal_L_target_voirT.to_csv(df_tal_L_targetICP_voirT_name,index=True,index_label='subjName')



    ##########################################################################################################################
    def prep_ICP_Mesh(self,path,curRegion,target_name,saveOriMesh,saveTalMesh,saveRmesh,saveTalLMesh,saveTargetMesh):

        """
          writing the transformed mesh, according to the boolean passed
        """
        df_tal_name = path + '/tal/df_tal.csv'
        df_tal_L_name = path + '/tal/df_tal_L.csv'
        df_allICP_name =  path + os.sep + curRegion + '/ICP/transRotMat_' + curRegion + '.txt'
        df_tal_L_target_name = path + os.sep + curRegion + '/ICP/transRotMat_tal_L_' +target_name+'_'+curRegion+'.txt' 
        oriMesh_path =  path + os.sep + curRegion + os.sep + "mesh_ori/" 
        talMesh_path = path + os.sep + curRegion + os.sep + "mesh_tal/" 
        Rmesh_path = path + os.sep + curRegion + os.sep + "mesh_R/"
        talLMesh_path = path + os.sep + curRegion + os.sep + "mesh_tal_L/" 
        targetMesh_path = path + os.sep + curRegion + os.sep + "mesh_" + target_name + os.sep


        df_tal = pd.read_csv(df_tal_name,index_col=0,header=0)
        df_tal_L = pd.read_csv(df_tal_L_name,index_col=0,header=0)
        df_allICP = pd.read_csv(df_allICP_name,index_col=0,header=0)
        df_target = pd.read_csv(df_tal_L_target_name,index_col=0,header=0)
        nameList_tal = df_tal_L.index.to_list()
        nameList_ICP = df_allICP.index.to_list()
        if saveOriMesh:
            if not os.path.exists(oriMesh_path):
                os.makedirs(oriMesh_path)
        if saveTalMesh:
            if not os.path.exists(talMesh_path):
                os.makedirs(talMesh_path)
        if saveTalLMesh:
            if not os.path.exists(talLMesh_path):
                os.makedirs(talLMesh_path)
        if saveTargetMesh:
            if not os.path.exists(targetMesh_path):
                os.makedirs(targetMesh_path)

        for name in nameList_tal:
            print('curSubj: '+name)
            if saveOriMesh:
                if (name[0] == 'L'):
                    inName = path + os.sep + curRegion + '/left/' + name + '.bck' 
                    curOutName = oriMesh_path + '/' + name + '.mesh'
                    self.cur_MA_tools.bckMesh(inName,oriMesh_path,curOutName)
                if (name[0] == 'f'):
                    new_name = name[5:] # remove 'flip_' from the name
                    inName = path + os.sep + curRegion + '/right/' + new_name + '.bck' 
                    curOutName = oriMesh_path + '/' + name + '.mesh'
                    self.cur_MA_tools.bckMesh(inName,oriMesh_path,curOutName)
            if saveTalMesh:
                inName = oriMesh_path + name + '.mesh'
                if inName:
                    curOutName = talMesh_path + name + '.mesh'
                    s_name = self.curTools.stripSubjName(name)
#                    s_name = int(s_name)
                    print(s_name)
                    trans = df_tal.loc[s_name].tolist()
                    result_trans = self.curTools.parseTransRotLine_to_array(trans)
                    # Save the transformation to a text file                    
                    result_trans_path = talMesh_path + name + '_tal.txt'
                    np.savetxt(result_trans_path, result_trans, delimiter=' ')                    
                    self.cur_MA_tools.meshTrans(inName,talMesh_path,curOutName,result_trans_path)
                    os.remove(result_trans_path)
                else:
                    print('Need to generate ' + inName)
            if saveTalLMesh:
                inName = oriMesh_path + name + '.mesh'
                if inName:
                    curOutName = talLMesh_path + name + '.mesh'                
                    trans = df_tal_L.loc[name].tolist()
                    result_trans = self.curTools.parseTransRotLine_to_array(trans)
                    # Save the transformation to a text file                    
                    result_trans_path = talLMesh_path + name + '_tal.txt'
                    np.savetxt(result_trans_path, result_trans, delimiter=' ')                    
                    self.cur_MA_tools.meshTrans(inName,talLMesh_path,curOutName,result_trans_path)
                    os.remove(result_trans_path)
                    # test function meshTrans_Matrix
#                    translation_vector,rotation_matrix = self.curTools.parseTransRotLine(trans)
#                    self.cur_MA_tools.meshTrans_Matrix(inName,curOutName,translation_vector,rotation_matrix)
                else:
                    print('Need to generate: '+inName)
            if name in nameList_ICP:
                if saveTargetMesh:
                    inName = oriMesh_path + name + '.mesh'
                    if inName:
                        curOutName = targetMesh_path + name + '.mesh'               
                        trans = df_target.loc[name].tolist()
                        result_trans = self.curTools.parseTransRotLine_to_array(trans)
                        # Save the transformation to a text file                    
                        result_trans_path = targetMesh_path + name + '_tal.txt'
                        np.savetxt(result_trans_path, result_trans, delimiter=' ')                    
                        self.cur_MA_tools.meshTrans(inName,targetMesh_path,curOutName,result_trans_path)
                        os.remove(result_trans_path)
                    else:
                        print('Need to generate: '+inName)
            else:
                print(name + ' not in ICP names, check if the bucket has a problem.')


    ##########################################################################################################################
    def add_target_Mesh(self,curRegion,target_name,pathOldMesh,pathNewMesh,pathCombRoot,df_tal_L_target_name,old_db_version,new_db_version):
        """
          generate the meshes transformed to the target space, for addition of a new base to an old base
        """
        df_target = pd.read_csv(df_tal_L_target_name,index_col=0,header=0)
        names_target = df_target.index.to_list()

        targetMesh_path = pathCombRoot + os.sep + curRegion + os.sep + "mesh_" + target_name + os.sep
        if not os.path.exists(targetMesh_path):
            os.makedirs(targetMesh_path)
        
        # get list of old meshes
        files = os.listdir(pathOldMesh)
        mesh_files = [file for file in files if file.endswith('.mesh')]
        names_old_mesh = [os.path.splitext(file)[0] for file in mesh_files]
        if old_db_version=='v1':
            names_old_mesh = ['flip-' + name if name.startswith('R') else name for name in names_old_mesh]
  
        # get list of new meshes
        files = os.listdir(pathNewMesh)
        mesh_files = [file for file in files if file.endswith('.mesh')]
        names_new_mesh = [os.path.splitext(file)[0] for file in mesh_files]
        if new_db_version=='v1':
            names_new_mesh = ['flip-' + name if name.startswith('R') else name for name in names_new_mesh]

        # compose and write old meshes 
        for name in names_old_mesh:
            if name in names_target:
                mesh_subj_name = name
                if old_db_version=='v1':
                    mesh_subj_name = name.replace('flip-', '') if name.startswith('flip-') else name
                in_mesh_name = pathOldMesh + mesh_subj_name + '.mesh' 
                trans = df_target.loc[name].tolist()
                in_trans_name = pathCombRoot + name + '_tal.txt'
                np.savetxt(in_trans_name, trans, delimiter=' ')   
                out_mesh_name = targetMesh_path + name + '.mesh'  
                self.cur_MA_tools.meshTrans(in_mesh_name,targetMesh_path,out_mesh_name,in_trans_name)
                os.remove(in_trans_name)
            else:
                print(name + ' not in comb transRotMat name list.')
                                
        # compose and write new meshes        
        for name in names_new_mesh:
            if name in names_target:
                mesh_subj_name = name
                if new_db_version=='v1':
                    mesh_subj_name = name.replace('flip-', '') if name.startswith('flip-') else name
                in_mesh_name = pathNewMesh + mesh_subj_name + '.mesh'
                trans = df_target.loc[name].tolist()
                in_trans_name = pathCombRoot + name + '_tal.txt'
                np.savetxt(in_trans_name, trans, delimiter=' ')   
                out_mesh_name = targetMesh_path + name + '.mesh'  
                self.cur_MA_tools.meshTrans(in_mesh_name,targetMesh_path,out_mesh_name,in_trans_name)
                os.remove(in_trans_name)
            else:
                print(name + ' not in comb transRotMat name list.')


    ##########################################################################################################################
    def prep_ISO_Mesh(self,isomapName,weightName,coordScale,numCoord,outMesh_Dir,targetMesh_path):
        """
            writing the meshes according to a given isomap
        """
        curIso = pd.read_csv(isomapName,index_col=0,header=0) 
        isomapNames = curIso.index.to_list()
        spam_df, coordVal, coordSubj_names = self.cur_MA_tools.readSpam(weightName)
        coordScale_ori = coordScale
        coordScale = coordScale / (coordVal[numCoord-1]-coordVal[0])    # coordScale adjusted to variance    

        for name in isomapNames:
            print(name)
            isoVal = curIso.loc[name]
            isoVal = float(isoVal)
            print(isoVal)
            curFactor = coordScale * (isoVal - coordVal[0])              # coordScale adjusted to minCoord   
            firstLine = ((0.0, curFactor, 0.0))
            curFactorFlip = coordScale * (coordVal[numCoord-1] - isoVal) # coordScale adjusted to maxCoord     
            firstLineFlip = ((0.0, curFactorFlip, 0.0))
            oneTrm = np.vstack((firstLine,np.eye((3))))
            oneTrmFlip = np.vstack((firstLineFlip,np.eye((3))))

            oneTrmFileName = outMesh_Dir + 'curTrans.txt'
            oneTrmFileNameFlip = outMesh_Dir + 'curTransFlip.txt'
            np.savetxt(oneTrmFileName,oneTrm)
            np.savetxt(oneTrmFileNameFlip,oneTrmFlip)

            inName = targetMesh_path + name + '.mesh'    
            outMeshName = outMesh_Dir + 'coord_' + str(coordScale_ori) + '_' + name + '.mesh' 
            outMeshName_flip = outMesh_Dir + 'reverse_coord_' + str(coordScale_ori) + '_' + name + '.mesh' 

            self.cur_MA_tools.meshTrans(inName,outMesh_Dir,outMeshName,trans=oneTrmFileName)
            self.cur_MA_tools.meshTrans(inName,outMesh_Dir,outMeshName_flip,trans=oneTrmFileNameFlip)

            os.remove(oneTrmFileName)
            os.remove(oneTrmFileNameFlip)



    ##########################################################################################################################
    def prep_MA_ima(self,path,curRegion,weightName,out_ima_dir,df_tal_L_target_voirT_name):
        """
            converting bck to iamges and transforming them to df_tal_L_target_voirT space
        """
        df_spam, coord_values, coordSubj_names = self.cur_MA_tools.readSpam(weightName)
        nameList_weight = df_spam.index.to_list()
        df_voirT = pd.read_csv(df_tal_L_target_voirT_name,index_col=0,header=0) 

        print('____________________________prep_MA_ima____________________________')

        for name in nameList_weight:

            spamBckLoc = ''
            print(name)
            composedTrm = df_voirT.loc[name].tolist()
            print(composedTrm)
            oneTrmFileName = out_ima_dir + name + '_curTrans.txt'
            np.savetxt(oneTrmFileName,composedTrm)
#            imaOutName = out_ima_dir + name + 'temp.nii.gz'
            if (name[0] == 'L'):    
                spamBckLoc = path + os.sep + curRegion + '/left/'
            if (name[0] == 'f'):
                new_name = name[5:] # remove 'flip_' from the name
                spamBckLoc = path + os.sep + curRegion + '/right/'

            self.cur_MA_tools.getTransformedBckIma(name,composedTrm,oneTrmFileName,out_ima_dir,spamBckLoc,out_ima_dir,process="Anatomy")
            os.remove(oneTrmFileName)


    ##########################################################################################################################
    def prep_ADD_MA_ima(self,pathOldRoot,pathNewRoot,pathCombRoot,curRegion,weightName,out_ima_dir,df_tal_L_target_voirT_name):
        """
            converting bck to iamges and transforming them to df_tal_L_target_voirT space
            for *adding* old and new data sets
        """
        df_spam, coord_values, coordSubj_names = self.cur_MA_tools.readSpam(weightName)
        nameList_weight = df_spam.index.to_list()
        removeFlip_list = [item[len('flip-'):] if item.startswith('flip-') else item for item in nameList_weight]   
        df_voirT = pd.read_csv(df_tal_L_target_voirT_name,index_col=0,header=0) 

        # get list of old and new bcks
        path_old_left = pathOldRoot + os.sep + curRegion + os.sep + 'left' + os.sep
        path_old_right = pathOldRoot + os.sep + curRegion + os.sep + 'right' + os.sep
        path_new_left = pathNewRoot + os.sep + curRegion + os.sep + 'left' + os.sep
        path_new_right = pathNewRoot + os.sep + curRegion + os.sep + 'right' + os.sep

        files_old_left = os.listdir(path_old_left)
        old_left_bck_files = [file for file in files_old_left if file.endswith('.bck')]
        names_old_left_bck = [os.path.splitext(file)[0] for file in old_left_bck_files]
        files_old_right = os.listdir(path_old_right)
        old_right_bck_files = [file for file in files_old_right if file.endswith('.bck')]
        names_old_right_bck = [os.path.splitext(file)[0] for file in old_right_bck_files]

        files_new_left = os.listdir(path_new_left)
        new_left_bck_files = [file for file in files_new_left if file.endswith('.bck')]
        names_new_left_bck = [os.path.splitext(file)[0] for file in new_left_bck_files]
        files_new_right = os.listdir(path_new_right)
        new_right_bck_files = [file for file in files_new_right if file.endswith('.bck')]
        names_new_right_bck = [os.path.splitext(file)[0] for file in new_right_bck_files]

        print('____________________________prep_MA_ima____________________________')
        for name in removeFlip_list:
            spamBckLoc = ''
            print('curName: '+name)
            weight_name = 'flip-' + name if name.startswith('R') else name
            composedTrm = df_voirT.loc[weight_name].tolist()
            oneTrmFileName = out_ima_dir + weight_name + '_curTrans.txt'
            np.savetxt(oneTrmFileName,composedTrm)

            if name in names_old_left_bck:   
                spamBckLoc = pathOldRoot + os.sep + curRegion + '/left/'
            if name in names_old_right_bck:   
                spamBckLoc = pathOldRoot + os.sep + curRegion + '/right/'
            if name in names_new_left_bck:   
                spamBckLoc = pathNewRoot + os.sep + curRegion + '/left/'
            if name in names_new_right_bck:   
                spamBckLoc = pathNewRoot + os.sep + curRegion + '/right/'
            self.cur_MA_tools.getTransformedBckIma(weight_name,composedTrm,oneTrmFileName,out_ima_dir,spamBckLoc,out_ima_dir,process="Anatomy")
            os.remove(oneTrmFileName)



    ##########################################################################################################################
    def compose_MA_ima(self,weightName,out_ima_dir,outMA_ima_dir):
        """
            composing the MA iamges according to a given weight
        """
        df_spam, coord_values, coordSubj_names = self.cur_MA_tools.readSpam(weightName)

        print(coord_values)
        print(coordSubj_names)


        numCoord = df_spam.shape[1]
        nameList_weight = df_spam.index.to_list()
        oneIma = out_ima_dir + nameList_weight[0] + '.nii.gz'
        subject = aims.read(oneIma)
        spamTable = list()
        for i in range(numCoord):   
            curSpam = aims.Volume (subject.getSizeX(), subject.getSizeY(), subject.getSizeZ(), dtype='FLOAT')  #init image length
            curSpam.header()['voxel_size'] = subject.header()['voxel_size']                                    #init image voxelSize
            curSpam.fill(0.)
            spamTable.append(curSpam)
        c = aims.__getattribute__('ShallowConverter_Volume_S16_Volume_FLOAT')            
        for element in nameList_weight: 
            oneIma = out_ima_dir + element + '.nii.gz'         
            print(oneIma)
            
            if os.path.exists(oneIma):
                subject = aims.read(oneIma)
                fsubject = c()(subject)
                for i in range(numCoord):
                    curWeight = df_spam.loc[element,df_spam.columns[i]]                     
                    if curWeight > 0.00001:       # weight thresh
                        spamTable[i] = spamTable[i] + fsubject * curWeight
            else:
                print('File cannot be found!')          
                      
        for i in range(numCoord):
            curSpamName = 'spam' + str(i+1) + '.nii' 
            outSpamName = outMA_ima_dir + curSpamName
            aims.write(spamTable[i], outSpamName)

        # remove .minf files
        files_to_delete = [os.path.join(outMA_ima_dir, filename) for filename in os.listdir(outMA_ima_dir) if filename.endswith('.minf')]
        for file_path in files_to_delete:
            os.remove(file_path)

            






    def compose_MA_mesh(self, weightName, outMA_ima_dir, inv_file, smoothingFactor, coordScale, outMA_mesh_dir, targetMesh_path, aimsThreshold, target_name):
        """
        Writing the MA meshes using standard Python formatting.
        """
        df_spam, coordVal, coordSubj = self.cur_MA_tools.readSpam(weightName)
        numCoord = df_spam.shape[1]
        
        # Adjust scale based on coordinate variance
        coord_range = coordVal[numCoord-1] - coordVal[0]
        coordScale = coordScale / coord_range

        for i in range(numCoord):
            idx = i + 1
            print(f'Cur i is: {i}')

            # --- 1. PREPARE TRANSFORMATION FILES ---
            curFactor = coordScale * (coordVal[i] - coordVal[0])
            curFactorFlip = coordScale * (coordVal[numCoord-1] - coordVal[i])
            
            oneTrm = np.vstack(((0.0, curFactor, 0.0), np.eye(3)))
            oneTrmFlip = np.vstack(((0.0, curFactorFlip, 0.0), np.eye(3)))
            
            trm_file = f"{outMA_mesh_dir}curTrans.txt"
            trm_file_flip = f"{outMA_mesh_dir}curTransFlip.txt"
            
            np.savetxt(trm_file, oneTrm)
            np.savetxt(trm_file_flip, oneTrmFlip)

            # --- 2. DEFINE FILE PATHS (f-strings) ---
            curSpamIma = f"{outMA_ima_dir}spam{idx}.nii"
            curSpamMesh = f"{outMA_mesh_dir}spam{idx}.mesh"
            curSpamGZ = f"{outMA_mesh_dir}gspam{idx}.nii.gz"
            curSpamGT = f"{outMA_mesh_dir}gtpam{idx}.nii.gz"
            tempSpamMesh = f"{outMA_mesh_dir}temp{idx}.mesh"
            
            print(f"Cur ima.......................{curSpamIma}")

            # --- 3. EXECUTE AIMS COMMANDS ---
            # Gaussian Smoothing
            print('....................Gaussian...................')
            os.system(f'AimsGaussianSmoothing -i {curSpamIma} -o {curSpamGZ} -x {smoothingFactor} -y {smoothingFactor} -z {smoothingFactor}')
            
            # Thresholding
            print('....................threshold.....................')
            os.system(f'AimsThreshold -i {curSpamGZ} -o {curSpamGT} -b -t {aimsThreshold}')
            
            # Meshing
            print('....................mesh.....................')       
            os.system(f'AimsMesh -i {curSpamGT} -o {tempSpamMesh} --deciMaxError 0.5 --deciMaxClearance 1 --smooth --smoothIt 20')
            
            # Concatenate/Finalize Mesh
            matching_files = glob.glob(f"{outMA_mesh_dir}temp{idx}*.mesh")
            if len(matching_files) > 0:
                print('....................zcat.....................') 
                os.system(f'AimsZCat -i {outMA_mesh_dir}temp{idx}*.mesh -o {curSpamMesh}')
            else:
                print(f"⚠️ WARNING: No mesh generated for index {idx}. Threshold {aimsThreshold} might be too high.")
                # Cleanup and skip transformation steps for this coordinate
                #os.system(f'rm {outMA_mesh_dir}g*') 
                #continue

            # Cleanup intermediate mesh files
            os.system(f'rm {outMA_mesh_dir}temp*.mesh {outMA_mesh_dir}temp*.mesh.minf')

            # --- 4. TRANSFORMATIONS (Space & Coordinates) ---
            # Move to inv_voitT space
            self.cur_MA_tools.meshTrans(curSpamMesh, outMA_mesh_dir, curSpamMesh, trans=inv_file)
            os.system(f'rm {outMA_mesh_dir}g*')

            # Move to coordinates
            coordMesh = f"{outMA_mesh_dir}ISO_spam{idx}.mesh"
            coordMeshFlip = f"{outMA_mesh_dir}ISO_inverse_spam{idx}.mesh"
            self.cur_MA_tools.meshTrans(curSpamMesh, outMA_mesh_dir, fileOutName=coordMesh, trans=trm_file)
            self.cur_MA_tools.meshTrans(curSpamMesh, outMA_mesh_dir, fileOutName=coordMeshFlip, trans=trm_file_flip)

            # --- 5. REPRESENTATIVE SUBJECTS ---
            if coordSubj[i] != 'none':
                subj = coordSubj[i]
                curCoordMesh = f"{targetMesh_path}{subj}.mesh"
                
                # File definitions
                f_out = f"{outMA_mesh_dir}to_coord_{subj}.mesh"
                f_out_flip = f"{outMA_mesh_dir}to_coord_inverse_{subj}.mesh"
                f_smooth = f"{outMA_mesh_dir}to_coord_Smooth_{subj}.mesh"
                f_smooth_flip = f"{outMA_mesh_dir}to_coord_inverse_Smooth_{subj}.mesh"
                f_target_smooth = f"{outMA_mesh_dir}to_{target_name}_Smooth_{subj}.mesh"

                # Apply Transformations
                self.cur_MA_tools.meshTrans(curCoordMesh, outMA_mesh_dir, f_out, trans=trm_file)
                self.cur_MA_tools.meshTrans(curCoordMesh, outMA_mesh_dir, f_out_flip, trans=trm_file_flip)

                # Smoothing Iterations
                self.cur_MA_tools.meshSmooth(curCoordMesh, outMA_mesh_dir, fileOutName=f_target_smooth, numIteration='200')
                self.cur_MA_tools.meshSmooth(f_out, outMA_mesh_dir, fileOutName=f_smooth, numIteration='200')
                self.cur_MA_tools.meshSmooth(f_out_flip, outMA_mesh_dir, fileOutName=f_smooth_flip, numIteration='200')

                # Cleanup unsmoothed versions
                for f in [f_out, f_out_flip]:
                    if os.path.exists(f): os.remove(f)
                
                os.system(f'rm {outMA_mesh_dir}*.minf')

            # Final cleanup for the loop iteration
            os.system(f'rm {trm_file} {trm_file_flip}')


    ##########################################################################################################################
    def compose_MA_mesh_old(self,weightName,outMA_ima_dir,inv_file,smoothingFactor,coordScale,outMA_mesh_dir,targetMesh_path,aimsThreshold,target_name):
        """
            writing the MA meshes
        """
        df_spam, coordVal, coordSubj = self.cur_MA_tools.readSpam(weightName)
        numCoord = df_spam.shape[1]
        coordScale = coordScale / (coordVal[numCoord-1]-coordVal[0])    # coordScale adjusted to variance

        for i in range(numCoord):
            print('Cur i is: '+str(i))

            # prepare trm files
            curFactor = coordScale * (coordVal[i] - coordVal[0])                # coordScale adjusted to minCoord   
            firstLine = ((0.0, curFactor, 0.0))
            curFactorFlip = coordScale * (coordVal[numCoord-1] - coordVal[i])   # coordScale adjusted to maxCoord  
            firstLineFlip = ((0.0, curFactorFlip, 0.0))
            oneTrm = np.vstack((firstLine,np.eye((3))))
            oneTrmFlip = np.vstack((firstLineFlip,np.eye((3))))
            oneTrmFileName = outMA_mesh_dir + 'curTrans.txt'
            oneTrmFileNameFlip = outMA_mesh_dir + 'curTransFlip.txt'
            np.savetxt(oneTrmFileName,oneTrm)
            np.savetxt(oneTrmFileNameFlip,oneTrmFlip) 
            curSpamMesh = outMA_mesh_dir + 'spam' + str(i+1) + '.mesh'        

            ###############################  generate MA meshes  ################################      
            curSpamIma = outMA_ima_dir + 'spam' + str(i+1) + '.nii'
            print(f"Cur ima.......................{curSpamIma}")
            curSpamGZ = outMA_mesh_dir + 'gspam' + str(i+1) + '.nii.gz'
            curSpamGT = outMA_mesh_dir + 'gtpam' + str(i+1) + '.nii.gz' 
            tempSpamMesh = outMA_mesh_dir + 'temp' + str(i+1) + '.mesh'    
            gaussianSmoothCmd = 'AimsGaussianSmoothing -i ' + curSpamIma + ' -o ' + curSpamGZ + ' -x '+str(smoothingFactor)+' -y '+str(smoothingFactor)+' -z '+str(smoothingFactor)
            thresholdCmd = 'AimsThreshold -i ' + curSpamGZ + ' -o ' + curSpamGT + ' -b -t ' + str(aimsThreshold) 
            meshCmd = 'AimsMesh -i ' + curSpamGT + ' -o ' + tempSpamMesh + ' --deciMaxError 0.5 --deciMaxClearance 1  --smooth --smoothIt 20'
            zcatCmd = 'AimsZCat  -i ' +outMA_mesh_dir+'temp' + str(i+1) + '*.mesh' + ' -o ' + curSpamMesh
            print('....................Gaussian...................')
            os.system(gaussianSmoothCmd)
            print('....................threshold.....................')
            os.system(thresholdCmd)
            print('....................mesh.....................')       
            os.system(meshCmd)
            #if not os.path.exists(tempSpamMesh):
            #    print(f"⚠️ WARNING: No mesh generated for index {i+1}. Threshold {aimsThreshold} might be too high.")
            #    continue 
            # ---------------------------

            print('....................zcat.....................') 
            os.system(zcatCmd)
            os.system('rm '+outMA_mesh_dir+'temp*.mesh')
            os.system('rm '+outMA_mesh_dir+'temp*.mesh.minf')

            ###############################   moving MA meshes to inv_voitT space  #################################                
            self.cur_MA_tools.meshTrans(curSpamMesh,outMA_mesh_dir,curSpamMesh,trans=inv_file)
            os.system(('rm ' + outMA_mesh_dir + 'g*'))    

            ################################  moving MA meshes to coordanates  ##################################           
            coordMesh = outMA_mesh_dir+'ISO_spam' + str(i+1) + '.mesh' 
            coordMeshFlip = outMA_mesh_dir+'ISO_inverse_spam' + str(i+1) + '.mesh' 
            self.cur_MA_tools.meshTrans(curSpamMesh,outMA_mesh_dir,fileOutName=coordMesh,trans=oneTrmFileName)  #transform curSpamMesh 
            self.cur_MA_tools.meshTrans(curSpamMesh,outMA_mesh_dir,fileOutName=coordMeshFlip,trans=oneTrmFileNameFlip)  #transform curSpamMesh flip

            ################################################  add represnetative subjects  #############################################
            if (coordSubj[i] != 'none'):
                curCoordMesh = targetMesh_path + coordSubj[i] + '.mesh'
                fileOutName = outMA_mesh_dir + 'to_coord_' + coordSubj[i] + '.mesh'
                fileOutNameFlip = outMA_mesh_dir +  'to_coord_inverse_' + coordSubj[i] + '.mesh'
                fileOutNameSmooth = outMA_mesh_dir + 'to_coord_Smooth_' + coordSubj[i] + '.mesh'
                fileOutNameSmoothFlip = outMA_mesh_dir + 'to_coord_inverse_Smooth_' + coordSubj[i] + '.mesh'                
                smoothCoordSubj = outMA_mesh_dir + 'to_' + target_name + '_Smooth_' + coordSubj[i] + '.mesh'
                smoothCoordSubjFlip = outMA_mesh_dir + 'to_' + target_name + '_inverse_Smooth_' + coordSubj[i] + '.mesh'

                self.cur_MA_tools.meshTrans(curCoordMesh,outMA_mesh_dir,fileOutName,trans=oneTrmFileName)  #transform curCoordSubj
                self.cur_MA_tools.meshTrans(curCoordMesh,outMA_mesh_dir,fileOutNameFlip,trans=oneTrmFileNameFlip)  #transform curCoordSubj flip

                # Add a smoother version of the meshes by defaut, numIteration 200 instead of 100 by default
                self.cur_MA_tools.meshSmooth(curCoordMesh,outMA_mesh_dir,fileOutName=smoothCoordSubj,numIteration='200')   # to_target_subj   
                self.cur_MA_tools.meshSmooth(fileOutName,outMA_mesh_dir,fileOutName=fileOutNameSmooth,numIteration='200')  # to_coord_subj
                self.cur_MA_tools.meshSmooth(fileOutNameFlip,outMA_mesh_dir,fileOutName=fileOutNameSmoothFlip,numIteration='200')  # to_coord_subj_flip


#                # Remove unsmoothed meshes
                if os.path.exists(fileOutName):
                    os.remove(fileOutName)
                if os.path.exists(fileOutNameFlip):    
                    os.remove(fileOutNameFlip)     
                os.system(('rm ' + outMA_mesh_dir + '*.minf'))     


            os.system(('rm ' + outMA_mesh_dir + 'curTrans.txt'))
            os.system(('rm ' + outMA_mesh_dir + 'curTransFlip.txt'))
 














 

