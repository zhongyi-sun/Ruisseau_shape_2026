import os
import numpy as np 
import pandas as pd
from general_tools import general_tools
from MA_basic_tools import MA_basic_tools
from MA_tools import MA_tools
import shutil
from soma import aims


class shape_visualisation:
    def __init__(self,arg1):
        self.arg1 = arg1
        self.curTools = general_tools(1)
        self.cur_MA_tools = MA_tools(1)


    ##########################################################################################################################
    def prep_target_mesh(self,inMeshDir,inTrmFile,center,shapeInDir,outDir,combPath,curRegion):

        """
          writing the mesh transformed to target space
        """
        curMA_basic_U = MA_basic_tools(1) 
        target_name = center
        regionICPPath = shapeInDir

        oriTalLoc = combPath + 'tal/' 
        df_tal_L_name = oriTalLoc + 'df_tal_L.csv'
        df_allICP_name = combPath+curRegion+'/ICP/transRotMat_' + curRegion + '.txt'

        df_tal_L_targetICP_name = regionICPPath + '/transRotMat_tal_L_' + target_name + '_' + curRegion + '.txt'
        df_tal_L_targetICP_voirT_name = regionICPPath + '/transRotMat_tal_L_' + target_name + '_voirT_' + curRegion + '.txt'
        curMA_basic_U.prep_ICP_Trm(df_tal_L_name,df_allICP_name,target_name,df_tal_L_targetICP_name,df_tal_L_targetICP_voirT_name)  
        df_target = pd.read_csv(df_tal_L_targetICP_name,index_col=0,header=0)

        for file_name in os.listdir(inMeshDir):
            if file_name.endswith('mesh'):
                curOutName = outDir + '/' + file_name
                print(curOutName) 
                base_name, extension = os.path.splitext(file_name)
                name = base_name
                trans = df_target.loc[name].tolist()
                result_trans = self.curTools.parseTransRotLine_to_array(trans)
                # Save the transformation to a text file                    
                result_trans_path = outDir + name + '_tal.txt'
                np.savetxt(result_trans_path, result_trans, delimiter=' ')
                file_name_complete_in = inMeshDir + '/' +  file_name
                self.cur_MA_tools.meshTrans(file_name_complete_in,outDir,curOutName,result_trans_path)
                os.remove(result_trans_path)


    def prep_spread_mesh(self,inName,outDir,weightName,bckInDir,coordScale,numCoord):

        """
          writing the mesh transformed to shape spread space
        """
        curMA_basic_U = MA_basic_tools(1)         
        outMeshDir = outDir + '/'
        curMA_basic_U.prep_ISO_Mesh(inName,weightName,coordScale,numCoord,outMeshDir,bckInDir)  













 

