import yaml
import sys
import os
import shutil
import pandas as pd
import numpy as np
#from itertools import chain

from database_projection import database_projection
from shape_coord_weight import shape_coord_weight
from shape_visualisation import shape_visualisation
from MA_tools import MA_tools
from MA_basic_tools import MA_basic_tools
from general_tools import general_tools

def main(argv):
    #curYaml = 'config_ataxia_shape_relabel_redo.yaml' # 'config_ataxia_shape.yaml'
    curYaml = 'config_Champollion_sca_shape.yaml' # 'config_ataxia_shape.yaml'
    with open(curYaml, 'r') as file:
        config = yaml.safe_load(file)
    curTools = general_tools(1)

    ###########################################
    ######### defining subprocesses  ##########
    ###########################################
    callChangeSubjPrefix = config['global']['callChangeSubjPrefix']
    callGetSubjNames = config['global']['callGetSubjNames']
    callGetProjection = config['global']['callGetProjection']
    callGetWeight = config['global']['callGetWeight']
    callPrepProjectionMesh = config['global']['callPrepProjectionMesh']
    callGetMeshSpread = config['global']['callGetMeshSpread']
    callGenMA_ima = config['global']['callGenMA_ima'] 
    callGenMA_mesh = config['global']['callGenMA_mesh']   

    ###########################################
    # datastruture and filenaming convention  #
    ###########################################
    generalRoot = config['global']['generalRoot']             
    curRegion = config['global']['curRegion']

    shapeStudyRoot = generalRoot + '/Shape_study/'


#############################################################################################
#############################################################################################
    # Add prefix to CSV subjID
    if callChangeSubjPrefix:     
        input_csv = config['functions']['callChangeSubjPrefix']['input_csv']
        output_csv = config['functions']['callChangeSubjPrefix']['output_csv']
        prefix = config['functions']['callChangeSubjPrefix']['prefix']

        input_csv_name = generalRoot + input_csv
        output_csv_name = generalRoot + output_csv         
        curTools.add_prefix_to_csv_index(input_csv_name, output_csv_name, prefix)


#############################################################################################
#############################################################################################
    # Prepare bck and tal files for ICP calculation

    if callGetSubjNames:    
        curBaseOne = config['functions']['callGetSubjNames']['curBaseOne']
        curBaseTwo = config['functions']['callGetSubjNames']['curBaseTwo']
        outProjName_dir = shapeStudyRoot
        bckNames = '/bck/subjNames_bck.txt'

        if len(curBaseOne) !=0:
            supInfo = '' # add any supplimentary information to base, eg: _test1
            inFileName = generalRoot + curBaseOne + '/' + curRegion + bckNames
            if (os.path.exists(outProjName_dir)==0):
                os.mkdir(outProjName_dir)
            outFileName = outProjName_dir + curBaseOne  + supInfo + '_subjNames.txt' 

            print('Reading... ' + inFileName)
            names = curTools.retrieveNames(inFileName)
            newNames = curTools.get_add_flip_to_Rsubjects(names)
            with open(outFileName, 'w') as file:
                for line in newNames:
                    file.write(line + '\n')
            print('Writing... ' + outFileName)

        if len(curBaseTwo) !=0:
            supInfo = '' # add any supplimentary information to base, eg: _test1
            inFileName = generalRoot + curBaseTwo + '/' + curRegion + bckNames
            if (os.path.exists(outProjName_dir)==0):
                os.mkdir(outProjName_dir)
            outFileName = outProjName_dir + curBaseTwo  + supInfo + '_subjNames.txt' 

            print('Reading... ' + inFileName)
            names = curTools.retrieveNames(inFileName)
            newNames = curTools.get_add_flip_to_Rsubjects(names)
            with open(outFileName, 'w') as file:
                for line in newNames:
                    file.write(line + '\n')
            print('Writing... ' + outFileName)


    if callGetProjection:
        curBaseTwo = config['functions']['callGetProjection']['curBaseTwo']        
        projectFrom = config['functions']['callGetProjection']['projectFrom']
        projectTo =  config['functions']['callGetProjection']['projectTo']
        typeDist =  config['functions']['callGetProjection']['typeDist']
        typeProjection =  config['functions']['callGetProjection']['typeProjection']
        subjectsOne = config['functions']['callGetProjection']['subjectsOne'] 
        subjectsTwo = config['functions']['callGetProjection']['subjectsTwo']
        combDir = config['functions']['callGetProjection']['combDir'] 
        shapeValTwo = config['functions']['callGetProjection']['shapeValTwo'] 
        outName = config['functions']['callGetProjection']['outName']
        outNameSubj = config['functions']['callGetProjection']['outNameSubj']

        subjectsOneFile = shapeStudyRoot + subjectsOne
        subjectsTwoFile = shapeStudyRoot + subjectsTwo
        regionShapeDir = shapeStudyRoot + curRegion
        projectionDir = regionShapeDir+'/'+projectFrom+'_project_to_'+projectTo+'/'
        # create projection dir if needed        
        if (os.path.exists(regionShapeDir)==0):
            os.mkdir(regionShapeDir)
        if (os.path.exists(projectionDir)==0):
            os.mkdir(projectionDir)
        # get the shape description values 
        shapeValTwoFile = generalRoot+curBaseTwo+'/'+curRegion+'/'+shapeValTwo
        # get the distance
        distanceFile = generalRoot+combDir+'/'+curRegion+'/ICP/distMat_'+curRegion+'.txt'
        # outputFileName
        outFileName = projectionDir+outName
        outFileNameSubj = projectionDir+outNameSubj

        print(outFileName)
        print(outFileNameSubj)

        curProjectionU = database_projection(1)
        values,subj = curProjectionU.get_projection(typeDist=typeDist,typeProjection=typeProjection,subjectsOneFile=subjectsOneFile,subjectsTwoFile=subjectsTwoFile,shapeValTwoFile=shapeValTwoFile,distanceFile=distanceFile,projectionDir=projectionDir)
        values.to_csv(outFileName,index_label='subjName', header=['1'])
        subj.to_csv(outFileNameSubj,index_label='subjName', header=['1'])


    if callGetWeight:
        shapeInDir = config['functions']['callGetWeight']['shapeInDir']
        inName = config['functions']['callGetWeight']['inName']
        shapeCol = config['functions']['callGetWeight']['shapeCol']
        outName = config['functions']['callGetWeight']['outName']
        rewriteShape = config['functions']['callGetWeight']['rewriteShape']
        numCoord = config['functions']['callGetWeight']['numCoord']

        shapeName = generalRoot + shapeInDir + '/' + inName
        weightName = generalRoot + shapeInDir + '/' + outName
        if rewriteShape != 'none':
            rewriteShape = generalRoot + shapeInDir + '/' + rewriteShape

        coordWeightU = shape_coord_weight(1)
        coordWeightU.get_norm_weight(shapeName=shapeName,weightName=weightName,rewriteShape=rewriteShape,shapeCol=shapeCol,numCoord=numCoord)


    if callPrepProjectionMesh:
        shapeInDir = config['functions']['callPrepProjectionMesh']['shapeInDir']        
        inBckDir = config['functions']['callPrepProjectionMesh']['inBckDir']
        combPath = config['functions']['callPrepProjectionMesh']['combPath']
        curTrmFile = config['functions']['callPrepProjectionMesh']['curTrmFile']
        center = config['functions']['callPrepProjectionMesh']['center']
        shapeInDir = generalRoot + shapeInDir
        inMeshDir = generalRoot + inBckDir
        inTrmFile = generalRoot + curTrmFile
        combPath = generalRoot + combPath

        shapeVisuU = shape_visualisation(1)
        shapeVisuU.prep_target_mesh(inMeshDir=inMeshDir,inTrmFile=inTrmFile,center=center,shapeInDir=shapeInDir,outDir=spreadDir,combPath=combPath,curRegion=curRegion)


    if callGetMeshSpread:
        bckInDir = config['functions']['callGetMeshSpread']['bckInDir']
        shapeInDir = config['functions']['callGetMeshSpread']['shapeInDir']
        inName = config['functions']['callGetMeshSpread']['inName']
        outDir = config['functions']['callGetMeshSpread']['outDir']
        curWeightFile = config['functions']['callGetMeshSpread']['curWeightFile']
        coordScale = config['functions']['callGetMeshSpread']['coordScale']
        numCoord = config['functions']['callGetMeshSpread']['numCoord']        
        
        shapeInDir = generalRoot + shapeInDir
        bckInDir = generalRoot + bckInDir + '/'
        inName = shapeInDir + '/' + inName        
        spreadDir = shapeInDir + outDir
        if (os.path.exists(spreadDir)==0):
            os.mkdir(spreadDir)
        weightName = generalRoot + curWeightFile

        shapeVisuU = shape_visualisation(1)
        shapeVisuU.prep_spread_mesh(inName=inName,outDir=spreadDir,weightName=weightName,bckInDir=bckInDir,coordScale=coordScale,numCoord=numCoord)


    if callGenMA_ima:
        curWeightFile = config['functions']['callGenMA_ima']['curWeightFile']
        ima_in_dir = config['functions']['callGenMA_ima']['ima_in_dir']
        imaMA_out_dir = config['functions']['callGenMA_ima']['imaMA_out_dir']

        weightName = generalRoot + curWeightFile
        out_ima_dir = generalRoot + ima_in_dir + '/'
        outMA_ima_dir = generalRoot + imaMA_out_dir + '/'

        if (os.path.exists(outMA_ima_dir)==0):
            os.mkdir(outMA_ima_dir)

        curMA_basic_U = MA_basic_tools(1)
        curMA_basic_U.compose_MA_ima(weightName,out_ima_dir,outMA_ima_dir)


    if callGenMA_mesh:
        curWeightFile = config['functions']['callGenMA_mesh']['curWeightFile']
        imaMA_in_dir = config['functions']['callGenMA_mesh']['imaMA_in_dir']
        meshMA_out_dir = config['functions']['callGenMA_mesh']['meshMA_out_dir']
        targetMesh_path = config['functions']['callGenMA_mesh']['targetMesh_path']
        target_name = config['functions']['callGenMA_mesh']['target_name']
        smoothingFactor = config['functions']['callGenMA_mesh']['smoothingFactor']
        coordScale = config['functions']['callGenMA_mesh']['coordScale']
        aimsThreshold = config['functions']['callGenMA_mesh']['aimsThreshold']


        weightName = generalRoot + curWeightFile
        outMA_ima_dir = generalRoot + imaMA_in_dir + '/'
        inv_file = generalRoot + '/voirTalairachInv.trm'
        outMA_mesh_dir = generalRoot + meshMA_out_dir + '/'
        targetMesh_path = generalRoot + targetMesh_path + '/' 

        if (os.path.exists(outMA_mesh_dir)==0):
            os.mkdir(outMA_mesh_dir)

        curMA_basic_U = MA_basic_tools(1)
        curMA_basic_U.compose_MA_mesh(weightName,outMA_ima_dir,inv_file,smoothingFactor,coordScale,outMA_mesh_dir,targetMesh_path,aimsThreshold,target_name)


if __name__ == "__main__":
    main(sys.argv[1:])

