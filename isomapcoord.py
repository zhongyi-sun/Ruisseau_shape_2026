import sys
import os
import numpy as np 
import pandas as pd
import math
#from Isomap_dist import Isomap_dist
from sklearn.manifold import Isomap
import general_tools

import itertools


class isomapcoord:
    """ calls sklearn isomap algorithm, modified to accept distance matrix
    """

    def __init__(self,arg1):
        self.arg1 = arg1
    

    def getIsomap(self,sulciName,ICPfilePath,ISOfilePath,typeIso,numNeig,filterOut,typeDist,numDim):
        """ recompose the distance calculation results
        """
        curDist = []
        curDistName = ''
        outisonameAll = ''
        
        if filterOut:
            curDistName = ICPfilePath + typeDist + 'Dist' + sulciName + 'outlierRemoved.txt'
            outisonameAll = ISOfilePath+'isomap'+typeIso+sulciName+'k'+str(numNeig)+'d'+str(numDim)+'dist'+typeDist+'.txt'            
        else:
            curDistName = ICPfilePath + typeDist + 'Dist' + sulciName + '.txt'
            outisonameAll = ISOfilePath+'isomap'+typeIso+sulciName+'k'+str(numNeig)+'d'+str(numDim)+'dist'+typeDist+'_keepOut.txt'               
        print(curDistName)                
#        curDist = pd.read_csv(curDistName,sep=' ',index_col=0)
        curDist = pd.read_csv(curDistName,index_col=0)

        subjNames = curDist.index
        dimNames = np.arange(1,numDim+1)
        #call isomap_dist, modified sklearn function to accept distance matrix
#        isoN = Isomap_dist(n_neighbors=numNeig, n_components=numDim).fit_transform(curDist.values)

        isoN = Isomap(n_neighbors=numNeig,n_components=numDim,metric='precomputed').fit_transform(curDist.values)
        isoallDF = pd.DataFrame(isoN,index=subjNames,columns=dimNames) 
#        isoallDF.to_csv(outisonameAll,sep=' ')     
        isoallDF.to_csv(outisonameAll)  
        outisonameDim = ''
        for i in range(numDim):
            curDim = isoallDF.iloc[:,i]
            if filterOut:
                outisonameDim = ISOfilePath+'isomap'+typeIso+sulciName+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(i+1)+'_dist'+typeDist+'.txt'            
            else:
                outisonameDim = ISOfilePath+'isomap'+typeIso+sulciName+'k'+str(numNeig)+'d'+str(numDim)+'_'+str(i+1)+'_dist'+typeDist+'_keepOut.txt' 
#            curDim.to_csv(outisonameDim,sep=' ')
            curDim.to_csv(outisonameDim)


    def coordSubjCount(self,isomapData,minCoord,maxCoord,numCoord):
        """
          called by getCoordAndSubj to return the number of subjects at the coord extremities
        """
        choiceCoord = "threshold" # "number" returns the number of subjects at extremities
                                  # "threshold" returns the number of subjects within a defined threshold
        isomapNames = isomapData.index
        # define distCoord: distance to each coord
        coordVal = np.linspace(minCoord,maxCoord,num=numCoord)  
        numrow = isomapData.shape[0]  
        distCoord = np.arange(numrow*numCoord).reshape((numrow,numCoord))
        distCoord = np.array(distCoord,dtype='float')      
        for i in range(numrow):
            distCoord[i] = abs(coordVal - isomapData.iloc[i].values)
        distCoord = pd.DataFrame(distCoord, columns=coordVal)

        # define assignCoord: assigning each subj to its minimun distance coord
        assignCoord = distCoord.idxmin(axis=1)
        assignCoord = pd.DataFrame(assignCoord,columns=['coordNum'])
        assignCoord.index = isomapNames

        # count the number of subjects assigned to the two extremities of coord
        count = assignCoord.groupby('coordNum').size()
        lastpos = count.shape[0] - 1
        numMin = count.iloc[0]       # number of minCoord subj
        numMax = count.iloc[lastpos] # number of maxCoord subj    

        # count the number of elements in the first and last column of the distCoord 
        # when the value is less than threshold
        threshold = abs( (maxCoord-minCoord)/numCoord ) / 3  # tried 1.5, 2, 3
        count_min = np.sum(distCoord.iloc[:, 0] <= threshold)
        count_max = np.sum(distCoord.iloc[:, -1] <= threshold)

        if (choiceCoord == "number"):
            return numMin, numMax
        if (choiceCoord == "threshold"):
            return count_min, count_max               
            

    def getCoordAndSubj(self,curDistName,maxDistName,isomapName,coordFileName,outDetailName,outSimpleName,coordStep,coordPercent,minExtremitySubject,numCoord,distThresh): 
#        curDist = pd.read_csv(curDistName,index_col=0)
        curMaxDist = pd.read_csv(maxDistName,index_col=0)        
        curIso = pd.read_csv(isomapName,index_col=0) 
        
        print('isomapName: '+isomapName)
        
        
        ############################################################################################
        ## get the coord and write to coord file

        # get number of subject needed at min max coord: numsubjThresh
        numsubjThresh = -1
        if (coordPercent != -1):    
            numSubj = curIso.shape[0]
            numsubjThresh = coordPercent * numSubj
            numsubjThresh = np.ceil(numsubjThresh)
        else:
            numsubjThresh = minExtremitySubject           
        # get initial min max coord
        startMinCoord = curIso.values.min()
        startMaxCoord = curIso.values.max()
        minCoord,maxCoord = startMinCoord, startMaxCoord
        # get initial number of subjects at min max coord
        numMin, numMax = -1, -1
        step = (startMaxCoord - startMinCoord)/coordStep      
        numMin, numMax = self.coordSubjCount(isomapData=curIso,minCoord=startMinCoord,maxCoord=startMaxCoord,numCoord=numCoord)     
        # find the min max coord where there are numsubjThresh subjects
        while ((numMin < numsubjThresh) | (numMax < numsubjThresh)):
            # note that update step here will change the value of step as we go closer to the center
            #print('step in while is :' + str(step))
            #step = (maxCoord - minCoord)/coordStep
            if (numMin < numsubjThresh):
                minCoord = minCoord + step
            if (numMax < numsubjThresh):
                maxCoord = maxCoord - step                            
            numMin, numMax = self.coordSubjCount(isomapData=curIso,minCoord=minCoord,maxCoord=maxCoord,numCoord=numCoord)     
            
            print('numMin, numMax:')
            print(str(numMin)+'  '+str(numMax))
              
        coordVal = []
        coordVal = np.linspace(minCoord,maxCoord,num=numCoord)
        np.savetxt(coordFileName,coordVal)

        #############################################################################################
        ## get list of subj close to each coord and write to simple and detailed corrdSubj output file
        
        #curThresh = step * distThresh
        curThresh = ((maxCoord - minCoord)/(numCoord - 1)) * distThresh
        orig_stdout = sys.stdout
        f = open(outDetailName, 'w')
        sys.stdout = f        
        coordSubj = np.array([])
        curCoordCount = 1
        representativeList = []        
        for element in coordVal:
            curMinT = element - curThresh    
            curMaxT = element + curThresh 
            select = curIso[(curIso >= curMinT) & (curIso < curMaxT)]   
            curList = select.dropna()                         
#            print("........................................")
#            print("curCoord: " + str(curCoordCount) + "  " + str(element))
#            print("minThresh  maxThresh :" + str(curMinT) + "  " + str(curMaxT)) 

            representative = ''
            if not curList.empty:
                if curList.shape[0] > 2:
                    curSubj = curList.index.values                                                            
                    curDistLoc = pd.DataFrame(curMaxDist,index=curSubj,columns=curSubj)                 
                    colSum = curDistLoc.sum(axis=1)
                    colSum = pd.DataFrame(colSum)                 
                    sortedColSum = colSum.sort_values(colSum.columns[0])
                    
                    print(sortedColSum)
                    sortedNames = sortedColSum.index.values
                    representative = sortedNames[0]                    
                if curList.shape[0] == 2:
                    curDistLoc = curList - element
                    curDistLoc = curDistLoc.abs()
                    sortedDist = curDistLoc.sort_values(curDistLoc.columns[0])                
                    representative = sortedDist.index[0]
                    sortedNames = sortedDist.index.values
                    print(sortedDist)   
                if curList.shape[0] == 1:
                    representative =  curList.index[0]   
                    print(representative)                 
            else:
                print('No subject found.')
#                representative = str(curCoordCount)
                representative = "none"
            curCoordCount = curCoordCount + 1 
            print()
            representativeList.append(representative)

        # close the system output to the detailed subjet file
        sys.stdout = orig_stdout
        f.close()
        print(representativeList)
        outSimple = open(outSimpleName,'w')
        outSimple.write(str(representativeList))
        outSimple.close        


###################################################################################################
# Implementation that get the coordinates through sorted isomap, get the representative points 
# through the isomaps instead of the distance matrix: distMax
###################################################################################################
    def getCoordAndSubj_new(self,isomapName,coordFileName,outDetailName,outSimpleName,coordPercent,minExtremitySubject,numCoord,coordDim,distThresh):
        print(isomapName)
#        curIso = pd.read_csv(isomapName,sep=' ',index_col=0) 
        curIso = pd.read_csv(isomapName,index_col=0) 

        numSubj = curIso.shape[0]
        numsubjThresh = -1
        if (coordPercent != -1):
            numsubjThresh = coordPercent * numSubj
        else:
            numsubjThresh = minExtremitySubject
        numsubjThresh = math.ceil(numsubjThresh) 
        numsubjIndex = numsubjThresh - 1

        ######  get the coord  ######
        curIsoVal = curIso.explode(coordDim, ignore_index=True)
        curIsoVal = curIsoVal[coordDim].values.astype(float)    
        nth_smallest = np.partition(curIsoVal, numsubjIndex)[numsubjIndex]
        nth_largest = -np.partition(-curIsoVal, numsubjIndex)[numsubjIndex]
        coordVal = np.linspace(nth_smallest,nth_largest,num=numCoord)
        np.savetxt(coordFileName,coordVal)

        ######  get the coord subj  ######
        out_name_detail = outDetailName
        out_name_simple = outSimpleName
        file_mode = "w"           
        with open(out_name_detail, file_mode) as file:
            coordVal_str = "\n".join(map(str, coordVal))
            file.write(coordVal_str+"\n")
            curThresh = ((nth_largest - nth_smallest)/(numCoord - 1)) * distThresh   
            representativeList = []
            curCoord = 1
            for element in coordVal:
                file.write("curCoord: "+str(curCoord)+"  "+str(element))
                curCoord = curCoord + 1
                curMinT = element - curThresh    
                curMaxT = element + curThresh 
                select = curIso[(curIso >= curMinT) & (curIso <= curMaxT)]
                select = select.sort_values(coordDim)
                curList = select.dropna()  
                curSubj = curList.index.values
                if len(curSubj) != 0:
                    representativeList.append(curSubj[0])
                else:
                    representativeList.append("none")
                #write to detailed file
                curList_str = "\n".join(map(str, curList))
                file.write(curList_str+"\n")
                file.write("minThresh  maxThresh: "+str(curMinT) + " "+str(curMaxT)+"\n")
                file.write("Index\tcoordDim\n") # Write column name as header
                for index, value in curList.iterrows():
                    value_str = "{:.2f}".format(value[str(coordDim)])
                    file.write(f"{index}\t{value_str}\n")
        #write to simple coord name file
        with open(out_name_simple, file_mode) as file:   
#            representativeList_str = "\n".join(map(str, representativeList))
#            file.write(representativeList_str)
            file.write(str(representativeList))


    def getNoneNormWeightOriWeight(self,curDistName,isomapName,coordFileName,noneNormWeightName,oriNormWeightName,coordSubjName,numCoord,scale):
        """ get the non-normalized and the normalized weight given an isomap file
        """
#        coordSubj = pd.read_csv(coordSubjName,sep=' ',header=None)
        coordSubj = pd.read_csv(coordSubjName,header=None)

        coordSubj = coordSubj.iloc[0]
        coordSubj = [s.replace('[', '').replace(']', '').replace("'", "").replace(",","") for s in coordSubj]
#        curDist = pd.read_csv(curDistName,sep=' ',index_col=0)
#        curIso = pd.read_csv(isomapName,sep=' ',index_col=0) 
##        curIso = pd.read_csv(isomapName,sep=' ',index_col=0,header=None) # problem! one line of numbers written as colnames
        curIso = pd.read_csv(isomapName,index_col=0) 
        curDist = pd.read_csv(curDistName,index_col=0)

        subjNames = curIso.index
        numSubj = curIso.shape[0]
        curCoord = np.loadtxt(coordFileName)
        minCoord = curCoord.min()
        maxCoord = curCoord.max()
        curVar = (((maxCoord - minCoord)/(numCoord -1)) * scale) ** 2     
        isoTile = np.tile(curIso,(1,numCoord))
        coordTile = np.tile(curCoord,(numSubj,1))
        curDistToCoord = np.abs(isoTile - coordTile)
        dimDist = np.exp(-curDistToCoord**2/curVar)
        # get normalized weight
        sumCoordDist = dimDist.sum(axis=0)
        sumCoordDistTile = np.tile(sumCoordDist,(numSubj,1))      
        normDist = dimDist / sumCoordDistTile
        # add row and col names, save
        coord_Subj_list = [str(curCoord) + ' ' + coordSubj for curCoord, coordSubj in zip(curCoord, coordSubj)]
        dimDist = pd.DataFrame(dimDist)
        normDist = pd.DataFrame(normDist)
        dimDist.index = subjNames
        dimDist.columns = coord_Subj_list
        normDist.index = subjNames
        normDist.columns = coord_Subj_list       
        dimDist.to_csv(noneNormWeightName,header=coord_Subj_list)
        normDist.to_csv(oriNormWeightName,header=coord_Subj_list)


    def getNoneNormWeightOriWeight_oriFormat(self,curDistName,isomapName,coordFileName,noneNormWeightName,oriNormWeightName,numCoord,scale):
        """ get the non-normalized weight given an isomap file
        """
        curIso = pd.read_csv(isomapName,index_col=0) 
        curDist = pd.read_csv(curDistName,index_col=0)
        

        subjNames = curIso.index
        numSubj = curIso.shape[0]
        curCoord = np.loadtxt(coordFileName)
        minCoord = curCoord.min()
        maxCoord = curCoord.max()
        curVar = (((maxCoord - minCoord)/(numCoord -1)) * scale) ** 2     
        isoTile = np.tile(curIso,(1,numCoord))
        coordTile = np.tile(curCoord,(numSubj,1))
        curDistToCoord = isoTile - coordTile
        curDistToCoord = np.abs(curDistToCoord)
        dimDist = np.exp(-curDistToCoord**2/curVar)

        # get normalized weight
        sumCoordDist = dimDist.sum(axis=0)
        sumCoordDistTile = np.tile(sumCoordDist,(numSubj,1))      
        normDist = dimDist / sumCoordDistTile

        # add row and col names, save
        dimDist = pd.DataFrame(dimDist)
        normDist = pd.DataFrame(normDist)
        dimDist.index = subjNames
        dimDist.columns = curCoord
        normDist.index = subjNames
        normDist.columns = curCoord
        dimDist.to_csv(noneNormWeightName,index=False)
        normDist.to_csv(oriNormWeightName,index=False)


    def getNormWeight(self,subjNames,noneNormWeightName,normWeightOutFile):
        """ 
        calculate the weight based on a given subject list and the noneNormWeight
        write the subjectNameFile and the normWeightOutFile 
        
        """
        numSubj = len(subjNames)
#        dimDist = pd.read_csv(noneNormWeightName,sep=' ',index_col=0)
        dimDist = pd.read_csv(noneNormWeightName,index_col=0)           
        curCoord = dimDist.columns
        dimDist = dimDist.loc[subjNames] # update dimDist      
        # get normalized weight
        sumCoordDist = dimDist.sum(axis=0)     
        normDist = dimDist.div(sumCoordDist, axis=1)   
        # add row and col names, save       
        normDist = pd.DataFrame(normDist, index=subjNames, columns=curCoord)
#        normDist.to_csv(normWeightOutFile,sep=' ',index=True)
        normDist.to_csv(normWeightOutFile,index=True)


    def getAnatomyProjectionNames(self,baseLoc,curAct):
        bckLoc = os.path.join(baseLoc, curAct)
        activation_dirs = ['left', 'right']
        file_ext = '.bck'
        projection_file_list = []

        for activation_dir in activation_dirs:
            full_activation_dir = os.path.join(bckLoc, activation_dir)
            if not os.path.exists(full_activation_dir):
                continue  # Skip if the directory doesn't exist

            projection_file_list.extend([os.path.splitext(f)[0] for f in os.listdir(full_activation_dir) if f.endswith(file_ext)])

        # Add 'flip-' prefix to the right-side projection files
        projection_file_list.extend(['flip-' + file for file in projection_file_list if file.startswith('right-')])
        return projection_file_list


    def getHCPprojectionNames(self,activation,curAct,subProcess):
        curList,pbList = [],[]
        if process in ['HCPactivation', 'HCPmask', 'HCPbundleM', 'HCPcurvature', 'HCPmyelin']:
            for element in os.listdir(activation):
                if process == 'HCPactivation': 
                    inName = f"{activation}{element}/{curAct}/{curAct}_cope{subProcess}_tstat.nii.gz" 
                    # inName = activation+element+'/'+curAct+'/'+curAct+'_cope'+subProcess+'_tstat.nii.gz'
                elif process == 'HCPmask':
                    inName = f"{activation}{element}/t1mri/default_acquisition/default_analysis/segmentation/{curAct}wm.nii.gz" 
                elif process == 'HCPbundleM':
                    inName = f"{activation}{element}/probMap/{element}_{curAct}_{subProcess}.nii.gz"
                elif process == 'HCPcurvature':
                    inName = f"{activation}{element}/{element}.curvature_MSMAll.32k_fs_LR.nii.gz"
                elif process == 'HCPmyelin':
                    inName = f"{activation}{element}/{element}.MyelinMap_BC_MSMAll.32k_fs_LR.nii.gz"
                if os.path.isfile(inName):
                    curList.append(element)
                else:
                    pbList.append(element)
            curList = ['L' + s for s in curList] + ['flip-R' + s for s in curList]
            pbList = ['L' + s for s in pbList] + ['flip-R' + s for s in pbList]
        if process == 'HCPbundle':
            for element in os.listdir(activation):
                inNameL = f"{activation}{element}/tract/3000/dbundles/bundleMapsReferential/left-hemisphere/{subProcess}_Left.nii.gz"  
                inNameR = f"{activation}{element}/tract/3000/dbundles/bundleMapsReferential/right-hemisphere/{subProcess}_Right.nii.gz" 
                if os.path.isfile(inNameL):
                    curList.append('L' + element)
                else:
                    pbList.append('L' + element)
                if os.path.isfile(inNameR):
                    curList.append('flip-R' + element)
                else:
                    pbList.append('flip-R' + element)
        return curList, pbList
    

    ################################################  Projection functionality  #################################################
    def getSubsetNameNormWeight(self, oldWeightPath, newSubjNames, newWeightSavePath):
        """
        Given an old weight and a new subject set smaller or equal to the old set, filters and re-normalizes weights. 
        Filters out the subset of old weight containing only the new subjects, then normalize it by deviding the new subset
        by the sum of its columns
        Maintains bit-for-bit identity if the subject list is unchanged.
        """
        # Load with index name preservation
        df_old = pd.read_csv(oldWeightPath, index_col=0)
        
        # Filter subjects
        valid_names = [name for name in newSubjNames if name in df_old.index]
        df_new = df_old.loc[valid_names].copy()
        
        # Smart Re-normalization
        # Calculate sums for each column
        column_sums = df_new.sum(axis=0)
        # Check if actually need to re-normalize.
        if not np.allclose(column_sums, 1.0, atol=1e-15): # use tolerance (1e-15) to see if the sum is basically 1.0
            print("Re-normalizing subset...")
            column_sums = column_sums.replace(0, 1)
            df_normalized = df_new.divide(column_sums, axis=1)
        else:
            print("Subject list unchanged or already normalized. Skipping division to preserve precision.")
            df_normalized = df_new

        # Save with Strict Formatting
        # Use float_format to control the number of digits written to the CSV
        df_normalized.to_csv(
            newWeightSavePath, 
            index=True, 
            header=True,
            sep=',', 
            float_format='%.16g' # '%g' is better for mixed scientific notation
        )
        
        return df_normalized

