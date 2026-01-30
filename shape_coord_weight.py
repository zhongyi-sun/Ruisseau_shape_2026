import sys
import os
import numpy as np 
import pandas as pd
import math

from general_tools import general_tools


class shape_coord_weight:
    """ generate coordinates and weights for a given shape descriptor
        a more general implementation of the isomapcoord.py
    """

    def __init__(self,arg1):
        self.arg1 = arg1    


    def find_min_max(self,shape_descriptors, threshold_percentage=0.1, num_coord=10,step=0.01):
#    def find_min_max(self,shape_descriptors, threshold_percentage=0.05, num_coord=10,step=0.01):
        """ given a shape descriptor, the percentage of subjects at the extremities, the
            number of coordinates, the size of each step, find the min and max coordinates 
        """
        min_value = np.min(shape_descriptors)
        max_value = np.max(shape_descriptors)    
        threshold_count = int(threshold_percentage * len(shape_descriptors)) 
        print('Threshold count: '+str(threshold_count))
        print('min: ' + str(min_value))
        print('max: ' + str(max_value))

        num_intervals = num_coord - 1  # There are num_coord points and num_coord-1 intervals
        while True:
            # Define the range and intervals
            range_value = max_value - min_value
            intervals = np.linspace(min_value, max_value, num_coord)
            print('Current intervals:')
            print(intervals)

            # Count subjects in the first and last intervals
            first_interval_count = np.sum((shape_descriptors >= intervals[0]) & (shape_descriptors <= intervals[1]))
            last_interval_count = np.sum((shape_descriptors >= intervals[-2]) & (shape_descriptors <= intervals[-1]))
            
            first_interval_count = int(first_interval_count)
            last_interval_count = int(last_interval_count)
                        
            print('minSubjCount and maxSubjCount: '+ str(first_interval_count)+' '+str(last_interval_count))

            # Check if conditions are met
            if (first_interval_count >= threshold_count) and (last_interval_count >= threshold_count):               
                return intervals
        
            # Adjust min and max values if conditions are not met
            if (first_interval_count < threshold_count):          
                min_value += step * range_value  # Adjust the min as needed
            if (last_interval_count < threshold_count):                
                max_value -= step * range_value  # Adjust the max as needed


    def get_norm_weight_legacy(self,shapeName,weightName,rewriteShape,shapeCol=0,numCoord=10,scale=1,):
        """ generate a weight file given the shape descriptor (isomap or others)
            shapeCol defines the column to be used in the shape descriptor
        """      
        ori_shape = pd.read_csv(shapeName,index_col=0)

        # select the relevant column in the shape file
        if isinstance(shapeCol, int):
            # Keep old functionality: Select by position
            shape = pd.DataFrame(ori_shape.iloc[:, shapeCol])
        elif isinstance(shapeCol, str):
            # New functionality: Select by column name
            if shapeCol in ori_shape.columns:
                shape = pd.DataFrame(ori_shape[shapeCol])
            else:
                raise ValueError(f"Column '{shapeCol}' not found in {shapeName}")
        else:
            raise TypeError("shapeCol must be an integer (index) or a string (column name)")

        # get the coordinate intervals
        intervals = self.find_min_max(shape,num_coord = numCoord)

        # Convert intervals from numpy array to a pandas DataFrame
        coord = pd.DataFrame(intervals, columns=['number'], index=[f'coord_{i}' for i in range(len(intervals))])

        # results which store the subject name of each corrdinate
        results = pd.DataFrame(columns=['coord', 'min_distance', 'subject_name'])
        
        for index, row in coord.iterrows():
            number = row['number']    
            distances = (shape - number).abs() # Calculate the absolute distance to each subject's value
            min_distance = distances.min()  # Find the minimum distance
            min_distance = min_distance[0]  # access content of multiIndex
            subject_name = distances.idxmin() # Find the corresponding subject name
            subject_name = subject_name[0]  # access content of multiIndex

            results = results.append({  # Store the results
                'coord': number,
                'min_distance': min_distance,
                'subject_name': subject_name
            }, ignore_index=True)
        coord_Subj_list = [f"{coord} {subject_name}" for coord, subject_name in zip(results['coord'], results['subject_name'])]

        # get the normalized weight
        maxCoord = results['coord'].iloc[-1]
        minCoord = results['coord'].iloc[0]
        curVar = (((maxCoord - minCoord)/(numCoord -1)) * scale) ** 2  
        isoTile = np.tile(shape,(1,numCoord))
        coordTile = np.tile(results['coord'],(len(shape),1))
        curDistToCoord = np.abs(isoTile - coordTile)
        dimDist = np.exp(-curDistToCoord**2/curVar)
        sumCoordDist = dimDist.sum(axis=0)
        sumCoordDistTile = np.tile(sumCoordDist,(len(shape),1)) 
        normDist = dimDist / sumCoordDistTile
        # Convert normDist to dataframe, add row and col names
        normDist = pd.DataFrame(normDist)
        normDist.index = shape.index
        normDist.columns = coord_Subj_list    
        normDist.to_csv(weightName,header=coord_Subj_list)
        print('Writing...')
        print(weightName)

        # rewrite shape if needed
        if rewriteShape != 'none':
            shape.to_csv(rewriteShape)


    def get_norm_weight(self, shapeName, weightName, rewriteShape, shapeCol=0, numCoord=10, scale=1):
            """ generate a weight file given the shape descriptor (isomap or others)
                shapeCol defines the column to take in the shape descriptor file
            """      
            ori_shape = pd.read_csv(shapeName, index_col=0)
            
            #########  Column Selection Logic  #########
            if isinstance(shapeCol, int):
                shape = pd.DataFrame(ori_shape.iloc[:, shapeCol])
            elif isinstance(shapeCol, str):
                if shapeCol in ori_shape.columns:
                    shape = pd.DataFrame(ori_shape[shapeCol])
                else:
                    raise ValueError(f"Column '{shapeCol}' not found in {shapeName}")
            else:
                raise TypeError("shapeCol must be an integer (index) or a string (column name)")

            # get the coordinate intervals
            intervals = self.find_min_max(shape, num_coord=numCoord)

            # Convert intervals to DataFrame
            coord = pd.DataFrame(intervals, columns=['number'], index=[f'coord_{i}' for i in range(len(intervals))])

            # Storage for results
            results_list = [] # Use a list instead of empty DF for speed/compatibility
            for index, row in coord.iterrows():
                number = row['number']    
                distances = (shape.iloc[:, 0] - number).abs() 
                
                min_distance = distances.min()
                subject_name = distances.idxmin()

                results_list.append({
                    'coord': number,
                    'min_distance': min_distance,
                    'subject_name': subject_name
                })
                
            results = pd.DataFrame(results_list)
            coord_Subj_list = [f"{c} {s}" for c, s in zip(results['coord'], results['subject_name'])]

            #############  Gaussian Weighting Logic  ##############
            maxCoord = results['coord'].iloc[-1]
            minCoord = results['coord'].iloc[0]
            curVar = (((maxCoord - minCoord) / (numCoord - 1)) * scale) ** 2  
            
            # Note: Using broadcasting here is more efficient than np.tile
            iso_vals = shape.values # (N, 1)
            coord_vals = results['coord'].values # (M,)
            
            curDistToCoord = np.abs(iso_vals - coord_vals) # Result is (N, M)
            dimDist = np.exp(-curDistToCoord**2 / curVar)
            
            # Normalization
            sumCoordDist = dimDist.sum(axis=0)
            normDist = dimDist / sumCoordDist # Broadcasting handles the division
            
            # --- Finalize and Save ---
            normDist_df = pd.DataFrame(normDist, index=shape.index, columns=coord_Subj_list)
            normDist_df.to_csv(weightName)
            
            print(f"Writing... {weightName}")

            if rewriteShape != 'none':
                shape.to_csv(rewriteShape)