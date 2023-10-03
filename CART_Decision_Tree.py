# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:51:18 2023

@author: sayan
"""

import numpy as np
import warnings


warnings.filterwarnings("ignore")


class CART:
    
    
    def __init__(self):
        self.delta=None
        self.frame=None
        self.target_column_name=None
        self.total_size=None
        self.leaf_size=None
        self.root=None
   

    class Node:
        def __init__(self,is_output_label,output_label,feature):
            self.is_output_label=is_output_label
            self.output_label=output_label
            self.feature=feature
            self.children={}
        
        def add_child(self,node,branch_name):
            self.children[branch_name]=node
        
    
    
    def predict_for_row(self,data,row,root):
        
        while root.is_output_label==False:
            feat=root.feature
            
            val=data.iloc[row][feat]
            split_point=list(root.children.keys())[0]
                    
            if val <= split_point:
                root = root.children[split_point]
            else:
                root=root.children[split_point+0.0001]
        return root.output_label

    
   
    def predict(self,x_test,y_test):
      
        y_pred=[]
        
        
        for row in range(len(x_test)):
            y_pred.append(self.predict_for_row(x_test,row,self.root))
            
     
        return y_pred
    
    
    def fit(self,frame,target,total_size,delta,leaf_size):
        self.delta=delta
        self.frame=frame
        self.target_column_name=target
        self.total_size=total_size
        self.leaf_size=leaf_size
        self.root = self.create_tree(self.frame,self.target_column_name,self.total_size)
        return self.root





    def get_current_root_and_split_point(self,frame,target):
   
        mappings={}
    
        for col in frame.columns:
            if col == target:
                continue

      
            for u in np.arange(min(frame[col]),max(frame[col]), 0.5):
                split=u
                left=frame[frame[col]<=split]
                right=frame[frame[col]>split]
                gini= self.get_gini_index_of_set(left, target) * left.shape[0]/frame.shape[0] + self.get_gini_index_of_set(right, target) * right.shape[0]/frame.shape[0]
            
                pair=(col,split)
                mappings[pair]=gini
       
    
            min_value = float('inf')  
            min_key = None 

            for key, value in mappings.items():
                if value < min_value:
                    min_value = value
                    min_key = key
   
            return min_key[0],min_key[1]






    def get_output_label_of_pure_set(self,frame,target):
            return frame.iloc[0][target]
        
    
    
    def get_gini_index_of_set(self,frame,target):
        size=len(frame)
        unique_counts=frame[target].value_counts()
        unique_vals=unique_counts.index
        #print(unique_counts)
        gini=0
        for u in unique_vals:
            gini = gini + (   (unique_counts[u]/size) **2 )
          
        gini = 1-gini
        return gini       
    




    def create_tree(self,df,target,total_size):
        gini=self.get_gini_index_of_set(df,target)
   
        if gini == 0:
            leaf_node = self.Node(True,self.get_output_label_of_pure_set(df,target), None)
            return leaf_node
    
        if gini !=0 and df.shape[0] < self.leaf_size:
            one_counts=len(df[target]==1)
            zero_counts=len(df[target]==0)
            if one_counts < zero_counts:
                max_prob_class=0
            else:
                max_prob_class=1
                leaf_node = self.Node(True,max_prob_class, None)
                return leaf_node
        
          
        
   
    
        root_feature,split_point=self.get_current_root_and_split_point(df,target)
        left_subset=df[df[root_feature]<=split_point]
        right_subset=df[df[root_feature]>split_point]
   
    
   
        root = self.Node(False,None,root_feature)
    
        if len(left_subset) > 0:
            left=self.create_tree(left_subset,target,total_size)
            root.add_child(left,split_point)
    
        if len(right_subset) > 0:
            right=self.create_tree(right_subset,target,total_size)
            root.add_child(right,split_point+self.delta)
    
   
    
        return root

    



 
    


        
    


    










