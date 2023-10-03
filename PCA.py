# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:35:24 2023

@author: sayan
"""



import numpy as np
import pandas as pd
import scipy.linalg as la
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")



class PCA:
    
    def __init__(self):
        self.a=None
        self.eigvals=None
        self.eigvecs=None
        
        
        
        
        
        
    def compress(self,df,target_variance,target_column_name):
        
       y=df[target_column_name]
       x=df.drop(columns=[target_column_name])
       data=x
       cov=np.cov(data.T)
      
      
       self.eigvals, self.eigvecs = la.eig(cov)
       
       sorted_indexes = np.argsort(self.eigvals,axis=0)[::-1]
       self.eigvals = self.eigvals[sorted_indexes]
       self.eigvecs = self.eigvecs[:,sorted_indexes]
       
       total_var=sum(self.eigvals)
      
       
       
       var=[]
       columns=[]
       percentage=100
       for i in range(0,len(self.eigvals)):
           var.append(percentage)
           columns.append(i)
           percentage = percentage - self.eigvals[i]/total_var*100
           
           

       
      
       
       
       
       target=(target_variance) * total_var/100
     
       sum_var=0
       cols=0
       for i in range(0,len(self.eigvals)):
           sum_var=sum_var+self.eigvals[i]
           cols=cols+1
           
           if(sum_var>=target):
               break
      
       new_base=self.eigvecs[:,0:cols]
       x=new_base.T.dot(data.T)
       x=x.T
       dataframe = pd.DataFrame.from_records(x)
       
       dataframe=dataframe.assign(income=y)
       return dataframe