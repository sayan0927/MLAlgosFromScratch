# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 21:13:39 2023

@author: sayan
"""

import numpy as np



import warnings
warnings.filterwarnings("ignore")




class LogisticRegression:
    
    def __init__(self):
        self.w=None
        self.b=None
        self.costs=None
        self.epochs=None
        
        
    def fit(self,x_train,y_train,num_iterations,learning_Rate):
        
        X=x_train.values
        y=y_train.values
        
      
     
        N=len(X)
        self.w=np.zeros((X.shape[1]))
        self.b=0
        self.costs=[]
        self.epochs=[]
        for i in range(num_iterations):
     
         
             Z = np.dot(self.w.T,X.T)+self.b
             h = (1/(1+np.exp(-Z)))  
     
             cost=-(1/N)*np.sum(y*np.log(h)+(1-y)*np.log(1-h))  
             
             dw = 1/N*np.dot(X.T,(h-y).T)
             db = 1/N*np.sum(h-y)  
             
             self.w = self.w - learning_Rate*dw
             self.b = self.b - learning_Rate*db
     
             self.costs.append(cost)
             self.epochs.append(i)
       
            
        
        return 
    
    def predicted_class(self,x):
        if x<0.5:
            return 0
        return 1
    
    
    
    def predict(self,w,b,x_test):
        X=x_test.values
        
        
        Z = np.dot(w.T,X.T)+b
        h = 1/(1+np.exp(-Z)) 
        prediction = np.vectorize(LogisticRegression().predicted_class)
        
        y_pred = prediction(h)
        
        return y_pred
        