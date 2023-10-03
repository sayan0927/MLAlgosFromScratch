

import numpy as np

import warnings
warnings.filterwarnings("ignore")


class SVM:
    
    def __init__(self,learning_rate,iterations,lambd):
        self.learning_rate=learning_rate
        self.lambd=lambd
        self.iterations=iterations
        self.w=None
        self.bias=None
        
                    
                    
    def fit(self,x_train,y_train):
          samples,features=x_train.shape
         
          self.w = np.zeros(features)
          self.bias=0
          y_train[y_train==-1]=0
          
          costs=[]
          iteration=[]
         
        
          for i in range(self.iterations):
              
              epoch_cost=0
              
              for idx in range(len(x_train)):
                  x_i=x_train[idx]
                  y_i=y_train[idx]
                  
                  dw=0
                  db=0
                  
                  if y_train[idx]==1:
                      cost = max(0,1 - (np.dot(x_i,self.w) -  self.bias))
                      if cost<=0:
                        dw=0
                        db=0
                      else:
                        dw = -x_i
                        db=1
                      epoch_cost=epoch_cost+cost
                    
                    
                    
                
                  if y_train[idx]==0:
                      cost = max(0,1 + (np.dot(x_i,self.w) -  self.bias))
                      if cost<=0:
                          dw=0
                          db=0
                      else:
                          dw=x_i
                          db=-1
                      epoch_cost=epoch_cost+cost
                        
                        
                  self.w = self.w - self.learning_rate * dw
                  self.bias = self.bias - self.learning_rate * db
                  
                  
                  
                  
                  
              costs.append(epoch_cost/len(x_train))
              iteration.append(i)
              
          return       
                        

                                    
        
            
    
        
        
    def predict(self,x_test):
        pred=np.dot(x_test,self.w) - self.bias
        return pred








