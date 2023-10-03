# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 20:46:52 2023

@author: sayan
"""
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import random

class KMeans:
    
    def __init__(self):
        self.iterations=None
        self.assigned_clusters=None
        self.cluster_centroids=None
        self.sse=None
        self.bcss=None
        self.iterations=None
        
        
        
    def fit(self,data,k,iterations):
        
        dimensions=data.shape[1]
        assigned_clusters=np.zeros((data.shape[0],1))
        cluster_centroids=np.zeros((k,dimensions))
        
        for i in range(k):
            idx=random.randrange(data.shape[0])
            cluster_centroids[i]=data[idx]
        
        
        
        for i in range(iterations):
            for row in range(data.shape[0]):
                cluster = self.assign_cluster(data[row][:], cluster_centroids, k,dimensions)
                assigned_clusters[row] = int(cluster)
                
            cluster_centroids = self.update_cluster_centroids(data, assigned_clusters, cluster_centroids, k, dimensions)
            
       
        error = self.get_error(data, assigned_clusters, cluster_centroids, k, dimensions)
        bcss=self.get_bcss(k, cluster_centroids, data, dimensions, assigned_clusters)
        
        self.sse=error
        self.bcss=bcss
        
        self.assigned_clusters=assigned_clusters
        self.cluster_centroids=cluster_centroids
        
        
        return self.recombine(data, assigned_clusters)
    
    
    
    def recombine(self,datapoints,assigned_cluster):
        newdata = np.column_stack((datapoints,assigned_cluster))
        return newdata
    
    def assign_cluster(self,datapoint,cluster_centroids,k,dimensions):
        
        
        minimum=1000000
        min_idx=-1
        
        
        for i in range(0,k):
            dist=self.get_distance(datapoint, cluster_centroids[i][:],dimensions)
            if dist < minimum:
                minimum=dist
                min_idx=i
                
        return min_idx
    
    def get_distance(self,datapoint,centroid,dimensions):
        
        dist=0
        
        for i in range(0,dimensions):
            dist = dist + (   (datapoint[i] - centroid[i])**2  )
        
        return math.sqrt(dist)
        


           
    def update_cluster_centroids(self,data,assigned_clusters,cluster_centroids,k,dimensions):
        
        for i in range(0,k):
            temp=np.zeros((1,dimensions))
            a=0
            for row in range(data.shape[0]):
                if assigned_clusters[row] == i:
                    temp=temp+data[row]
                    a=a+1
            cluster_centroids[i]=temp/a
        return cluster_centroids
    
    
    def get_error(self,data,assigned_clusters,cluster_centroids,k,dimensions):
        
        error=0
        
        for row in range(data.shape[0]):
            datapoint = data[row]
            cluster = assigned_clusters[row][0]
            centroid = cluster_centroids[int(cluster)]
            dist = self.get_distance(datapoint, centroid, dimensions)
            error = error + dist**2
        return error


    def get_bcss(self,k,cluster_centroids,data,dimensions,assigned_cluster):
        
        sample_mean=self.get_sample_mean(data, dimensions)
        cluster_sizes={}
        
        for i in range(k):
            cluster_sizes[i]=0
        
        for row in range(len(assigned_cluster)):
            cluster=assigned_cluster[row][0]
            cluster_sizes[cluster] = cluster_sizes[cluster]+1

        bcss=0
        for i in range(k):
            centroid=cluster_centroids[i]
            dist = self.get_distance(centroid, sample_mean, dimensions)
            bcss = bcss + cluster_sizes[i] * (dist**2)
        
       
        return bcss
    
    
    def get_sample_mean(self,data,dimensions):
        
        mean=np.sum(data,axis=0)
        mean=mean/data.shape[0]
        return mean