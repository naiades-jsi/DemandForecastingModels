#!/usr/bin/env python
# coding: utf-8



# pip install -U numpy




import numpy as np 
import pandas as pd 
import numpy.ma as ma
import sys
sys.path.append('/home/costa/predictive_ensemble_dynamics/utils')
import operator_calculations as op_calc
import stats
import delay_embedding as embed
import clustering_methods as cl




df = pd.read_csv('data_rambla.csv')




Values = df['Values'].values




import numpy.ma as ma




X = ma.masked_invalid(Values).reshape(-1,1)




X = X.reshape(-1,1)




import matplotlib.pyplot as plt




X.shape
plt.plot(X)




plt.figure(figsize=(10,10))
plt.plot(X[:1000])
plt.show()




#to get error estimates in the manuscript we split the trajectory into non-overlapping segments

n_seed_range=np.arange(25,210,25) #number of partitions to examine
range_Ks =  np.arange(1,100,dtype=int) #range of delays to study
h_K=np.zeros((len(range_Ks),len(n_seed_range)))
for k,K in enumerate(range_Ks):
    traj_matrix = embed.trajectory_matrix(X,K=K-1)
    for ks,n_seeds in enumerate(n_seed_range):
        labels=cl.kmeans_knn_partition(traj_matrix,n_seeds)
        h = op_calc.get_entropy(labels)
        h_K[k,ks]=h
        print('Computed for {} delays and {} seeds.'.format(K,n_seeds))




plt.plot(n_seed_range,h_K.T)
plt.xlabel('N',fontsize=15)
plt.ylabel('h (nats/s)',fontsize=15)
plt.show()




plt.plot(h_K[:,0],marker='o')
plt.suptitle('Entropy Rambla')
plt.savefig('InformationRambla.jpg')
plt.show()






