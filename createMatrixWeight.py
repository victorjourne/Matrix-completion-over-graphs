# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 20:25:56 2017

@author: KD5299
"""
import pandas as pd
from itertools import *
import numpy as np
import matplotlib.pyplot as plt
import cvxpy
users = {'sex' :['H','F'],
         'age' :['<20','>30']}
movies = {'genre':['Am','Gu','Co'],
          'sortie':['>2000','<1980']}
tuplesU = list(product(*users.values()))
tuplesM = list(product(*movies.values()))

index = pd.MultiIndex.from_tuples(tuplesU, 
                                    names=users.keys())
columns = pd.MultiIndex.from_tuples(tuplesM, 
                                  names=movies.keys())

scores = [[3,2,4,2,1,3],[2,1,3,3,2,5],[2,5,2,4,1,2],[1,4,1,2,5,3]]
# buid table of scores by groups
scoresGroup = pd.DataFrame(index=index,columns=columns,data = scores)

scoresGroup.ix[('>30','H')].ix[('Am','>2000')]
# define nb of users in each groups
nbU = np.random.randint(10,14,len(tuplesU))
nbM = np.random.randint(8,12,len(tuplesM))

# Build matrix A
# buid columns and index with hierachical level
u = []
it=0
for g,n in zip(tuplesU,nbU):
    u+=[g+ (i,) for i in range(it,it+n)]
    it +=n
indexU =  pd.MultiIndex.from_tuples(u, 
                                    names=list(users.keys())+['users'])
m = []
it=0
for g,n in zip(tuplesM,nbM):
    m+=[g+(i,) for i in range(it,it+n)]
    it +=n
columnsM =  pd.MultiIndex.from_tuples(m, 
                                    names=list(movies.keys())+['movies'])    

# Mtrice A creation  
A = pd.DataFrame(index=indexU,columns = columnsM)
print(A)
# To access to a block: example of functions pandas
A.loc[('>30','H'),('Am','>2000')] = scoresGroup.ix[('>30','H')].ix[('Am','>2000')]
A.index.get_level_values('sex')
# fill the matrix according to scoresGroup
for blockM,blockU in product(tuplesM,tuplesU):
#    print blockM,blockU 
    A.loc[blockU,blockM] = scoresGroup.ix[blockU].ix[blockM]
# shuffle lign and columns
A = A.sample(frac=1,axis=0)
A = A.sample(frac=1,axis=1)
A.index.get_level_values('users')
A.columns.get_level_values('movies')
# Fill out matrix A
def mask(u, v, proportion = 0.3):
    mat_mask = np.random.binomial(1, proportion, size =  (u, v))
    print("We observe {} per cent of the entries of a {}*{} matrix".format(100 * mat_mask.mean(),u, v))
    return mat_mask
mat_mask = mask(*A.shape,proportion=0.1)
A_mask = mat_mask*A
# Create matrix of weigths

WeightU = pd.DataFrame(index = indexU,columns=indexU,data=0)
for row,col in zip(tuplesU,tuplesU):#product(tuplesU,tuplesU)
    nbOfCommonFeature = len(set(row) & set(col))
    WeightU.loc[row,col] = 1 #nbOfCommonFeature

WeightM = pd.DataFrame(index = columnsM,columns=columnsM,data=0)
for row,col in zip(tuplesM,tuplesM):
    nbOfCommonFeature = len(set(row) & set(col))
    WeightM.loc[row,col] = 1 #nbOfCommonFeature
# shuffle lign and columns according to A shuffling
WeightU = WeightU.loc[A.index,A.index]
WeightM = WeightM.loc[A.columns,A.columns]
#Build matrix L
DM = np.diag(WeightM.sum(axis=0))
LM = (DM-WeightM).values
DU = np.diag(WeightU.sum(axis=0))
LU = (DU-WeightU).values
# try
VU,PU = np.linalg.eigh(LU)
VU1 = np.diag(np.sqrt(np.abs(VU)))
PU1 = np.dot(PU,VU1)
#np.dot(PU1,PU1.T) ==LU
VM,PM = np.linalg.eigh(LM)
VM1 = np.diag(np.sqrt(np.abs(VM)))
PM1 = np.dot(PM,VM1)
#np.dot(PM1,PM1.T) ==LM
# Resolution
# test on users
np.trace(np.dot(np.dot(PU1.T,A).T,np.dot(PU1.T,A)))
np.trace(np.dot(A.T,np.dot(LU,A)))
# test on movies
np.trace(np.dot(A,np.dot(LM,A.T)))
np.trace(np.dot(np.dot(A,PM1),np.dot(A,PM1).T))

np.dot(LM,(A.T).values)
np.dot((A.T).values,LU)

from cvxpy import *
print(installed_solvers())
X = Variable(*A.shape)

obj = Minimize(norm(X, 'nuc')+norm(X*PM1, 'fro')+norm((PU1.T)*X, 'fro'))#
constraints = [mul_elemwise(mat_mask, X) == mul_elemwise(mat_mask, np.array(A))]
prob = Problem(obj, constraints)
prob.solve(solver=SCS)
X.value

A_rebuild = pd.DataFrame(index = A.index,columns=A.columns,data=np.round(X.value,1))
def rmse(A,B):
    rmse = ((A-B).values**2).mean()
    print("RMSE: %.2f" %rmse)
#    return rmse
rmse(A,A_rebuild)


# build Graph of users
import networkx as nx
G=nx.Graph()
# for all combinations between users, update weight graphs
for tupleEdge in combinations(WeightU.index.get_level_values('users'),2):
    weight = WeightU.xs(tupleEdge[0], level='users', axis=0).xs(tupleEdge[1], level='users', axis=1).values
    G.add_edge(*tupleEdge,weight=weight)
# 
elarge=[(a,b) for (a,b,d) in G.edges(data=True) if d['weight'] ==1]
#esmall=[(a,b) for (a,b,d) in G.edges(data=True) if d['weight'] ]

pos=nx.spring_layout(G) # positions for all nodes

# nodes
nx.draw_networkx_nodes(G,pos,node_size=700)

# edges
nx.draw_networkx_edges(G,pos,edgelist=elarge,
                    width=1)
#nx.draw_networkx_edges(G,pos,edgelist=esmall,
#                    width=1,alpha=0.5,edge_color='b',style='dashed')

# labels
nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')

plt.axis('off')
plt.savefig("weighted_graph.png") # save as png
plt.show() # display