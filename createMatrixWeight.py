# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 20:25:56 2017

@author: KD5299
"""
import pandas as pd
from itertools import *
import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *

category = {'users' : {'sex' :['H','F'],
                     'age' :['<20','>30']},

            'movies' : {'genre':['Am','Gu','Co'],
                      'sortie':['>2000','<1980']}
            }
            
tuplesU = list(product(*category['users'].values()))
tuplesM = list(product(*category['movies'].values()))

index = pd.MultiIndex.from_tuples(tuplesU, 
                                    names=category['users'].keys())
columns = pd.MultiIndex.from_tuples(tuplesM, 
                                  names=category['movies'].keys())

scores = [[3,2,4,3,1,3],[2,1,3,2,2,5],[2,5,2,4,1,2],[1,4,1,2,5,3]]
# buid table of scores by groups
scoresGroup = pd.DataFrame(index=index,columns=columns,data = scores)
scoresGroup.ix[('>30','H')].ix[('Am','>2000')]
# define nb of users in each groups
nbU = np.random.randint(10,14,len(tuplesU))
nbM = np.random.randint(8,12,len(tuplesM))

# Build matrix A
# buid columns and index with hierachical level
def build_individus(name,nbInd):
    tuplesInd = list(product(*category[name].values()))
    u = []
    it=0
    for g,n in zip(tuplesInd,nbInd):
        u+=[g+ (i,) for i in range(it,it+n)]
        it +=n
    index =  pd.MultiIndex.from_tuples(u, 
                                        names=list(category[name].keys())+[name])
    return index

# Mtrice Users/Movies creation  
def build_matriceUM(nbU,nbM):
    A = pd.DataFrame(index=    build_individus('users',nbU),
                 columns = build_individus('movies',nbM))
    # fill the matrix according to scoresGroup
    for blockM,blockU in product(tuplesM,tuplesU):
        A.loc[blockU,blockM] = scoresGroup.ix[blockU].ix[blockM]
    # shuffle lign and columns
    A = A.sample(frac=1,axis=0)
    A = A.sample(frac=1,axis=1)
    return A
    
UM = build_matriceUM(nbU,nbM)
print(UM.head())
# To access to a block: example of functions pandas
UM.loc[('>30','H'),('Am','>2000')] = scoresGroup.ix[('>30','H')].ix[('Am','>2000')]
UM.index.get_level_values('sex')

UM.index.get_level_values('users')
A.columns.get_level_values('movies')
# Fill out matrix A
def mask(u, v, proportion = 0.3):
    mat_mask = np.random.binomial(1, proportion, size =  (u, v))
    print("We observe {} per cent of the entries of a {}*{} matrix".format(100 * mat_mask.mean(),u, v))
    return mat_mask
mat_mask = mask(*UM.shape,proportion=0.2)
UM_mask = mat_mask*UM
# CreUMte matrix of weigths
def build_weight(index,group):
    Weight = pd.DataFrame(index = index,columns=index,data=0)
    for row,col in zip(group,group):#product(tuplesU,tuplesU)
        Weight.loc[row,col]= 1 #nbOfCommonFeature
    return Weight

WeightU = build_weight(UM.index,tuplesU)
WeightM = build_weight(UM.columns,tuplesM)

#Build matrix L
def build_L(Weight):
    D =  np.diag(Weight.sum(axis=0))
    L = (D-Weight).values
    return L
LM = build_L(WeightM)
LU = build_L(WeightU)
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
np.trace(np.dot(np.dot(PU1.T,UM).T,np.dot(PU1.T,UM)))
np.trace(np.dot(UM.T,np.dot(LU,UM)))
# test on movies
np.trace(np.dot(UM,np.dot(LM,UM.T)))
np.trace(np.dot(np.dot(UM,PM1),np.dot(UM,PM1).T))

np.dot(LM,(UM.T).values)
np.dot((A.T).values,LU)

# find the solution
print(installed_solvers())
X = Variable(*UM.shape)

obj = Minimize(norm(X, 'nuc')+norm(X*PM1, 'fro')+norm((PU1.T)*X, 'fro'))#
constraints = [mul_elemwise(mat_mask, X) == mul_elemwise(mat_mask, np.array(UM))]
prob = Problem(obj, constraints)
prob.solve(solver=SCS)
A_rebuild = pd.DataFrame(index = UM.index,columns=UM.columns,data=np.round(X.value,1))
def rmse(A,B):
    rmse = ((A-B).values**2).mean()
    print("RMSE: %.2f" %rmse)
#    return rmse
rmse(UM,UM_rebuild)


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