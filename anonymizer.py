"""
run mondrian with given parameters
"""

# !/usr/bin/env python
# coding=utf-8
from mondrian import mondrian
import sys, copy, random, pdb
import numpy as np
from random import randrange
from scipy.spatial import distance as oklid
from sklearn.neighbors import LocalOutlierFactor
from collections import deque
import copy
from vptree3 import getir
import numpy as np
RELAX=False
def euclidean(p1, p2):
    x= np.linalg.norm(np.asanyarray(p1)-np.asanyarray(p2))
    return x

def hesapla(partition, index):
 
    gec=[]
    for row in partition:
        gec.append(row[index])


    width = float(max(gec)-min(gec))
    return width

def get_result_one(data,k):
  
    r = 0
    dp=0
    ncp=0
    (a,b)=getir(data,k)
    return (a,b)

