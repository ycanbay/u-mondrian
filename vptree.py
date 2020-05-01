import numpy as np
from random import randrange
from scipy.spatial import distance as oklid

from scipy.spatial import distance as oklid
from sklearn.neighbors import LocalOutlierFactor
from collections import deque
import copy
global RESULT
RESULT = []

import sys

def hesapla(partition, index):

    gec=[]
    for row in partition:
        gec.append(row[index])


    width = float(max(gec)-min(gec))
    return width


left1 = []
right1 = []




class VPTree(object):

    def __init__(self, points, k):

        self.mu = None
        self.point=points[:]
        self.allow=10
    def add_record(self, record):

        self.point.append(record)

    def add_multiple_record(self, records):
 
        for record in records:
            self.add_record(record)

a=10
def build_tree(partition, k, a):
    if a == 0:
        RESULT.append(partition)
        return

    vp = partition.point[randrange(len(partition.point))]
    distances = [oklid.euclidean(vp, p) for p in partition.point]
    partition.mu = np.median(distances)
    if partition.mu == 0:
        return
    lhs = VPTree([], k)
    rhs = VPTree([], k)
    left_points = []
    right_points = []

    for point, distance in zip(partition.point, distances):
        if distance >= partition.mu:
            rhs.add_record(point)
        else:
            lhs.add_record(point)

  

    x=k-1
    y=k*2

    if len(lhs.point) < k  or len(rhs.point) < k:
       a = 0
    if len(rhs.point) >= x and len(rhs.point) < y:
       a = 0

    build_tree(lhs, k, a)
    build_tree(rhs, k, a)


def init(data, k):

    global GL_K, RESULT, RES, QI_LEN, QI_DICT, QI_RANGE, QI_ORDER, bi
    GL_K = k
    RESULT = []
    RES = []


def getir(data,k):
    init(data, k)

    self = VPTree(data, k)

    build_tree(self, k, 10)

    g = []
    dp = 0
    ncp = 0
    l = 0
    out = []
    m = 0
    r = 0
    for partition in RESULT:
        print(len(partition.point))
        af = []
        sirali = []
        temp = []
        ufak = []
        ufak = copy.deepcopy(partition.point)

        for i in range(int(k * 0.4), int(k * 0.6) + 1):
            clf = LocalOutlierFactor(n_neighbors=i)
            y_pred = clf.fit_predict(ufak)
            X_scores = clf.negative_outlier_factor_
            temp.append(-X_scores)
        q = np.asanyarray(temp).T

        for i, row in enumerate(ufak):
            row.append(np.max(q[i]))

        sor = sorted(ufak, key=lambda x: x[5])

        for row in sor:
            del row[5]
        minimum = sor[0]

        fark = []
        gg = []

        for row in ufak:
            fark.append(oklid.euclidean(np.array(minimum), np.array(row)))

        for i, row in enumerate(partition.point):
            row.append(fark[i])

        lastList = sorted(partition.point, key=lambda x: x[5])

        buyuk = lastList[k:]

        kucuk = lastList[:k]
        for row in kucuk:
            del row[5]

        if len(buyuk) != 0:
            for t in buyuk:
                del t[5]

        for row in buyuk:
            out.append(row)

        rncp = 0
        for index in range(5):
            rncp += hesapla(kucuk, index)

        rncp *= len(kucuk)
        ncp += rncp
        dp += len(kucuk) ** 2
        m = m + 1
    print(m)
    return (out, (dp, m, ncp))
