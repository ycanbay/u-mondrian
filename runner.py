import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
import anonymizer as mn
import copy
import gc

print(__doc__)
res=[]
comp1 = []

with open("adult-5.csv") as f:
    mylist = []
    for row in csv.reader(f, delimiter=","):
        del row[5]
        i=0
        for x in row:
            if x*1 != x:
                print("warning")
            x=int(x)
            row[i]=x
            i=i+1
        mylist.append(row)


y = MinMaxScaler(copy=True, feature_range=(0, 1)).fit(mylist)
y = y.transform(mylist)
X = y.tolist()
p=[]


sonuc=[]
zam=[]

ed = copy.deepcopy(X)
for h in [5,10,20,30,40,50,60,70]:
    ilkList = copy.deepcopy(X)
    for i, row in enumerate(X):
        p.append(i)
    for i in [1,2,3,4,5]:

        (result, ilkSonuc) = mn.get_result_one(ilkList, ed, h, p)
        ilkList=result[0]

        if ilkList == []:
            break

        p = result[1]
        sonuc.append((h,len(X)-len(p),len(p), ilkSonuc[0],ilkSonuc[1],ilkSonuc[2]))


with open("sonucum7_3.csv", "w") as s:
    for row in sonuc:
        s.write(str(row) + '\n')



