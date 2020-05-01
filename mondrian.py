"""
main module of mondrian
"""
from sklearn.neighbors import LocalOutlierFactor


# coding=utf-8
import matplotlib.pyplot as plt
import copy
from scipy.spatial import distance

from numpy import linalg as la
import pdb
import time
from utils.utility import cmp_str
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.spatial import distance

# warning all these variables should be re-inited, if
# you want to run mondrian with different parameters
QI_LEN = 5
GL_K = 0
RESULT = []
RES=[]
QI_RANGE = []
QI_DICT = []
QI_ORDER = []


class Partition(object):

    """
    Class for Group (or EC), which is used to keep records
    self.member: records in group
    self.low: lower point, use index to avoid negative values
    self.high: higher point, use index to avoid negative values
    self.allow: show if partition can be split on this QI
    """

    def __init__(self, data, low, high):
        """
        split_tuple = (index, low, high)
        """
        self.low = list(low)
        self.high = list(high)
        self.member = data[:]
        self.allow = [1] * QI_LEN

    def add_record(self, record, dim):
        """
        add one record to member
        """
        self.member.append(record)

    def add_multiple_record(self, records, dim):
        """
        add multiple records (list) to partition
        """
        for record in records:
            self.add_record(record, dim)

    def __len__(self):
        """
        return number of records
        """
        return len(self.member)


def get_normalized_width(partition, index):
    """
    return Normalized width of partition
    similar to NCP
    """
    d_order = QI_ORDER[index]
    width = float(d_order[partition.high[index]]) - float(d_order[partition.low[index]])
    return width * 1.0 / QI_RANGE[index]

def hesapla(partition, index):
    """
    return Normalized width of partition
    similar to NCP
    """
    gec=[]
    for row in partition.member:
        gec.append(row[index])

    d_order = QI_ORDER[index]
    width = float(max(gec)-min(gec))
    return width * 1.0 / QI_RANGE[index]

def choose_dimension(partition):
    """
    chooss dim with largest norm_width from all attributes.
    This function can be upgraded with other distance function.
    """
    max_width = -1
    max_dim = -1
    for dim in range(QI_LEN):
        if partition.allow[dim] == 0:
            continue
        norm_width = get_normalized_width(partition, dim)
        if norm_width > max_width:
            max_width = norm_width
            max_dim = dim
    if max_width > 1:
        pdb.set_trace()
    return max_dim


def frequency_set(partition, dim):
    """
    get the frequency_set of partition on dim
    """
    frequency = {}
    for record in partition.member:
        try:
            frequency[record[dim]] += 1
        except KeyError:
            frequency[record[dim]] = 1
    return frequency


def find_median(partition, dim):
    """
    find the middle of the partition, return splitVal
    """
    # use frequency set to get median
    frequency = frequency_set(partition, dim)
    splitVal = 0
    nextVal = 0
    value_list = list(frequency.keys())
    value_list.sort()
    total = sum(frequency.values())
    middle = total / 2

    if middle < GL_K or len(value_list) <= 1:
        try:
            return ('', '', value_list[0], value_list[-1])
        except IndexError:
            return ('', '', '', '')
    index = 0
    split_index = 0
    for i, qi_value in enumerate(value_list):
        index += frequency[qi_value]
        if index >= middle:
            splitVal = qi_value
            split_index = i
            break
    else:
        print("Error: cannot find splitVal")
    try:
        nextVal = value_list[split_index + 1]
    except IndexError:

        nextVal = splitVal
    return (splitVal, nextVal, value_list[0], value_list[-1])


def anonymize_strict(partition):
    """
    recursively partition groups until not allowable
    """

    allow_count = sum(partition.allow)


    if allow_count == 0:
        RESULT.append(partition)
        return
    for index in range(allow_count):
        # choose attrubite from domain
        dim = choose_dimension(partition)
        if dim == -1:
            print("Error: dim=-1")
            pdb.set_trace()
        (splitVal, nextVal, low, high) = find_median(partition, dim)
        # Update parent low and high
        if low is not '':
            partition.low[dim] = QI_DICT[dim][low]
            partition.high[dim] = QI_DICT[dim][high]
        if splitVal == '' or splitVal == nextVal:
            # cannot split
            partition.allow[dim] = 0
            continue
        # split the group from median
        mean = QI_DICT[dim][splitVal]
        lhs_high = partition.high[:]
        rhs_low = partition.low[:]
        lhs_high[dim] = mean
        rhs_low[dim] = QI_DICT[dim][nextVal]
        lhs = Partition([], partition.low, lhs_high)
        rhs = Partition([], rhs_low, partition.high)
        for record in partition.member:
            pos = QI_DICT[dim][record[dim]]
            if pos <= mean:
                # lhs = [low, mean]
                lhs.add_record(record, dim)
            else:
                # rhs = (mean, high]
                rhs.add_record(record, dim)
        # check is lhs and rhs satisfy k-anonymity
        if len(lhs) < GL_K or len(rhs) < GL_K:
            partition.allow[dim] = 0
            continue
        # anonymize sub-partition
        anonymize_strict(lhs)
        anonymize_strict(rhs)
        return

    RESULT.append(partition)


def anonymize_relaxed(partition):
    """
    recursively partition groups until not allowable
    """
    if sum(partition.allow) == 0:
        # can not split
        RESULT.append(partition)
        return
    allow_count = sum(partition.allow)
    for index in range(allow_count):
        # choose attrubite from domain
        dim = choose_dimension(partition)
        if dim == -1:
            print("Error: dim=-1")
            pdb.set_trace()
        # use frequency set to get median
        (splitVal, nextVal, low, high) = find_median(partition, dim)
        # Update parent low and high
        if low is not '':
            partition.low[dim] = QI_DICT[dim][low]
            partition.high[dim] = QI_DICT[dim][high]
        if splitVal == '':
            # cannot split
            partition.allow[dim] = 0
            anonymize_relaxed(partition)
            return
        # split the group from median
        mean = QI_DICT[dim][splitVal]
        lhs_high = partition.high[:]
        rhs_low = partition.low[:]
        lhs_high[dim] = mean
        rhs_low[dim] = QI_DICT[dim][nextVal]
        lhs = Partition([], partition.low, lhs_high)
        rhs = Partition([], rhs_low, partition.high)
        mid_set = []
        for record in partition.member:
            pos = QI_DICT[dim][record[dim]]
            if pos < mean:
                # lhs = [low, mean)
                lhs.add_record(record, dim)
            elif pos > mean:
                # rhs = (mean, high]
                rhs.add_record(record, dim)
            else:
                # mid_set keep the means
                mid_set.append(record)


        half_size = (len(partition) // 2)

        for i in range(half_size - len(lhs)):
            record = mid_set.pop()
            lhs.add_record(record, dim)
        if len(mid_set) > 0:
            rhs.low[dim] = mean
            rhs.add_multiple_record(mid_set, dim)
        # It's not necessary now.

        if len(lhs) < GL_K or len(rhs) < GL_K:
            partition.allow[dim] = 0
            continue
        # anonymize sub-partition
        anonymize_relaxed(lhs)
        anonymize_relaxed(rhs)
        return
    RESULT.append(partition)

def init(data, k, QI_num=5):
    """
    reset global variables
    """
    global GL_K, RESULT, RES, QI_LEN, QI_DICT, QI_RANGE, QI_ORDER, bi
    if QI_num <= 0:
        QI_LEN = len(data[0]) - 1
    else:
        QI_LEN = QI_num
    GL_K = k
    RESULT = []
    RES = []

    # static values
    QI_DICT = []
    QI_ORDER = []
    QI_RANGE = []
    att_values = []
    for i in range(QI_LEN):
        att_values.append(set())
        QI_DICT.append(dict())
    for record in data:
        for i in range(QI_LEN):
            att_values[i].add(record[i])
    for i in range(QI_LEN):
        value_list = list(att_values[i])
        value_list.sort()
        #print(value_list)
        QI_RANGE.append(float(value_list[-1]) - float(value_list[0]))
        QI_ORDER.append(list(value_list))
        for index, qi_value in enumerate(value_list):
            QI_DICT[i][qi_value] = index



def mondrian(data, mylist, k, p, relax=False, QI_num=5):

    init(data, k, QI_num)
    for i, h in enumerate(data):
        h.append(p[i])
    result = []
    data_size = len(data)
    low = [0] * QI_LEN
    high = [(len(t) - 1) for t in QI_ORDER]
    whole_partition = Partition(data, low, high)
    # begin mondrian
    start_time = time.time()
    anonymize_strict(whole_partition)
    rtime = float(time.time() - start_time)

    dp = 0
    ncp=0
    m=0
    ak=0
    f=[]
    g=[]
    r=[]
    e=[]
    c=0
    out = []
    kucuk = []
    abz = []
    for res in RESULT:
        af = []
        sirali=[]
        temp=[]
        ufak=[]

        for row in res.member:
            sirali.append(row[5])

        ufak=copy.deepcopy(res.member)
        for row in ufak:
            del row[5]


        for i in range(int(k*0.4),int(k*0.6)+1):
            clf = LocalOutlierFactor(n_neighbors=i, contamination="auto")
            y_pred = clf.fit_predict(ufak)
            X_scores = clf.negative_outlier_factor_
            temp.append(-X_scores)
        q = np.asanyarray(temp).T

        for i,row in enumerate(ufak):
            row.append(np.max(q[i]))


        sor=sorted(ufak, key=lambda x: x[5])

        for row in sor:
            del row[5]
        minimum=sor[0]

        fark=[]
        gg=[]


        for row in ufak:
            fark.append(distance.euclidean(np.array(minimum), np.array(row)))

        for i,row in enumerate(res.member):
            row.append(fark[i])


        lastList=sorted(res.member, key=lambda x: x[6])

        x=lastList[k:]

        for row in x:
            out.append(row[5])

        kucuk=lastList[:k]
        for row in kucuk:
            del row[6]
            del row[5]
        res.member=kucuk


        if len(x) !=0:
            for t in x:
                del t[6]
                del t[5]
                g.append(t)


        m=m+1
        rncp = 0.0


        for index in range(QI_LEN):
            rncp += hesapla(res, index)

        rncp *= len(res)
        ncp += rncp
        dp += len(res) ** 2

    return ((g,out), (ncp, m, dp))
