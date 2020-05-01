"""
run mondrian with given parameters
"""

# !/usr/bin/env python
# coding=utf-8
from mondrian import mondrian
import sys, copy, random, pdb

RELAX=False


def get_result_one(data, mylist, k, p):
   
    result, eval_result = mondrian(data, mylist, k, p)

    return (result, eval_result)

