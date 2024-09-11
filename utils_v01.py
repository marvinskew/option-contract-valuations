import os
import sys
import argparse
import numpy as np
from math import pow
from functools import wraps
from scipy.stats import binom, norm


class QriskBeutralProbError(Exception):
    default_msg= "Error related to risk neutral probabilities"
    def __init__(self, msg= None):
        self.msg = msg
        super().__init__(msg, default_msg)

class BS_NormalCumDistError(Exception):
    default_msg= "Error related to Black-Scholes Normal Cumulative distribution of d1 & d2"
    def __init__(self, msg= None):
        self.msg = msg
        super().__init__(msg, default_msg)

def compute_execution_time(func):
    @wraps(func)
    def inner(*args, **kwargs):
        try:
            print(f"{func.__name__} starts execution ...")
            start_time =  time.time()
            fn = func(*args, **kwargs)
            end_time = time.time()
            print(f"{func.__name__} ends execution...")
        except Exception as ex:
            print("Error was thrown as :",ex)
    return inner
