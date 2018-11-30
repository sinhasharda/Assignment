# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:02:34 2018

@author: Sharda.sinha
"""
#import libraries
import os

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split