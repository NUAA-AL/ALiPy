"""
Implement several selected state-of-the-art methods.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import pickle
import sys
import warnings

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, \
    rbf_kernel
from sklearn.neighbors import kneighbors_graph

from ..utils import interface




