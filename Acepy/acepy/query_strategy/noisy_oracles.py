"""
Pre-defined query strategy for noisy oracles.
There are 2 categories of methods.
1. Evaluate oracles.
    1.1 Always query from the best oracle
    1.2 Query from the most appropriate oracle
        according to the selected instance and label.
2. Evaluate labels. Labels are obtained from multiple noisy oracles.
And the algorithm tries to obtain the accurate label for each instance.
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division

import collections
import copy
import warnings
import cvxpy
import os