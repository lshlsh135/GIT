# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:34:48 2017

@author: SH-NoteBook
"""
import numpy as np
from sklearn import tree

X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
clf.predict([[2., 2.]])


