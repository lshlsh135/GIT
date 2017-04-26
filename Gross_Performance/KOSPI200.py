# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:36:51 2017

@author: SH-NoteBook
"""

import numpy as np
import pandas as pd
result = pd.DataFrame(np.zeros((1,15)))
for i in range(0,15):
    
    raw_data = pd.read_pickle('raw_data')
    operation = pd.read_pickle('operation')
    data = raw_data[raw_data[i+3]==1]
    data = pd.concat([data,operation[i+3]],axis=1,join='inner',ignore_index=True)
    data=data[data[18].notnull()]
    result.loc[0,i]=np.sum(data[18])

