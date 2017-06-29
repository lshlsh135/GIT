# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:35:31 2017

@author: SH-NoteBook
"""

import pandas as pd
import numpy as np


return_data = np.zeros((5,194))
return_data = pd.DataFrame(return_data)
rtn_daily = pd.read_excel('A_trend_factor.xlsm',sheetname='일별주가1')
rtn_daily.to_pickle('rtn_daily')
rtn_daily = pd.read_pickle('rtn_daily')
