# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:35:31 2017

@author: SH-NoteBook
"""

import pandas as pd
import numpy as np


return_data = np.zeros((5,194))
return_data = pd.DataFrame(return_data)
#rtn_daily1 = pd.read_excel('A_trend_factor.xlsm',sheetname='일별주가1',header=None)
#monthly_date = pd.read_excel('A_trend_factor.xlsm',sheetname='월말날짜1',header=None)
#rtn_daily1.to_pickle('rtn_daily1')
monthly_date = pd.read_pickle('monthly_date')
rtn_daily = pd.read_pickle('rtn_daily')
rtn_daily1 = pd.read_pickle('rtn_daily1')



a=rtn_daily.iloc[0,:999]
monthly_date.iloc[0,0]



#이거다이거
rtn_daily.columns.get_loc(monthly_date.iloc[0,0])





monthly_date.loc[:,:10]
monthly_date.index[monthly_date.iloc[0,0]]
monthly_date.index.duplicated()

rtn_daily1.index.get_loc(monthly_date.iloc[0,0])
