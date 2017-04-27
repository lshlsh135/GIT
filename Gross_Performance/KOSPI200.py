# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:36:51 2017

@author: SH-NoteBook
"""
#==============================================================================
#KOSPI200 
#==============================================================================
import numpy as np
import pandas as pd
result_operation = pd.DataFrame(np.zeros((1,15)))
result_net_income = pd.DataFrame(np.zeros((1,15)))
raw_data = pd.read_pickle('raw_data')
operation = pd.read_pickle('operation')
#net_income = pd.read_excel('KOSPI200.xlsm',sheetname='당기순이익',header=None)
raw_data_size = pd.read_pickle('raw_data_size')
net_income = pd.read_pickle('net_income')

for i in range(0,15):
    data = raw_data[raw_data[i+3]==1]
    data = pd.concat([data,operation[i+3]],axis=1,join='inner',ignore_index=True)
    data=data[data[18].notnull()]
    result_operation.loc[0,i]=np.sum(data[18])
    
for i in range(0,15):
    data = raw_data[raw_data[i+3]==1]
    data = pd.concat([data,net_income[i+3]],axis=1,join='inner',ignore_index=True)
    data=data[data[18].notnull()]
    result_net_income.loc[0,i]=np.sum(data[18])
    
#==============================================================================
# 코스피 대중소
#==============================================================================
import numpy as np
import pandas as pd
result_operation = pd.DataFrame(np.zeros((1,15)))
result_net_income = pd.DataFrame(np.zeros((1,15)))
raw_data = pd.read_pickle('raw_data')
operation = pd.read_pickle('operation')
#net_income = pd.read_excel('KOSPI200.xlsm',sheetname='당기순이익',header=None)
raw_data_size = pd.read_pickle('raw_data_size')
net_income = pd.read_pickle('net_income')

for i in range(0,15):
    data = raw_data_size[(raw_data_size[i+3]==2)|(raw_data_size[i+3]==3)]
    data = pd.concat([data,operation[i+3]],axis=1,join='inner',ignore_index=True)
    data=data[data[18].notnull()]
    result_operation.loc[0,i]=np.sum(data[18])
    
for i in range(0,15):
    data = raw_data_size[(raw_data_size[i+3]==2)|(raw_data_size[i+3]==3)]
    data = pd.concat([data,net_income[i+3]],axis=1,join='inner',ignore_index=True)
    data=data[data[18].notnull()]
    result_net_income.loc[0,i]=np.sum(data[18])




#==============================================================================
# KOSPI200 적자기업 제외
#==============================================================================
import numpy as np
import pandas as pd
result_operation = pd.DataFrame(np.zeros((1,15)))
result_net_income = pd.DataFrame(np.zeros((1,15)))
raw_data = pd.read_pickle('raw_data')
operation = pd.read_pickle('operation')
#net_income = pd.read_excel('KOSPI200.xlsm',sheetname='당기순이익',header=None)
raw_data_size = pd.read_pickle('raw_data_size')
net_income = pd.read_pickle('net_income')

for i in range(0,15):
    data = raw_data[raw_data[i+3]==1]
    data = pd.concat([data,operation[i+3]],axis=1,join='inner',ignore_index=True)
    data=data[data[18].notnull()]
    data=data[data[18]>0]
    result_operation.loc[0,i]=np.sum(data[18])
    
for i in range(0,15):
    data = raw_data[raw_data[i+3]==1]
    data = pd.concat([data,net_income[i+3]],axis=1,join='inner',ignore_index=True)
    data=data[data[18].notnull()]
    data=data[data[18]>0]
    result_net_income.loc[0,i]=np.sum(data[18])
    
#==============================================================================
# 코스피 대중소 적자기업 제
#==============================================================================
import numpy as np
import pandas as pd
result_operation = pd.DataFrame(np.zeros((1,15)))
result_net_income = pd.DataFrame(np.zeros((1,15)))
raw_data = pd.read_pickle('raw_data')
operation = pd.read_pickle('operation')
#net_income = pd.read_excel('KOSPI200.xlsm',sheetname='당기순이익',header=None)
raw_data_size = pd.read_pickle('raw_data_size')
net_income = pd.read_pickle('net_income')

for i in range(0,15):
    data = raw_data_size[(raw_data_size[i+3]==2)|(raw_data_size[i+3]==3)]
    data = pd.concat([data,operation[i+3]],axis=1,join='inner',ignore_index=True)
    data=data[data[18].notnull()]
    data=data[data[18]>0]
    result_operation.loc[0,i]=np.sum(data[18])
    
for i in range(0,15):
    data = raw_data_size[(raw_data_size[i+3]==2)|(raw_data_size[i+3]==3)]
    data = pd.concat([data,net_income[i+3]],axis=1,join='inner',ignore_index=True)
    data=data[data[18].notnull()]
    data=data[data[18]>0]
    result_net_income.loc[0,i]=np.sum(data[18])


