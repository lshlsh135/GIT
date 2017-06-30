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

n=9
beta_3_temp = pd.DataFrame(np.zeros((1,12)))
for i in range(0,12):
    #df.columns.get_loc() 이걸 하면 column 위치를 알 수 있다!!
    rebalancing_date_column=rtn_daily.columns.get_loc(monthly_date.iloc[0,n-i])
    
    #3일 moving_average를 마지막 가격으로 나누워준 값
    # iloc을 이용해서 column 범위를 구하면 원하는 column +1을 해줘야하네
    ma_3=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-2:rebalancing_date_column+1],axis=1)/3/rtn_daily.iloc[:,rebalancing_date_column])
    ma_5=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-4:rebalancing_date_column+1],axis=1)/5/rtn_daily.iloc[:,rebalancing_date_column])
    ma_10=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-9:rebalancing_date_column+1],axis=1)/10/rtn_daily.iloc[:,rebalancing_date_column])
    ma_20=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-19:rebalancing_date_column+1],axis=1)/20/rtn_daily.iloc[:,rebalancing_date_column])
    ma_50=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-49:rebalancing_date_column+1],axis=1)/50/rtn_daily.iloc[:,rebalancing_date_column])
    ma_100=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-99:rebalancing_date_column+1],axis=1)/100/rtn_daily.iloc[:,rebalancing_date_column])
    ma_200=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-199:rebalancing_date_column+1],axis=1)/200/rtn_daily.iloc[:,rebalancing_date_column])
    ma_400=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-399:rebalancing_date_column+1],axis=1)/400/rtn_daily.iloc[:,rebalancing_date_column])
    ma_600=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-599:rebalancing_date_column+1],axis=1)/600/rtn_daily.iloc[:,rebalancing_date_column])
    ma_800=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-799:rebalancing_date_column+1],axis=1)/800/rtn_daily.iloc[:,rebalancing_date_column])
    ma_1000=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-999:rebalancing_date_column+1],axis=1)/1000/rtn_daily.iloc[:,rebalancing_date_column])
    
    
    ma_3=ma_3[ma_3[0].notnull()]
    future_rtn = pd.DataFrame(rtn_daily.loc[:,monthly_date.iloc[0,n+1-i]])
    future_rtn = future_rtn[future_rtn[monthly_date.iloc[0,n+1-i]].notnull()]
    ma_3_temp = pd.concat([ma_3,future_rtn],axis=1)
    ma_3_temp = ma_3_temp.assign(product=ma_3_temp.iloc[:,0]*ma_3_temp.iloc[:,1])
    ma_3_temp = ma_3_temp[ma_3_temp['product'].notnull()]
    
    beta_3 = pd.DataFrame(np.dot(np.linalg.inv(np.dot(ma_3.T,ma_3)),np.dot(ma_3_temp[0].T,ma_3_temp.iloc[:,1])))
    beta_3_temp.iloc[0,i]=beta_3.iloc[0,0]

beta_3=np.average(beta_3_temp)    
beta_3 * ma_3


