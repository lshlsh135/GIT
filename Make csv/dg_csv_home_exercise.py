# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:26:48 2017

@author: SH-NoteBook
"""
# =============================================================================
# 일딴 대충 만들어놓은 dg_csv_home 가지고 연습해보자..
# =============================================================================
import mysql.connector
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


engine = create_engine('mysql+mysqlconnector://root:22tkdgns@@@localhost/exercise')
sql = 'select * from kospi_ex' 
sql_2 = 'select * from kosdaq_data'
#table에 있는 data를 pandas dataframe type으로 바꾸는 함수 : pd.read_sql
#그런데 여기서도 sqlalchemy의 engine이 쓰임 ;;
kospi = pd.read_sql(sql,engine)
kosdaq = pd.read_sql(sql_2,engine)

whole_data = pd.concat([kospi,kosdaq],axis=0,ignore_index=True)
whole_data = whole_data.replace(['NaN'],np.nan)    
whole_data = whole_data[whole_data['equity'].notnull()]
whole_data = whole_data[whole_data['market_cap'].notnull()]
whole_data['1/per'] =  int(whole_data['equity'])/whole_data['market_cap']
temp = whole_data.iloc[0:5,:]

temp['trd_date'] =='2001-02-28'
temp['co_nm'] =='조흥은행'
temp['gicode'] =='A000010'
trd_date 도 varchar로 해야할듯?
숫자들은 double로..
