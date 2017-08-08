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
from sqlalchemy import create_engine


engine = create_engine('mysql+mysqlconnector://root:22tkdgns@@@localhost/test')
sql = 'select * from kospi_ex' 
#table에 있는 data를 pandas dataframe type으로 바꾸는 함수 : pd.read_sql
#그런데 여기서도 sqlalchemy의 engine이 쓰임 ;;
kospi_ex = pd.read_sql(sql,engine)
ni=dg_csv_home.loc[:,['date','name','ni']]
