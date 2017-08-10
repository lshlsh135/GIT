# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:25:57 2017

@author: SH-NoteBook
"""
import pandas as pd
import cx_Oracle


#이거 두개 반드시 선언!
cx0=cx_Oracle.makedsn("localhost",1521,"xe")
connection = cx_Oracle.connect("lshlsh135","2tkdgns2",cx0)


#DATA를 가져온다!!
kospi = pd.read_sql("""select * from kospi_ex""",con=connection)
kosdaq = pd.read_sql("""select * from kosdaq_ex""",con=connection)
rebalancing_date = pd.read_sql("""select * from rebalancing_date""",con=connection)

raw_data = pd.concat([kospi,kosdaq],axis=0,ignore_index=True)

raw_data['1/PER'] = raw_data['NI']/raw_data['MARKET_CAP'] 
aa= raw_data.iloc[0:1000,:]

bb = raw_data[raw_data['TRD_DATE']=='2007-03-30']
