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
df_ora = pd.read_sql("""select * from kospi_ex""",con=connection)

df_ora=df_ora[df_ora['CO_NM']=='삼성전자']
df_ora['per'] = df_ora['NI']/df_ora['MARKET_CAP']
