# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 13:11:20 2017

@author: SH-NoteBook
"""

import pandas as pd
import numpy as np
import cx_Oracle
from one_factor import one_factor  # file path가 같은곳에 one_factor.py가 있고 그안에 class one_factor를 불러옴
#이거 두개 반드시 선언!
cx0=cx_Oracle.makedsn("localhost",1521,"xe")
connection = cx_Oracle.connect("lshlsh135","2tkdgns2",cx0) #이게 실행이 안될때가 있는데
#그때는 services에 들어가서 oracle listner를 실행시켜줘야함


#DATA를 가져온다!!
kospi = pd.read_sql("""select * from kospi_ex""",con=connection)
kosdaq = pd.read_sql("""select * from kosdaq_ex""",con=connection)
rebalancing_date = pd.read_sql("""select * from rebalancing_date""",con=connection)

raw_data = pd.concat([kospi,kosdaq],axis=0,ignore_index=True).drop_duplicates()   #왜인지 모르겠는데 db에 중복된 정보가 들어가있네 ..? 
a=one_factor(raw_data,rebalancing_date)
aaa=a.per()    

