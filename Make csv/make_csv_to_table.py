# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:38:15 2017

@author: SH-NoteBook
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 16:01:22 2017

@author: SH-NoteBook
"""

#==============================================================================
# http://stackoverflow.com/questions/10154633/load-csv-data-into-mysql-in-python
#==============================================================================
# database 생성
import mysql.connector
#database 입력 안하고 실행해도 접속이 가능하다
samsung =  mysql.connector.connect(host='localhost',user='root',password='22tkdgns@@') 
curs = samsung.cursor()  #samsung에 cursor을 생성하고 그걸 curs라고 하자
sql = 'CREATE DATABASE TEST' #TEST라는 database를 생성
curs.execute(sql)  #실행!!



#==============================================================================
# fns_jd_example table 생성
#==============================================================================
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine

### 이거로 엔진 설치하고 하니깐 존나 테이블로 받아옴 개 행복
engine = create_engine('mysql+mysqlconnector://root:22tkdgns@@@localhost/test')
# csv를 읽어올때 한글이 primary key라면 encoding = 'cp949'를 넣어줘야함 ㅎㅎㅎㅎ
csv_data = pd.read_csv('dg_csv_example.csv',encoding='CP949')   
csv_data.to_sql('fns_jd_example',engine, if_exists = 'replace')

#==============================================================================
# big_size table 생성
#==============================================================================
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine

### 이거로 엔진 설치하고 하니깐 존나 테이블로 받아옴 개 행복
engine = create_engine('mysql+mysqlconnector://root:22tkdgns@@@localhost/test')
# csv를 읽어올때 한글이 primary key라면 encoding = 'cp949'를 넣어줘야함 ㅎㅎㅎㅎ
csv_data = pd.read_csv('big_size.csv',encoding='CP949')   
csv_data.to_sql('big_size',engine, if_exists = 'replace')

#==============================================================================
# csv - > table 만든거를 다시 pandas dataFrame으로, 다시 Table로 가능
#==============================================================================
import mysql.connector
import pandas as pd
from sqlalchemy import create_engine


engine = create_engine('mysql+mysqlconnector://root:22tkdgns@@@localhost/test')
sql = 'select * from raw_data' 
#table에 있는 data를 pandas dataframe type으로 바꾸는 함수 : pd.read_sql
#그런데 여기서도 sqlalchemy의 engine이 쓰임 ;;
samsung1 = pd.read_sql(sql,engine)
a=samsung1.iloc[:,[1,2,3,4]]
#DataFrame 을 다시 table로 저장도 가능
a.to_sql('a',engine, if_exists = 'replace')

#==============================================================================
# 집받 생성
#==============================================================================
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine

### 이거로 엔진 설치하고 하니깐 존나 테이블로 받아옴 개 행복
engine = create_engine('mysql+mysqlconnector://root:22tkdgns@@@localhost/test')
# csv를 읽어올때 한글이 primary key라면 encoding = 'cp949'를 넣어줘야함 ㅎㅎㅎㅎ
csv_data = pd.read_csv('dg_csv_home.csv',encoding='CP949')   #파일명은 영어로 해야함.
csv_data.to_sql('dg_csv_home',engine, if_exists = 'replace')

#==============================================================================
# 집받 생성 pymysql로 해보기
#==============================================================================
import mysql.connector
import pandas as pd
from sqlalchemy import create_engine


engine = create_engine('mysql+mysqlconnector://root:22tkdgns@@@localhost/test')
sql = 'select * from dg_csv_home' 
#table에 있는 data를 pandas dataframe type으로 바꾸는 함수 : pd.read_sql
#그런데 여기서도 sqlalchemy의 engine이 쓰임 ;;
dg_csv_home = pd.read_sql(sql,engine)
a=samsung1.iloc[:,[1,2,3,4]]
#DataFrame 을 다시 table로 저장도 가능
a.to_sql('a',engine, if_exists = 'replace')
