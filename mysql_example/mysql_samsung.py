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
# table 생성
#==============================================================================
import csv
import pandas as pd
import mysql.connector
import sqlite3
from sqlalchemy import create_engine
#database 입력 했음
samsung =  mysql.connector.connect(host='localhost',database='test',user='root',password='22tkdgns@@') 
curs = samsung.cursor()  #samsung에 cursor을 생성하고 그걸 curs라고 하자
sql = """CREATE TABLE SAMSUNG (
            DATE INT,
            z FLOAT,
            a FLOAT,
            b FLOAT,
            c FLOAT)"""

curs.execute(sql)
### 이거로 엔진 설치하고 하니깐 존나 테이블로 받아옴 개 행복
engine = create_engine('mysql+mysqlconnector://root:22tkdgns@@@localhost/test')

csv_data = pd.read_csv('mysql_samsung.csv')
csv_data.to_sql('samgsung1',engine, if_exists = 'replace')
