# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 08:05:00 2017
https://dev.mysql.com/doc/connector-python/en/connector-python-example-ddl.html
@author: SH-NoteBook
"""
#==============================================================================
# http://pythonstudy.xyz/python/article/202-MySQL-%EC%BF%BC%EB%A6%AC
#==============================================================================
import mysql.connector
conn = mysql.connector.connect(host='localhost',database='classicmodels',user='root',password='22tkdgns@@')
curs = conn.cursor(dictionary=True) # 딕셔너리 뭔가 좋아보임
curs2 = conn.cursor()

sql = "select * from customers"
curs.execute(sql)
curs2.execute(sql)

rows = curs.fetchall()
rows2 = curs2.fetchall()

print(rows)
print(rows2)
conn.close()
#==============================================================================
# 
#==============================================================================
import mysql.connector
from mysql.connector import Error
 
 
def connect():
    """ Connect to MySQL database """
    try:
        conn = mysql.connector.connect(host='localhost',
                                       database='python_mysql',
                                       user='root',
                                       password='22tkdgns@@')
        if conn.is_connected():
            print('Connected to MySQL database')
 
    except Error as e:
        print(e)
 
    finally:
        conn.close()
 
 
if __name__ == '__main__':
    connect()
    
#==============================================================================
#  https://planetsantosh.wordpress.com/2015/04/19/import-excel-to-mysql-using-python/
#==============================================================================
import xlrd
import mysql.connector

book = xlrd.open_workbook('msci_rawdata.xlsx')

conn = mysql.connector.connect(host='localhost',database='classicmodels',user='root',password='22tkdgns@@')