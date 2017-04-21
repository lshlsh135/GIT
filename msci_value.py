# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:12:08 2017

@author: SH-NoteBook
"""

#==============================================================================
# MSCI Value = > Trailing PBR 역수, PBR 12m forward 역수, 배당수익률
#==============================================================================
import pandas as pd
import numpy as np


raw_data = pd.read_pickle('raw_data')
size = pd.read_pickle('size')
ni = pd.read_pickle('ni')
rtn = pd.read_pickle('rtn')
equity = pd.read_pickle('equity')
operate = pd.read_pickle('operate')  
kospi = pd.read_pickle('kospi')
beta_week = pd.read_pickle('beta_week')
ni_12fw = pd.read_pickle('ni_12fw')
cash_div = pd.read_pickle('cash_div')




for i in range(0,3):
    for j in range(0,3):
        locals()['return_data_{}{}'.format(i,j)] = pd.DataFrame(np.zeros((1,63)))
        locals()['data_name_{}{}'.format(i,j)] = pd.DataFrame(np.zeros((80,63)))
        
for n in range(3,66):
    n=65
    data_big = raw_data[(raw_data[n] == 1)]
    data_big = data_big.loc[:,[1,n]]
    data = pd.concat([data_big, size[n], equity[n], ni_12fw[n],cash_div[n]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size','equity','ni_12fw','cash_div']
    data['1/pbr']=data['equity']/data['size']
    data['1/per']=data['ni_12fw']/data['size']
    data['1/div_yield']=data['size']/data['cash_div']
    data = data.replace([np.inf, -np.inf],np.nan)  
    data=data[data['1/pbr'].notnull()]    # per가 NAN인 Row 제외
    data=data[data['1/pbr']>0]
    data=data[data['1/per'].notnull()]
    data=data[data['1/div_yield'].notnull()]    # per가 NAN인 Row 제외
    data=data[data['1/div_yield']>0]
    
    m_pbr=np.mean(data['1/pbr'])
    std_pbr=np.std(data['1/pbr'])
    data1=(data['1/pbr']-m_pbr)/std_pbr
          
    m_per=np.mean(data['1/per'])
    std_per=np.std(data['1/per'])
    data2=(data['1/per']-m_per)/std_per
          
    m_div=np.mean(data['1/div_yield'])
    std_div=np.std(data['1/div_yield'])
    data3=(data['1/div_yield']-m_div)/std_div
          
    data=data.assign(z_score=data1+data2+data3)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    data_size= len(data)     # Row count    
    data=data.assign(rnk_pbr=np.floor(data['pbr'].rank(method='first')/(data_size/3+1/3)))
    data=data.assign(rnk_roe=np.floor(data.groupby(['rnk_pbr'])['roe'].rank(method='first')/(data_size/9+1/3)))
    

    for i in range(0,3):
        for j in range(0,3):
                locals()['data_{}{}'.format(i,j)] = data[(data['rnk_pbr']==i)&(data['rnk_roe']==j)]  
                locals()['data_{}{}'.format(i,j)] = pd.concat([locals()['data_{}{}'.format(i,j)],rtn[n-3]],axis=1,join='inner',ignore_index=True) #수익률 매칭
                locals()['data_name_{}{}'.format(i,j)][n-3] = locals()['data_{}{}'.format(i,j)][0].reset_index(drop=True)
                locals()['return_data_{}{}'.format(i,j)].iloc[0,n-3]=np.mean(locals()['data_{}{}'.format(i,j)][9])

    if n == 65 : 
        pass

return_final=pd.DataFrame(np.zeros((3,3)))    

for i in range(0,3):
    for j in range(0,3):
        return_final.iloc[i,j] = np.product(locals()['return_data_{}{}'.format(i,j)],axis=1)   #np.product로 하면 누적, np.average
        
#종목수

stock_number = pd.DataFrame(np.zeros((9,63))) #주식수 세보자
for n in range(3,66):
   for i in range(0,3):
       for j in range(0,3):
           stock_number.loc[[3*(i+1)+j-3],[n-3]] = len(locals()['data_name_{}{}'.format(i,j)][locals()['data_name_{}{}'.format(i,j)][n-3].notnull()])
         
    
    # 종목 변화율
for n in range(3,65):
    for i in range(0,3):
        for j in range(0,3):
            len1 = len(locals()['data_name_{}{}'.format(i,j)][locals()['data_name_{}{}'.format(i,j)][n-2].notnull()])
            aaa=locals()['data_name_{}{}'.format(i,j)].loc[:,[n-3,n-2]]
            bbb=pd.DataFrame(aaa.stack().value_counts())
            len2=len(bbb[bbb[0]==2])
            locals()['data_name_{}{}'.format(i,j)].loc[79,n-2]=(len1-len2)/len1 #turnover
                  
                  
# 각 return_data들을 하나의 matrix로 모으기 -> 엑셀에 편하게 담기 위해서
return_data = np.zeros((9,63))
return_data = pd.DataFrame(return_data)
for i in range(0,3):
        for j in range(0,3):
            return_data.loc[[3*(i+1)+j-3]] = locals()['return_data_{}{}'.format(i,j)].loc[[0],:].values
# turnover도 모아보자     
turnvoer_data = np.zeros((9,63))
turnvoer_data = pd.DataFrame(turnvoer_data) 
for i in range(0,3):
        for j in range(0,3):
            turnvoer_data.loc[[3*(i+1)+j-3]] = locals()['data_name_{}{}'.format(i,j)].loc[[79],:].values

