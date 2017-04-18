# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:33:51 2017

@author: SH-NoteBook
"""

#==============================================================================
# Dependent double sorting 1) PBR 2) ROE & OPerating Income    3x3
#==============================================================================
# 27 portfolio

import pandas as pd
import numpy as np


raw_data = pd.read_pickle('raw_data')
size = pd.read_pickle('size')
ni = pd.read_pickle('ni')
rtn = pd.read_pickle('rtn')
rtn_week = pd.read_pickle('rtn_week')
equity = pd.read_pickle('equity')
kospi = pd.read_pickle('kospi')
operate = pd.read_pickle('operate')
operate_income = pd.read_pickle('operate_income')
beta_week = pd.read_pickle('beta_week')

return_data = np.zeros((5,63))
return_data = pd.DataFrame(return_data)

for i in range(0,3):
    for j in range(0,3):
        for z in range(0,3):
            locals()['return_data_{}{}{}'.format(i,j,z)] = pd.DataFrame(np.zeros((1,63)))
            locals()['beta_{}{}{}'.format(i,j,z)] = pd.DataFrame(np.zeros((1,822)))
            locals()['beta_sum_{}{}{}'.format(i,j,z)] = pd.DataFrame(np.zeros((1,822)))
            locals()['data_name_{}{}{}'.format(i,j,z)] = pd.DataFrame(np.zeros((80,63)))
            

        
for n in range(3,66):
    
    data_big = raw_data[(raw_data[n] == 1)|(raw_data[n] == 2)|(raw_data[n]==3)]
    data_big = data_big.loc[:,[1,n]]
    data = pd.concat([data_big, size[n], equity[n], operate_income[n],ni[n]],axis=1,join='inner',ignore_index=True)
    data1 = pd.concat([data_big, size[n], equity[n], ni[n]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size','equity','operate_income','ni']
    
    data['pbr']=data['size']/data['equity']
    data['oe']=data['operate_income']/data['equity']
    data['roe']=data['ni']/data['equity']
    
    
    data=data[data['oe']>0]
    data=data[data['oe'].notnull()]
    data=data[data['pbr'].notnull()]    # per가 NAN인 Row 제외
    data=data.query('pbr>0')
    
    data=data[data['roe']>0]
    data=data[data['roe'].notnull()] 
    
    
    data_size= len(data)     # Row count    
    data=data.assign(rnk_pbr=np.floor(data['pbr'].rank(method='first')/(data_size/3+1/3)))
    data=data.assign(rnk_oe=np.floor(data.groupby(['rnk_pbr'])['oe'].rank(method='first')/(data_size/9+1/3)))
    data=data.assign(rnk_roe=np.floor(data.groupby(['rnk_pbr'])['roe'].rank(method='first')/(data_size/9+1/3)))
    

    for i in range(0,3):
        for j in range(0,3):
            for z in range(0,3):
                    locals()['data_{}{}{}'.format(i,j,z)] = data[(data['rnk_pbr']==i)&(data['rnk_oe']==j)&(data['rnk_roe']==z)]  
                    locals()['data_{}{}{}'.format(i,j,z)] = pd.concat([locals()['data_{}{}{}'.format(i,j,z)],rtn[n-3]],axis=1,join='inner',ignore_index=True) #수익률 매칭
                    locals()['data_name_{}{}{}'.format(i,j,z)][n-3] = locals()['data_{}{}{}'.format(i,j,z)][0].reset_index(drop=True)
                    locals()['data_{}{}{}'.format(i,j,z)]=locals()['data_{}{}{}'.format(i,j,z)][locals()['data_{}{}{}'.format(i,j,z)][12].notnull()]
                    locals()['data_{}{}{}'.format(i,j,z)]=locals()['data_{}{}{}'.format(i,j,z)][locals()['data_{}{}{}'.format(i,j,z)][12]>0]
                    locals()['return_data_{}{}{}'.format(i,j,z)].iloc[0,n-3]=np.mean(locals()['data_{}{}{}'.format(i,j,z)][12])
                    locals()['beta_rtn_{}{}{}'.format(i,j,z)] = pd.concat([locals()['data_{}{}{}'.format(i,j,z)],rtn_week.iloc[:,13*(n-2)-13:13*(n-2)-1]],axis=1,join='inner',ignore_index=True)
                    locals()['beta_{}{}{}'.format(i,j,z)] = pd.concat([locals()['data_{}{}{}'.format(i,j,z)],beta_week.iloc[:,13*(n-2)-13:13*(n-2)]],axis=1,join='inner',ignore_index=True)
                    locals()['beta_sum_{}{}{}'.format(i,j,z)].iloc[:,13*(n-2)-13] = np.sum(locals()['beta_{}{}{}'.format(i,j,z)][13]/len(locals()['beta_{}{}{}'.format(i,j,z)]))
                    
                    for p in range(1,12):
                        locals()['beta_rtn_{}{}{}'.format(i,j,z)][13+p] = locals()['beta_rtn_{}{}{}'.format(i,j,z)][13+p]*locals()['beta_rtn_{}{}{}'.format(i,j,z)][12+p]
                    for k in range(1,13):
                        locals()['beta_sum_{}{}{}'.format(i,j,z)].iloc[:,13*(n-2)-13+k] = np.transpose(locals()['beta_{}{}{}'.format(i,j,z)][13+k]).dot(locals()['beta_rtn_{}{}{}'.format(i,j,z)][12+k]/np.sum(locals()['beta_rtn_{}{}{}'.format(i,j,z)][12+k]))
                    
                        
    if n == 65 : 
        pass

return_final_0=pd.DataFrame(np.zeros((3,3)))
return_final_1=pd.DataFrame(np.zeros((3,3)))    
return_final_2=pd.DataFrame(np.zeros((3,3)))        

for i in range(0,3):
    for j in range(0,3):
        for z in range(0,3):
                if i==0:
                    return_final_0.iloc[j,z] = np.product(locals()['return_data_0{}{}'.format(j,z)],axis=1)   #np.product로 하면 누적, np.average
                elif i==1:
                    return_final_1.iloc[j,z] = np.product(locals()['return_data_1{}{}'.format(j,z)],axis=1)
                elif i==2:
                    return_final_2.iloc[j,z] = np.product(locals()['return_data_2{}{}'.format(j,z)],axis=1)
# 종목 변화율
for n in range(3,65):
    for i in range(0,3):
        for j in range(0,3):
            for z in range(0,3):
                len1 = len(locals()['data_name_{}{}{}'.format(i,j,z)][locals()['data_name_{}{}{}'.format(i,j,z)][n-2].notnull()])
                aaa=locals()['data_name_{}{}{}'.format(i,j,z)].loc[:,[n-3,n-2]]
                bbb=pd.DataFrame(aaa.stack().value_counts())
                len2=len(bbb[bbb[0]==2])
                locals()['data_name_{}{}{}'.format(i,j,z)].loc[100,n-2]=(len1-len2)/len1    