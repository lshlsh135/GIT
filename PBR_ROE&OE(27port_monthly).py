# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 09:23:56 2017

@author: SH-NoteBook
"""

#==============================================================================
# Dependent double sorting 1) PBR 2) ROE & OPerating Income    3x3
#==============================================================================
#월별 수익률 기록(가정은 리밸런싱 했을때의 비중이 고정되는게 아니고 중간중간 변화되는거 관찰)
import pandas as pd
import numpy as np

rtn_m = pd.read_pickle('rtn_m')
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


return_data = np.zeros((5,189))
return_data = pd.DataFrame(return_data)

for i in range(0,3):
    for j in range(0,3):
        for z in range(0,3):
            locals()['return_data_{}{}{}'.format(i,j,z)] = pd.DataFrame(np.zeros((1,189)))
            locals()['data_name_{}{}{}'.format(i,j,z)] = pd.DataFrame(np.zeros((80,189)))
        

        
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
                locals()['data_{}{}{}'.format(i,j,z)] = pd.concat([locals()['data_{}{}{}'.format(i,j,z)],rtn_m.loc[:,[3*n+3,3*n+4,3*n+5]]],axis=1,join='inner',ignore_index=True) #수익률 매칭
                locals()['data_{}{}{}'.format(i,j,z)][15]=locals()['data_{}{}{}'.format(i,j,z)][12]*locals()['data_{}{}{}'.format(i,j,z)][13] #2개월 누적수익률
                locals()['data_{}{}{}'.format(i,j,z)][16]=locals()['data_{}{}{}'.format(i,j,z)][15]*locals()['data_{}{}{}'.format(i,j,z)][14] #3개월 누적수익률
#                locals()['data_{}{}{}'.format(i,j,z)][17]=locals()['data_{}{}{}'.format(i,j,z)][16]/locals()['data_{}{}{}'.format(i,j,z)][15] # 3개월 누적 / 2개월 누적 = 3월달 수익률
#                locals()['data_{}{}{}'.format(i,j,z)][18]=locals()['data_{}{}{}'.format(i,j,z)][15]/locals()['data_{}{}{}'.format(i,j,z)][12] # 2개월 누적수익률 / 1개월 수익률 = 2월달 수익률
                locals()['data_{}{}{}'.format(i,j,z)]=locals()['data_{}{}{}'.format(i,j,z)][locals()['data_{}{}{}'.format(i,j,z)][12].notnull()]  # 분기별 수익률은 있었는데 월별로 보니깐 중간에 없는 경우가 있음 미친 씨발
                locals()['data_{}{}{}'.format(i,j,z)]=locals()['data_{}{}{}'.format(i,j,z)][locals()['data_{}{}{}'.format(i,j,z)][13].notnull()]
                locals()['data_{}{}{}'.format(i,j,z)]=locals()['data_{}{}{}'.format(i,j,z)][locals()['data_{}{}{}'.format(i,j,z)][14].notnull()]
                locals()['data_{}{}{}'.format(i,j,z)]=locals()['data_{}{}{}'.format(i,j,z)][locals()['data_{}{}{}'.format(i,j,z)][15].notnull()]
                locals()['data_{}{}{}'.format(i,j,z)]=locals()['data_{}{}{}'.format(i,j,z)][locals()['data_{}{}{}'.format(i,j,z)][16].notnull()]
#                locals()['data_{}{}{}'.format(i,j,z)] = locals()['data_{}{}{}'.format(i,j,z)].replace(np.nan,0)  # 는 무슨 씨발 위에 0을 빼버렸더니 비중이 달라짐 (종목수가 n-1) 그래서 nan을 0 으로 해서 투자 종목으로 포함시켜야함 개 씨발 좆도 이거때문에 3시간 날렸다 ㅎㅎㅎㅎ
                #는 무슨 상장폐지 혹은 지주사로 변경하는건 기존에 공시나기 때문에 알수 있으므로 빼는게 맞다 개 씨발 ㅠㅠ
                locals()['data_name_{}{}{}'.format(i,j,z)][n-3] = locals()['data_{}{}{}'.format(i,j,z)][0].reset_index(drop=True)
                locals()['return_data_{}{}{}'.format(i,j,z)].iloc[:,[3*n-9,3*n+1-9,3*n+2-9]]=np.mean(locals()['data_{}{}{}'.format(i,j,z)].iloc[:,[12,15,16]]).values.reshape(1,3)
                
                a=locals()['return_data_{}{}{}'.format(i,j,z)].iloc[:,[3*n-9,3*n+1-9,3*n+2-9]]
                
                locals()['return_data_{}{}{}'.format(i,j,z)].iloc[:,3*n+2-9] = a[3*n+2-9]/a[3*n+1-9]
                locals()['return_data_{}{}{}'.format(i,j,z)].iloc[:,3*n+1-9] = a[3*n+1-9]/a[3*n-9]
      

      
    if n == 65 : 
        pass

#최종 누적수익률 구하기
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
                
#189개의 월별 수익률 하나의 matrix로 병합
return_data_0=pd.DataFrame(np.zeros((9,189)))
return_data_1=pd.DataFrame(np.zeros((9,189)))    
return_data_2=pd.DataFrame(np.zeros((9,189)))     
for i in range(0,3):
        for j in range(0,3):
            for z in range(0,3):
                if i==0:
                    return_data_0.loc[[3*(j+1)+z-3]] = locals()['return_data_0{}{}'.format(j,z)].loc[[0],:].values
                if i==1:
                    return_data_1.loc[[3*(j+1)+z-3]] = locals()['return_data_1{}{}'.format(j,z)].loc[[0],:].values
                if i==2:
                    return_data_2.loc[[3*(j+1)+z-3]] = locals()['return_data_2{}{}'.format(j,z)].loc[[0],:].values








