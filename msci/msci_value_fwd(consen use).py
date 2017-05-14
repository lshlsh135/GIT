# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:12:08 2017

@author: SH-NoteBook
"""
#forward 당기순이익을 받아서 사용 -> 없는 데이터가 너무 많다.
#==============================================================================
# MSCI Value = > Trailing PBR 역수, PBR 12m forward 역수, 배당수익률
#==============================================================================
import pandas as pd
import numpy as np


raw_data = pd.read_pickle('raw_data')
#size = pd.read_pickle('size')  #시가총액
ni = pd.read_pickle('ni')
rtn = pd.read_pickle('rtn')
equity = pd.read_pickle('equity')
operate = pd.read_pickle('operate')  
kospi = pd.read_pickle('kospi')
beta_week = pd.read_pickle('beta_week')
ni_12fw = pd.read_pickle('ni_12fw')
cash_div = pd.read_pickle('cash_div')
size_FIF=pd.read_pickle('size_FIF')  #자기주식 제외 시가총액
size_FIF_insider=pd.read_pickle('size_FIF_insider') #자기주식, 최대주주 주식 제외 시가총


return_data = np.zeros((5,63))
return_data = pd.DataFrame(return_data)
for n in range(3,66):
     #65마지막 분기
    data_big = raw_data[(raw_data[n] == 3)]
    data_big = data_big.loc[:,[1,n]]
    data = pd.concat([data_big, size_FIF_insider[n], equity[n], ni_12fw[n],cash_div[n]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size','equity','ni_12fw','cash_div']
    data['size']=data['size']/1000    #size 단위 thousand
    data['1/pbr']=data['equity']/data['size']
    data['1/per']=data['ni_12fw']/data['size']
    data['1/div_yield']=data['size']/data['cash_div']
    data = data.replace([np.inf, -np.inf],np.nan)  
    data=data[data['1/pbr'].notnull()]    # per가 NAN인 Row 제외
#    data=data[data['1/pbr']>0]
    data=data[data['1/per'].notnull()]
    data=data[data['1/div_yield'].notnull()]    # per가 NAN인 Row 제외
#    data=data[data['1/div_yield']>0]
#####################################################################   
#    양 끝단 제거 안하고 해본
#    m_pbr=np.mean(data['1/pbr'])
#    std_pbr=np.std(data['1/pbr'])
#    data1=(data['1/pbr']-m_pbr)/std_pbr
#          
#    m_per=np.mean(data['1/per'])
#    std_per=np.std(data['1/per'])
#    data2=(data['1/per']-m_per)/std_per
#          
#    m_div=np.mean(data['1/div_yield'])
#    std_div=np.std(data['1/div_yield'])
#    data3=(data['1/div_yield']-m_div)/std_div
#          
#    data=data.assign(z_score=data1+data2+data3)
#    data[data['name']=='삼성전자']
######################################################################
    #양 끝 5% 제거
    market_capital=np.sum(data['size'])
    pbr_min=np.percentile(data['1/pbr'],5)
    pbr_max=np.percentile(data['1/pbr'],95)
    per_min=np.percentile(data['1/per'],5)
    per_max=np.percentile(data['1/per'],95)
    div_min=np.percentile(data['1/div_yield'],5)
    div_max=np.percentile(data['1/div_yield'],95)
    
    data_inv_pbr=data[(data['1/pbr']>pbr_min)&(data['1/pbr']<pbr_max)]
    
    data_q=data[(data['1/pbr']>pbr_min)&(data['1/pbr']<pbr_max)
    &(data['1/per']>per_min)&(data['1/per']<per_max)&(data['1/div_yield']<div_max)
    &(data['1/div_yield']>div_min)]
    #시가총액비중 구함 (양끝단  5% 제거)
    data_q=data_q.assign(market_weight=data_q['size']/market_capital)
    mu_inv_pbr=np.sum(data_q['1/pbr']*data_q['market_weight'])
    mu_inv_per=np.sum(data_q['1/per']*data_q['market_weight'])
    mu_inv_div=np.sum(data_q['1/div_yield']*data_q['market_weight'])
    
    std_inv_pbr=np.sqrt(np.square(data_q['1/pbr']-mu_inv_pbr)*data_q['market_weight'])
    std_inv_per=np.sqrt(np.square(data_q['1/per']-mu_inv_per)*data_q['market_weight'])
    std_inv_div=np.sqrt(np.square(data_q['1/div_yield']-mu_inv_div)*data_q['market_weight'])
    
    data1=(data_q['1/pbr']-mu_inv_pbr)/std_inv_pbr
    data2=(data_q['1/per']-mu_inv_per)/std_inv_per
    data3=(data_q['1/div_yield']-mu_inv_div)/std_inv_div
          
    data_q=data_q.assign(z_score=data1+data2+data3)
    
    data=data_q[data_q['z_score']>0]
    
    data_size= len(data)     # Row count    
    
    data = pd.concat([data,rtn[n-3]],axis=1,join='inner',ignore_index=True) #수익률 매칭
    return_data.iloc[0,n-3]=np.mean(data[11])
    if n == 65 : 
        pass
    return_final=np.product(return_data,axis=1)

#==============================================================================
# MSCI Value = > Trailing PBR 역수, PBR 12m forward 역수, 배당수익률
# 양끝단 2.5% 자르고, 대중소 z-score 상위 30%만.
#==============================================================================
import pandas as pd
import numpy as np


raw_data = pd.read_pickle('raw_data')
#size = pd.read_pickle('size')  #시가총액
ni = pd.read_pickle('ni')
rtn = pd.read_pickle('rtn')
equity = pd.read_pickle('equity')
operate = pd.read_pickle('operate')  
kospi = pd.read_pickle('kospi')
beta_week = pd.read_pickle('beta_week')
ni_12fw = pd.read_pickle('ni_12fw')
cash_div = pd.read_pickle('cash_div')
size_FIF=pd.read_pickle('size_FIF')  #자기주식 제외 시가총액
size_FIF_insider=pd.read_pickle('size_FIF_insider') #자기주식, 최대주주 주식 제외 시가총


return_data = np.zeros((5,63))
return_data = pd.DataFrame(return_data)
for n in range(3,66):
    #65마지막 분기
    data_big = raw_data[(raw_data[n] == 3)]
    data_big = data_big.loc[:,[1,n]]
    data = pd.concat([data_big, size_FIF_insider[n], equity[n], ni_12fw[n],cash_div[n]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size','equity','ni_12fw','cash_div']
    data['size']=data['size']/1000    #size 단위 thousand
    data['1/pbr']=data['equity']/data['size']
    data['1/per']=data['ni_12fw']/data['size']
    data['1/div_yield']=data['size']/data['cash_div']
    data = data.replace([np.inf, -np.inf],np.nan)  
    data=data[data['1/pbr'].notnull()]    # per가 NAN인 Row 제외
#    data=data[data['1/pbr']>0]
    data=data[data['1/per'].notnull()]
    data=data[data['1/div_yield'].notnull()]    # per가 NAN인 Row 제외
#    data=data[data['1/div_yield']>0]
#####################################################################   
#    양 끝단 제거 안하고 해본
#    m_pbr=np.mean(data['1/pbr'])
#    std_pbr=np.std(data['1/pbr'])
#    data1=(data['1/pbr']-m_pbr)/std_pbr
#          
#    m_per=np.mean(data['1/per'])
#    std_per=np.std(data['1/per'])
#    data2=(data['1/per']-m_per)/std_per
#          
#    m_div=np.mean(data['1/div_yield'])
#    std_div=np.std(data['1/div_yield'])
#    data3=(data['1/div_yield']-m_div)/std_div
#          
#    data=data.assign(z_score=data1+data2+data3)
#    data[data['name']=='삼성전자']
######################################################################
    #양 끝 2.5% 제거
    market_capital=np.sum(data['size'])
    pbr_min=np.percentile(data['1/pbr'],2.5)
    pbr_max=np.percentile(data['1/pbr'],97.5)
    per_min=np.percentile(data['1/per'],2.5)
    per_max=np.percentile(data['1/per'],97.5)
    div_min=np.percentile(data['1/div_yield'],2.5)
    div_max=np.percentile(data['1/div_yield'],97.5)
    
        
    data_q=data[(data['1/pbr']>pbr_min)&(data['1/pbr']<pbr_max)
    &(data['1/per']>per_min)&(data['1/per']<per_max)&(data['1/div_yield']<div_max)
    &(data['1/div_yield']>div_min)]
    #시가총액비중 구함 (양끝단  5% 제거)
    data_q=data_q.assign(market_weight=data_q['size']/market_capital)
    mu_inv_pbr=np.sum(data_q['1/pbr']*data_q['market_weight'])
    mu_inv_per=np.sum(data_q['1/per']*data_q['market_weight'])
    mu_inv_div=np.sum(data_q['1/div_yield']*data_q['market_weight'])
    
    std_inv_pbr=np.sqrt(np.square(data_q['1/pbr']-mu_inv_pbr)*data_q['market_weight'])
    std_inv_per=np.sqrt(np.square(data_q['1/per']-mu_inv_per)*data_q['market_weight'])
    std_inv_div=np.sqrt(np.square(data_q['1/div_yield']-mu_inv_div)*data_q['market_weight'])
   
    data1=(data_q['1/pbr']-mu_inv_pbr)/std_inv_pbr
    data2=(data_q['1/per']-mu_inv_per)/std_inv_per
    data3=(data_q['1/div_yield']-mu_inv_div)/std_inv_div
          
    data_q=data_q.assign(z_score=data1+data2+data3)
    
    data=data_q[data_q['z_score']>0]
    
    
    #z_score 상위 30%
    z_max=np.percentile(data['z_score'],70)
    data=data[(data['z_score']>=z_max)]
    market_capital=np.sum(data['size'])
    data=data.assign(market_weight2=data_q['size']/market_capital)
    
    
    
    
    data = pd.concat([data,rtn[n-3]],axis=1,join='inner',ignore_index=True)
    #수익률 매칭
    return_data.iloc[0,n-3]=np.sum(data[12]*data[11])
    if n == 65 : 
        pass
    return_final=np.product(return_data,axis=1)


#==============================================================================
# MSCI Value = > Trailing PBR 역수, PBR 12m forward 역수, 배당수익률
# 양끝단 제거x, 대중소 z-score 상위 30%만.
#==============================================================================
import pandas as pd
import numpy as np


raw_data = pd.read_pickle('raw_data')
#size = pd.read_pickle('size')  #시가총액
ni = pd.read_pickle('ni')
rtn = pd.read_pickle('rtn')
equity = pd.read_pickle('equity')
operate = pd.read_pickle('operate')  
kospi = pd.read_pickle('kospi')
beta_week = pd.read_pickle('beta_week')
ni_12fw = pd.read_pickle('ni_12fw')
cash_div = pd.read_pickle('cash_div')
size_FIF=pd.read_pickle('size_FIF')  #자기주식 제외 시가총액
size_FIF_insider=pd.read_pickle('size_FIF_insider') #자기주식, 최대주주 주식 제외 시가총


return_data = np.zeros((5,63))
return_data = pd.DataFrame(return_data)
for n in range(3,66):
    #65마지막 분기
    data_big = raw_data[(raw_data[n] == 1)|(raw_data[n] == 2)|(raw_data[n] == 3)]
    data_big = data_big.loc[:,[1,n]]
    data = pd.concat([data_big, size_FIF_insider[n], equity[n], ni_12fw[n],cash_div[n]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size','equity','ni_12fw','cash_div']
    data['size']=data['size']/1000    #size 단위 thousand
    data['1/pbr']=data['equity']/data['size']
    data['1/per']=data['ni_12fw']/data['size']
    data['1/div_yield']=data['size']/data['cash_div']
    data = data.replace([np.inf, -np.inf],np.nan)  
    data=data[data['1/pbr'].notnull()]    # per가 NAN인 Row 제외
#    data=data[data['1/pbr']>0]
    data=data[data['1/per'].notnull()]
    data=data[data['1/div_yield'].notnull()]    # per가 NAN인 Row 제외
#    data=data[data['1/div_yield']>0]
#####################################################################   
#    양 끝단 제거 안하고 해본
#    m_pbr=np.mean(data['1/pbr'])
#    std_pbr=np.std(data['1/pbr'])
#    data1=(data['1/pbr']-m_pbr)/std_pbr
#          
#    m_per=np.mean(data['1/per'])
#    std_per=np.std(data['1/per'])
#    data2=(data['1/per']-m_per)/std_per
#          
#    m_div=np.mean(data['1/div_yield'])
#    std_div=np.std(data['1/div_yield'])
#    data3=(data['1/div_yield']-m_div)/std_div
#          
#    data=data.assign(z_score=data1+data2+data3)
#    data[data['name']=='삼성전자']
######################################################################
    #양 끝 2.5% 제거
    market_capital=np.sum(data['size'])
#    pbr_min=np.percentile(data['1/pbr'],2.5)
#    pbr_max=np.percentile(data['1/pbr'],97.5)
#    per_min=np.percentile(data['1/per'],2.5)
#    per_max=np.percentile(data['1/per'],97.5)
#    div_min=np.percentile(data['1/div_yield'],2.5)
#    div_max=np.percentile(data['1/div_yield'],97.5)
    
        
#    data_q=data[(data['1/pbr']>pbr_min)&(data['1/pbr']<pbr_max)
#    &(data['1/per']>per_min)&(data['1/per']<per_max)&(data['1/div_yield']<div_max)
#    &(data['1/div_yield']>div_min)]
    #시가총액비중 구함 (양끝단  5% 제거)
    data_q=data
    data_q=data_q.assign(market_weight=data_q['size']/market_capital)
    mu_inv_pbr=np.sum(data_q['1/pbr']*data_q['market_weight'])
    mu_inv_per=np.sum(data_q['1/per']*data_q['market_weight'])
    mu_inv_div=np.sum(data_q['1/div_yield']*data_q['market_weight'])
    
    std_inv_pbr=np.sqrt(np.square(data_q['1/pbr']-mu_inv_pbr)*data_q['market_weight'])
    std_inv_per=np.sqrt(np.square(data_q['1/per']-mu_inv_per)*data_q['market_weight'])
    std_inv_div=np.sqrt(np.square(data_q['1/div_yield']-mu_inv_div)*data_q['market_weight'])
   
    data1=(data_q['1/pbr']-mu_inv_pbr)/std_inv_pbr
    data2=(data_q['1/per']-mu_inv_per)/std_inv_per
    data3=(data_q['1/div_yield']-mu_inv_div)/std_inv_div
          
    data_q=data_q.assign(z_score=data1+data2+data3)
    
    data=data_q[data_q['z_score'].notnull()]
    
    
    #z_score 상위 30%
    z_max=np.percentile(data['z_score'],70)
    data=data[(data['z_score']>=z_max)]
    market_capital=np.sum(data['size'])
    data=data.assign(market_weight2=data_q['size']/market_capital)
    
    
    
    
    data = pd.concat([data,rtn[n-3]],axis=1,join='inner',ignore_index=True)
    #수익률 매칭
    return_data.iloc[0,n-3]=np.sum(data[12]*data[11])
    if n == 65 : 
        pass
    return_final=np.product(return_data,axis=1)



















