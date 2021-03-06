# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:18:48 2017

@author: SH-NoteBook
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:14:33 2017

@author: SH-NoteBook
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:14:03 2017

@author: SH-NoteBook
"""
#200103~201703
#2 삼성전자 모두 포함하게 
#forward 를 미리 안다고 가정해서 다음기 당기순이익을 받아서 사용
#당기순이익이나 자기자본, 현금배당이 NAN인것들 제외
#==============================================================================
# MSCI Value = > Trailing PBR 역수, PBR 12m forward 역수, 배당수익률 역수
# 양끝 5% 제거하지 않음~!, pbr, per, div 각각 독립적으로 z-score 구해서 산술평균
# 각 size 별 z_score 상위 30 % 골라서 투자
#==============================================================================
import pandas as pd
import numpy as np

#raw_data_kq.to_pickle('raw_data_kq')
#size_kq.to_pickle('size_kq')
#ni_kq.to_pickle('ni_kq')
#rtn_kq.to_pickle('rtn_kq')
#equity_kq.to_pickle('equity_kq')
#cash_div_kq.to_pickle('cash_div_kq')
#size_FIF_wisefn_kq.to_pickle('size_FIF_wisefn_kq')
#raw_data_=pd.read_excel('exercise_v02.xlsx',sheetname='Raw_data1',header=None)
#raw_data_kq=pd.read_excel('msci_rawdata_kq2.xlsm',sheetname='Raw_data_kq1',header=None)  # 편입시장
#size_kq = pd.read_excel('msci_rawdata_kq2.xlsm',sheetname='시가총액1',header=None)  #시가총액
#ni_kq = pd.read_excel('msci_rawdata_kq2.xlsm',sheetname='당기순이익1',header=None)  # 당기순이익
#rtn_kq = pd.read_excel('msci_rawdata_kq2.xlsm',sheetname='수익률1',header=None)  #수익률
#equity_kq = pd.read_excel('msci_rawdata_kq2.xlsm',sheetname='자본총계1',header=None)  #자본총걔
#size_FIF_wisefn_kq = pd.read_excel('msci_rawdata_kq2.xlsm',sheetname='유통주식수x수정주가1',header=None)  #free floating 시가총액
#cash_div_kq = pd.read_excel('msci_rawdata_kq2.xlsm',sheetname='현금배당액1',header=None)
#
kospi_quarter = pd.read_pickle('kospi_quarter')
raw_data = pd.read_pickle('raw_data')
size = pd.read_pickle('size')  #시가총액
ni = pd.read_pickle('ni') # 당기순이익
rtn = pd.read_pickle('rtn')
equity = pd.read_pickle('equity') #자본총계
cash_div = pd.read_pickle('cash_div')
size_FIF_wisefn=pd.read_pickle('size_FIF_wisefn') #시가총액

raw_data_kq = pd.read_pickle('raw_data_kq')
size_kq = pd.read_pickle('size_kq')  #시가총액
ni_kq = pd.read_pickle('ni_kq') # 당기순이익
rtn_kq = pd.read_pickle('rtn_kq')
equity_kq = pd.read_pickle('equity_kq') #자본총계
cash_div_kq = pd.read_pickle('cash_div_kq')
size_FIF_wisefn_kq=pd.read_pickle('size_FIF_wisefn_kq') #시가총액

#소형주 + KOSDAQ 하기 위해 새로운 rawdata 생성(primary key 때문에)                   
raw_data_sum=pd.concat([raw_data,raw_data_kq],axis=0,ignore_index=True)
rtn_sum=pd.concat([rtn,rtn_kq],axis=0,ignore_index=True)
size_sum=pd.concat([size,size_kq],axis=0,ignore_index=True)  
ni_sum=pd.concat([ni,ni_kq],axis=0,ignore_index=True)
equity_sum=pd.concat([equity,equity_kq],axis=0,ignore_index=True)
size_FIF_wisefn_sum=pd.concat([size_FIF_wisefn,size_FIF_wisefn_kq],axis=0,ignore_index=True)
cash_div_sum=pd.concat([cash_div,cash_div_kq],axis=0,ignore_index=True)
#size_FIF=pd.read_pickle('size_FIF')  #자기주식 제외 시가총액
#size_FIF_insider=pd.read_pickle('size_FIF_insider') #자기주식, 최대주주 주식 제외 시가총
#size_FIF_wisefn = pd.read_excel('msci_rawdata.xlsx',sheetname='유통주식수x수정주가1',header=None) # wisefn에서 산출해주는 유통비율 이용
#size_FIF_wisefn.to_pickle('size_FIF_wisefn')

turnover = pd.DataFrame(np.zeros((1,1)))
return_data = np.zeros((5,64))
return_data = pd.DataFrame(return_data)
data_name=pd.DataFrame(np.zeros((1000,64)))
# 매 분기 수익률을 기록하기 위해 quarter_data를 만듬
quarter_data = pd.DataFrame(np.zeros((1000,128)))
for n in range(3,67):
    #66마지막 분기
    data_big = raw_data[(raw_data[n] == 1)]
    data_big = data_big.loc[:,[1,n]]
    data = pd.concat([data_big, size_FIF_wisefn[n], equity[n], ni[n+1],cash_div[n],size[n]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size_FIF_wisefn','equity','ni_12fw','cash_div','size']
    result_temp = data
    samsung = pd.DataFrame(data.loc[390,:]).transpose()
    #만약 삼전의 재무재표중 없는게 있다면 14 column을 맞추기 위해 0을 넣어버림
    if (np.isnan(result_temp.loc[390]['equity']))|(np.isnan(result_temp.loc[390]['ni_12fw']))|(np.isnan(result_temp.loc[390]['cash_div'])):
        samsung = pd.DataFrame(result_temp.loc[390,:]).transpose()
        samsung['1/pbr'] = 0
        samsung['1/per'] = 0
        samsung['div_yield'] = 0
        samsung['pbr_z'] = 0
        samsung['per_z'] = 0
        samsung['div_z'] = 0
        samsung['z_score'] = 0
    
    data = data[data['equity'].notnull()]
    data = data[data['ni_12fw'].notnull()]
    data = data[data['cash_div'].notnull()]
    
    # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
    # 시총비중 구할떄는 free-float
    data['size_FIF_wisefn']=data['size_FIF_wisefn']/1000    #size 단위 thousand
    data['1/pbr']=data['equity']/data['size']
    data['1/per']=data['ni_12fw']/data['size']
    data['div_yield']=data['cash_div']/data['size']
    
    # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
    data = data.replace([np.inf, -np.inf],np.nan)  
    
    # Null 값 제거
    data_per = data[data['1/per'].notnull()]
    data_pbr = data[data['1/pbr'].notnull()]
    data_div = data[data['div_yield'].notnull()]

    #양 끝 5% 구함
#    market_capital=np.sum(data['size'])
#    pbr_min=np.percentile(data_pbr['1/pbr'],5)
#    pbr_max=np.percentile(data_pbr['1/pbr'],95)
#    per_min=np.percentile(data_per['1/per'],5)
#    per_max=np.percentile(data_per['1/per'],95)
#    div_min=np.percentile(data_div['div_yield'],5)
#    div_max=np.percentile(data_div['div_yield'],95)

    
    #양끝단  5% 제거
#    data_pbr = data_pbr[(data_pbr['1/pbr']>pbr_min)&(data_pbr['1/pbr']<pbr_max)]
#    data_per = data_per[(data_per['1/per']>per_min)&(data_per['1/per']<per_max)]
#    data_div = data_div[(data_div['div_yield']>div_min)&(data_div['div_yield']<div_max)]

    # 시가총액비중 구함 
    data_pbr_cap = np.sum(data_pbr['size_FIF_wisefn'])
    data_per_cap = np.sum(data_per['size_FIF_wisefn'])
    data_div_cap = np.sum(data_div['size_FIF_wisefn'])

    data_pbr = data_pbr.assign(market_weight=data_pbr['size_FIF_wisefn']/data_pbr_cap)
    data_per = data_per.assign(market_weight=data_per['size_FIF_wisefn']/data_per_cap)
    data_div = data_div.assign(market_weight=data_div['size_FIF_wisefn']/data_div_cap)
    
    # 시총가중 평균 
    mu_inv_pbr=np.sum(data_pbr['1/pbr']*data_pbr['market_weight'])
    mu_inv_per=np.sum(data_per['1/per']*data_per['market_weight'])
    mu_inv_div=np.sum(data_div['div_yield']*data_div['market_weight'])
    
    # 시총 가중 표준편자
    std_inv_pbr=np.sqrt(np.sum(np.square(data_pbr['1/pbr']-mu_inv_pbr)*data_pbr['market_weight']))
    std_inv_per=np.sqrt(np.sum(np.square(data_per['1/per']-mu_inv_per)*data_per['market_weight']))
    std_inv_div=np.sqrt(np.sum(np.square(data_div['div_yield']-mu_inv_div)*data_div['market_weight']))
    
    data1=(data_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
    data1.name= 'pbr_z'
    data2=(data_per['1/per']-mu_inv_per)/std_inv_per
    data2.name= 'per_z'
    data3=(data_div['div_yield']-mu_inv_div)/std_inv_div
    data3.name= 'div_z'
          
    result = pd.concat([data, data1, data2, data3], axis = 1)
    
    # np.nanmean : nan 값 포함해서 평균 내기!!
    result = result.assign(z_score=np.nanmean(result.iloc[:,[10,11,12]],axis=1))
#    result_temp = result

    
    # z_score > 0 인것이 가치주라고 msci에서 하고있음
    result =result[result['z_score'].notnull()]
    
    #상위 65%로 결정하면 삼성전자가 n=64,65,66일때 모두 포함이 된다.
    z_score1_max=np.percentile(result['z_score'],50)
    result =result[result['z_score']>z_score1_max]
    
    
#    result = pd.concat([result,pd.DataFrame(result_temp.loc[390,:]).transpose()],axis=0)
    #어쨋든 삼전이 탈락했다면
    if np.sum(result['name']=="삼성전자")!=1:
        samsung['1/pbr'] = 0
        samsung['1/per'] = 0
        samsung['div_yield'] = 0
        samsung['pbr_z'] = 0
        samsung['per_z'] = 0
        samsung['div_z'] = 0
        samsung['z_score'] = 0
        result = pd.concat([result,samsung],axis=0)
        
    #중복 rows 1개 빼고 다 제거 
    result = result.drop_duplicates()
    
    

    result1 = pd.concat([result,rtn[n-3]],axis=1,join='inner',ignore_index=True) #수익률 매칭
    
    data_big = raw_data[(raw_data[n] == 2)]
    data_big = data_big.loc[:,[1,n]]
    data = pd.concat([data_big, size_FIF_wisefn[n], equity[n], ni[n+1],cash_div[n],size[n]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size_FIF_wisefn','equity','ni_12fw','cash_div','size']
    
    data = data[data['equity'].notnull()]
    data = data[data['ni_12fw'].notnull()]
    data = data[data['cash_div'].notnull()]
    # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
    # 시총비중 구할떄는 free-float
    data['size_FIF_wisefn']=data['size_FIF_wisefn']/1000    #size 단위 thousand
    data['1/pbr']=data['equity']/data['size']
    data['1/per']=data['ni_12fw']/data['size']
    data['div_yield']=data['cash_div']/data['size']
    
    # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
    data = data.replace([np.inf, -np.inf],np.nan)  
    
    # Null 값 제거
    data_per = data[data['1/per'].notnull()]
    data_pbr = data[data['1/pbr'].notnull()]
    data_div = data[data['div_yield'].notnull()]

    #양 끝 5% 구함
#    market_capital=np.sum(data['size'])
#    pbr_min=np.percentile(data_pbr['1/pbr'],5)
#    pbr_max=np.percentile(data_pbr['1/pbr'],95)
#    per_min=np.percentile(data_per['1/per'],5)
#    per_max=np.percentile(data_per['1/per'],95)
#    div_min=np.percentile(data_div['div_yield'],5)
#    div_max=np.percentile(data_div['div_yield'],95)

    
    #양끝단  5% 제거
#    data_pbr = data_pbr[(data_pbr['1/pbr']>pbr_min)&(data_pbr['1/pbr']<pbr_max)]
#    data_per = data_per[(data_per['1/per']>per_min)&(data_per['1/per']<per_max)]
#    data_div = data_div[(data_div['div_yield']>div_min)&(data_div['div_yield']<div_max)]

    # 시가총액비중 구함 
    data_pbr_cap = np.sum(data_pbr['size_FIF_wisefn'])
    data_per_cap = np.sum(data_per['size_FIF_wisefn'])
    data_div_cap = np.sum(data_div['size_FIF_wisefn'])

    data_pbr = data_pbr.assign(market_weight=data_pbr['size_FIF_wisefn']/data_pbr_cap)
    data_per = data_per.assign(market_weight=data_per['size_FIF_wisefn']/data_per_cap)
    data_div = data_div.assign(market_weight=data_div['size_FIF_wisefn']/data_div_cap)
    
    # 시총가중 평균 
    mu_inv_pbr=np.sum(data_pbr['1/pbr']*data_pbr['market_weight'])
    mu_inv_per=np.sum(data_per['1/per']*data_per['market_weight'])
    mu_inv_div=np.sum(data_div['div_yield']*data_div['market_weight'])
    
    # 시총 가중 표준편자
    std_inv_pbr=np.sqrt(np.sum(np.square(data_pbr['1/pbr']-mu_inv_pbr)*data_pbr['market_weight']))
    std_inv_per=np.sqrt(np.sum(np.square(data_per['1/per']-mu_inv_per)*data_per['market_weight']))
    std_inv_div=np.sqrt(np.sum(np.square(data_div['div_yield']-mu_inv_div)*data_div['market_weight']))
    
    data1=(data_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
    data1.name= 'pbr_z'
    data2=(data_per['1/per']-mu_inv_per)/std_inv_per
    data2.name= 'per_z'
    data3=(data_div['div_yield']-mu_inv_div)/std_inv_div
    data3.name= 'div_z'
          
    result = pd.concat([data, data1, data2, data3], axis = 1)
    
    # np.nanmean : nan 값 포함해서 평균 내기!!
    result = result.assign(z_score=np.nanmean(result.iloc[:,[10,11,12]],axis=1))
    
    # z_score > 0 인것이 가치주라고 msci에서 하고있음
    result =result[result['z_score'].notnull()]
    z_score2_max=np.percentile(result['z_score'],50)
    result =result[result['z_score']>z_score2_max]
    

    result2 = pd.concat([result,rtn[n-3]],axis=1,join='inner',ignore_index=True) #수익률 매칭                      
              

    
    data_big = raw_data_sum[(raw_data_sum[n] == 3)]
    data_big = data_big.loc[:,[1,n]]
    data1 = pd.concat([data_big, size_FIF_wisefn[n], equity[n], ni[n+1],cash_div[n],size[n]],axis=1,join='inner',ignore_index=True)
    data1.columns = ['name','group','size_FIF_wisefn','equity','ni_12fw','cash_div','size']
    data1=data1[data1['size']>100000000000]
  #KOSDAQ
    data_big = raw_data_sum[(raw_data_sum[n] == 'KOSDAQ')]                   
    data_big = data_big.loc[:,[1,n]]
    data2 = pd.concat([data_big, size_FIF_wisefn_sum[n], equity_sum[n], ni_sum[n+1],cash_div_sum[n],size_sum[n]],axis=1,join='inner',ignore_index=True)
    data2.columns = ['name','group','size_FIF_wisefn','equity','ni_12fw','cash_div','size']
    
    data = pd.concat([data1,data2],axis=0)
    data=data[data['size']>100000000000]
        
    data = data[data['equity'].notnull()]
    data = data[data['ni_12fw'].notnull()]
    data = data[data['cash_div'].notnull()]
    # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
    # 시총비중 구할떄는 free-float
    data['size_FIF_wisefn']=data['size_FIF_wisefn']/1000    #size 단위 thousand
    data['1/pbr']=data['equity']/data['size']
    data['1/per']=data['ni_12fw']/data['size']
    data['div_yield']=data['cash_div']/data['size']
    
    # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
    data = data.replace([np.inf, -np.inf],np.nan)  
    
    # Null 값 제거
    data_per = data[data['1/per'].notnull()]
    data_pbr = data[data['1/pbr'].notnull()]
    data_div = data[data['div_yield'].notnull()]


    # 시가총액비중 구함 
    data_pbr_cap = np.sum(data_pbr['size_FIF_wisefn'])
    data_per_cap = np.sum(data_per['size_FIF_wisefn'])
    data_div_cap = np.sum(data_div['size_FIF_wisefn'])

    data_pbr = data_pbr.assign(market_weight=data_pbr['size_FIF_wisefn']/data_pbr_cap)
    data_per = data_per.assign(market_weight=data_per['size_FIF_wisefn']/data_per_cap)
    data_div = data_div.assign(market_weight=data_div['size_FIF_wisefn']/data_div_cap)
    
    # 시총가중 평균 
    mu_inv_pbr=np.sum(data_pbr['1/pbr']*data_pbr['market_weight'])
    mu_inv_per=np.sum(data_per['1/per']*data_per['market_weight'])
    mu_inv_div=np.sum(data_div['div_yield']*data_div['market_weight'])
    
    # 시총 가중 표준편자
    std_inv_pbr=np.sqrt(np.sum(np.square(data_pbr['1/pbr']-mu_inv_pbr)*data_pbr['market_weight']))
    std_inv_per=np.sqrt(np.sum(np.square(data_per['1/per']-mu_inv_per)*data_per['market_weight']))
    std_inv_div=np.sqrt(np.sum(np.square(data_div['div_yield']-mu_inv_div)*data_div['market_weight']))
    
    data1=(data_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
    data1.name= 'pbr_z'
    data2=(data_per['1/per']-mu_inv_per)/std_inv_per
    data2.name= 'per_z'
    data3=(data_div['div_yield']-mu_inv_div)/std_inv_div
    data3.name= 'div_z'
          
    result = pd.concat([data, data1, data2, data3], axis = 1)
    
    # np.nanmean : nan 값 포함해서 평균 내기!!
    result = result.assign(z_score=np.nanmean(result.iloc[:,[10,11,12]],axis=1))
    
    # z_score > 0 인것이 가치주라고 msci에서 하고있음
    result =result[result['z_score'].notnull()]
    z_score3_max=np.percentile(result['z_score'],80)
    result =result[result['z_score']>z_score3_max]
    

    result3 = pd.concat([result,rtn_sum[n-3]],axis=1,join='inner',ignore_index=True) #수익률 매칭  
                  
                           
                       
#대형주+중형주+소형주+KOSDAQ                       
    result = pd.concat([result1,result2,result3])
    #n=33 일때 국민은행 같은게 지주사로 전환되면서 수익률이 0으로 나와버림 NAN이랑은 다른개념 
    #사전에 공시가 되기 때문에 거를수 있을거라 판단해서 제외
    #일딴 여기서 제거해보고 , z 값 산출 전에 제거하는법 생각
    result = result[result[14]!=0]
    #매 분기 수익률을 기록
    quarter_data[[2*(n-3),2*(n-3)+1]] = result.iloc[:,[0,14]].reset_index(drop=True)
    market_capital=np.sum(result[2])
    result=result.assign(market_weight2=result[2]/market_capital)          
    
    #동일가중
    return_data.iloc[0,n-3]=np.mean(result[14])
#시총가중
#    return_data.iloc[0,n-3]=np.sum(result[14]*result['market_weight2'])
    data_name[n-3]=result[0].reset_index(drop=True)
#    return_data.iloc[0,n-3]=np.sum(result[13]*result[14])    
    if n == 66 : 
        pass
    return_final=np.product(return_data,axis=1)

    # 삼성전자가 몇분기동안 포함되었는지 확인
np.sum(np.sum(data_name=="삼성전자"))

average_return = np.mean(return_data,axis=1)
std_return = np.std(return_data,axis=1)
average_return/std_return
#turnover
for n in range(3,66):
    
    len1 = len(data_name[data_name[n-2].notnull()])
    aaa=data_name.loc[:,[n-3,n-2]]
    bbb=pd.DataFrame(aaa.stack().value_counts())
    len2=len(bbb[bbb[0]==2])
    data_name.loc[999,n-2]=(len1-len2)/len1
    qqqqq=data_name.iloc[999,1:]
    turnover=np.mean(qqqqq)

#승률
diff = return_data - np.tile(kospi_quarter,(5,1))
column_lengh = len(diff.columns)
diff = diff>0
#true == 1 , False == 0 으로 판단하기 때문에 다 더하면 가장 끝 column에 0보다 큰 것들 갯수가 남음
diff = diff.cumsum(axis=1)
win_rate = diff[column_lengh-1]/column_lengh



