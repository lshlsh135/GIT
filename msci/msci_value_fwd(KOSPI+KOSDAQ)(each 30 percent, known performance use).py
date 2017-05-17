# -*- coding: utf-8 -*-
"""
Created on Tue May 16 14:21:32 2017

@author: SH-NoteBook
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 15 09:33:49 2017

@author: SH-NoteBook
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:34:29 2017

@author: SH-NoteBook
"""

#먼저 코스피에서 각 size 별 z_score 상위 30%가 되는 값을 골라서 코스닥에도 적용

#1. 
#2. 삼성전자 모두 포함하게.



#forward 를 미리 안다고 가정해서 다음기 당기순이익을 받아서 사용
#==============================================================================
# MSCI Value = > Trailing PBR 역수, PBR 12m forward 역수, 배당수익률 역수
# 양끝 5% 제거, pbr, per, div 각각 독립적으로 z-score 구해서 산술평균
# 각 size 별 z_score 상위 30 % 골라서 투자
#==============================================================================
import pandas as pd
import numpy as np



#raw_data_kq = pd.read_excel('msci_rawdata_kq.xlsm',sheetname='Raw_data_kq1',header=None)
#raw_data_kq.to_pickle('raw_Data_kq')
#size_kq = pd.read_excel('msci_rawdata_kq.xlsm',sheetname='시가총액1',header=None)
#size_kq.to_pickle('size_kq')
#ni_kq = pd.read_excel('msci_rawdata_kq.xlsm',sheetname='당기순이익1',header=None)
#ni_kq.to_pickle('ni_kq')
#rtn_kq = pd.read_excel('msci_rawdata_kq.xlsm',sheetname='수익률1',header=None)
#rtn_kq.to_pickle('rtn_kq')
#equity_kq = pd.read_excel('msci_rawdata_kq.xlsm',sheetname='자본총계1',header=None)
#equity_kq.to_pickle('equity_kq')
#size_FIF_wisefn_kq = pd.read_excel('msci_rawdata_kq.xlsm',sheetname='유통주식수x수정주가1',header=None) # wisefn에서 산출해주는 유통비율 이용
#size_FIF_wisefn_kq.to_pickle('size_FIF_wisefn_kq')
#cash_div_kq = pd.read_excel('msci_rawdata_kq.xlsm',sheetname='현금배당액1',header=None) 
#cash_div_kq.to_pickle('cash_div_kq')


raw_data_kq=pd.read_pickle('raw_data_kq')  # 편입시장
size_kq = pd.read_pickle('size_kq')  #시가총액
ni_kq = pd.read_pickle('ni_kq')  # 당기순이익
rtn_kq = pd.read_pickle('rtn_kq')  #수익률
equity_kq = pd.read_pickle('equity_kq')  #자본총걔
size_FIF_wisefn_kq = pd.read_pickle('size_FIF_wisefn_kq')  #free floating 시가총액
cash_div_kq = pd.read_pickle('cash_div_kq')


raw_data = pd.read_pickle('raw_data')
size = pd.read_pickle('size')  #시가총액
ni = pd.read_pickle('ni') # 당기순이익
rtn = pd.read_pickle('rtn')
equity = pd.read_pickle('equity') #자본총계
operate = pd.read_pickle('operate')  
kospi = pd.read_pickle('kospi')
beta_week = pd.read_pickle('beta_week')
ni_12fw = pd.read_pickle('ni_12fw')
cash_div = pd.read_pickle('cash_div')
#size_FIF=pd.read_pickle('size_FIF')  #자기주식 제외 시가총액
size_FIF_insider=pd.read_pickle('size_FIF_insider') #자기주식, 최대주주 주식 제외 시가총
#size_FIF_wisefn = pd.read_excel('msci_rawdata.xlsx',sheetname='유통주식수x수정주가1',header=None) # wisefn에서 산출해주는 유통비율 이용
#size_FIF_wisefn.to_pickle('size_FIF_wisefn')
size_FIF_wisefn=pd.read_pickle('size_FIF_wisefn') #시가총액

return_data = np.zeros((5,63))
return_data = pd.DataFrame(return_data)
data_name=pd.DataFrame(np.zeros((500,63)))
for n in range(3,66):   #N=65마지막 분기 (2016년 3분기)
#대형주
        
    data_big = raw_data[(raw_data[n] == 1)]
    data_big = data_big.loc[:,[1,n]]
    data = pd.concat([data_big, size_FIF_wisefn[n], equity[n], ni[n+1],cash_div[n],size[n]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size_FIF_wisefn','equity','ni_12fw','cash_div','size']
    # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
    # 시총비중 구할떄는 free-float
    data['size_FIF_wisefn']=data['size_FIF_wisefn']/1000    #size 단위 thousand
    data['1/pbr']=data['equity']/data['size']
    data['1/per']=data['ni_12fw']/data['size']
    data['1/div_yield']=data['size']/data['cash_div']
    
    # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
    data = data.replace([np.inf, -np.inf],np.nan)  
    
    # Null 값 제거
    data_per = data[data['1/per'].notnull()]
    data_pbr = data[data['1/pbr'].notnull()]
    data_div = data[data['1/div_yield'].notnull()]

    #양 끝 5% 구함
#    market_capital=np.sum(data['size'])
    pbr_min=np.percentile(data_pbr['1/pbr'],5)
    pbr_max=np.percentile(data_pbr['1/pbr'],95)
    per_min=np.percentile(data_per['1/per'],5)
    per_max=np.percentile(data_per['1/per'],95)
    div_min=np.percentile(data_div['1/div_yield'],5)
    div_max=np.percentile(data_div['1/div_yield'],95)

    
    #양끝단  5% 제거
    data_pbr = data_pbr[(data_pbr['1/pbr']>pbr_min)&(data_pbr['1/pbr']<pbr_max)]
    data_per = data_per[(data_per['1/per']>per_min)&(data_per['1/per']<per_max)]
    data_div = data_div[(data_div['1/div_yield']>div_min)&(data_div['1/div_yield']<div_max)]

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
    mu_inv_div=np.sum(data_div['1/div_yield']*data_div['market_weight'])
    
    # 시총 가중 표준편자
    std_inv_pbr=np.sqrt(np.sum(np.square(data_pbr['1/pbr']-mu_inv_pbr)*data_pbr['market_weight']))
    std_inv_per=np.sqrt(np.sum(np.square(data_per['1/per']-mu_inv_per)*data_per['market_weight']))
    std_inv_div=np.sqrt(np.sum(np.square(data_div['1/div_yield']-mu_inv_div)*data_div['market_weight']))
    
    data1=(data_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
    data1.name= 'pbr_z'
    data2=(data_per['1/per']-mu_inv_per)/std_inv_per
    data2.name= 'per_z'
    data3=(data_div['1/div_yield']-mu_inv_div)/std_inv_div
    data3.name= 'div_z'
          
    result = pd.concat([data, data1, data2, data3], axis = 1)
    
    # np.nanmean : nan 값 포함해서 평균 내기!!
    result = result.assign(z_score=np.nanmean(result.iloc[:,[10,11,12]],axis=1))
    
    # z_score > 0 인것이 가치주라고 msci에서 하고있음
    result =result[result['z_score'].notnull()]
    z_score_max=np.percentile(result['z_score'],70)
    result =result[result['z_score']>z_score_max]
    

    result1 = pd.concat([result,rtn[n-3]],axis=1,join='inner',ignore_index=True) #수익률 매칭
#중형주
    data_big = raw_data[(raw_data[n] == 2)]
    data_big = data_big.loc[:,[1,n]]
    data = pd.concat([data_big, size_FIF_wisefn[n], equity[n], ni[n+1],cash_div[n],size[n]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size_FIF_wisefn','equity','ni_12fw','cash_div','size']
    # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
    # 시총비중 구할떄는 free-float
    data['size_FIF_wisefn']=data['size_FIF_wisefn']/1000    #size 단위 thousand
    data['1/pbr']=data['equity']/data['size']
    data['1/per']=data['ni_12fw']/data['size']
    data['1/div_yield']=data['size']/data['cash_div']
    
    # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
    data = data.replace([np.inf, -np.inf],np.nan)  
    
    # Null 값 제거
    data_per = data[data['1/per'].notnull()]
    data_pbr = data[data['1/pbr'].notnull()]
    data_div = data[data['1/div_yield'].notnull()]

    #양 끝 5% 구함
    #    market_capital=np.sum(data['size'])
    pbr_min=np.percentile(data_pbr['1/pbr'],5)
    pbr_max=np.percentile(data_pbr['1/pbr'],95)
    per_min=np.percentile(data_per['1/per'],5)
    per_max=np.percentile(data_per['1/per'],95)
    div_min=np.percentile(data_div['1/div_yield'],5)
    div_max=np.percentile(data_div['1/div_yield'],95)

    
    #양끝단  5% 제거
    data_pbr = data_pbr[(data_pbr['1/pbr']>pbr_min)&(data_pbr['1/pbr']<pbr_max)]
    data_per = data_per[(data_per['1/per']>per_min)&(data_per['1/per']<per_max)]
    data_div = data_div[(data_div['1/div_yield']>div_min)&(data_div['1/div_yield']<div_max)]

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
    mu_inv_div=np.sum(data_div['1/div_yield']*data_div['market_weight'])
    
    # 시총 가중 표준편자
    std_inv_pbr=np.sqrt(np.sum(np.square(data_pbr['1/pbr']-mu_inv_pbr)*data_pbr['market_weight']))
    std_inv_per=np.sqrt(np.sum(np.square(data_per['1/per']-mu_inv_per)*data_per['market_weight']))
    std_inv_div=np.sqrt(np.sum(np.square(data_div['1/div_yield']-mu_inv_div)*data_div['market_weight']))
    
    data1=(data_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
    data1.name= 'pbr_z'
    data2=(data_per['1/per']-mu_inv_per)/std_inv_per
    data2.name= 'per_z'
    data3=(data_div['1/div_yield']-mu_inv_div)/std_inv_div
    data3.name= 'div_z'
          
    result = pd.concat([data, data1, data2, data3], axis = 1)
    
    # np.nanmean : nan 값 포함해서 평균 내기!!
    result = result.assign(z_score=np.nanmean(result.iloc[:,[10,11,12]],axis=1))
    
    # z_score > 0 인것이 가치주라고 msci에서 하고있음
    result =result[result['z_score'].notnull()]
    z_score_max=np.percentile(result['z_score'],70)
    result =result[result['z_score']>z_score_max]
    

    result2 = pd.concat([result,rtn[n-3]],axis=1,join='inner',ignore_index=True) #수익률 매칭                      
                       
#소형주    
    data_big = raw_data[(raw_data[n] == 3)]
    data_big = data_big.loc[:,[1,n]]
    data = pd.concat([data_big, size_FIF_wisefn[n], equity[n], ni[n+1],cash_div[n],size[n]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size_FIF_wisefn','equity','ni_12fw','cash_div','size']
    # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
    # 시총비중 구할떄는 free-float
    data['size_FIF_wisefn']=data['size_FIF_wisefn']/1000    #size 단위 thousand
    data['1/pbr']=data['equity']/data['size']
    data['1/per']=data['ni_12fw']/data['size']
    data['1/div_yield']=data['size']/data['cash_div']
    
    # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
    data = data.replace([np.inf, -np.inf],np.nan)  
    
    # Null 값 제거
    data_per = data[data['1/per'].notnull()]
    data_pbr = data[data['1/pbr'].notnull()]
    data_div = data[data['1/div_yield'].notnull()]

    #양 끝 5% 구함
#    market_capital=np.sum(data['size'])
    pbr_min=np.percentile(data_pbr['1/pbr'],5)
    pbr_max=np.percentile(data_pbr['1/pbr'],95)
    per_min=np.percentile(data_per['1/per'],5)
    per_max=np.percentile(data_per['1/per'],95)
    div_min=np.percentile(data_div['1/div_yield'],5)
    div_max=np.percentile(data_div['1/div_yield'],95)

    
    #양끝단  5% 제거
    data_pbr = data_pbr[(data_pbr['1/pbr']>pbr_min)&(data_pbr['1/pbr']<pbr_max)]
    data_per = data_per[(data_per['1/per']>per_min)&(data_per['1/per']<per_max)]
    data_div = data_div[(data_div['1/div_yield']>div_min)&(data_div['1/div_yield']<div_max)]

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
    mu_inv_div=np.sum(data_div['1/div_yield']*data_div['market_weight'])
    
    # 시총 가중 표준편자
    std_inv_pbr=np.sqrt(np.sum(np.square(data_pbr['1/pbr']-mu_inv_pbr)*data_pbr['market_weight']))
    std_inv_per=np.sqrt(np.sum(np.square(data_per['1/per']-mu_inv_per)*data_per['market_weight']))
    std_inv_div=np.sqrt(np.sum(np.square(data_div['1/div_yield']-mu_inv_div)*data_div['market_weight']))
    
    data1=(data_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
    data1.name= 'pbr_z'
    data2=(data_per['1/per']-mu_inv_per)/std_inv_per
    data2.name= 'per_z'
    data3=(data_div['1/div_yield']-mu_inv_div)/std_inv_div
    data3.name= 'div_z'
          
    result = pd.concat([data, data1, data2, data3], axis = 1)
    
    # np.nanmean : nan 값 포함해서 평균 내기!!
    result = result.assign(z_score=np.nanmean(result.iloc[:,[10,11,12]],axis=1))
    
    # z_score > 0 인것이 가치주라고 msci에서 하고있음
    result =result[result['z_score'].notnull()]
    z_score_max=np.percentile(result['z_score'],70)
    result =result[result['z_score']>z_score_max]
    
    
    result3 = pd.concat([result,rtn[n-3]],axis=1,join='inner',ignore_index=True) #수익률 매칭  
 #KOSDAQ
    data_big = raw_data_kq[(raw_data_kq[n] == 'KOSDAQ')]                   
    data_big = data_big.loc[:,[1,n]]
    data = pd.concat([data_big, size_FIF_wisefn_kq[n], equity_kq[n], ni_kq[n+1],cash_div_kq[n],size_kq[n]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size_FIF_wisefn','equity','ni_12fw','cash_div','size']
    # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
    # 시총비중 구할떄는 free-float
    data['size_FIF_wisefn']=data['size_FIF_wisefn']/1000    #size 단위 thousand
    data['1/pbr']=data['equity']/data['size']
    data['1/per']=data['ni_12fw']/data['size']
    data['1/div_yield']=data['size']/data['cash_div']
    
    # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
    data = data.replace([np.inf, -np.inf],np.nan)  
    
    # Null 값 제거
    data_per = data[data['1/per'].notnull()]
    data_pbr = data[data['1/pbr'].notnull()]
    data_div = data[data['1/div_yield'].notnull()]

    #양 끝 5% 구함
#    market_capital=np.sum(data['size'])
    pbr_min=np.percentile(data_pbr['1/pbr'],5)
    pbr_max=np.percentile(data_pbr['1/pbr'],95)
    per_min=np.percentile(data_per['1/per'],5)
    per_max=np.percentile(data_per['1/per'],95)
    div_min=np.percentile(data_div['1/div_yield'],5)
    div_max=np.percentile(data_div['1/div_yield'],95)

    
    #양끝단  5% 제거
    data_pbr = data_pbr[(data_pbr['1/pbr']>pbr_min)&(data_pbr['1/pbr']<pbr_max)]
    data_per = data_per[(data_per['1/per']>per_min)&(data_per['1/per']<per_max)]
    data_div = data_div[(data_div['1/div_yield']>div_min)&(data_div['1/div_yield']<div_max)]

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
    mu_inv_div=np.sum(data_div['1/div_yield']*data_div['market_weight'])
    
    # 시총 가중 표준편자
    std_inv_pbr=np.sqrt(np.sum(np.square(data_pbr['1/pbr']-mu_inv_pbr)*data_pbr['market_weight']))
    std_inv_per=np.sqrt(np.sum(np.square(data_per['1/per']-mu_inv_per)*data_per['market_weight']))
    std_inv_div=np.sqrt(np.sum(np.square(data_div['1/div_yield']-mu_inv_div)*data_div['market_weight']))
    
    data1=(data_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
    data1.name= 'pbr_z'
    data2=(data_per['1/per']-mu_inv_per)/std_inv_per
    data2.name= 'per_z'
    data3=(data_div['1/div_yield']-mu_inv_div)/std_inv_div
    data3.name= 'div_z'
          
    result = pd.concat([data, data1, data2, data3], axis = 1)
    
    # np.nanmean : nan 값 포함해서 평균 내기!!
    result = result.assign(z_score=np.nanmean(result.iloc[:,[10,11,12]],axis=1))
    
    # z_score > 0 인것이 가치주라고 msci에서 하고있음
    result =result[result['z_score'].notnull()]
    z_score_max=np.percentile(result['z_score'],70)
    result =result[result['z_score']>z_score_max]
    

    result4 = pd.concat([result,rtn_kq[n-3]],axis=1,join='inner',ignore_index=True) #수익률 매칭  


     
                       
                       
#대형주+중형주+소형주+KOSDAQ                       
    result = pd.concat([result1,result2,result3,result4])    
    market_capital=np.sum(result[2])
    result=result.assign(market_weight2=result[2]/market_capital)          
    
    #동일가중
    return_data.iloc[0,n-3]=np.mean(result[14])
#시총가중
#    return_data.iloc[0,n-3]=np.sum(result[14]*result['market_weight2'])
    data_name[n-3]=result[0].reset_index(drop=True)
#    return_data.iloc[0,n-3]=np.sum(result[13]*result[14])    
    if n == 65 : 
        pass
    return_final=np.product(return_data,axis=1)

    # 삼성전자가 몇분기동안 포함되었는지 확인
np.sum(np.sum(data_name=="삼성전자"))



#2 삼성전자 모두 포함하게 
#forward 를 미리 안다고 가정해서 다음기 당기순이익을 받아서 사용
#==============================================================================
# MSCI Value = > Trailing PBR 역수, PBR 12m forward 역수, 배당수익률 역수
# 양끝 5% 제거, pbr, per, div 각각 독립적으로 z-score 구해서 산술평균
# 각 size 별 z_score 상위 30 % 골라서 투자
#==============================================================================
import pandas as pd
import numpy as np

raw_data_kq=pd.read_pickle('raw_data_kq')  # 편입시장
size_kq = pd.read_pickle('size_kq')  #시가총액
ni_kq = pd.read_pickle('ni_kq')  # 당기순이익
rtn_kq = pd.read_pickle('rtn_kq')  #수익률
equity_kq = pd.read_pickle('equity_kq')  #자본총걔
size_FIF_wisefn_kq = pd.read_pickle('size_FIF_wisefn_kq')  #free floating 시가총액
cash_div_kq = pd.read_pickle('cash_div_kq')





raw_data = pd.read_pickle('raw_data')
size = pd.read_pickle('size')  #시가총액
ni = pd.read_pickle('ni') # 당기순이익
rtn = pd.read_pickle('rtn')
equity = pd.read_pickle('equity') #자본총계
operate = pd.read_pickle('operate')  
kospi = pd.read_pickle('kospi')
beta_week = pd.read_pickle('beta_week')
ni_12fw = pd.read_pickle('ni_12fw')
cash_div = pd.read_pickle('cash_div')
#size_FIF=pd.read_pickle('size_FIF')  #자기주식 제외 시가총액
size_FIF_insider=pd.read_pickle('size_FIF_insider') #자기주식, 최대주주 주식 제외 시가총
#size_FIF_wisefn = pd.read_excel('msci_rawdata.xlsx',sheetname='유통주식수x수정주가1',header=None) # wisefn에서 산출해주는 유통비율 이용
#size_FIF_wisefn.to_pickle('size_FIF_wisefn')
size_FIF_wisefn=pd.read_pickle('size_FIF_wisefn') #시가총액

return_data = np.zeros((5,63))
return_data = pd.DataFrame(return_data)
data_name=pd.DataFrame(np.zeros((500,63)))
for n in range(3,66):
    #65마지막 분기
    data_big = raw_data[(raw_data[n] == 1)]
    data_big = data_big.loc[:,[1,n]]
    data = pd.concat([data_big, size_FIF_wisefn[n], equity[n], ni[n+1],cash_div[n],size[n]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size_FIF_wisefn','equity','ni_12fw','cash_div','size']
    # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
    # 시총비중 구할떄는 free-float
    data['size_FIF_wisefn']=data['size_FIF_wisefn']/1000    #size 단위 thousand
    data['1/pbr']=data['equity']/data['size']
    data['1/per']=data['ni_12fw']/data['size']
    data['1/div_yield']=data['size']/data['cash_div']
    
    # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
    data = data.replace([np.inf, -np.inf],np.nan)  
    
    # Null 값 제거
    data_per = data[data['1/per'].notnull()]
    data_pbr = data[data['1/pbr'].notnull()]
    data_div = data[data['1/div_yield'].notnull()]

    #양 끝 5% 구함
#    market_capital=np.sum(data['size'])
    pbr_min=np.percentile(data_pbr['1/pbr'],5)
    pbr_max=np.percentile(data_pbr['1/pbr'],95)
    per_min=np.percentile(data_per['1/per'],5)
    per_max=np.percentile(data_per['1/per'],95)
    div_min=np.percentile(data_div['1/div_yield'],5)
    div_max=np.percentile(data_div['1/div_yield'],95)

    
    #양끝단  5% 제거
    data_pbr = data_pbr[(data_pbr['1/pbr']>pbr_min)&(data_pbr['1/pbr']<pbr_max)]
    data_per = data_per[(data_per['1/per']>per_min)&(data_per['1/per']<per_max)]
    data_div = data_div[(data_div['1/div_yield']>div_min)&(data_div['1/div_yield']<div_max)]

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
    mu_inv_div=np.sum(data_div['1/div_yield']*data_div['market_weight'])
    
    # 시총 가중 표준편자
    std_inv_pbr=np.sqrt(np.sum(np.square(data_pbr['1/pbr']-mu_inv_pbr)*data_pbr['market_weight']))
    std_inv_per=np.sqrt(np.sum(np.square(data_per['1/per']-mu_inv_per)*data_per['market_weight']))
    std_inv_div=np.sqrt(np.sum(np.square(data_div['1/div_yield']-mu_inv_div)*data_div['market_weight']))
    
    data1=(data_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
    data1.name= 'pbr_z'
    data2=(data_per['1/per']-mu_inv_per)/std_inv_per
    data2.name= 'per_z'
    data3=(data_div['1/div_yield']-mu_inv_div)/std_inv_div
    data3.name= 'div_z'
          
    result = pd.concat([data, data1, data2, data3], axis = 1)
    
    # np.nanmean : nan 값 포함해서 평균 내기!!
    result = result.assign(z_score=np.nanmean(result.iloc[:,[10,11,12]],axis=1))
    result_temp = result

    
    # z_score > 0 인것이 가치주라고 msci에서 하고있음
    result =result[result['z_score'].notnull()]
    z_score1_max=np.percentile(result['z_score'],70)
    result =result[result['z_score']>z_score1_max]
    
    result = pd.concat([result,pd.DataFrame(result_temp.loc[390,:]).transpose()],axis=0)
    
    #중복 rows 1개 빼고 다 제거 
    result = result.drop_duplicates()
    
    

    result1 = pd.concat([result,rtn[n-3]],axis=1,join='inner',ignore_index=True) #수익률 매칭
    
    data_big = raw_data[(raw_data[n] == 2)]
    data_big = data_big.loc[:,[1,n]]
    data = pd.concat([data_big, size_FIF_wisefn[n], equity[n], ni[n+1],cash_div[n],size[n]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size_FIF_wisefn','equity','ni_12fw','cash_div','size']
    # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
    # 시총비중 구할떄는 free-float
    data['size_FIF_wisefn']=data['size_FIF_wisefn']/1000    #size 단위 thousand
    data['1/pbr']=data['equity']/data['size']
    data['1/per']=data['ni_12fw']/data['size']
    data['1/div_yield']=data['size']/data['cash_div']
    
    # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
    data = data.replace([np.inf, -np.inf],np.nan)  
    
    # Null 값 제거
    data_per = data[data['1/per'].notnull()]
    data_pbr = data[data['1/pbr'].notnull()]
    data_div = data[data['1/div_yield'].notnull()]

    #양 끝 5% 구함
#    market_capital=np.sum(data['size'])
    pbr_min=np.percentile(data_pbr['1/pbr'],5)
    pbr_max=np.percentile(data_pbr['1/pbr'],95)
    per_min=np.percentile(data_per['1/per'],5)
    per_max=np.percentile(data_per['1/per'],95)
    div_min=np.percentile(data_div['1/div_yield'],5)
    div_max=np.percentile(data_div['1/div_yield'],95)

    
    #양끝단  5% 제거
    data_pbr = data_pbr[(data_pbr['1/pbr']>pbr_min)&(data_pbr['1/pbr']<pbr_max)]
    data_per = data_per[(data_per['1/per']>per_min)&(data_per['1/per']<per_max)]
    data_div = data_div[(data_div['1/div_yield']>div_min)&(data_div['1/div_yield']<div_max)]

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
    mu_inv_div=np.sum(data_div['1/div_yield']*data_div['market_weight'])
    
    # 시총 가중 표준편자
    std_inv_pbr=np.sqrt(np.sum(np.square(data_pbr['1/pbr']-mu_inv_pbr)*data_pbr['market_weight']))
    std_inv_per=np.sqrt(np.sum(np.square(data_per['1/per']-mu_inv_per)*data_per['market_weight']))
    std_inv_div=np.sqrt(np.sum(np.square(data_div['1/div_yield']-mu_inv_div)*data_div['market_weight']))
    
    data1=(data_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
    data1.name= 'pbr_z'
    data2=(data_per['1/per']-mu_inv_per)/std_inv_per
    data2.name= 'per_z'
    data3=(data_div['1/div_yield']-mu_inv_div)/std_inv_div
    data3.name= 'div_z'
          
    result = pd.concat([data, data1, data2, data3], axis = 1)
    
    # np.nanmean : nan 값 포함해서 평균 내기!!
    result = result.assign(z_score=np.nanmean(result.iloc[:,[10,11,12]],axis=1))
    
    # z_score > 0 인것이 가치주라고 msci에서 하고있음
    result =result[result['z_score'].notnull()]
    z_score2_max=np.percentile(result['z_score'],70)
    result =result[result['z_score']>z_score2_max]
    

    result2 = pd.concat([result,rtn[n-3]],axis=1,join='inner',ignore_index=True) #수익률 매칭                      
                       
    
    data_big = raw_data[(raw_data[n] == 3)]
    data_big = data_big.loc[:,[1,n]]
    data = pd.concat([data_big, size_FIF_wisefn[n], equity[n], ni[n+1],cash_div[n],size[n]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size_FIF_wisefn','equity','ni_12fw','cash_div','size']
    # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
    # 시총비중 구할떄는 free-float
    data['size_FIF_wisefn']=data['size_FIF_wisefn']/1000    #size 단위 thousand
    data['1/pbr']=data['equity']/data['size']
    data['1/per']=data['ni_12fw']/data['size']
    data['1/div_yield']=data['size']/data['cash_div']
    
    # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
    data = data.replace([np.inf, -np.inf],np.nan)  
    
    # Null 값 제거
    data_per = data[data['1/per'].notnull()]
    data_pbr = data[data['1/pbr'].notnull()]
    data_div = data[data['1/div_yield'].notnull()]

    #양 끝 5% 구함
#    market_capital=np.sum(data['size'])
    pbr_min=np.percentile(data_pbr['1/pbr'],5)
    pbr_max=np.percentile(data_pbr['1/pbr'],95)
    per_min=np.percentile(data_per['1/per'],5)
    per_max=np.percentile(data_per['1/per'],95)
    div_min=np.percentile(data_div['1/div_yield'],5)
    div_max=np.percentile(data_div['1/div_yield'],95)

    
    #양끝단  5% 제거
    data_pbr = data_pbr[(data_pbr['1/pbr']>pbr_min)&(data_pbr['1/pbr']<pbr_max)]
    data_per = data_per[(data_per['1/per']>per_min)&(data_per['1/per']<per_max)]
    data_div = data_div[(data_div['1/div_yield']>div_min)&(data_div['1/div_yield']<div_max)]

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
    mu_inv_div=np.sum(data_div['1/div_yield']*data_div['market_weight'])
    
    # 시총 가중 표준편자
    std_inv_pbr=np.sqrt(np.sum(np.square(data_pbr['1/pbr']-mu_inv_pbr)*data_pbr['market_weight']))
    std_inv_per=np.sqrt(np.sum(np.square(data_per['1/per']-mu_inv_per)*data_per['market_weight']))
    std_inv_div=np.sqrt(np.sum(np.square(data_div['1/div_yield']-mu_inv_div)*data_div['market_weight']))
    
    data1=(data_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
    data1.name= 'pbr_z'
    data2=(data_per['1/per']-mu_inv_per)/std_inv_per
    data2.name= 'per_z'
    data3=(data_div['1/div_yield']-mu_inv_div)/std_inv_div
    data3.name= 'div_z'
          
    result = pd.concat([data, data1, data2, data3], axis = 1)
    
    # np.nanmean : nan 값 포함해서 평균 내기!!
    result = result.assign(z_score=np.nanmean(result.iloc[:,[10,11,12]],axis=1))
    
    # z_score > 0 인것이 가치주라고 msci에서 하고있음
    result =result[result['z_score'].notnull()]
    z_score3_max=np.percentile(result['z_score'],70)
    result =result[result['z_score']>z_score3_max]
    

    result3 = pd.concat([result,rtn[n-3]],axis=1,join='inner',ignore_index=True) #수익률 매칭  
                     
    
            
#KOSDAQ
    data_big = raw_data_kq[(raw_data_kq[n] == 'KOSDAQ')]                   
    data_big = data_big.loc[:,[1,n]]
    data = pd.concat([data_big, size_FIF_wisefn_kq[n], equity_kq[n], ni_kq[n+1],cash_div_kq[n],size_kq[n]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size_FIF_wisefn','equity','ni_12fw','cash_div','size']
    # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
    # 시총비중 구할떄는 free-float
    data['size_FIF_wisefn']=data['size_FIF_wisefn']/1000    #size 단위 thousand
    data['1/pbr']=data['equity']/data['size']
    data['1/per']=data['ni_12fw']/data['size']
    data['1/div_yield']=data['size']/data['cash_div']
    
    # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
    data = data.replace([np.inf, -np.inf],np.nan)  
    
    # Null 값 제거
    data_per = data[data['1/per'].notnull()]
    data_pbr = data[data['1/pbr'].notnull()]
    data_div = data[data['1/div_yield'].notnull()]

    #양 끝 5% 구함
#    market_capital=np.sum(data['size'])
    pbr_min=np.percentile(data_pbr['1/pbr'],5)
    pbr_max=np.percentile(data_pbr['1/pbr'],95)
    per_min=np.percentile(data_per['1/per'],5)
    per_max=np.percentile(data_per['1/per'],95)
    div_min=np.percentile(data_div['1/div_yield'],5)
    div_max=np.percentile(data_div['1/div_yield'],95)

    
    #양끝단  5% 제거
    data_pbr = data_pbr[(data_pbr['1/pbr']>pbr_min)&(data_pbr['1/pbr']<pbr_max)]
    data_per = data_per[(data_per['1/per']>per_min)&(data_per['1/per']<per_max)]
    data_div = data_div[(data_div['1/div_yield']>div_min)&(data_div['1/div_yield']<div_max)]

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
    mu_inv_div=np.sum(data_div['1/div_yield']*data_div['market_weight'])
    
    # 시총 가중 표준편자
    std_inv_pbr=np.sqrt(np.sum(np.square(data_pbr['1/pbr']-mu_inv_pbr)*data_pbr['market_weight']))
    std_inv_per=np.sqrt(np.sum(np.square(data_per['1/per']-mu_inv_per)*data_per['market_weight']))
    std_inv_div=np.sqrt(np.sum(np.square(data_div['1/div_yield']-mu_inv_div)*data_div['market_weight']))
    
    data1=(data_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
    data1.name= 'pbr_z'
    data2=(data_per['1/per']-mu_inv_per)/std_inv_per
    data2.name= 'per_z'
    data3=(data_div['1/div_yield']-mu_inv_div)/std_inv_div
    data3.name= 'div_z'
          
    result = pd.concat([data, data1, data2, data3], axis = 1)
    
    # np.nanmean : nan 값 포함해서 평균 내기!!
    result = result.assign(z_score=np.nanmean(result.iloc[:,[10,11,12]],axis=1))
    
    # z_score > 0 인것이 가치주라고 msci에서 하고있음
    result =result[result['z_score'].notnull()]
    z_score4_max=np.mean([z_score1_max,z_score2_max,z_score3_max])
    z_score_kq_max=np.percentile(result['z_score'],70)
    result =result[result['z_score']>z_score_kq_max]
    

    result4 = pd.concat([result,rtn_kq[n-3]],axis=1,join='inner',ignore_index=True) #수익률 매칭  


     
                       
                       
#대형주+중형주+소형주+KOSDAQ                       
    result = pd.concat([result1,result2,result3,result4])    
    market_capital=np.sum(result[2])
    result=result.assign(market_weight2=result[2]/market_capital)          
    
    #동일가중
    return_data.iloc[0,n-3]=np.mean(result[14])
#시총가중
#    return_data.iloc[0,n-3]=np.sum(result[14]*result['market_weight2'])
    data_name[n-3]=result[0].reset_index(drop=True)
#    return_data.iloc[0,n-3]=np.sum(result[13]*result[14])    
    if n == 65 : 
        pass
    return_final=np.product(return_data,axis=1)

    # 삼성전자가 몇분기동안 포함되었는지 확인
np.sum(np.sum(data_name=="삼성전자"))


