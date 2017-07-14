# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:21:38 2017

@author: SH-NoteBook
"""

#섹터별로 표준화

#200103~201703 2 5 8 11 월 말 리밸런싱
#12 month forward 당기순이익 있는건 그대로 쓰고 없다면 trailing 당기순이익 사용 
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
#ni_12m_fw_kq.to_pickle('ni_12m_fw_kq')
#raw_data_=pd.read_excel('exercise_v02.xlsx',sheetname='Raw_data1',header=None)
#raw_data_kq=pd.read_excel('msci_rawdata_kq2.xlsm',sheetname='Raw_data_kq1',header=None)  # 편입시장
#size_kq = pd.read_excel('msci_rawdata_kq2.xlsm',sheetname='시가총액1',header=None)  #시가총액
#ni_kq = pd.read_excel('msci_rawdata_kq2.xlsm',sheetname='당기순이익MAIN1',header=None)  # 당기순이익
#ni_12m_fw_kq = pd.read_excel('msci_rawdata_kq2.xlsm',sheetname='월별당기순이익CON1',header=None)  # 당기순이익
#rtn_kq = pd.read_excel('msci_rawdata_kq2.xlsm',sheetname='수익률1',header=None)  #수익률
#equity_kq = pd.read_excel('msci_rawdata_kq2.xlsm',sheetname='자본총계1',header=None)  #자본총걔
#size_FIF_wisefn_kq = pd.read_excel('msci_rawdata_kosdaq_25811.xlsm',sheetname='유통주식수x수정주가1',header=None)  #free floating 시가총액
#cash_div_kq = pd.read_excel('msci_rawdata_kq2.xlsm',sheetname='현금배당액1',header=None)
#sector_kq = pd.read_excel('msci_rawdata_kosdaq_25811.xlsm',sheetname='업종1',header=None)
#sector_kq.to_pickle('sector_kq')
#return_dividend_kq = pd.read_excel('msci_rawdata_kosdaq_25811.xlsm',sheetname='3개월전대비수익률(배당금포함)1',header=None)
#return_dividend_kq.to_pickle('return_dividend_kq')
#cash_div_rtn = pd.read_excel('msci_rawdata_kospi_25811.xlsm',sheetname='연말현금배당수익률1',header=None)
#cash_div_rtn.to_pickle('cash_div_rtn')
#cash_div_rtn_kq = pd.read_excel('msci_rawdata_kosdaq_25811.xlsm',sheetname='연말현금배당수익률1',header=None)
#cash_div_rtn_kq.to_pickle('cash_div_rtn_kq')
#rtn_month = pd.read_excel('msci_rawdata_kospi_25811.xlsm',sheetname='월별수익률1',header=None)
#rtn_month.to_pickle('rtn_month')
#rtn_month_kq = pd.read_excel('msci_rawdata_kosdaq_25811.xlsm',sheetname='월별수익률1',header=None)
#rtn_month_kq.to_pickle('rtn_month_kq')




kospi_quarter = pd.read_pickle('kospi_quarter')
raw_data = pd.read_pickle('raw_data')
size = pd.read_pickle('size')  #시가총액
ni = pd.read_pickle('ni') # 당기순이익
ni_12m_fw = pd.read_pickle('ni_12m_fw')
rtn = pd.read_pickle('rtn')
equity = pd.read_pickle('equity') #자본총계
cash_div = pd.read_pickle('cash_div')
size_FIF_wisefn=pd.read_pickle('size_FIF_wisefn') #시가총액
sector=pd.read_pickle('sector') 
return_dividend = pd.read_pickle('return_dividend') #배당고려수익률
cash_div_rtn = pd.read_pickle('cash_div_rtn') #연말현금배당수익률
rtn_month = pd.read_pickle('rtn_month') #월별수익률

raw_data_kq = pd.read_pickle('raw_data_kq')
size_kq = pd.read_pickle('size_kq')  #시가총액
ni_kq = pd.read_pickle('ni_kq') # 당기순이익
ni_12m_fw_kq = pd.read_pickle('ni_12m_fw_kq') 
rtn_kq = pd.read_pickle('rtn_kq')
equity_kq = pd.read_pickle('equity_kq') #자본총계
cash_div_kq = pd.read_pickle('cash_div_kq')
size_FIF_wisefn_kq=pd.read_pickle('size_FIF_wisefn_kq') #시가총액
sector_kq=pd.read_pickle('sector_kq') #시가총액
return_dividend_kq = pd.read_pickle('return_dividend_kq')
cash_div_rtn_kq = pd.read_pickle('cash_div_rtn_kq') #연말현금배당수익률
rtn_month_kq = pd.read_pickle('rtn_month_kq') #월별수익률

#소형주 + KOSDAQ 하기 위해 새로운 rawdata 생성(primary key 때문에)                   
raw_data_sum=pd.concat([raw_data,raw_data_kq],axis=0,ignore_index=True)
rtn_sum=pd.concat([rtn,rtn_kq],axis=0,ignore_index=True)
size_sum=pd.concat([size,size_kq],axis=0,ignore_index=True)
ni_12m_fw_sum = pd.concat([ni_12m_fw,ni_12m_fw_kq],axis=0,ignore_index=True)
ni_sum=pd.concat([ni,ni_kq],axis=0,ignore_index=True)
equity_sum=pd.concat([equity,equity_kq],axis=0,ignore_index=True)
size_FIF_wisefn_sum=pd.concat([size_FIF_wisefn,size_FIF_wisefn_kq],axis=0,ignore_index=True)
cash_div_sum=pd.concat([cash_div,cash_div_kq],axis=0,ignore_index=True)
sector_sum=pd.concat([sector,sector_kq],axis=0,ignore_index=True)
return_dividend_sum = pd.concat([return_dividend,return_dividend_kq],axis=0,ignore_index=True)
cash_div_rtn_sum = pd.concat([cash_div_rtn,cash_div_rtn_kq],axis=0,ignore_index=True)
rtn_month_sum = pd.concat([rtn_month,rtn_month_kq],axis=0,ignore_index=True)
#size_FIF=pd.read_pickle('size_FIF')  #자기주식 제외 시가총액
#size_FIF_insider=pd.read_pickle('size_FIF_insider') #자기주식, 최대주주 주식 제외 시가총
#size_FIF_wisefn = pd.read_excel('msci_rawdata.xlsx',sheetname='유통주식수x수정주가1',header=None) # wisefn에서 산출해주는 유통비율 이용
#size_FIF_wisefn.to_pickle('size_FIF_wisefn')

turnover = pd.DataFrame(np.zeros((1,1)))
portfolio_cash_rtn = pd.DataFrame(np.zeros((1,16)))
return_data = np.zeros((5,65))
return_data = pd.DataFrame(return_data)
return_month_data = pd.DataFrame(np.zeros((1,195)))
data_name=pd.DataFrame(np.zeros((1000,65)))
kosdaq_count = pd.DataFrame(np.zeros((1,65)))
# 매 분기 수익률을 기록하기 위해 quarter_data를 만듬
quarter_data = pd.DataFrame(np.zeros((1000,195)))
sector_data = pd.DataFrame(np.zeros((1000,130)))
group_data = pd.DataFrame(np.zeros((1000,130)))
result_cash = pd.DataFrame(np.zeros((200,16)))


z=0 #연말현금배당수익률을 저장하기 위해 ... 아래 if문있음
for n in range(3,68):
    #66마지막 분기
    data_big = raw_data_sum[(raw_data_sum[n] == 1)|(raw_data_sum[n] == 2)|(raw_data_sum[n] == 3)|(raw_data_sum[n] == 'KOSDAQ')]
    data_big = data_big.loc[:,[1,n]]
    #ni_12m_fw_sum 쓰면 fwd per, 그냥 ni_sum 쓰면 trailing
    #rtn_sum은 그냥, rtn_dividend_sum은 배당고려 수익률
    data = pd.concat([data_big, size_FIF_wisefn_sum[n], equity_sum[n], ni_12m_fw_sum[n],cash_div_sum[n],size_sum[n],rtn_sum[n-3],sector_sum[n],rtn_month_sum[3*(n-3)],rtn_month_sum[3*(n-3)+1],rtn_month_sum[3*(n-3)+2]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size_FIF_wisefn','equity','ni_12fw','cash_div','size','return','sector','return_month1','return_month2','return_month3']
    data=data[data['size']>100000000000]
    #상폐, 지주사전환, 분할상장 때문에 생기는 수익률 0 제거
    data=data[data['return']!=0]
    result_temp = data
    samsung = pd.DataFrame(data.loc[390,:]).transpose()

    data = data[data['equity'].notnull()]
    data = data[data['ni_12fw'].notnull()]
    data = data[data['cash_div'].notnull()]
    
    #IT 섹터
    if np.sum(data['sector']=='IT')>0:
        data_IT = data[data['sector']=="IT"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_IT['size_FIF_wisefn']=data_IT['size_FIF_wisefn']/1000    #size 단위 thousand
        data_IT['1/pbr']=data_IT['equity']/data_IT['size']
        data_IT['1/per']=data_IT['ni_12fw']/data_IT['size']
        data_IT['div_yield']=data_IT['cash_div']/data_IT['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_IT_IT = data_IT.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_IT_per = data_IT[data_IT['1/per'].notnull()]
        data_IT_pbr = data_IT[data_IT['1/pbr'].notnull()]
        data_IT_div = data_IT[data_IT['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_IT_pbr_cap = np.sum(data_IT_pbr['size_FIF_wisefn'])
        data_IT_per_cap = np.sum(data_IT_per['size_FIF_wisefn'])
        data_IT_div_cap = np.sum(data_IT_div['size_FIF_wisefn'])
    
        data_IT_pbr = data_IT_pbr.assign(market_weight=data_IT_pbr['size_FIF_wisefn']/data_IT_pbr_cap)
        data_IT_per = data_IT_per.assign(market_weight=data_IT_per['size_FIF_wisefn']/data_IT_per_cap)
        data_IT_div = data_IT_div.assign(market_weight=data_IT_div['size_FIF_wisefn']/data_IT_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_IT_pbr['1/pbr']*data_IT_pbr['market_weight'])
        mu_inv_per=np.sum(data_IT_per['1/per']*data_IT_per['market_weight'])
        mu_inv_div=np.sum(data_IT_div['div_yield']*data_IT_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_IT_pbr['1/pbr']-mu_inv_pbr)*data_IT_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_IT_per['1/per']-mu_inv_per)*data_IT_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_IT_div['div_yield']-mu_inv_div)*data_IT_div['market_weight']))
        
        data_IT1=(data_IT_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_IT1.name= 'pbr_z'
        data_IT2=(data_IT_per['1/per']-mu_inv_per)/std_inv_per
        data_IT2.name= 'per_z'
        data_IT3=(data_IT_div['div_yield']-mu_inv_div)/std_inv_div
        data_IT3.name= 'div_z'
              
        result_IT = pd.concat([data_IT, data_IT1, data_IT2, data_IT3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_IT = result_IT.assign(z_score=np.nanmean(result_IT.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        result_IT =result_IT[result_IT['z_score'].notnull()]
    
    
    #건강관리 섹터
    if np.sum(data['sector']=='건강관리')>0:
        data_건강관리 = data[data['sector']=='건강관리']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_건강관리['size_FIF_wisefn']=data_건강관리['size_FIF_wisefn']/1000    #size 단위 thousand
        data_건강관리['1/pbr']=data_건강관리['equity']/data_건강관리['size']
        data_건강관리['1/per']=data_건강관리['ni_12fw']/data_건강관리['size']
        data_건강관리['div_yield']=data_건강관리['cash_div']/data_건강관리['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_건강관리_건강관리 = data_건강관리.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_건강관리_per = data_건강관리[data_건강관리['1/per'].notnull()]
        data_건강관리_pbr = data_건강관리[data_건강관리['1/pbr'].notnull()]
        data_건강관리_div = data_건강관리[data_건강관리['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_건강관리_pbr_cap = np.sum(data_건강관리_pbr['size_FIF_wisefn'])
        data_건강관리_per_cap = np.sum(data_건강관리_per['size_FIF_wisefn'])
        data_건강관리_div_cap = np.sum(data_건강관리_div['size_FIF_wisefn'])
    
        data_건강관리_pbr = data_건강관리_pbr.assign(market_weight=data_건강관리_pbr['size_FIF_wisefn']/data_건강관리_pbr_cap)
        data_건강관리_per = data_건강관리_per.assign(market_weight=data_건강관리_per['size_FIF_wisefn']/data_건강관리_per_cap)
        data_건강관리_div = data_건강관리_div.assign(market_weight=data_건강관리_div['size_FIF_wisefn']/data_건강관리_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_건강관리_pbr['1/pbr']*data_건강관리_pbr['market_weight'])
        mu_inv_per=np.sum(data_건강관리_per['1/per']*data_건강관리_per['market_weight'])
        mu_inv_div=np.sum(data_건강관리_div['div_yield']*data_건강관리_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_건강관리_pbr['1/pbr']-mu_inv_pbr)*data_건강관리_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_건강관리_per['1/per']-mu_inv_per)*data_건강관리_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_건강관리_div['div_yield']-mu_inv_div)*data_건강관리_div['market_weight']))
        
        data_건강관리1=(data_건강관리_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_건강관리1.name= 'pbr_z'
        data_건강관리2=(data_건강관리_per['1/per']-mu_inv_per)/std_inv_per
        data_건강관리2.name= 'per_z'
        data_건강관리3=(data_건강관리_div['div_yield']-mu_inv_div)/std_inv_div
        data_건강관리3.name= 'div_z'
              
        result_건강관리 = pd.concat([data_건강관리, data_건강관리1, data_건강관리2, data_건강관리3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_건강관리 = result_건강관리.assign(z_score=np.nanmean(result_건강관리.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        result_건강관리 =result_건강관리[result_건강관리['z_score'].notnull()]
        
       #경기관련소비재 섹터
    if np.sum(data['sector']=='경기관련소비재')>0:
        data_경기관련소비재 = data[data['sector']=='경기관련소비재']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_경기관련소비재['size_FIF_wisefn']=data_경기관련소비재['size_FIF_wisefn']/1000    #size 단위 thousand
        data_경기관련소비재['1/pbr']=data_경기관련소비재['equity']/data_경기관련소비재['size']
        data_경기관련소비재['1/per']=data_경기관련소비재['ni_12fw']/data_경기관련소비재['size']
        data_경기관련소비재['div_yield']=data_경기관련소비재['cash_div']/data_경기관련소비재['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_경기관련소비재_경기관련소비재 = data_경기관련소비재.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_경기관련소비재_per = data_경기관련소비재[data_경기관련소비재['1/per'].notnull()]
        data_경기관련소비재_pbr = data_경기관련소비재[data_경기관련소비재['1/pbr'].notnull()]
        data_경기관련소비재_div = data_경기관련소비재[data_경기관련소비재['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_경기관련소비재_pbr_cap = np.sum(data_경기관련소비재_pbr['size_FIF_wisefn'])
        data_경기관련소비재_per_cap = np.sum(data_경기관련소비재_per['size_FIF_wisefn'])
        data_경기관련소비재_div_cap = np.sum(data_경기관련소비재_div['size_FIF_wisefn'])
    
        data_경기관련소비재_pbr = data_경기관련소비재_pbr.assign(market_weight=data_경기관련소비재_pbr['size_FIF_wisefn']/data_경기관련소비재_pbr_cap)
        data_경기관련소비재_per = data_경기관련소비재_per.assign(market_weight=data_경기관련소비재_per['size_FIF_wisefn']/data_경기관련소비재_per_cap)
        data_경기관련소비재_div = data_경기관련소비재_div.assign(market_weight=data_경기관련소비재_div['size_FIF_wisefn']/data_경기관련소비재_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_경기관련소비재_pbr['1/pbr']*data_경기관련소비재_pbr['market_weight'])
        mu_inv_per=np.sum(data_경기관련소비재_per['1/per']*data_경기관련소비재_per['market_weight'])
        mu_inv_div=np.sum(data_경기관련소비재_div['div_yield']*data_경기관련소비재_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_경기관련소비재_pbr['1/pbr']-mu_inv_pbr)*data_경기관련소비재_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_경기관련소비재_per['1/per']-mu_inv_per)*data_경기관련소비재_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_경기관련소비재_div['div_yield']-mu_inv_div)*data_경기관련소비재_div['market_weight']))
        
        data_경기관련소비재1=(data_경기관련소비재_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_경기관련소비재1.name= 'pbr_z'
        data_경기관련소비재2=(data_경기관련소비재_per['1/per']-mu_inv_per)/std_inv_per
        data_경기관련소비재2.name= 'per_z'
        data_경기관련소비재3=(data_경기관련소비재_div['div_yield']-mu_inv_div)/std_inv_div
        data_경기관련소비재3.name= 'div_z'
              
        result_경기관련소비재 = pd.concat([data_경기관련소비재, data_경기관련소비재1, data_경기관련소비재2, data_경기관련소비재3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_경기관련소비재 = result_경기관련소비재.assign(z_score=np.nanmean(result_경기관련소비재.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        result_경기관련소비재 =result_경기관련소비재[result_경기관련소비재['z_score'].notnull()]
        
    #금융 섹터
    if np.sum(data['sector']=='금융')>0:
        data_금융 = data[data['sector']=='금융']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_금융['size_FIF_wisefn']=data_금융['size_FIF_wisefn']/1000    #size 단위 thousand
        data_금융['1/pbr']=data_금융['equity']/data_금융['size']
        data_금융['1/per']=data_금융['ni_12fw']/data_금융['size']
        data_금융['div_yield']=data_금융['cash_div']/data_금융['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_금융_금융 = data_금융.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_금융_per = data_금융[data_금융['1/per'].notnull()]
        data_금융_pbr = data_금융[data_금융['1/pbr'].notnull()]
        data_금융_div = data_금융[data_금융['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_금융_pbr_cap = np.sum(data_금융_pbr['size_FIF_wisefn'])
        data_금융_per_cap = np.sum(data_금융_per['size_FIF_wisefn'])
        data_금융_div_cap = np.sum(data_금융_div['size_FIF_wisefn'])
    
        data_금융_pbr = data_금융_pbr.assign(market_weight=data_금융_pbr['size_FIF_wisefn']/data_금융_pbr_cap)
        data_금융_per = data_금융_per.assign(market_weight=data_금융_per['size_FIF_wisefn']/data_금융_per_cap)
        data_금융_div = data_금융_div.assign(market_weight=data_금융_div['size_FIF_wisefn']/data_금융_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_금융_pbr['1/pbr']*data_금융_pbr['market_weight'])
        mu_inv_per=np.sum(data_금융_per['1/per']*data_금융_per['market_weight'])
        mu_inv_div=np.sum(data_금융_div['div_yield']*data_금융_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_금융_pbr['1/pbr']-mu_inv_pbr)*data_금융_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_금융_per['1/per']-mu_inv_per)*data_금융_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_금융_div['div_yield']-mu_inv_div)*data_금융_div['market_weight']))
        
        data_금융1=(data_금융_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_금융1.name= 'pbr_z'
        data_금융2=(data_금융_per['1/per']-mu_inv_per)/std_inv_per
        data_금융2.name= 'per_z'
        data_금융3=(data_금융_div['div_yield']-mu_inv_div)/std_inv_div
        data_금융3.name= 'div_z'
              
        result_금융 = pd.concat([data_금융, data_금융1, data_금융2, data_금융3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_금융 = result_금융.assign(z_score=np.nanmean(result_금융.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        result_금융 =result_금융[result_금융['z_score'].notnull()]
       
    #산업재 섹터
    if np.sum(data['sector']=='산업재')>0:
        data_산업재 = data[data['sector']=='산업재']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_산업재['size_FIF_wisefn']=data_산업재['size_FIF_wisefn']/1000    #size 단위 thousand
        data_산업재['1/pbr']=data_산업재['equity']/data_산업재['size']
        data_산업재['1/per']=data_산업재['ni_12fw']/data_산업재['size']
        data_산업재['div_yield']=data_산업재['cash_div']/data_산업재['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_산업재_산업재 = data_산업재.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_산업재_per = data_산업재[data_산업재['1/per'].notnull()]
        data_산업재_pbr = data_산업재[data_산업재['1/pbr'].notnull()]
        data_산업재_div = data_산업재[data_산업재['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_산업재_pbr_cap = np.sum(data_산업재_pbr['size_FIF_wisefn'])
        data_산업재_per_cap = np.sum(data_산업재_per['size_FIF_wisefn'])
        data_산업재_div_cap = np.sum(data_산업재_div['size_FIF_wisefn'])
    
        data_산업재_pbr = data_산업재_pbr.assign(market_weight=data_산업재_pbr['size_FIF_wisefn']/data_산업재_pbr_cap)
        data_산업재_per = data_산업재_per.assign(market_weight=data_산업재_per['size_FIF_wisefn']/data_산업재_per_cap)
        data_산업재_div = data_산업재_div.assign(market_weight=data_산업재_div['size_FIF_wisefn']/data_산업재_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_산업재_pbr['1/pbr']*data_산업재_pbr['market_weight'])
        mu_inv_per=np.sum(data_산업재_per['1/per']*data_산업재_per['market_weight'])
        mu_inv_div=np.sum(data_산업재_div['div_yield']*data_산업재_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_산업재_pbr['1/pbr']-mu_inv_pbr)*data_산업재_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_산업재_per['1/per']-mu_inv_per)*data_산업재_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_산업재_div['div_yield']-mu_inv_div)*data_산업재_div['market_weight']))
        
        data_산업재1=(data_산업재_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_산업재1.name= 'pbr_z'
        data_산업재2=(data_산업재_per['1/per']-mu_inv_per)/std_inv_per
        data_산업재2.name= 'per_z'
        data_산업재3=(data_산업재_div['div_yield']-mu_inv_div)/std_inv_div
        data_산업재3.name= 'div_z'
              
        result_산업재 = pd.concat([data_산업재, data_산업재1, data_산업재2, data_산업재3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_산업재 = result_산업재.assign(z_score=np.nanmean(result_산업재.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        result_산업재 =result_산업재[result_산업재['z_score'].notnull()]
        
    #소재 섹터
    if np.sum(data['sector']=='소재')>0:
        data_소재 = data[data['sector']=='소재']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_소재['size_FIF_wisefn']=data_소재['size_FIF_wisefn']/1000    #size 단위 thousand
        data_소재['1/pbr']=data_소재['equity']/data_소재['size']
        data_소재['1/per']=data_소재['ni_12fw']/data_소재['size']
        data_소재['div_yield']=data_소재['cash_div']/data_소재['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_소재_소재 = data_소재.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_소재_per = data_소재[data_소재['1/per'].notnull()]
        data_소재_pbr = data_소재[data_소재['1/pbr'].notnull()]
        data_소재_div = data_소재[data_소재['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_소재_pbr_cap = np.sum(data_소재_pbr['size_FIF_wisefn'])
        data_소재_per_cap = np.sum(data_소재_per['size_FIF_wisefn'])
        data_소재_div_cap = np.sum(data_소재_div['size_FIF_wisefn'])
    
        data_소재_pbr = data_소재_pbr.assign(market_weight=data_소재_pbr['size_FIF_wisefn']/data_소재_pbr_cap)
        data_소재_per = data_소재_per.assign(market_weight=data_소재_per['size_FIF_wisefn']/data_소재_per_cap)
        data_소재_div = data_소재_div.assign(market_weight=data_소재_div['size_FIF_wisefn']/data_소재_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_소재_pbr['1/pbr']*data_소재_pbr['market_weight'])
        mu_inv_per=np.sum(data_소재_per['1/per']*data_소재_per['market_weight'])
        mu_inv_div=np.sum(data_소재_div['div_yield']*data_소재_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_소재_pbr['1/pbr']-mu_inv_pbr)*data_소재_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_소재_per['1/per']-mu_inv_per)*data_소재_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_소재_div['div_yield']-mu_inv_div)*data_소재_div['market_weight']))
        
        data_소재1=(data_소재_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_소재1.name= 'pbr_z'
        data_소재2=(data_소재_per['1/per']-mu_inv_per)/std_inv_per
        data_소재2.name= 'per_z'
        data_소재3=(data_소재_div['div_yield']-mu_inv_div)/std_inv_div
        data_소재3.name= 'div_z'
              
        result_소재 = pd.concat([data_소재, data_소재1, data_소재2, data_소재3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_소재 = result_소재.assign(z_score=np.nanmean(result_소재.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        result_소재 =result_소재[result_소재['z_score'].notnull()]
          
    #에너지 섹터
    if np.sum(data['sector']=='에너지')>0:
        data_에너지 = data[data['sector']=='에너지']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_에너지['size_FIF_wisefn']=data_에너지['size_FIF_wisefn']/1000    #size 단위 thousand
        data_에너지['1/pbr']=data_에너지['equity']/data_에너지['size']
        data_에너지['1/per']=data_에너지['ni_12fw']/data_에너지['size']
        data_에너지['div_yield']=data_에너지['cash_div']/data_에너지['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_에너지_에너지 = data_에너지.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_에너지_per = data_에너지[data_에너지['1/per'].notnull()]
        data_에너지_pbr = data_에너지[data_에너지['1/pbr'].notnull()]
        data_에너지_div = data_에너지[data_에너지['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_에너지_pbr_cap = np.sum(data_에너지_pbr['size_FIF_wisefn'])
        data_에너지_per_cap = np.sum(data_에너지_per['size_FIF_wisefn'])
        data_에너지_div_cap = np.sum(data_에너지_div['size_FIF_wisefn'])
    
        data_에너지_pbr = data_에너지_pbr.assign(market_weight=data_에너지_pbr['size_FIF_wisefn']/data_에너지_pbr_cap)
        data_에너지_per = data_에너지_per.assign(market_weight=data_에너지_per['size_FIF_wisefn']/data_에너지_per_cap)
        data_에너지_div = data_에너지_div.assign(market_weight=data_에너지_div['size_FIF_wisefn']/data_에너지_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_에너지_pbr['1/pbr']*data_에너지_pbr['market_weight'])
        mu_inv_per=np.sum(data_에너지_per['1/per']*data_에너지_per['market_weight'])
        mu_inv_div=np.sum(data_에너지_div['div_yield']*data_에너지_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_에너지_pbr['1/pbr']-mu_inv_pbr)*data_에너지_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_에너지_per['1/per']-mu_inv_per)*data_에너지_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_에너지_div['div_yield']-mu_inv_div)*data_에너지_div['market_weight']))
        
        data_에너지1=(data_에너지_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_에너지1.name= 'pbr_z'
        data_에너지2=(data_에너지_per['1/per']-mu_inv_per)/std_inv_per
        data_에너지2.name= 'per_z'
        data_에너지3=(data_에너지_div['div_yield']-mu_inv_div)/std_inv_div
        data_에너지3.name= 'div_z'
              
        result_에너지 = pd.concat([data_에너지, data_에너지1, data_에너지2, data_에너지3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_에너지 = result_에너지.assign(z_score=np.nanmean(result_에너지.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        result_에너지 =result_에너지[result_에너지['z_score'].notnull()]
         
    #유틸리티 섹터
    if np.sum(data['sector']=='유틸리티')>0:
        data_유틸리티 = data[data['sector']=='유틸리티']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_유틸리티['size_FIF_wisefn']=data_유틸리티['size_FIF_wisefn']/1000    #size 단위 thousand
        data_유틸리티['1/pbr']=data_유틸리티['equity']/data_유틸리티['size']
        data_유틸리티['1/per']=data_유틸리티['ni_12fw']/data_유틸리티['size']
        data_유틸리티['div_yield']=data_유틸리티['cash_div']/data_유틸리티['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_유틸리티_유틸리티 = data_유틸리티.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_유틸리티_per = data_유틸리티[data_유틸리티['1/per'].notnull()]
        data_유틸리티_pbr = data_유틸리티[data_유틸리티['1/pbr'].notnull()]
        data_유틸리티_div = data_유틸리티[data_유틸리티['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_유틸리티_pbr_cap = np.sum(data_유틸리티_pbr['size_FIF_wisefn'])
        data_유틸리티_per_cap = np.sum(data_유틸리티_per['size_FIF_wisefn'])
        data_유틸리티_div_cap = np.sum(data_유틸리티_div['size_FIF_wisefn'])
    
        data_유틸리티_pbr = data_유틸리티_pbr.assign(market_weight=data_유틸리티_pbr['size_FIF_wisefn']/data_유틸리티_pbr_cap)
        data_유틸리티_per = data_유틸리티_per.assign(market_weight=data_유틸리티_per['size_FIF_wisefn']/data_유틸리티_per_cap)
        data_유틸리티_div = data_유틸리티_div.assign(market_weight=data_유틸리티_div['size_FIF_wisefn']/data_유틸리티_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_유틸리티_pbr['1/pbr']*data_유틸리티_pbr['market_weight'])
        mu_inv_per=np.sum(data_유틸리티_per['1/per']*data_유틸리티_per['market_weight'])
        mu_inv_div=np.sum(data_유틸리티_div['div_yield']*data_유틸리티_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_유틸리티_pbr['1/pbr']-mu_inv_pbr)*data_유틸리티_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_유틸리티_per['1/per']-mu_inv_per)*data_유틸리티_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_유틸리티_div['div_yield']-mu_inv_div)*data_유틸리티_div['market_weight']))
        
        data_유틸리티1=(data_유틸리티_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_유틸리티1.name= 'pbr_z'
        data_유틸리티2=(data_유틸리티_per['1/per']-mu_inv_per)/std_inv_per
        data_유틸리티2.name= 'per_z'
        data_유틸리티3=(data_유틸리티_div['div_yield']-mu_inv_div)/std_inv_div
        data_유틸리티3.name= 'div_z'
              
        result_유틸리티 = pd.concat([data_유틸리티, data_유틸리티1, data_유틸리티2, data_유틸리티3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_유틸리티 = result_유틸리티.assign(z_score=np.nanmean(result_유틸리티.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        result_유틸리티 =result_유틸리티[result_유틸리티['z_score'].notnull()]
          
    #전기통신서비스 섹터
    if np.sum(data['sector']=='전기통신서비스')>0:
        data_정기통신서비스 = data[data['sector']=='전기통신서비스']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_정기통신서비스['size_FIF_wisefn']=data_정기통신서비스['size_FIF_wisefn']/1000    #size 단위 thousand
        data_정기통신서비스['1/pbr']=data_정기통신서비스['equity']/data_정기통신서비스['size']
        data_정기통신서비스['1/per']=data_정기통신서비스['ni_12fw']/data_정기통신서비스['size']
        data_정기통신서비스['div_yield']=data_정기통신서비스['cash_div']/data_정기통신서비스['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_정기통신서비스_정기통신서비스 = data_정기통신서비스.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_정기통신서비스_per = data_정기통신서비스[data_정기통신서비스['1/per'].notnull()]
        data_정기통신서비스_pbr = data_정기통신서비스[data_정기통신서비스['1/pbr'].notnull()]
        data_정기통신서비스_div = data_정기통신서비스[data_정기통신서비스['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_정기통신서비스_pbr_cap = np.sum(data_정기통신서비스_pbr['size_FIF_wisefn'])
        data_정기통신서비스_per_cap = np.sum(data_정기통신서비스_per['size_FIF_wisefn'])
        data_정기통신서비스_div_cap = np.sum(data_정기통신서비스_div['size_FIF_wisefn'])
    
        data_정기통신서비스_pbr = data_정기통신서비스_pbr.assign(market_weight=data_정기통신서비스_pbr['size_FIF_wisefn']/data_정기통신서비스_pbr_cap)
        data_정기통신서비스_per = data_정기통신서비스_per.assign(market_weight=data_정기통신서비스_per['size_FIF_wisefn']/data_정기통신서비스_per_cap)
        data_정기통신서비스_div = data_정기통신서비스_div.assign(market_weight=data_정기통신서비스_div['size_FIF_wisefn']/data_정기통신서비스_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_정기통신서비스_pbr['1/pbr']*data_정기통신서비스_pbr['market_weight'])
        mu_inv_per=np.sum(data_정기통신서비스_per['1/per']*data_정기통신서비스_per['market_weight'])
        mu_inv_div=np.sum(data_정기통신서비스_div['div_yield']*data_정기통신서비스_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_정기통신서비스_pbr['1/pbr']-mu_inv_pbr)*data_정기통신서비스_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_정기통신서비스_per['1/per']-mu_inv_per)*data_정기통신서비스_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_정기통신서비스_div['div_yield']-mu_inv_div)*data_정기통신서비스_div['market_weight']))
        
        data_정기통신서비스1=(data_정기통신서비스_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_정기통신서비스1.name= 'pbr_z'
        data_정기통신서비스2=(data_정기통신서비스_per['1/per']-mu_inv_per)/std_inv_per
        data_정기통신서비스2.name= 'per_z'
        data_정기통신서비스3=(data_정기통신서비스_div['div_yield']-mu_inv_div)/std_inv_div
        data_정기통신서비스3.name= 'div_z'
              
        result_정기통신서비스 = pd.concat([data_정기통신서비스, data_정기통신서비스1, data_정기통신서비스2, data_정기통신서비스3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_정기통신서비스 = result_정기통신서비스.assign(z_score=np.nanmean(result_정기통신서비스.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        result_정기통신서비스 =result_정기통신서비스[result_정기통신서비스['z_score'].notnull()]
        
    #필수소비재 섹터
    if np.sum(data['sector']=='필수소비재')>0:
        data_필수소비재 = data[data['sector']=='필수소비재']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_필수소비재['size_FIF_wisefn']=data_필수소비재['size_FIF_wisefn']/1000    #size 단위 thousand
        data_필수소비재['1/pbr']=data_필수소비재['equity']/data_필수소비재['size']
        data_필수소비재['1/per']=data_필수소비재['ni_12fw']/data_필수소비재['size']
        data_필수소비재['div_yield']=data_필수소비재['cash_div']/data_필수소비재['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_필수소비재_필수소비재 = data_필수소비재.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_필수소비재_per = data_필수소비재[data_필수소비재['1/per'].notnull()]
        data_필수소비재_pbr = data_필수소비재[data_필수소비재['1/pbr'].notnull()]
        data_필수소비재_div = data_필수소비재[data_필수소비재['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_필수소비재_pbr_cap = np.sum(data_필수소비재_pbr['size_FIF_wisefn'])
        data_필수소비재_per_cap = np.sum(data_필수소비재_per['size_FIF_wisefn'])
        data_필수소비재_div_cap = np.sum(data_필수소비재_div['size_FIF_wisefn'])
    
        data_필수소비재_pbr = data_필수소비재_pbr.assign(market_weight=data_필수소비재_pbr['size_FIF_wisefn']/data_필수소비재_pbr_cap)
        data_필수소비재_per = data_필수소비재_per.assign(market_weight=data_필수소비재_per['size_FIF_wisefn']/data_필수소비재_per_cap)
        data_필수소비재_div = data_필수소비재_div.assign(market_weight=data_필수소비재_div['size_FIF_wisefn']/data_필수소비재_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_필수소비재_pbr['1/pbr']*data_필수소비재_pbr['market_weight'])
        mu_inv_per=np.sum(data_필수소비재_per['1/per']*data_필수소비재_per['market_weight'])
        mu_inv_div=np.sum(data_필수소비재_div['div_yield']*data_필수소비재_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_필수소비재_pbr['1/pbr']-mu_inv_pbr)*data_필수소비재_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_필수소비재_per['1/per']-mu_inv_per)*data_필수소비재_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_필수소비재_div['div_yield']-mu_inv_div)*data_필수소비재_div['market_weight']))
        
        data_필수소비재1=(data_필수소비재_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_필수소비재1.name= 'pbr_z'
        data_필수소비재2=(data_필수소비재_per['1/per']-mu_inv_per)/std_inv_per
        data_필수소비재2.name= 'per_z'
        data_필수소비재3=(data_필수소비재_div['div_yield']-mu_inv_div)/std_inv_div
        data_필수소비재3.name= 'div_z'
              
        result_필수소비재 = pd.concat([data_필수소비재, data_필수소비재1, data_필수소비재2, data_필수소비재3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_필수소비재 = result_필수소비재.assign(z_score=np.nanmean(result_필수소비재.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        result_필수소비재 =result_필수소비재[result_필수소비재['z_score'].notnull()]
    
    result = pd.concat([result_IT,result_건강관리,result_경기관련소비재,result_금융,result_산업재,result_소재,result_에너지,result_유틸리티,result_정기통신서비스,result_필수소비재],axis=0)
    
    
    
    
    #상위 65%로 결정하면 삼성전자가 n=64,65,66일때 모두 포함이 된다.
#    z_score1_max=np.percentile(result['z_score'],50)
#    result =result[result['z_score']>z_score1_max]
    result=result.assign(rnk=result['z_score'].rank(method='first',ascending=False)) 
    
#    result = pd.concat([result,pd.DataFrame(result_temp.loc[390,:]).transpose()],axis=0)

    #중복 rows 1개 빼고 다 제거 
    result = result.drop_duplicates()
    result = result[result['rnk']<201] 
    

#    result = pd.concat([result,rtn_sum[n-3]],axis=1,join='inner',ignore_index=True) #수익률 매칭
    
  
                       
#대형주+중형주+소형주+KOSDAQ                       
    
    #코스닥이 몇종목 포함되었는지 기록
    try : 
        kosdaq_number = result['group']=='KOSDAQ'
    except :
        kosdaq_number = result['group']==0
    kosdaq_number2 = pd.DataFrame(kosdaq_number.cumsum(axis=0))
    len3 = len(kosdaq_number2)
    kosdaq_count.iloc[0,n-3] = kosdaq_number2.iloc[len3-1,0]
    
    

    #n=33 일때 국민은행 같은게 지주사로 전환되면서 수익률이 0으로 나와버림 NAN이랑은 다른개념 
    #사전에 공시가 되기 때문에 거를수 있을거라 판단해서 제외
    #일딴 여기서 제거해보고 , z 값 산출 전에 제거하는법 생각
#    result = result[result[15]!=0]
    #매 분기 수익률을 기록
    quarter_data[[3*(n-3),3*(n-3)+1,3*(n-3)+2]] = result.iloc[:,[0,1,7]].reset_index(drop=True)
    market_capital=np.sum(result['size_FIF_wisefn'])
    result=result.assign(market_weight2=result['size_FIF_wisefn']/market_capital)          
    
    #연말현금배당수익률 저장
    if (n>4)&((n-4)%4==2):
        result_cash_temp= pd.concat([result['name'],cash_div_rtn_sum[(n+2)/4-2]],axis=1)
        result_cash_temp=result_cash_temp[result_cash_temp['name'].notnull()]
        result_cash[[z,z+1]] = result_cash_temp.iloc[:,[0,1]].reset_index(drop=True)
        z=z+2
        
    
    #동일가중
    return_data.iloc[0,n-3]=np.mean(result['return'])

    #월별 수익률 구하기
    result = result.assign(gross_return_2 = result['return_month1']*result['return_month2'])
    #아래처럼 구하면 누적수익률이 달라짐
#    return_month_data[[3*(n-3),3*(n-3)+1,3*(n-3)+2]]=pd.DataFrame(np.mean(result[['return_month1','return_month2','return_month3']])).transpose()
    #forward yield같은 느낌으로 구함
    return_month_data[3*(n-3)] = np.mean(result['return_month1'])
    return_month_data[3*(n-3)+1] = np.mean(result['gross_return_2'])/return_month_data[3*(n-3)]
    return_month_data[3*(n-3)+2] = np.mean(result['return'])/np.mean(result['gross_return_2'])

    #시총가중
#    return_data.iloc[0,n-3]=np.sum(result['return']*result['market_weight2'])

    data_name[n-3]=result['name'].reset_index(drop=True)
    #섹터별 비중 구함
    sector_data[[2*(n-3),2*(n-3)+1]]=result.groupby('sector').size().reset_index(drop=False)
    group_data[[2*(n-3),2*(n-3)+1]]=result.groupby('group').size().reset_index(drop=False)
#    return_data.iloc[0,n-3]=np.sum(result[13]*result[14])    
    if n == 67 : 
        pass
    return_final=np.product(return_data,axis=1)

    # 삼성전자가 몇분기동안 포함되었는지 확인
np.sum(np.sum(data_name=="삼성전자"))

average_return = np.mean(return_data,axis=1)
std_return = np.std(return_data,axis=1)
average_return/std_return
#turnover
for n in range(3,67):
    
    len1 = len(data_name[data_name[n-2].notnull()])
    aaa=data_name.loc[:,[n-3,n-2]]
    bbb=pd.DataFrame(aaa.stack().value_counts())
    len2=len(bbb[bbb[0]==2])
    data_name.loc[999,n-2]=(len1-len2)/len1
    turnover_quarter=data_name.iloc[999,1:]
    turnover=np.mean(turnover_quarter)

#turnvoer에 2% 곱해서 거래비용 계산하기
#첫기에는 거래비용이 100%이다
turnover_temp = pd.DataFrame(np.ones((1,1)))
turnover_quarter = pd.DataFrame(turnover_quarter).transpose().reset_index(drop=True)
turnover_quarter = pd.concat([turnover_temp,turnover_quarter],axis=1)
turnover_quarter = turnover_quarter * 0.01
return_diff = return_data - np.tile(turnover_quarter,(5,1))
return_transaction_cost_final=np.product(return_diff,axis=1)

#승률
diff = return_data - np.tile(kospi_quarter,(5,1))
column_lengh = len(diff.columns)
diff = diff>0
#true == 1 , False == 0 으로 판단하기 때문에 다 더하면 가장 끝 column에 0보다 큰 것들 갯수가 남음
diff = diff.cumsum(axis=1)
win_rate = diff[column_lengh-1]/column_lengh


#섹터별 비중구하기 마지막
#초기 기준이 되는 full index 설정
sector_data = sector_data.iloc[0:10,:]
#종목수가 적어지면 ex) 25개  섹터 10개중에서 안걸리는 섹터가 있어서 index 10개를 직접 썻다.
sector_data_temp = sector_data.set_index([['IT','건강관리','경기관련소비재','금융','산업재','소재','에너지','유틸리티','필수소비재','전기통신서비스']])
#초기값 설정
sector_data_count = sector_data_temp.iloc[0:10,1]
sector_data_sum = np.sum(sector_data_count)
sector_data_count = sector_data_count/sector_data_sum

#sector 저장
for n in range(1,65):
    sector_data_temp = sector_data.set_index([2*(n)],drop=False)
    sector_row_lengh = len(sector_data[2*n][sector_data[2*n].notnull()])
    sector_data_count = pd.concat([sector_data_count,sector_data_temp.iloc[0:sector_row_lengh,2*(n)+1]],axis=1)
    sector_data_sum = np.sum(sector_data_count[2*n+1],axis=0)
    sector_data_count[2*n+1] = sector_data_count[2*n+1]/sector_data_sum
    
#그룹별 비중구하기 마지막
#초기 기준이 되는 full 그룹 설정
group_data_temp = group_data.set_index([0],drop=False)
#초기값 설정
group_data_count = group_data_temp.iloc[0:4,1]
group_data_sum = np.sum(group_data_count)
group_data_count = group_data_count/group_data_sum

#group 저장
for n in range(1,65):
    group_data_temp = group_data.set_index([2*(n)],drop=False)
    group_row_lengh = len(group_data[2*n][group_data[2*n].notnull()])
    group_data_count = pd.concat([group_data_count,group_data_temp.iloc[0:group_row_lengh,2*(n)+1]],axis=1)
    group_data_sum = np.sum(group_data_count[2*n+1],axis=0)
    group_data_count[2*n+1] = group_data_count[2*n+1]/group_data_sum
    
        

#포트폴리오 연말현금배당수익률 : 한글과 숫자 columns 다 있어도 np.sum 하면 숫자만 알아서 됨
for i in range(0,16):
    portfolio_cash_rtn[i] =   np.sum(result_cash[2*i+1])/len(result_cash[2*i+1][result_cash[2*i+1].notnull()])

