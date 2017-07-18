# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 08:29:11 2017

@author: SH-NoteBook
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:27:12 2017

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
#sector_mid = pd.read_excel('msci_rawdata_kospi_25811.xlsm',sheetname='섹터중분류1',header=None)
#sector_mid.to_pickle('sector_mid')
#sector_mid_kq = pd.read_excel('msci_rawdata_kosdaq_25811.xlsm',sheetname='섹터중분류1',header=None)
#sector_mid_kq.to_pickle('sector_mid_kq')
#sector_mid_rtn_month = pd.read_excel('wics 중분류 모멘텀 수익률.xlsx',sheetname='월별수익률1',header=None)
#sector_mid_rtn_month.to_pickle('sector_mid_rtn_month')



kospi_quarter = pd.read_pickle('kospi_quarter')
raw_data = pd.read_pickle('raw_data')
size = pd.read_pickle('size')  #시가총액
ni = pd.read_pickle('ni') # 당기순이익
ni_12m_fw = pd.read_pickle('ni_12m_fw')
rtn = pd.read_pickle('rtn')
equity = pd.read_pickle('equity') #자본총계
cash_div = pd.read_pickle('cash_div')
size_FIF_wisefn=pd.read_pickle('size_FIF_wisefn') #시가총액
sector=pd.read_pickle('sector_mid')   #중분류
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
sector_kq=pd.read_pickle('sector_mid_kq') #중분류
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
#섹터 월별수익률
#sector_rtn_month = pd.read_pickle('sector_rtn_month')
sector_mid_rtn_month = pd.read_pickle('sector_mid_rtn_month')
z=0 #연말현금배당수익률을 저장하기 위해 ... 아래 if문있음
for n in range(3,68):
    #66마지막 분기

        
        
    data_big = raw_data_sum[(raw_data_sum[n] == 1)]
    data_big = data_big.loc[:,[1,n]]
    #ni_12m_fw_sum 쓰면 fwd per, 그냥 ni_sum 쓰면 trailing
    #rtn_sum은 그냥, rtn_dividend_sum은 배당고려 수익률
    data = pd.concat([data_big, size_FIF_wisefn_sum[n], equity_sum[n], ni_12m_fw_sum[n],cash_div_sum[n],size_sum[n],rtn_sum[n-3],sector_sum[n],rtn_month_sum[3*(n-3)],rtn_month_sum[3*(n-3)+1],rtn_month_sum[3*(n-3)+2]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size_FIF_wisefn','equity','ni_12fw','cash_div','size','return','sector','return_month1','return_month2','return_month3']
    #섹터 시가총액 비중으로 z_score 종목 짜르기 위해서 섹터별 시총 합 계산
    data_sector_ratio = data.groupby('sector').sum()
    #각 섹터별 시가총액비중을 구하는데  ceil 로 올림해서 구한다음 아래에서 rank로 짜르지뭐
    data_sector_ratio = data_sector_ratio.assign(sector_ratio = np.ceil(data_sector_ratio['size']*100/np.sum(data_sector_ratio['size'])))
    data=data[data['size']>100000000000]
    #상폐, 지주사전환, 분할상장 때문에 생기는 수익률 0 제거
    data=data[data['return']!=0]
    result_temp = data
    samsung = pd.DataFrame(data.loc[390,:]).transpose()

    data = data[data['equity'].notnull()]
    data = data[data['ni_12fw'].notnull()]
    data = data[data['cash_div'].notnull()]
    
    for i in range(1,30):
        locals()['result_{}'.format(i)] = pd.DataFrame(np.zeros((200,19)))

    
    
    a=1
        #에너지 섹터
    if (np.sum(data['sector']=='에너지')>0):
        data_에너지 = data[data['sector']=='에너지']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
#        data_에너지['size_FIF_wisefn']=data_에너지['size_FIF_wisefn']/1000    #size 단위 thousand
        data_에너지.loc[:,'size_FIF_wisefn']=data_에너지.loc[:,'size_FIF_wisefn']/1000        
        data_에너지['1/pbr']=data_에너지['equity']/data_에너지['size']
        data_에너지['1/per']=data_에너지['ni_12fw']/data_에너지['size']
        data_에너지['div_yield']=data_에너지['cash_div']/data_에너지['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_에너지 = data_에너지.replace([np.inf, -np.inf],np.nan)  
        
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
        locals()['result_{}'.format(a)] =result_에너지[result_에너지['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['에너지','sector_ratio']),:]
        
        a=a+1
        
    #소재 섹터
    if (np.sum(data['sector']=='소재')>0):
        data_소재 = data[data['sector']=='소재']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_소재['size_FIF_wisefn']=data_소재['size_FIF_wisefn']/1000    #size 단위 thousand
        data_소재['1/pbr']=data_소재['equity']/data_소재['size']
        data_소재['1/per']=data_소재['ni_12fw']/data_소재['size']
        data_소재['div_yield']=data_소재['cash_div']/data_소재['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_소재 = data_소재.replace([np.inf, -np.inf],np.nan)  
        
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
        locals()['result_{}'.format(a)] =result_소재[result_소재['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['소재','sector_ratio']),:]
        
        
        a=a+1    
    #자본재 섹터
    if (np.sum(data['sector']=='자본재')>0):
        data_자본재 = data[data['sector']=="자본재"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_자본재['size_FIF_wisefn']=data_자본재['size_FIF_wisefn']/1000    #size 단위 thousand
        data_자본재['1/pbr']=data_자본재['equity']/data_자본재['size']
        data_자본재['1/per']=data_자본재['ni_12fw']/data_자본재['size']
        data_자본재['div_yield']=data_자본재['cash_div']/data_자본재['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_자본재 = data_자본재.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_자본재_per = data_자본재[data_자본재['1/per'].notnull()]
        data_자본재_pbr = data_자본재[data_자본재['1/pbr'].notnull()]
        data_자본재_div = data_자본재[data_자본재['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_자본재_pbr_cap = np.sum(data_자본재_pbr['size_FIF_wisefn'])
        data_자본재_per_cap = np.sum(data_자본재_per['size_FIF_wisefn'])
        data_자본재_div_cap = np.sum(data_자본재_div['size_FIF_wisefn'])
    
        data_자본재_pbr = data_자본재_pbr.assign(market_weight=data_자본재_pbr['size_FIF_wisefn']/data_자본재_pbr_cap)
        data_자본재_per = data_자본재_per.assign(market_weight=data_자본재_per['size_FIF_wisefn']/data_자본재_per_cap)
        data_자본재_div = data_자본재_div.assign(market_weight=data_자본재_div['size_FIF_wisefn']/data_자본재_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_자본재_pbr['1/pbr']*data_자본재_pbr['market_weight'])
        mu_inv_per=np.sum(data_자본재_per['1/per']*data_자본재_per['market_weight'])
        mu_inv_div=np.sum(data_자본재_div['div_yield']*data_자본재_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_자본재_pbr['1/pbr']-mu_inv_pbr)*data_자본재_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_자본재_per['1/per']-mu_inv_per)*data_자본재_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_자본재_div['div_yield']-mu_inv_div)*data_자본재_div['market_weight']))
        
        data_자본재1=(data_자본재_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_자본재1.name= 'pbr_z'
        data_자본재2=(data_자본재_per['1/per']-mu_inv_per)/std_inv_per
        data_자본재2.name= 'per_z'
        data_자본재3=(data_자본재_div['div_yield']-mu_inv_div)/std_inv_div
        data_자본재3.name= 'div_z'
              
        result_자본재 = pd.concat([data_자본재, data_자본재1, data_자본재2, data_자본재3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_자본재 = result_자본재.assign(z_score=np.nanmean(result_자본재.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_자본재[result_자본재['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['자본재','sector_ratio']),:]
        a=a+1
    
    
    #상업서비스와공급품 섹터
    if (np.sum(data['sector']=='상업서비스와공급품')>0):
        data_상업서비스와공급품 = data[data['sector']=='상업서비스와공급품']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_상업서비스와공급품['size_FIF_wisefn']=data_상업서비스와공급품['size_FIF_wisefn']/1000    #size 단위 thousand
        data_상업서비스와공급품['1/pbr']=data_상업서비스와공급품['equity']/data_상업서비스와공급품['size']
        data_상업서비스와공급품['1/per']=data_상업서비스와공급품['ni_12fw']/data_상업서비스와공급품['size']
        data_상업서비스와공급품['div_yield']=data_상업서비스와공급품['cash_div']/data_상업서비스와공급품['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_상업서비스와공급품 = data_상업서비스와공급품.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_상업서비스와공급품_per = data_상업서비스와공급품[data_상업서비스와공급품['1/per'].notnull()]
        data_상업서비스와공급품_pbr = data_상업서비스와공급품[data_상업서비스와공급품['1/pbr'].notnull()]
        data_상업서비스와공급품_div = data_상업서비스와공급품[data_상업서비스와공급품['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_상업서비스와공급품_pbr_cap = np.sum(data_상업서비스와공급품_pbr['size_FIF_wisefn'])
        data_상업서비스와공급품_per_cap = np.sum(data_상업서비스와공급품_per['size_FIF_wisefn'])
        data_상업서비스와공급품_div_cap = np.sum(data_상업서비스와공급품_div['size_FIF_wisefn'])
    
        data_상업서비스와공급품_pbr = data_상업서비스와공급품_pbr.assign(market_weight=data_상업서비스와공급품_pbr['size_FIF_wisefn']/data_상업서비스와공급품_pbr_cap)
        data_상업서비스와공급품_per = data_상업서비스와공급품_per.assign(market_weight=data_상업서비스와공급품_per['size_FIF_wisefn']/data_상업서비스와공급품_per_cap)
        data_상업서비스와공급품_div = data_상업서비스와공급품_div.assign(market_weight=data_상업서비스와공급품_div['size_FIF_wisefn']/data_상업서비스와공급품_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_상업서비스와공급품_pbr['1/pbr']*data_상업서비스와공급품_pbr['market_weight'])
        mu_inv_per=np.sum(data_상업서비스와공급품_per['1/per']*data_상업서비스와공급품_per['market_weight'])
        mu_inv_div=np.sum(data_상업서비스와공급품_div['div_yield']*data_상업서비스와공급품_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_상업서비스와공급품_pbr['1/pbr']-mu_inv_pbr)*data_상업서비스와공급품_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_상업서비스와공급품_per['1/per']-mu_inv_per)*data_상업서비스와공급품_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_상업서비스와공급품_div['div_yield']-mu_inv_div)*data_상업서비스와공급품_div['market_weight']))
        
        data_상업서비스와공급품1=(data_상업서비스와공급품_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_상업서비스와공급품1.name= 'pbr_z'
        data_상업서비스와공급품2=(data_상업서비스와공급품_per['1/per']-mu_inv_per)/std_inv_per
        data_상업서비스와공급품2.name= 'per_z'
        data_상업서비스와공급품3=(data_상업서비스와공급품_div['div_yield']-mu_inv_div)/std_inv_div
        data_상업서비스와공급품3.name= 'div_z'
              
        result_상업서비스와공급품 = pd.concat([data_상업서비스와공급품, data_상업서비스와공급품1, data_상업서비스와공급품2, data_상업서비스와공급품3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_상업서비스와공급품 = result_상업서비스와공급품.assign(z_score=np.nanmean(result_상업서비스와공급품.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_상업서비스와공급품[result_상업서비스와공급품['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['상업서비스와공급품','sector_ratio']),:]
        
        
        
        a=a+1
       #운송 섹터
    if (np.sum(data['sector']=='운송')>0):
        data_운송 = data[data['sector']=='운송']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_운송['size_FIF_wisefn']=data_운송['size_FIF_wisefn']/1000    #size 단위 thousand
        data_운송['1/pbr']=data_운송['equity']/data_운송['size']
        data_운송['1/per']=data_운송['ni_12fw']/data_운송['size']
        data_운송['div_yield']=data_운송['cash_div']/data_운송['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_운송 = data_운송.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_운송_per = data_운송[data_운송['1/per'].notnull()]
        data_운송_pbr = data_운송[data_운송['1/pbr'].notnull()]
        data_운송_div = data_운송[data_운송['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_운송_pbr_cap = np.sum(data_운송_pbr['size_FIF_wisefn'])
        data_운송_per_cap = np.sum(data_운송_per['size_FIF_wisefn'])
        data_운송_div_cap = np.sum(data_운송_div['size_FIF_wisefn'])
    
        data_운송_pbr = data_운송_pbr.assign(market_weight=data_운송_pbr['size_FIF_wisefn']/data_운송_pbr_cap)
        data_운송_per = data_운송_per.assign(market_weight=data_운송_per['size_FIF_wisefn']/data_운송_per_cap)
        data_운송_div = data_운송_div.assign(market_weight=data_운송_div['size_FIF_wisefn']/data_운송_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_운송_pbr['1/pbr']*data_운송_pbr['market_weight'])
        mu_inv_per=np.sum(data_운송_per['1/per']*data_운송_per['market_weight'])
        mu_inv_div=np.sum(data_운송_div['div_yield']*data_운송_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_운송_pbr['1/pbr']-mu_inv_pbr)*data_운송_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_운송_per['1/per']-mu_inv_per)*data_운송_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_운송_div['div_yield']-mu_inv_div)*data_운송_div['market_weight']))
        
        data_운송1=(data_운송_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_운송1.name= 'pbr_z'
        data_운송2=(data_운송_per['1/per']-mu_inv_per)/std_inv_per
        data_운송2.name= 'per_z'
        data_운송3=(data_운송_div['div_yield']-mu_inv_div)/std_inv_div
        data_운송3.name= 'div_z'
              
        result_운송 = pd.concat([data_운송, data_운송1, data_운송2, data_운송3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_운송 = result_운송.assign(z_score=np.nanmean(result_운송.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_운송[result_운송['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['운송','sector_ratio']),:]
        a=a+1
    #자동차와부품 섹터
    if (np.sum(data['sector']=='자동차와부품')>0):
        data_자동차와부품 = data[data['sector']=='자동차와부품']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_자동차와부품['size_FIF_wisefn']=data_자동차와부품['size_FIF_wisefn']/1000    #size 단위 thousand
        data_자동차와부품['1/pbr']=data_자동차와부품['equity']/data_자동차와부품['size']
        data_자동차와부품['1/per']=data_자동차와부품['ni_12fw']/data_자동차와부품['size']
        data_자동차와부품['div_yield']=data_자동차와부품['cash_div']/data_자동차와부품['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_자동차와부품 = data_자동차와부품.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_자동차와부품_per = data_자동차와부품[data_자동차와부품['1/per'].notnull()]
        data_자동차와부품_pbr = data_자동차와부품[data_자동차와부품['1/pbr'].notnull()]
        data_자동차와부품_div = data_자동차와부품[data_자동차와부품['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_자동차와부품_pbr_cap = np.sum(data_자동차와부품_pbr['size_FIF_wisefn'])
        data_자동차와부품_per_cap = np.sum(data_자동차와부품_per['size_FIF_wisefn'])
        data_자동차와부품_div_cap = np.sum(data_자동차와부품_div['size_FIF_wisefn'])
    
        data_자동차와부품_pbr = data_자동차와부품_pbr.assign(market_weight=data_자동차와부품_pbr['size_FIF_wisefn']/data_자동차와부품_pbr_cap)
        data_자동차와부품_per = data_자동차와부품_per.assign(market_weight=data_자동차와부품_per['size_FIF_wisefn']/data_자동차와부품_per_cap)
        data_자동차와부품_div = data_자동차와부품_div.assign(market_weight=data_자동차와부품_div['size_FIF_wisefn']/data_자동차와부품_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_자동차와부품_pbr['1/pbr']*data_자동차와부품_pbr['market_weight'])
        mu_inv_per=np.sum(data_자동차와부품_per['1/per']*data_자동차와부품_per['market_weight'])
        mu_inv_div=np.sum(data_자동차와부품_div['div_yield']*data_자동차와부품_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_자동차와부품_pbr['1/pbr']-mu_inv_pbr)*data_자동차와부품_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_자동차와부품_per['1/per']-mu_inv_per)*data_자동차와부품_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_자동차와부품_div['div_yield']-mu_inv_div)*data_자동차와부품_div['market_weight']))
        
        data_자동차와부품1=(data_자동차와부품_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_자동차와부품1.name= 'pbr_z'
        data_자동차와부품2=(data_자동차와부품_per['1/per']-mu_inv_per)/std_inv_per
        data_자동차와부품2.name= 'per_z'
        data_자동차와부품3=(data_자동차와부품_div['div_yield']-mu_inv_div)/std_inv_div
        data_자동차와부품3.name= 'div_z'
              
        result_자동차와부품 = pd.concat([data_자동차와부품, data_자동차와부품1, data_자동차와부품2, data_자동차와부품3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_자동차와부품 = result_자동차와부품.assign(z_score=np.nanmean(result_자동차와부품.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_자동차와부품[result_자동차와부품['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['자동차와부품','sector_ratio']),:]
        a=a+1
    #내구소비재와의류 섹터
    if (np.sum(data['sector']=='내구소비재와의류')>0):
        data_내구소비재와의류 = data[data['sector']=='내구소비재와의류']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_내구소비재와의류['size_FIF_wisefn']=data_내구소비재와의류['size_FIF_wisefn']/1000    #size 단위 thousand
        data_내구소비재와의류['1/pbr']=data_내구소비재와의류['equity']/data_내구소비재와의류['size']
        data_내구소비재와의류['1/per']=data_내구소비재와의류['ni_12fw']/data_내구소비재와의류['size']
        data_내구소비재와의류['div_yield']=data_내구소비재와의류['cash_div']/data_내구소비재와의류['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_내구소비재와의류 = data_내구소비재와의류.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_내구소비재와의류_per = data_내구소비재와의류[data_내구소비재와의류['1/per'].notnull()]
        data_내구소비재와의류_pbr = data_내구소비재와의류[data_내구소비재와의류['1/pbr'].notnull()]
        data_내구소비재와의류_div = data_내구소비재와의류[data_내구소비재와의류['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_내구소비재와의류_pbr_cap = np.sum(data_내구소비재와의류_pbr['size_FIF_wisefn'])
        data_내구소비재와의류_per_cap = np.sum(data_내구소비재와의류_per['size_FIF_wisefn'])
        data_내구소비재와의류_div_cap = np.sum(data_내구소비재와의류_div['size_FIF_wisefn'])
    
        data_내구소비재와의류_pbr = data_내구소비재와의류_pbr.assign(market_weight=data_내구소비재와의류_pbr['size_FIF_wisefn']/data_내구소비재와의류_pbr_cap)
        data_내구소비재와의류_per = data_내구소비재와의류_per.assign(market_weight=data_내구소비재와의류_per['size_FIF_wisefn']/data_내구소비재와의류_per_cap)
        data_내구소비재와의류_div = data_내구소비재와의류_div.assign(market_weight=data_내구소비재와의류_div['size_FIF_wisefn']/data_내구소비재와의류_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_내구소비재와의류_pbr['1/pbr']*data_내구소비재와의류_pbr['market_weight'])
        mu_inv_per=np.sum(data_내구소비재와의류_per['1/per']*data_내구소비재와의류_per['market_weight'])
        mu_inv_div=np.sum(data_내구소비재와의류_div['div_yield']*data_내구소비재와의류_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_내구소비재와의류_pbr['1/pbr']-mu_inv_pbr)*data_내구소비재와의류_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_내구소비재와의류_per['1/per']-mu_inv_per)*data_내구소비재와의류_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_내구소비재와의류_div['div_yield']-mu_inv_div)*data_내구소비재와의류_div['market_weight']))
        
        data_내구소비재와의류1=(data_내구소비재와의류_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_내구소비재와의류1.name= 'pbr_z'
        data_내구소비재와의류2=(data_내구소비재와의류_per['1/per']-mu_inv_per)/std_inv_per
        data_내구소비재와의류2.name= 'per_z'
        data_내구소비재와의류3=(data_내구소비재와의류_div['div_yield']-mu_inv_div)/std_inv_div
        data_내구소비재와의류3.name= 'div_z'
              
        result_내구소비재와의류 = pd.concat([data_내구소비재와의류, data_내구소비재와의류1, data_내구소비재와의류2, data_내구소비재와의류3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_내구소비재와의류 = result_내구소비재와의류.assign(z_score=np.nanmean(result_내구소비재와의류.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_내구소비재와의류[result_내구소비재와의류['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['내구소비재와의류','sector_ratio']),:]
        a=a+1
    

    #호텔_레스토랑_레저 섹터
    if (np.sum(data['sector']=='호텔,레스토랑,레저등')>0):
        data_호텔_레스토랑_레저 = data[data['sector']=='호텔,레스토랑,레저등']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_호텔_레스토랑_레저['size_FIF_wisefn']=data_호텔_레스토랑_레저['size_FIF_wisefn']/1000    #size 단위 thousand
        data_호텔_레스토랑_레저['1/pbr']=data_호텔_레스토랑_레저['equity']/data_호텔_레스토랑_레저['size']
        data_호텔_레스토랑_레저['1/per']=data_호텔_레스토랑_레저['ni_12fw']/data_호텔_레스토랑_레저['size']
        data_호텔_레스토랑_레저['div_yield']=data_호텔_레스토랑_레저['cash_div']/data_호텔_레스토랑_레저['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_호텔_레스토랑_레저 = data_호텔_레스토랑_레저.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_호텔_레스토랑_레저_per = data_호텔_레스토랑_레저[data_호텔_레스토랑_레저['1/per'].notnull()]
        data_호텔_레스토랑_레저_pbr = data_호텔_레스토랑_레저[data_호텔_레스토랑_레저['1/pbr'].notnull()]
        data_호텔_레스토랑_레저_div = data_호텔_레스토랑_레저[data_호텔_레스토랑_레저['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_호텔_레스토랑_레저_pbr_cap = np.sum(data_호텔_레스토랑_레저_pbr['size_FIF_wisefn'])
        data_호텔_레스토랑_레저_per_cap = np.sum(data_호텔_레스토랑_레저_per['size_FIF_wisefn'])
        data_호텔_레스토랑_레저_div_cap = np.sum(data_호텔_레스토랑_레저_div['size_FIF_wisefn'])
    
        data_호텔_레스토랑_레저_pbr = data_호텔_레스토랑_레저_pbr.assign(market_weight=data_호텔_레스토랑_레저_pbr['size_FIF_wisefn']/data_호텔_레스토랑_레저_pbr_cap)
        data_호텔_레스토랑_레저_per = data_호텔_레스토랑_레저_per.assign(market_weight=data_호텔_레스토랑_레저_per['size_FIF_wisefn']/data_호텔_레스토랑_레저_per_cap)
        data_호텔_레스토랑_레저_div = data_호텔_레스토랑_레저_div.assign(market_weight=data_호텔_레스토랑_레저_div['size_FIF_wisefn']/data_호텔_레스토랑_레저_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_호텔_레스토랑_레저_pbr['1/pbr']*data_호텔_레스토랑_레저_pbr['market_weight'])
        mu_inv_per=np.sum(data_호텔_레스토랑_레저_per['1/per']*data_호텔_레스토랑_레저_per['market_weight'])
        mu_inv_div=np.sum(data_호텔_레스토랑_레저_div['div_yield']*data_호텔_레스토랑_레저_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_호텔_레스토랑_레저_pbr['1/pbr']-mu_inv_pbr)*data_호텔_레스토랑_레저_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_호텔_레스토랑_레저_per['1/per']-mu_inv_per)*data_호텔_레스토랑_레저_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_호텔_레스토랑_레저_div['div_yield']-mu_inv_div)*data_호텔_레스토랑_레저_div['market_weight']))
        
        data_호텔_레스토랑_레저1=(data_호텔_레스토랑_레저_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_호텔_레스토랑_레저1.name= 'pbr_z'
        data_호텔_레스토랑_레저2=(data_호텔_레스토랑_레저_per['1/per']-mu_inv_per)/std_inv_per
        data_호텔_레스토랑_레저2.name= 'per_z'
        data_호텔_레스토랑_레저3=(data_호텔_레스토랑_레저_div['div_yield']-mu_inv_div)/std_inv_div
        data_호텔_레스토랑_레저3.name= 'div_z'
              
        result_호텔_레스토랑_레저 = pd.concat([data_호텔_레스토랑_레저, data_호텔_레스토랑_레저1, data_호텔_레스토랑_레저2, data_호텔_레스토랑_레저3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_호텔_레스토랑_레저 = result_호텔_레스토랑_레저.assign(z_score=np.nanmean(result_호텔_레스토랑_레저.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_호텔_레스토랑_레저[result_호텔_레스토랑_레저['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['호텔,레스토랑,레저등','sector_ratio']),:]
        a=a+1
    #미디어 섹터
    if (np.sum(data['sector']=='미디어')>0):
        data_미디어 = data[data['sector']=='미디어']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_미디어['size_FIF_wisefn']=data_미디어['size_FIF_wisefn']/1000    #size 단위 thousand
        data_미디어['1/pbr']=data_미디어['equity']/data_미디어['size']
        data_미디어['1/per']=data_미디어['ni_12fw']/data_미디어['size']
        data_미디어['div_yield']=data_미디어['cash_div']/data_미디어['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_미디어 = data_미디어.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_미디어_per = data_미디어[data_미디어['1/per'].notnull()]
        data_미디어_pbr = data_미디어[data_미디어['1/pbr'].notnull()]
        data_미디어_div = data_미디어[data_미디어['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_미디어_pbr_cap = np.sum(data_미디어_pbr['size_FIF_wisefn'])
        data_미디어_per_cap = np.sum(data_미디어_per['size_FIF_wisefn'])
        data_미디어_div_cap = np.sum(data_미디어_div['size_FIF_wisefn'])
    
        data_미디어_pbr = data_미디어_pbr.assign(market_weight=data_미디어_pbr['size_FIF_wisefn']/data_미디어_pbr_cap)
        data_미디어_per = data_미디어_per.assign(market_weight=data_미디어_per['size_FIF_wisefn']/data_미디어_per_cap)
        data_미디어_div = data_미디어_div.assign(market_weight=data_미디어_div['size_FIF_wisefn']/data_미디어_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_미디어_pbr['1/pbr']*data_미디어_pbr['market_weight'])
        mu_inv_per=np.sum(data_미디어_per['1/per']*data_미디어_per['market_weight'])
        mu_inv_div=np.sum(data_미디어_div['div_yield']*data_미디어_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_미디어_pbr['1/pbr']-mu_inv_pbr)*data_미디어_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_미디어_per['1/per']-mu_inv_per)*data_미디어_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_미디어_div['div_yield']-mu_inv_div)*data_미디어_div['market_weight']))
        
        data_미디어1=(data_미디어_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_미디어1.name= 'pbr_z'
        data_미디어2=(data_미디어_per['1/per']-mu_inv_per)/std_inv_per
        data_미디어2.name= 'per_z'
        data_미디어3=(data_미디어_div['div_yield']-mu_inv_div)/std_inv_div
        data_미디어3.name= 'div_z'
              
        result_미디어 = pd.concat([data_미디어, data_미디어1, data_미디어2, data_미디어3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_미디어 = result_미디어.assign(z_score=np.nanmean(result_미디어.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_미디어[result_미디어['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['미디어','sector_ratio']),:]
        
        a=a+1
    #소매(유통) 섹터
    if (np.sum(data['sector']=='소매(유통)')>0):
        data_소매_유통 = data[data['sector']=='소매(유통)']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_소매_유통['size_FIF_wisefn']=data_소매_유통['size_FIF_wisefn']/1000    #size 단위 thousand
        data_소매_유통['1/pbr']=data_소매_유통['equity']/data_소매_유통['size']
        data_소매_유통['1/per']=data_소매_유통['ni_12fw']/data_소매_유통['size']
        data_소매_유통['div_yield']=data_소매_유통['cash_div']/data_소매_유통['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_소매_유통 = data_소매_유통.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_소매_유통_per = data_소매_유통[data_소매_유통['1/per'].notnull()]
        data_소매_유통_pbr = data_소매_유통[data_소매_유통['1/pbr'].notnull()]
        data_소매_유통_div = data_소매_유통[data_소매_유통['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_소매_유통_pbr_cap = np.sum(data_소매_유통_pbr['size_FIF_wisefn'])
        data_소매_유통_per_cap = np.sum(data_소매_유통_per['size_FIF_wisefn'])
        data_소매_유통_div_cap = np.sum(data_소매_유통_div['size_FIF_wisefn'])
    
        data_소매_유통_pbr = data_소매_유통_pbr.assign(market_weight=data_소매_유통_pbr['size_FIF_wisefn']/data_소매_유통_pbr_cap)
        data_소매_유통_per = data_소매_유통_per.assign(market_weight=data_소매_유통_per['size_FIF_wisefn']/data_소매_유통_per_cap)
        data_소매_유통_div = data_소매_유통_div.assign(market_weight=data_소매_유통_div['size_FIF_wisefn']/data_소매_유통_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_소매_유통_pbr['1/pbr']*data_소매_유통_pbr['market_weight'])
        mu_inv_per=np.sum(data_소매_유통_per['1/per']*data_소매_유통_per['market_weight'])
        mu_inv_div=np.sum(data_소매_유통_div['div_yield']*data_소매_유통_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_소매_유통_pbr['1/pbr']-mu_inv_pbr)*data_소매_유통_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_소매_유통_per['1/per']-mu_inv_per)*data_소매_유통_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_소매_유통_div['div_yield']-mu_inv_div)*data_소매_유통_div['market_weight']))
        
        data_소매_유통1=(data_소매_유통_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_소매_유통1.name= 'pbr_z'
        data_소매_유통2=(data_소매_유통_per['1/per']-mu_inv_per)/std_inv_per
        data_소매_유통2.name= 'per_z'
        data_소매_유통3=(data_소매_유통_div['div_yield']-mu_inv_div)/std_inv_div
        data_소매_유통3.name= 'div_z'
              
        result_소매_유통 = pd.concat([data_소매_유통, data_소매_유통1, data_소매_유통2, data_소매_유통3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_소매_유통 = result_소매_유통.assign(z_score=np.nanmean(result_소매_유통.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_소매_유통[result_소매_유통['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['소매(유통)','sector_ratio']),:]
        
        a=a+1
        
     #교육서비스 섹터
    if (np.sum(data['sector']=='교육서비스')>0):
        data_교육서비스 = data[data['sector']=="교육서비스"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_교육서비스['size_FIF_wisefn']=data_교육서비스['size_FIF_wisefn']/1000    #size 단위 thousand
        data_교육서비스['1/pbr']=data_교육서비스['equity']/data_교육서비스['size']
        data_교육서비스['1/per']=data_교육서비스['ni_12fw']/data_교육서비스['size']
        data_교육서비스['div_yield']=data_교육서비스['cash_div']/data_교육서비스['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_교육서비스 = data_교육서비스.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_교육서비스_per = data_교육서비스[data_교육서비스['1/per'].notnull()]
        data_교육서비스_pbr = data_교육서비스[data_교육서비스['1/pbr'].notnull()]
        data_교육서비스_div = data_교육서비스[data_교육서비스['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_교육서비스_pbr_cap = np.sum(data_교육서비스_pbr['size_FIF_wisefn'])
        data_교육서비스_per_cap = np.sum(data_교육서비스_per['size_FIF_wisefn'])
        data_교육서비스_div_cap = np.sum(data_교육서비스_div['size_FIF_wisefn'])
    
        data_교육서비스_pbr = data_교육서비스_pbr.assign(market_weight=data_교육서비스_pbr['size_FIF_wisefn']/data_교육서비스_pbr_cap)
        data_교육서비스_per = data_교육서비스_per.assign(market_weight=data_교육서비스_per['size_FIF_wisefn']/data_교육서비스_per_cap)
        data_교육서비스_div = data_교육서비스_div.assign(market_weight=data_교육서비스_div['size_FIF_wisefn']/data_교육서비스_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_교육서비스_pbr['1/pbr']*data_교육서비스_pbr['market_weight'])
        mu_inv_per=np.sum(data_교육서비스_per['1/per']*data_교육서비스_per['market_weight'])
        mu_inv_div=np.sum(data_교육서비스_div['div_yield']*data_교육서비스_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_교육서비스_pbr['1/pbr']-mu_inv_pbr)*data_교육서비스_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_교육서비스_per['1/per']-mu_inv_per)*data_교육서비스_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_교육서비스_div['div_yield']-mu_inv_div)*data_교육서비스_div['market_weight']))
        
        data_교육서비스1=(data_교육서비스_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_교육서비스1.name= 'pbr_z'
        data_교육서비스2=(data_교육서비스_per['1/per']-mu_inv_per)/std_inv_per
        data_교육서비스2.name= 'per_z'
        data_교육서비스3=(data_교육서비스_div['div_yield']-mu_inv_div)/std_inv_div
        data_교육서비스3.name= 'div_z'
              
        result_교육서비스 = pd.concat([data_교육서비스, data_교육서비스1, data_교육서비스2, data_교육서비스3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_교육서비스 = result_교육서비스.assign(z_score=np.nanmean(result_교육서비스.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_교육서비스[result_교육서비스['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['교육서비스','sector_ratio']),:]
        
        a=a+1
    
     #식품과기본식료품소매 섹터
    if (np.sum(data['sector']=='식품과기본식료품소매')>0):
        data_식품과기본식료품소매 = data[data['sector']=="식품과기본식료품소매"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_식품과기본식료품소매['size_FIF_wisefn']=data_식품과기본식료품소매['size_FIF_wisefn']/1000    #size 단위 thousand
        data_식품과기본식료품소매['1/pbr']=data_식품과기본식료품소매['equity']/data_식품과기본식료품소매['size']
        data_식품과기본식료품소매['1/per']=data_식품과기본식료품소매['ni_12fw']/data_식품과기본식료품소매['size']
        data_식품과기본식료품소매['div_yield']=data_식품과기본식료품소매['cash_div']/data_식품과기본식료품소매['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_식품과기본식료품소매 = data_식품과기본식료품소매.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_식품과기본식료품소매_per = data_식품과기본식료품소매[data_식품과기본식료품소매['1/per'].notnull()]
        data_식품과기본식료품소매_pbr = data_식품과기본식료품소매[data_식품과기본식료품소매['1/pbr'].notnull()]
        data_식품과기본식료품소매_div = data_식품과기본식료품소매[data_식품과기본식료품소매['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_식품과기본식료품소매_pbr_cap = np.sum(data_식품과기본식료품소매_pbr['size_FIF_wisefn'])
        data_식품과기본식료품소매_per_cap = np.sum(data_식품과기본식료품소매_per['size_FIF_wisefn'])
        data_식품과기본식료품소매_div_cap = np.sum(data_식품과기본식료품소매_div['size_FIF_wisefn'])
    
        data_식품과기본식료품소매_pbr = data_식품과기본식료품소매_pbr.assign(market_weight=data_식품과기본식료품소매_pbr['size_FIF_wisefn']/data_식품과기본식료품소매_pbr_cap)
        data_식품과기본식료품소매_per = data_식품과기본식료품소매_per.assign(market_weight=data_식품과기본식료품소매_per['size_FIF_wisefn']/data_식품과기본식료품소매_per_cap)
        data_식품과기본식료품소매_div = data_식품과기본식료품소매_div.assign(market_weight=data_식품과기본식료품소매_div['size_FIF_wisefn']/data_식품과기본식료품소매_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_식품과기본식료품소매_pbr['1/pbr']*data_식품과기본식료품소매_pbr['market_weight'])
        mu_inv_per=np.sum(data_식품과기본식료품소매_per['1/per']*data_식품과기본식료품소매_per['market_weight'])
        mu_inv_div=np.sum(data_식품과기본식료품소매_div['div_yield']*data_식품과기본식료품소매_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_식품과기본식료품소매_pbr['1/pbr']-mu_inv_pbr)*data_식품과기본식료품소매_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_식품과기본식료품소매_per['1/per']-mu_inv_per)*data_식품과기본식료품소매_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_식품과기본식료품소매_div['div_yield']-mu_inv_div)*data_식품과기본식료품소매_div['market_weight']))
        
        data_식품과기본식료품소매1=(data_식품과기본식료품소매_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_식품과기본식료품소매1.name= 'pbr_z'
        data_식품과기본식료품소매2=(data_식품과기본식료품소매_per['1/per']-mu_inv_per)/std_inv_per
        data_식품과기본식료품소매2.name= 'per_z'
        data_식품과기본식료품소매3=(data_식품과기본식료품소매_div['div_yield']-mu_inv_div)/std_inv_div
        data_식품과기본식료품소매3.name= 'div_z'
              
        result_식품과기본식료품소매 = pd.concat([data_식품과기본식료품소매, data_식품과기본식료품소매1, data_식품과기본식료품소매2, data_식품과기본식료품소매3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_식품과기본식료품소매 = result_식품과기본식료품소매.assign(z_score=np.nanmean(result_식품과기본식료품소매.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_식품과기본식료품소매[result_식품과기본식료품소매['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['식품과기본식료품소매','sector_ratio']),:]
        
        a=a+1
    
     #식품,음료,담배 섹터
    if (np.sum(data['sector']=='식품,음료,담배')>0):
        data_식품_음료_담배 = data[data['sector']=="식품,음료,담배"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_식품_음료_담배['size_FIF_wisefn']=data_식품_음료_담배['size_FIF_wisefn']/1000    #size 단위 thousand
        data_식품_음료_담배['1/pbr']=data_식품_음료_담배['equity']/data_식품_음료_담배['size']
        data_식품_음료_담배['1/per']=data_식품_음료_담배['ni_12fw']/data_식품_음료_담배['size']
        data_식품_음료_담배['div_yield']=data_식품_음료_담배['cash_div']/data_식품_음료_담배['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_식품_음료_담배 = data_식품_음료_담배.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_식품_음료_담배_per = data_식품_음료_담배[data_식품_음료_담배['1/per'].notnull()]
        data_식품_음료_담배_pbr = data_식품_음료_담배[data_식품_음료_담배['1/pbr'].notnull()]
        data_식품_음료_담배_div = data_식품_음료_담배[data_식품_음료_담배['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_식품_음료_담배_pbr_cap = np.sum(data_식품_음료_담배_pbr['size_FIF_wisefn'])
        data_식품_음료_담배_per_cap = np.sum(data_식품_음료_담배_per['size_FIF_wisefn'])
        data_식품_음료_담배_div_cap = np.sum(data_식품_음료_담배_div['size_FIF_wisefn'])
    
        data_식품_음료_담배_pbr = data_식품_음료_담배_pbr.assign(market_weight=data_식품_음료_담배_pbr['size_FIF_wisefn']/data_식품_음료_담배_pbr_cap)
        data_식품_음료_담배_per = data_식품_음료_담배_per.assign(market_weight=data_식품_음료_담배_per['size_FIF_wisefn']/data_식품_음료_담배_per_cap)
        data_식품_음료_담배_div = data_식품_음료_담배_div.assign(market_weight=data_식품_음료_담배_div['size_FIF_wisefn']/data_식품_음료_담배_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_식품_음료_담배_pbr['1/pbr']*data_식품_음료_담배_pbr['market_weight'])
        mu_inv_per=np.sum(data_식품_음료_담배_per['1/per']*data_식품_음료_담배_per['market_weight'])
        mu_inv_div=np.sum(data_식품_음료_담배_div['div_yield']*data_식품_음료_담배_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_식품_음료_담배_pbr['1/pbr']-mu_inv_pbr)*data_식품_음료_담배_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_식품_음료_담배_per['1/per']-mu_inv_per)*data_식품_음료_담배_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_식품_음료_담배_div['div_yield']-mu_inv_div)*data_식품_음료_담배_div['market_weight']))
        
        data_식품_음료_담배1=(data_식품_음료_담배_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_식품_음료_담배1.name= 'pbr_z'
        data_식품_음료_담배2=(data_식품_음료_담배_per['1/per']-mu_inv_per)/std_inv_per
        data_식품_음료_담배2.name= 'per_z'
        data_식품_음료_담배3=(data_식품_음료_담배_div['div_yield']-mu_inv_div)/std_inv_div
        data_식품_음료_담배3.name= 'div_z'
              
        result_식품_음료_담배 = pd.concat([data_식품_음료_담배, data_식품_음료_담배1, data_식품_음료_담배2, data_식품_음료_담배3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_식품_음료_담배 = result_식품_음료_담배.assign(z_score=np.nanmean(result_식품_음료_담배.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_식품_음료_담배[result_식품_음료_담배['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['식품,음료,담배','sector_ratio']),:]
        a=a+1
    
     #가정용품과개인용품 섹터
    if (np.sum(data['sector']=='가정용품과개인용품')>0):
        data_가정용품과개인용품 = data[data['sector']=="가정용품과개인용품"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_가정용품과개인용품['size_FIF_wisefn']=data_가정용품과개인용품['size_FIF_wisefn']/1000    #size 단위 thousand
        data_가정용품과개인용품['1/pbr']=data_가정용품과개인용품['equity']/data_가정용품과개인용품['size']
        data_가정용품과개인용품['1/per']=data_가정용품과개인용품['ni_12fw']/data_가정용품과개인용품['size']
        data_가정용품과개인용품['div_yield']=data_가정용품과개인용품['cash_div']/data_가정용품과개인용품['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_가정용품과개인용품 = data_가정용품과개인용품.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_가정용품과개인용품_per = data_가정용품과개인용품[data_가정용품과개인용품['1/per'].notnull()]
        data_가정용품과개인용품_pbr = data_가정용품과개인용품[data_가정용품과개인용품['1/pbr'].notnull()]
        data_가정용품과개인용품_div = data_가정용품과개인용품[data_가정용품과개인용품['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_가정용품과개인용품_pbr_cap = np.sum(data_가정용품과개인용품_pbr['size_FIF_wisefn'])
        data_가정용품과개인용품_per_cap = np.sum(data_가정용품과개인용품_per['size_FIF_wisefn'])
        data_가정용품과개인용품_div_cap = np.sum(data_가정용품과개인용품_div['size_FIF_wisefn'])
    
        data_가정용품과개인용품_pbr = data_가정용품과개인용품_pbr.assign(market_weight=data_가정용품과개인용품_pbr['size_FIF_wisefn']/data_가정용품과개인용품_pbr_cap)
        data_가정용품과개인용품_per = data_가정용품과개인용품_per.assign(market_weight=data_가정용품과개인용품_per['size_FIF_wisefn']/data_가정용품과개인용품_per_cap)
        data_가정용품과개인용품_div = data_가정용품과개인용품_div.assign(market_weight=data_가정용품과개인용품_div['size_FIF_wisefn']/data_가정용품과개인용품_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_가정용품과개인용품_pbr['1/pbr']*data_가정용품과개인용품_pbr['market_weight'])
        mu_inv_per=np.sum(data_가정용품과개인용품_per['1/per']*data_가정용품과개인용품_per['market_weight'])
        mu_inv_div=np.sum(data_가정용품과개인용품_div['div_yield']*data_가정용품과개인용품_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_가정용품과개인용품_pbr['1/pbr']-mu_inv_pbr)*data_가정용품과개인용품_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_가정용품과개인용품_per['1/per']-mu_inv_per)*data_가정용품과개인용품_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_가정용품과개인용품_div['div_yield']-mu_inv_div)*data_가정용품과개인용품_div['market_weight']))
        
        data_가정용품과개인용품1=(data_가정용품과개인용품_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_가정용품과개인용품1.name= 'pbr_z'
        data_가정용품과개인용품2=(data_가정용품과개인용품_per['1/per']-mu_inv_per)/std_inv_per
        data_가정용품과개인용품2.name= 'per_z'
        data_가정용품과개인용품3=(data_가정용품과개인용품_div['div_yield']-mu_inv_div)/std_inv_div
        data_가정용품과개인용품3.name= 'div_z'
              
        result_가정용품과개인용품 = pd.concat([data_가정용품과개인용품, data_가정용품과개인용품1, data_가정용품과개인용품2, data_가정용품과개인용품3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_가정용품과개인용품 = result_가정용품과개인용품.assign(z_score=np.nanmean(result_가정용품과개인용품.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_가정용품과개인용품[result_가정용품과개인용품['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['가정용품과개인용품','sector_ratio']),:]
        
        a=a+1
    
     #건강관리장비와서비스 섹터
    if (np.sum(data['sector']=='건강관리장비와서비스')>0):
        data_건강관리장비와서비스 = data[data['sector']=="건강관리장비와서비스"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_건강관리장비와서비스['size_FIF_wisefn']=data_건강관리장비와서비스['size_FIF_wisefn']/1000    #size 단위 thousand
        data_건강관리장비와서비스['1/pbr']=data_건강관리장비와서비스['equity']/data_건강관리장비와서비스['size']
        data_건강관리장비와서비스['1/per']=data_건강관리장비와서비스['ni_12fw']/data_건강관리장비와서비스['size']
        data_건강관리장비와서비스['div_yield']=data_건강관리장비와서비스['cash_div']/data_건강관리장비와서비스['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_건강관리장비와서비스 = data_건강관리장비와서비스.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_건강관리장비와서비스_per = data_건강관리장비와서비스[data_건강관리장비와서비스['1/per'].notnull()]
        data_건강관리장비와서비스_pbr = data_건강관리장비와서비스[data_건강관리장비와서비스['1/pbr'].notnull()]
        data_건강관리장비와서비스_div = data_건강관리장비와서비스[data_건강관리장비와서비스['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_건강관리장비와서비스_pbr_cap = np.sum(data_건강관리장비와서비스_pbr['size_FIF_wisefn'])
        data_건강관리장비와서비스_per_cap = np.sum(data_건강관리장비와서비스_per['size_FIF_wisefn'])
        data_건강관리장비와서비스_div_cap = np.sum(data_건강관리장비와서비스_div['size_FIF_wisefn'])
    
        data_건강관리장비와서비스_pbr = data_건강관리장비와서비스_pbr.assign(market_weight=data_건강관리장비와서비스_pbr['size_FIF_wisefn']/data_건강관리장비와서비스_pbr_cap)
        data_건강관리장비와서비스_per = data_건강관리장비와서비스_per.assign(market_weight=data_건강관리장비와서비스_per['size_FIF_wisefn']/data_건강관리장비와서비스_per_cap)
        data_건강관리장비와서비스_div = data_건강관리장비와서비스_div.assign(market_weight=data_건강관리장비와서비스_div['size_FIF_wisefn']/data_건강관리장비와서비스_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_건강관리장비와서비스_pbr['1/pbr']*data_건강관리장비와서비스_pbr['market_weight'])
        mu_inv_per=np.sum(data_건강관리장비와서비스_per['1/per']*data_건강관리장비와서비스_per['market_weight'])
        mu_inv_div=np.sum(data_건강관리장비와서비스_div['div_yield']*data_건강관리장비와서비스_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_건강관리장비와서비스_pbr['1/pbr']-mu_inv_pbr)*data_건강관리장비와서비스_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_건강관리장비와서비스_per['1/per']-mu_inv_per)*data_건강관리장비와서비스_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_건강관리장비와서비스_div['div_yield']-mu_inv_div)*data_건강관리장비와서비스_div['market_weight']))
        
        data_건강관리장비와서비스1=(data_건강관리장비와서비스_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_건강관리장비와서비스1.name= 'pbr_z'
        data_건강관리장비와서비스2=(data_건강관리장비와서비스_per['1/per']-mu_inv_per)/std_inv_per
        data_건강관리장비와서비스2.name= 'per_z'
        data_건강관리장비와서비스3=(data_건강관리장비와서비스_div['div_yield']-mu_inv_div)/std_inv_div
        data_건강관리장비와서비스3.name= 'div_z'
              
        result_건강관리장비와서비스 = pd.concat([data_건강관리장비와서비스, data_건강관리장비와서비스1, data_건강관리장비와서비스2, data_건강관리장비와서비스3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_건강관리장비와서비스 = result_건강관리장비와서비스.assign(z_score=np.nanmean(result_건강관리장비와서비스.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_건강관리장비와서비스[result_건강관리장비와서비스['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['건강관리장비와서비스','sector_ratio']),:]
        
        a=a+1
    
     #제약과생물공학 섹터
    if (np.sum(data['sector']=='제약과생물공학')>0):
        data_제약과생물공학 = data[data['sector']=="제약과생물공학"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_제약과생물공학['size_FIF_wisefn']=data_제약과생물공학['size_FIF_wisefn']/1000    #size 단위 thousand
        data_제약과생물공학['1/pbr']=data_제약과생물공학['equity']/data_제약과생물공학['size']
        data_제약과생물공학['1/per']=data_제약과생물공학['ni_12fw']/data_제약과생물공학['size']
        data_제약과생물공학['div_yield']=data_제약과생물공학['cash_div']/data_제약과생물공학['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_제약과생물공학 = data_제약과생물공학.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_제약과생물공학_per = data_제약과생물공학[data_제약과생물공학['1/per'].notnull()]
        data_제약과생물공학_pbr = data_제약과생물공학[data_제약과생물공학['1/pbr'].notnull()]
        data_제약과생물공학_div = data_제약과생물공학[data_제약과생물공학['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_제약과생물공학_pbr_cap = np.sum(data_제약과생물공학_pbr['size_FIF_wisefn'])
        data_제약과생물공학_per_cap = np.sum(data_제약과생물공학_per['size_FIF_wisefn'])
        data_제약과생물공학_div_cap = np.sum(data_제약과생물공학_div['size_FIF_wisefn'])
    
        data_제약과생물공학_pbr = data_제약과생물공학_pbr.assign(market_weight=data_제약과생물공학_pbr['size_FIF_wisefn']/data_제약과생물공학_pbr_cap)
        data_제약과생물공학_per = data_제약과생물공학_per.assign(market_weight=data_제약과생물공학_per['size_FIF_wisefn']/data_제약과생물공학_per_cap)
        data_제약과생물공학_div = data_제약과생물공학_div.assign(market_weight=data_제약과생물공학_div['size_FIF_wisefn']/data_제약과생물공학_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_제약과생물공학_pbr['1/pbr']*data_제약과생물공학_pbr['market_weight'])
        mu_inv_per=np.sum(data_제약과생물공학_per['1/per']*data_제약과생물공학_per['market_weight'])
        mu_inv_div=np.sum(data_제약과생물공학_div['div_yield']*data_제약과생물공학_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_제약과생물공학_pbr['1/pbr']-mu_inv_pbr)*data_제약과생물공학_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_제약과생물공학_per['1/per']-mu_inv_per)*data_제약과생물공학_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_제약과생물공학_div['div_yield']-mu_inv_div)*data_제약과생물공학_div['market_weight']))
        
        data_제약과생물공학1=(data_제약과생물공학_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_제약과생물공학1.name= 'pbr_z'
        data_제약과생물공학2=(data_제약과생물공학_per['1/per']-mu_inv_per)/std_inv_per
        data_제약과생물공학2.name= 'per_z'
        data_제약과생물공학3=(data_제약과생물공학_div['div_yield']-mu_inv_div)/std_inv_div
        data_제약과생물공학3.name= 'div_z'
              
        result_제약과생물공학 = pd.concat([data_제약과생물공학, data_제약과생물공학1, data_제약과생물공학2, data_제약과생물공학3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_제약과생물공학 = result_제약과생물공학.assign(z_score=np.nanmean(result_제약과생물공학.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_제약과생물공학[result_제약과생물공학['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['제약과생물공학','sector_ratio']),:]
        
        a=a+1
   
     #은행 섹터
    if (np.sum(data['sector']=='은행')>0):
        data_은행 = data[data['sector']=="은행"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_은행['size_FIF_wisefn']=data_은행['size_FIF_wisefn']/1000    #size 단위 thousand
        data_은행['1/pbr']=data_은행['equity']/data_은행['size']
        data_은행['1/per']=data_은행['ni_12fw']/data_은행['size']
        data_은행['div_yield']=data_은행['cash_div']/data_은행['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_은행 = data_은행.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_은행_per = data_은행[data_은행['1/per'].notnull()]
        data_은행_pbr = data_은행[data_은행['1/pbr'].notnull()]
        data_은행_div = data_은행[data_은행['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_은행_pbr_cap = np.sum(data_은행_pbr['size_FIF_wisefn'])
        data_은행_per_cap = np.sum(data_은행_per['size_FIF_wisefn'])
        data_은행_div_cap = np.sum(data_은행_div['size_FIF_wisefn'])
    
        data_은행_pbr = data_은행_pbr.assign(market_weight=data_은행_pbr['size_FIF_wisefn']/data_은행_pbr_cap)
        data_은행_per = data_은행_per.assign(market_weight=data_은행_per['size_FIF_wisefn']/data_은행_per_cap)
        data_은행_div = data_은행_div.assign(market_weight=data_은행_div['size_FIF_wisefn']/data_은행_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_은행_pbr['1/pbr']*data_은행_pbr['market_weight'])
        mu_inv_per=np.sum(data_은행_per['1/per']*data_은행_per['market_weight'])
        mu_inv_div=np.sum(data_은행_div['div_yield']*data_은행_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_은행_pbr['1/pbr']-mu_inv_pbr)*data_은행_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_은행_per['1/per']-mu_inv_per)*data_은행_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_은행_div['div_yield']-mu_inv_div)*data_은행_div['market_weight']))
        
        data_은행1=(data_은행_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_은행1.name= 'pbr_z'
        data_은행2=(data_은행_per['1/per']-mu_inv_per)/std_inv_per
        data_은행2.name= 'per_z'
        data_은행3=(data_은행_div['div_yield']-mu_inv_div)/std_inv_div
        data_은행3.name= 'div_z'
              
        result_은행 = pd.concat([data_은행, data_은행1, data_은행2, data_은행3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_은행 = result_은행.assign(z_score=np.nanmean(result_은행.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_은행[result_은행['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['은행','sector_ratio']),:]
        
        a=a+1
    
     #증권 섹터
    if (np.sum(data['sector']=='증권')>0):
        data_증권 = data[data['sector']=="증권"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_증권['size_FIF_wisefn']=data_증권['size_FIF_wisefn']/1000    #size 단위 thousand
        data_증권['1/pbr']=data_증권['equity']/data_증권['size']
        data_증권['1/per']=data_증권['ni_12fw']/data_증권['size']
        data_증권['div_yield']=data_증권['cash_div']/data_증권['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_증권 = data_증권.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_증권_per = data_증권[data_증권['1/per'].notnull()]
        data_증권_pbr = data_증권[data_증권['1/pbr'].notnull()]
        data_증권_div = data_증권[data_증권['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_증권_pbr_cap = np.sum(data_증권_pbr['size_FIF_wisefn'])
        data_증권_per_cap = np.sum(data_증권_per['size_FIF_wisefn'])
        data_증권_div_cap = np.sum(data_증권_div['size_FIF_wisefn'])
    
        data_증권_pbr = data_증권_pbr.assign(market_weight=data_증권_pbr['size_FIF_wisefn']/data_증권_pbr_cap)
        data_증권_per = data_증권_per.assign(market_weight=data_증권_per['size_FIF_wisefn']/data_증권_per_cap)
        data_증권_div = data_증권_div.assign(market_weight=data_증권_div['size_FIF_wisefn']/data_증권_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_증권_pbr['1/pbr']*data_증권_pbr['market_weight'])
        mu_inv_per=np.sum(data_증권_per['1/per']*data_증권_per['market_weight'])
        mu_inv_div=np.sum(data_증권_div['div_yield']*data_증권_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_증권_pbr['1/pbr']-mu_inv_pbr)*data_증권_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_증권_per['1/per']-mu_inv_per)*data_증권_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_증권_div['div_yield']-mu_inv_div)*data_증권_div['market_weight']))
        
        data_증권1=(data_증권_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_증권1.name= 'pbr_z'
        data_증권2=(data_증권_per['1/per']-mu_inv_per)/std_inv_per
        data_증권2.name= 'per_z'
        data_증권3=(data_증권_div['div_yield']-mu_inv_div)/std_inv_div
        data_증권3.name= 'div_z'
              
        result_증권 = pd.concat([data_증권, data_증권1, data_증권2, data_증권3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_증권 = result_증권.assign(z_score=np.nanmean(result_증권.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_증권[result_증권['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['증권','sector_ratio']),:]
        
        a=a+1
    
     #다각화된금융 섹터
    if (np.sum(data['sector']=='다각화된금융')>0):
        data_다각화된금융 = data[data['sector']=="다각화된금융"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_다각화된금융['size_FIF_wisefn']=data_다각화된금융['size_FIF_wisefn']/1000    #size 단위 thousand
        data_다각화된금융['1/pbr']=data_다각화된금융['equity']/data_다각화된금융['size']
        data_다각화된금융['1/per']=data_다각화된금융['ni_12fw']/data_다각화된금융['size']
        data_다각화된금융['div_yield']=data_다각화된금융['cash_div']/data_다각화된금융['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_다각화된금융 = data_다각화된금융.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_다각화된금융_per = data_다각화된금융[data_다각화된금융['1/per'].notnull()]
        data_다각화된금융_pbr = data_다각화된금융[data_다각화된금융['1/pbr'].notnull()]
        data_다각화된금융_div = data_다각화된금융[data_다각화된금융['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_다각화된금융_pbr_cap = np.sum(data_다각화된금융_pbr['size_FIF_wisefn'])
        data_다각화된금융_per_cap = np.sum(data_다각화된금융_per['size_FIF_wisefn'])
        data_다각화된금융_div_cap = np.sum(data_다각화된금융_div['size_FIF_wisefn'])
    
        data_다각화된금융_pbr = data_다각화된금융_pbr.assign(market_weight=data_다각화된금융_pbr['size_FIF_wisefn']/data_다각화된금융_pbr_cap)
        data_다각화된금융_per = data_다각화된금융_per.assign(market_weight=data_다각화된금융_per['size_FIF_wisefn']/data_다각화된금융_per_cap)
        data_다각화된금융_div = data_다각화된금융_div.assign(market_weight=data_다각화된금융_div['size_FIF_wisefn']/data_다각화된금융_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_다각화된금융_pbr['1/pbr']*data_다각화된금융_pbr['market_weight'])
        mu_inv_per=np.sum(data_다각화된금융_per['1/per']*data_다각화된금융_per['market_weight'])
        mu_inv_div=np.sum(data_다각화된금융_div['div_yield']*data_다각화된금융_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_다각화된금융_pbr['1/pbr']-mu_inv_pbr)*data_다각화된금융_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_다각화된금융_per['1/per']-mu_inv_per)*data_다각화된금융_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_다각화된금융_div['div_yield']-mu_inv_div)*data_다각화된금융_div['market_weight']))
        
        data_다각화된금융1=(data_다각화된금융_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_다각화된금융1.name= 'pbr_z'
        data_다각화된금융2=(data_다각화된금융_per['1/per']-mu_inv_per)/std_inv_per
        data_다각화된금융2.name= 'per_z'
        data_다각화된금융3=(data_다각화된금융_div['div_yield']-mu_inv_div)/std_inv_div
        data_다각화된금융3.name= 'div_z'
              
        result_다각화된금융 = pd.concat([data_다각화된금융, data_다각화된금융1, data_다각화된금융2, data_다각화된금융3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_다각화된금융 = result_다각화된금융.assign(z_score=np.nanmean(result_다각화된금융.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_다각화된금융[result_다각화된금융['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['다각화된금융','sector_ratio']),:]
        
        a=a+1
    
     #보험 섹터
    if (np.sum(data['sector']=='보험')>0):
        data_보험 = data[data['sector']=="보험"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_보험['size_FIF_wisefn']=data_보험['size_FIF_wisefn']/1000    #size 단위 thousand
        data_보험['1/pbr']=data_보험['equity']/data_보험['size']
        data_보험['1/per']=data_보험['ni_12fw']/data_보험['size']
        data_보험['div_yield']=data_보험['cash_div']/data_보험['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_보험 = data_보험.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_보험_per = data_보험[data_보험['1/per'].notnull()]
        data_보험_pbr = data_보험[data_보험['1/pbr'].notnull()]
        data_보험_div = data_보험[data_보험['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_보험_pbr_cap = np.sum(data_보험_pbr['size_FIF_wisefn'])
        data_보험_per_cap = np.sum(data_보험_per['size_FIF_wisefn'])
        data_보험_div_cap = np.sum(data_보험_div['size_FIF_wisefn'])
    
        data_보험_pbr = data_보험_pbr.assign(market_weight=data_보험_pbr['size_FIF_wisefn']/data_보험_pbr_cap)
        data_보험_per = data_보험_per.assign(market_weight=data_보험_per['size_FIF_wisefn']/data_보험_per_cap)
        data_보험_div = data_보험_div.assign(market_weight=data_보험_div['size_FIF_wisefn']/data_보험_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_보험_pbr['1/pbr']*data_보험_pbr['market_weight'])
        mu_inv_per=np.sum(data_보험_per['1/per']*data_보험_per['market_weight'])
        mu_inv_div=np.sum(data_보험_div['div_yield']*data_보험_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_보험_pbr['1/pbr']-mu_inv_pbr)*data_보험_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_보험_per['1/per']-mu_inv_per)*data_보험_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_보험_div['div_yield']-mu_inv_div)*data_보험_div['market_weight']))
        
        data_보험1=(data_보험_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_보험1.name= 'pbr_z'
        data_보험2=(data_보험_per['1/per']-mu_inv_per)/std_inv_per
        data_보험2.name= 'per_z'
        data_보험3=(data_보험_div['div_yield']-mu_inv_div)/std_inv_div
        data_보험3.name= 'div_z'
              
        result_보험 = pd.concat([data_보험, data_보험1, data_보험2, data_보험3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_보험 = result_보험.assign(z_score=np.nanmean(result_보험.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_보험[result_보험['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['보험','sector_ratio']),:]
        
        a=a+1
   
     #부동산 섹터
    if (np.sum(data['sector']=='부동산')>0):
        data_부동산 = data[data['sector']=="부동산"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_부동산['size_FIF_wisefn']=data_부동산['size_FIF_wisefn']/1000    #size 단위 thousand
        data_부동산['1/pbr']=data_부동산['equity']/data_부동산['size']
        data_부동산['1/per']=data_부동산['ni_12fw']/data_부동산['size']
        data_부동산['div_yield']=data_부동산['cash_div']/data_부동산['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_부동산 = data_부동산.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_부동산_per = data_부동산[data_부동산['1/per'].notnull()]
        data_부동산_pbr = data_부동산[data_부동산['1/pbr'].notnull()]
        data_부동산_div = data_부동산[data_부동산['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_부동산_pbr_cap = np.sum(data_부동산_pbr['size_FIF_wisefn'])
        data_부동산_per_cap = np.sum(data_부동산_per['size_FIF_wisefn'])
        data_부동산_div_cap = np.sum(data_부동산_div['size_FIF_wisefn'])
    
        data_부동산_pbr = data_부동산_pbr.assign(market_weight=data_부동산_pbr['size_FIF_wisefn']/data_부동산_pbr_cap)
        data_부동산_per = data_부동산_per.assign(market_weight=data_부동산_per['size_FIF_wisefn']/data_부동산_per_cap)
        data_부동산_div = data_부동산_div.assign(market_weight=data_부동산_div['size_FIF_wisefn']/data_부동산_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_부동산_pbr['1/pbr']*data_부동산_pbr['market_weight'])
        mu_inv_per=np.sum(data_부동산_per['1/per']*data_부동산_per['market_weight'])
        mu_inv_div=np.sum(data_부동산_div['div_yield']*data_부동산_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_부동산_pbr['1/pbr']-mu_inv_pbr)*data_부동산_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_부동산_per['1/per']-mu_inv_per)*data_부동산_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_부동산_div['div_yield']-mu_inv_div)*data_부동산_div['market_weight']))
        
        data_부동산1=(data_부동산_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_부동산1.name= 'pbr_z'
        data_부동산2=(data_부동산_per['1/per']-mu_inv_per)/std_inv_per
        data_부동산2.name= 'per_z'
        data_부동산3=(data_부동산_div['div_yield']-mu_inv_div)/std_inv_div
        data_부동산3.name= 'div_z'
              
        result_부동산 = pd.concat([data_부동산, data_부동산1, data_부동산2, data_부동산3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_부동산 = result_부동산.assign(z_score=np.nanmean(result_부동산.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_부동산[result_부동산['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['부동산','sector_ratio']),:]
        
        a=a+1
    
     #기타금융서비스 섹터
    if (np.sum(data['sector']=='기타금융서비스')>0):
        data_기타금융서비스 = data[data['sector']=="기타금융서비스"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_기타금융서비스['size_FIF_wisefn']=data_기타금융서비스['size_FIF_wisefn']/1000    #size 단위 thousand
        data_기타금융서비스['1/pbr']=data_기타금융서비스['equity']/data_기타금융서비스['size']
        data_기타금융서비스['1/per']=data_기타금융서비스['ni_12fw']/data_기타금융서비스['size']
        data_기타금융서비스['div_yield']=data_기타금융서비스['cash_div']/data_기타금융서비스['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_기타금융서비스 = data_기타금융서비스.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_기타금융서비스_per = data_기타금융서비스[data_기타금융서비스['1/per'].notnull()]
        data_기타금융서비스_pbr = data_기타금융서비스[data_기타금융서비스['1/pbr'].notnull()]
        data_기타금융서비스_div = data_기타금융서비스[data_기타금융서비스['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_기타금융서비스_pbr_cap = np.sum(data_기타금융서비스_pbr['size_FIF_wisefn'])
        data_기타금융서비스_per_cap = np.sum(data_기타금융서비스_per['size_FIF_wisefn'])
        data_기타금융서비스_div_cap = np.sum(data_기타금융서비스_div['size_FIF_wisefn'])
    
        data_기타금융서비스_pbr = data_기타금융서비스_pbr.assign(market_weight=data_기타금융서비스_pbr['size_FIF_wisefn']/data_기타금융서비스_pbr_cap)
        data_기타금융서비스_per = data_기타금융서비스_per.assign(market_weight=data_기타금융서비스_per['size_FIF_wisefn']/data_기타금융서비스_per_cap)
        data_기타금융서비스_div = data_기타금융서비스_div.assign(market_weight=data_기타금융서비스_div['size_FIF_wisefn']/data_기타금융서비스_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_기타금융서비스_pbr['1/pbr']*data_기타금융서비스_pbr['market_weight'])
        mu_inv_per=np.sum(data_기타금융서비스_per['1/per']*data_기타금융서비스_per['market_weight'])
        mu_inv_div=np.sum(data_기타금융서비스_div['div_yield']*data_기타금융서비스_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_기타금융서비스_pbr['1/pbr']-mu_inv_pbr)*data_기타금융서비스_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_기타금융서비스_per['1/per']-mu_inv_per)*data_기타금융서비스_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_기타금융서비스_div['div_yield']-mu_inv_div)*data_기타금융서비스_div['market_weight']))
        
        data_기타금융서비스1=(data_기타금융서비스_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_기타금융서비스1.name= 'pbr_z'
        data_기타금융서비스2=(data_기타금융서비스_per['1/per']-mu_inv_per)/std_inv_per
        data_기타금융서비스2.name= 'per_z'
        data_기타금융서비스3=(data_기타금융서비스_div['div_yield']-mu_inv_div)/std_inv_div
        data_기타금융서비스3.name= 'div_z'
              
        result_기타금융서비스 = pd.concat([data_기타금융서비스, data_기타금융서비스1, data_기타금융서비스2, data_기타금융서비스3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_기타금융서비스 = result_기타금융서비스.assign(z_score=np.nanmean(result_기타금융서비스.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_기타금융서비스[result_기타금융서비스['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['기타금융서비스','sector_ratio']),:]
        
        a=a+1
    
     #소프트웨어와서비스 섹터
    if (np.sum(data['sector']=='소프트웨어와서비스')>0):
        data_소프트웨어와서비스 = data[data['sector']=="소프트웨어와서비스"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_소프트웨어와서비스['size_FIF_wisefn']=data_소프트웨어와서비스['size_FIF_wisefn']/1000    #size 단위 thousand
        data_소프트웨어와서비스['1/pbr']=data_소프트웨어와서비스['equity']/data_소프트웨어와서비스['size']
        data_소프트웨어와서비스['1/per']=data_소프트웨어와서비스['ni_12fw']/data_소프트웨어와서비스['size']
        data_소프트웨어와서비스['div_yield']=data_소프트웨어와서비스['cash_div']/data_소프트웨어와서비스['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_소프트웨어와서비스 = data_소프트웨어와서비스.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_소프트웨어와서비스_per = data_소프트웨어와서비스[data_소프트웨어와서비스['1/per'].notnull()]
        data_소프트웨어와서비스_pbr = data_소프트웨어와서비스[data_소프트웨어와서비스['1/pbr'].notnull()]
        data_소프트웨어와서비스_div = data_소프트웨어와서비스[data_소프트웨어와서비스['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_소프트웨어와서비스_pbr_cap = np.sum(data_소프트웨어와서비스_pbr['size_FIF_wisefn'])
        data_소프트웨어와서비스_per_cap = np.sum(data_소프트웨어와서비스_per['size_FIF_wisefn'])
        data_소프트웨어와서비스_div_cap = np.sum(data_소프트웨어와서비스_div['size_FIF_wisefn'])
    
        data_소프트웨어와서비스_pbr = data_소프트웨어와서비스_pbr.assign(market_weight=data_소프트웨어와서비스_pbr['size_FIF_wisefn']/data_소프트웨어와서비스_pbr_cap)
        data_소프트웨어와서비스_per = data_소프트웨어와서비스_per.assign(market_weight=data_소프트웨어와서비스_per['size_FIF_wisefn']/data_소프트웨어와서비스_per_cap)
        data_소프트웨어와서비스_div = data_소프트웨어와서비스_div.assign(market_weight=data_소프트웨어와서비스_div['size_FIF_wisefn']/data_소프트웨어와서비스_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_소프트웨어와서비스_pbr['1/pbr']*data_소프트웨어와서비스_pbr['market_weight'])
        mu_inv_per=np.sum(data_소프트웨어와서비스_per['1/per']*data_소프트웨어와서비스_per['market_weight'])
        mu_inv_div=np.sum(data_소프트웨어와서비스_div['div_yield']*data_소프트웨어와서비스_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_소프트웨어와서비스_pbr['1/pbr']-mu_inv_pbr)*data_소프트웨어와서비스_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_소프트웨어와서비스_per['1/per']-mu_inv_per)*data_소프트웨어와서비스_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_소프트웨어와서비스_div['div_yield']-mu_inv_div)*data_소프트웨어와서비스_div['market_weight']))
        
        data_소프트웨어와서비스1=(data_소프트웨어와서비스_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_소프트웨어와서비스1.name= 'pbr_z'
        data_소프트웨어와서비스2=(data_소프트웨어와서비스_per['1/per']-mu_inv_per)/std_inv_per
        data_소프트웨어와서비스2.name= 'per_z'
        data_소프트웨어와서비스3=(data_소프트웨어와서비스_div['div_yield']-mu_inv_div)/std_inv_div
        data_소프트웨어와서비스3.name= 'div_z'
              
        result_소프트웨어와서비스 = pd.concat([data_소프트웨어와서비스, data_소프트웨어와서비스1, data_소프트웨어와서비스2, data_소프트웨어와서비스3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_소프트웨어와서비스 = result_소프트웨어와서비스.assign(z_score=np.nanmean(result_소프트웨어와서비스.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_소프트웨어와서비스[result_소프트웨어와서비스['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['소프트웨어와서비스','sector_ratio']),:]
        
        a=a+1
    
     #기술하드웨어와장비 섹터
    if (np.sum(data['sector']=='기술하드웨어와장비')>0):
        data_기술하드웨어와장비 = data[data['sector']=="기술하드웨어와장비"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_기술하드웨어와장비['size_FIF_wisefn']=data_기술하드웨어와장비['size_FIF_wisefn']/1000    #size 단위 thousand
        data_기술하드웨어와장비['1/pbr']=data_기술하드웨어와장비['equity']/data_기술하드웨어와장비['size']
        data_기술하드웨어와장비['1/per']=data_기술하드웨어와장비['ni_12fw']/data_기술하드웨어와장비['size']
        data_기술하드웨어와장비['div_yield']=data_기술하드웨어와장비['cash_div']/data_기술하드웨어와장비['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_기술하드웨어와장비 = data_기술하드웨어와장비.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_기술하드웨어와장비_per = data_기술하드웨어와장비[data_기술하드웨어와장비['1/per'].notnull()]
        data_기술하드웨어와장비_pbr = data_기술하드웨어와장비[data_기술하드웨어와장비['1/pbr'].notnull()]
        data_기술하드웨어와장비_div = data_기술하드웨어와장비[data_기술하드웨어와장비['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_기술하드웨어와장비_pbr_cap = np.sum(data_기술하드웨어와장비_pbr['size_FIF_wisefn'])
        data_기술하드웨어와장비_per_cap = np.sum(data_기술하드웨어와장비_per['size_FIF_wisefn'])
        data_기술하드웨어와장비_div_cap = np.sum(data_기술하드웨어와장비_div['size_FIF_wisefn'])
    
        data_기술하드웨어와장비_pbr = data_기술하드웨어와장비_pbr.assign(market_weight=data_기술하드웨어와장비_pbr['size_FIF_wisefn']/data_기술하드웨어와장비_pbr_cap)
        data_기술하드웨어와장비_per = data_기술하드웨어와장비_per.assign(market_weight=data_기술하드웨어와장비_per['size_FIF_wisefn']/data_기술하드웨어와장비_per_cap)
        data_기술하드웨어와장비_div = data_기술하드웨어와장비_div.assign(market_weight=data_기술하드웨어와장비_div['size_FIF_wisefn']/data_기술하드웨어와장비_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_기술하드웨어와장비_pbr['1/pbr']*data_기술하드웨어와장비_pbr['market_weight'])
        mu_inv_per=np.sum(data_기술하드웨어와장비_per['1/per']*data_기술하드웨어와장비_per['market_weight'])
        mu_inv_div=np.sum(data_기술하드웨어와장비_div['div_yield']*data_기술하드웨어와장비_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_기술하드웨어와장비_pbr['1/pbr']-mu_inv_pbr)*data_기술하드웨어와장비_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_기술하드웨어와장비_per['1/per']-mu_inv_per)*data_기술하드웨어와장비_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_기술하드웨어와장비_div['div_yield']-mu_inv_div)*data_기술하드웨어와장비_div['market_weight']))
        
        data_기술하드웨어와장비1=(data_기술하드웨어와장비_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_기술하드웨어와장비1.name= 'pbr_z'
        data_기술하드웨어와장비2=(data_기술하드웨어와장비_per['1/per']-mu_inv_per)/std_inv_per
        data_기술하드웨어와장비2.name= 'per_z'
        data_기술하드웨어와장비3=(data_기술하드웨어와장비_div['div_yield']-mu_inv_div)/std_inv_div
        data_기술하드웨어와장비3.name= 'div_z'
              
        result_기술하드웨어와장비 = pd.concat([data_기술하드웨어와장비, data_기술하드웨어와장비1, data_기술하드웨어와장비2, data_기술하드웨어와장비3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_기술하드웨어와장비 = result_기술하드웨어와장비.assign(z_score=np.nanmean(result_기술하드웨어와장비.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_기술하드웨어와장비[result_기술하드웨어와장비['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['기술하드웨어와장비','sector_ratio']),:]
        
        a=a+1
    
     #반도체와반도체장비 섹터
    if (np.sum(data['sector']=='반도체와반도체장비')>0):
        data_반도체와반도체장비 = data[data['sector']=="반도체와반도체장비"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_반도체와반도체장비['size_FIF_wisefn']=data_반도체와반도체장비['size_FIF_wisefn']/1000    #size 단위 thousand
        data_반도체와반도체장비['1/pbr']=data_반도체와반도체장비['equity']/data_반도체와반도체장비['size']
        data_반도체와반도체장비['1/per']=data_반도체와반도체장비['ni_12fw']/data_반도체와반도체장비['size']
        data_반도체와반도체장비['div_yield']=data_반도체와반도체장비['cash_div']/data_반도체와반도체장비['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_반도체와반도체장비 = data_반도체와반도체장비.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_반도체와반도체장비_per = data_반도체와반도체장비[data_반도체와반도체장비['1/per'].notnull()]
        data_반도체와반도체장비_pbr = data_반도체와반도체장비[data_반도체와반도체장비['1/pbr'].notnull()]
        data_반도체와반도체장비_div = data_반도체와반도체장비[data_반도체와반도체장비['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_반도체와반도체장비_pbr_cap = np.sum(data_반도체와반도체장비_pbr['size_FIF_wisefn'])
        data_반도체와반도체장비_per_cap = np.sum(data_반도체와반도체장비_per['size_FIF_wisefn'])
        data_반도체와반도체장비_div_cap = np.sum(data_반도체와반도체장비_div['size_FIF_wisefn'])
    
        data_반도체와반도체장비_pbr = data_반도체와반도체장비_pbr.assign(market_weight=data_반도체와반도체장비_pbr['size_FIF_wisefn']/data_반도체와반도체장비_pbr_cap)
        data_반도체와반도체장비_per = data_반도체와반도체장비_per.assign(market_weight=data_반도체와반도체장비_per['size_FIF_wisefn']/data_반도체와반도체장비_per_cap)
        data_반도체와반도체장비_div = data_반도체와반도체장비_div.assign(market_weight=data_반도체와반도체장비_div['size_FIF_wisefn']/data_반도체와반도체장비_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_반도체와반도체장비_pbr['1/pbr']*data_반도체와반도체장비_pbr['market_weight'])
        mu_inv_per=np.sum(data_반도체와반도체장비_per['1/per']*data_반도체와반도체장비_per['market_weight'])
        mu_inv_div=np.sum(data_반도체와반도체장비_div['div_yield']*data_반도체와반도체장비_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_반도체와반도체장비_pbr['1/pbr']-mu_inv_pbr)*data_반도체와반도체장비_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_반도체와반도체장비_per['1/per']-mu_inv_per)*data_반도체와반도체장비_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_반도체와반도체장비_div['div_yield']-mu_inv_div)*data_반도체와반도체장비_div['market_weight']))
        
        data_반도체와반도체장비1=(data_반도체와반도체장비_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_반도체와반도체장비1.name= 'pbr_z'
        data_반도체와반도체장비2=(data_반도체와반도체장비_per['1/per']-mu_inv_per)/std_inv_per
        data_반도체와반도체장비2.name= 'per_z'
        data_반도체와반도체장비3=(data_반도체와반도체장비_div['div_yield']-mu_inv_div)/std_inv_div
        data_반도체와반도체장비3.name= 'div_z'
              
        result_반도체와반도체장비 = pd.concat([data_반도체와반도체장비, data_반도체와반도체장비1, data_반도체와반도체장비2, data_반도체와반도체장비3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_반도체와반도체장비 = result_반도체와반도체장비.assign(z_score=np.nanmean(result_반도체와반도체장비.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_반도체와반도체장비[result_반도체와반도체장비['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['반도체와반도체장비','sector_ratio']),:]
        
        a=a+1
    
     #전자와 전기제품 섹터
    if (np.sum(data['sector']=='전자와 전기제품')>0):
        data_전자와_전기제품 = data[data['sector']=="전자와 전기제품"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_전자와_전기제품['size_FIF_wisefn']=data_전자와_전기제품['size_FIF_wisefn']/1000    #size 단위 thousand
        data_전자와_전기제품['1/pbr']=data_전자와_전기제품['equity']/data_전자와_전기제품['size']
        data_전자와_전기제품['1/per']=data_전자와_전기제품['ni_12fw']/data_전자와_전기제품['size']
        data_전자와_전기제품['div_yield']=data_전자와_전기제품['cash_div']/data_전자와_전기제품['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_전자와_전기제품 = data_전자와_전기제품.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_전자와_전기제품_per = data_전자와_전기제품[data_전자와_전기제품['1/per'].notnull()]
        data_전자와_전기제품_pbr = data_전자와_전기제품[data_전자와_전기제품['1/pbr'].notnull()]
        data_전자와_전기제품_div = data_전자와_전기제품[data_전자와_전기제품['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_전자와_전기제품_pbr_cap = np.sum(data_전자와_전기제품_pbr['size_FIF_wisefn'])
        data_전자와_전기제품_per_cap = np.sum(data_전자와_전기제품_per['size_FIF_wisefn'])
        data_전자와_전기제품_div_cap = np.sum(data_전자와_전기제품_div['size_FIF_wisefn'])
    
        data_전자와_전기제품_pbr = data_전자와_전기제품_pbr.assign(market_weight=data_전자와_전기제품_pbr['size_FIF_wisefn']/data_전자와_전기제품_pbr_cap)
        data_전자와_전기제품_per = data_전자와_전기제품_per.assign(market_weight=data_전자와_전기제품_per['size_FIF_wisefn']/data_전자와_전기제품_per_cap)
        data_전자와_전기제품_div = data_전자와_전기제품_div.assign(market_weight=data_전자와_전기제품_div['size_FIF_wisefn']/data_전자와_전기제품_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_전자와_전기제품_pbr['1/pbr']*data_전자와_전기제품_pbr['market_weight'])
        mu_inv_per=np.sum(data_전자와_전기제품_per['1/per']*data_전자와_전기제품_per['market_weight'])
        mu_inv_div=np.sum(data_전자와_전기제품_div['div_yield']*data_전자와_전기제품_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_전자와_전기제품_pbr['1/pbr']-mu_inv_pbr)*data_전자와_전기제품_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_전자와_전기제품_per['1/per']-mu_inv_per)*data_전자와_전기제품_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_전자와_전기제품_div['div_yield']-mu_inv_div)*data_전자와_전기제품_div['market_weight']))
        
        data_전자와_전기제품1=(data_전자와_전기제품_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_전자와_전기제품1.name= 'pbr_z'
        data_전자와_전기제품2=(data_전자와_전기제품_per['1/per']-mu_inv_per)/std_inv_per
        data_전자와_전기제품2.name= 'per_z'
        data_전자와_전기제품3=(data_전자와_전기제품_div['div_yield']-mu_inv_div)/std_inv_div
        data_전자와_전기제품3.name= 'div_z'
              
        result_전자와_전기제품 = pd.concat([data_전자와_전기제품, data_전자와_전기제품1, data_전자와_전기제품2, data_전자와_전기제품3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_전자와_전기제품 = result_전자와_전기제품.assign(z_score=np.nanmean(result_전자와_전기제품.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_전자와_전기제품[result_전자와_전기제품['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['전자와 전기제품','sector_ratio']),:]
        
        a=a+1
    
     #디스플레이 섹터
    if (np.sum(data['sector']=='디스플레이')>0):
        data_디스플레이 = data[data['sector']=="디스플레이"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_디스플레이['size_FIF_wisefn']=data_디스플레이['size_FIF_wisefn']/1000    #size 단위 thousand
        data_디스플레이['1/pbr']=data_디스플레이['equity']/data_디스플레이['size']
        data_디스플레이['1/per']=data_디스플레이['ni_12fw']/data_디스플레이['size']
        data_디스플레이['div_yield']=data_디스플레이['cash_div']/data_디스플레이['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_디스플레이 = data_디스플레이.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_디스플레이_per = data_디스플레이[data_디스플레이['1/per'].notnull()]
        data_디스플레이_pbr = data_디스플레이[data_디스플레이['1/pbr'].notnull()]
        data_디스플레이_div = data_디스플레이[data_디스플레이['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_디스플레이_pbr_cap = np.sum(data_디스플레이_pbr['size_FIF_wisefn'])
        data_디스플레이_per_cap = np.sum(data_디스플레이_per['size_FIF_wisefn'])
        data_디스플레이_div_cap = np.sum(data_디스플레이_div['size_FIF_wisefn'])
    
        data_디스플레이_pbr = data_디스플레이_pbr.assign(market_weight=data_디스플레이_pbr['size_FIF_wisefn']/data_디스플레이_pbr_cap)
        data_디스플레이_per = data_디스플레이_per.assign(market_weight=data_디스플레이_per['size_FIF_wisefn']/data_디스플레이_per_cap)
        data_디스플레이_div = data_디스플레이_div.assign(market_weight=data_디스플레이_div['size_FIF_wisefn']/data_디스플레이_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_디스플레이_pbr['1/pbr']*data_디스플레이_pbr['market_weight'])
        mu_inv_per=np.sum(data_디스플레이_per['1/per']*data_디스플레이_per['market_weight'])
        mu_inv_div=np.sum(data_디스플레이_div['div_yield']*data_디스플레이_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_디스플레이_pbr['1/pbr']-mu_inv_pbr)*data_디스플레이_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_디스플레이_per['1/per']-mu_inv_per)*data_디스플레이_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_디스플레이_div['div_yield']-mu_inv_div)*data_디스플레이_div['market_weight']))
        
        data_디스플레이1=(data_디스플레이_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_디스플레이1.name= 'pbr_z'
        data_디스플레이2=(data_디스플레이_per['1/per']-mu_inv_per)/std_inv_per
        data_디스플레이2.name= 'per_z'
        data_디스플레이3=(data_디스플레이_div['div_yield']-mu_inv_div)/std_inv_div
        data_디스플레이3.name= 'div_z'
              
        result_디스플레이 = pd.concat([data_디스플레이, data_디스플레이1, data_디스플레이2, data_디스플레이3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_디스플레이 = result_디스플레이.assign(z_score=np.nanmean(result_디스플레이.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_디스플레이[result_디스플레이['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['디스플레이','sector_ratio']),:]
        
        a=a+1
    
     #통신서비스 섹터
    if (np.sum(data['sector']=='통신서비스')>0):
        data_통신서비스 = data[data['sector']=="통신서비스"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_통신서비스['size_FIF_wisefn']=data_통신서비스['size_FIF_wisefn']/1000    #size 단위 thousand
        data_통신서비스['1/pbr']=data_통신서비스['equity']/data_통신서비스['size']
        data_통신서비스['1/per']=data_통신서비스['ni_12fw']/data_통신서비스['size']
        data_통신서비스['div_yield']=data_통신서비스['cash_div']/data_통신서비스['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_통신서비스 = data_통신서비스.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_통신서비스_per = data_통신서비스[data_통신서비스['1/per'].notnull()]
        data_통신서비스_pbr = data_통신서비스[data_통신서비스['1/pbr'].notnull()]
        data_통신서비스_div = data_통신서비스[data_통신서비스['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_통신서비스_pbr_cap = np.sum(data_통신서비스_pbr['size_FIF_wisefn'])
        data_통신서비스_per_cap = np.sum(data_통신서비스_per['size_FIF_wisefn'])
        data_통신서비스_div_cap = np.sum(data_통신서비스_div['size_FIF_wisefn'])
    
        data_통신서비스_pbr = data_통신서비스_pbr.assign(market_weight=data_통신서비스_pbr['size_FIF_wisefn']/data_통신서비스_pbr_cap)
        data_통신서비스_per = data_통신서비스_per.assign(market_weight=data_통신서비스_per['size_FIF_wisefn']/data_통신서비스_per_cap)
        data_통신서비스_div = data_통신서비스_div.assign(market_weight=data_통신서비스_div['size_FIF_wisefn']/data_통신서비스_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_통신서비스_pbr['1/pbr']*data_통신서비스_pbr['market_weight'])
        mu_inv_per=np.sum(data_통신서비스_per['1/per']*data_통신서비스_per['market_weight'])
        mu_inv_div=np.sum(data_통신서비스_div['div_yield']*data_통신서비스_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_통신서비스_pbr['1/pbr']-mu_inv_pbr)*data_통신서비스_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_통신서비스_per['1/per']-mu_inv_per)*data_통신서비스_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_통신서비스_div['div_yield']-mu_inv_div)*data_통신서비스_div['market_weight']))
        
        data_통신서비스1=(data_통신서비스_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_통신서비스1.name= 'pbr_z'
        data_통신서비스2=(data_통신서비스_per['1/per']-mu_inv_per)/std_inv_per
        data_통신서비스2.name= 'per_z'
        data_통신서비스3=(data_통신서비스_div['div_yield']-mu_inv_div)/std_inv_div
        data_통신서비스3.name= 'div_z'
              
        result_통신서비스 = pd.concat([data_통신서비스, data_통신서비스1, data_통신서비스2, data_통신서비스3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_통신서비스 = result_통신서비스.assign(z_score=np.nanmean(result_통신서비스.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_통신서비스[result_통신서비스['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['통신서비스','sector_ratio']),:]
        
        a=a+1
    
     #유틸리티 섹터
    if (np.sum(data['sector']=='유틸리티')>0):
        data_유틸리티 = data[data['sector']=="유틸리티"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_유틸리티['size_FIF_wisefn']=data_유틸리티['size_FIF_wisefn']/1000    #size 단위 thousand
        data_유틸리티['1/pbr']=data_유틸리티['equity']/data_유틸리티['size']
        data_유틸리티['1/per']=data_유틸리티['ni_12fw']/data_유틸리티['size']
        data_유틸리티['div_yield']=data_유틸리티['cash_div']/data_유틸리티['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_유틸리티 = data_유틸리티.replace([np.inf, -np.inf],np.nan)  
        
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
        locals()['result_{}'.format(a)] =result_유틸리티[result_유틸리티['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['유틸리티','sector_ratio']),:]
        
        a=a+1
    
    
    
    for y in range(2,a):    
        result_1 = pd.concat([result_1,locals()['result_{}'.format(y)]],axis=0,join='inner')
   
    
    result = result_1
    
    #상위 65%로 결정하면 삼성전자가 n=64,65,66일때 모두 포함이 된다.
#    z_score1_max=np.percentile(result['z_score'],50)
#    result =result[result['z_score']>z_score1_max]
    result=result.assign(rnk=result['z_score'].rank(method='first',ascending=False)) 
    
#    result = pd.concat([result,pd.DataFrame(result_temp.loc[390,:]).transpose()],axis=0)

    #중복 rows 1개 빼고 다 제거 
    result = result.drop_duplicates()
    result1 = result[result['rnk']<86] 
   
    
#############################################################################
## 중소형주 ###
    data_big = raw_data_sum[(raw_data_sum[n] == 2)|(raw_data_sum[n] == 3)|(raw_data_sum[n] == 'KOSDAQ')]
    data_big = data_big.loc[:,[1,n]]
    #ni_12m_fw_sum 쓰면 fwd per, 그냥 ni_sum 쓰면 trailing
    #rtn_sum은 그냥, rtn_dividend_sum은 배당고려 수익률
    data = pd.concat([data_big, size_FIF_wisefn_sum[n], equity_sum[n], ni_12m_fw_sum[n],cash_div_sum[n],size_sum[n],rtn_sum[n-3],sector_sum[n],rtn_month_sum[3*(n-3)],rtn_month_sum[3*(n-3)+1],rtn_month_sum[3*(n-3)+2]],axis=1,join='inner',ignore_index=True)
    data.columns = ['name','group','size_FIF_wisefn','equity','ni_12fw','cash_div','size','return','sector','return_month1','return_month2','return_month3']
    #섹터 시가총액 비중으로 z_score 종목 짜르기 위해서 섹터별 시총 합 계산
    data_sector_ratio = data.groupby('sector').sum()
    #각 섹터별 시가총액비중을 구하는데  ceil 로 올림해서 구한다음 아래에서 rank로 짜르지뭐
    data_sector_ratio = data_sector_ratio.assign(sector_ratio = np.ceil(data_sector_ratio['size']*100/np.sum(data_sector_ratio['size'])))
    data=data[data['size']>100000000000]
    #상폐, 지주사전환, 분할상장 때문에 생기는 수익률 0 제거
    data=data[data['return']!=0]
    result_temp = data
#    samsung = pd.DataFrame(data.loc[390,:]).transpose()

    data = data[data['equity'].notnull()]
    data = data[data['ni_12fw'].notnull()]
    data = data[data['cash_div'].notnull()]
    
    for i in range(1,30):
        locals()['result_{}'.format(i)] = pd.DataFrame(np.zeros((200,19)))

    
    
    a=1
        #에너지 섹터
    if (np.sum(data['sector']=='에너지')>0):
        data_에너지 = data[data['sector']=='에너지']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
#        data_에너지['size_FIF_wisefn']=data_에너지['size_FIF_wisefn']/1000    #size 단위 thousand
        data_에너지.loc[:,'size_FIF_wisefn']=data_에너지.loc[:,'size_FIF_wisefn']/1000        
        data_에너지['1/pbr']=data_에너지['equity']/data_에너지['size']
        data_에너지['1/per']=data_에너지['ni_12fw']/data_에너지['size']
        data_에너지['div_yield']=data_에너지['cash_div']/data_에너지['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_에너지 = data_에너지.replace([np.inf, -np.inf],np.nan)  
        
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
        locals()['result_{}'.format(a)] =result_에너지[result_에너지['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['에너지','sector_ratio']),:]
        
        a=a+1
        
    #소재 섹터
    if (np.sum(data['sector']=='소재')>0):
        data_소재 = data[data['sector']=='소재']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_소재['size_FIF_wisefn']=data_소재['size_FIF_wisefn']/1000    #size 단위 thousand
        data_소재['1/pbr']=data_소재['equity']/data_소재['size']
        data_소재['1/per']=data_소재['ni_12fw']/data_소재['size']
        data_소재['div_yield']=data_소재['cash_div']/data_소재['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_소재 = data_소재.replace([np.inf, -np.inf],np.nan)  
        
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
        locals()['result_{}'.format(a)] =result_소재[result_소재['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['소재','sector_ratio']),:]
        
        
        a=a+1    
    #자본재 섹터
    if (np.sum(data['sector']=='자본재')>0):
        data_자본재 = data[data['sector']=="자본재"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_자본재['size_FIF_wisefn']=data_자본재['size_FIF_wisefn']/1000    #size 단위 thousand
        data_자본재['1/pbr']=data_자본재['equity']/data_자본재['size']
        data_자본재['1/per']=data_자본재['ni_12fw']/data_자본재['size']
        data_자본재['div_yield']=data_자본재['cash_div']/data_자본재['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_자본재 = data_자본재.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_자본재_per = data_자본재[data_자본재['1/per'].notnull()]
        data_자본재_pbr = data_자본재[data_자본재['1/pbr'].notnull()]
        data_자본재_div = data_자본재[data_자본재['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_자본재_pbr_cap = np.sum(data_자본재_pbr['size_FIF_wisefn'])
        data_자본재_per_cap = np.sum(data_자본재_per['size_FIF_wisefn'])
        data_자본재_div_cap = np.sum(data_자본재_div['size_FIF_wisefn'])
    
        data_자본재_pbr = data_자본재_pbr.assign(market_weight=data_자본재_pbr['size_FIF_wisefn']/data_자본재_pbr_cap)
        data_자본재_per = data_자본재_per.assign(market_weight=data_자본재_per['size_FIF_wisefn']/data_자본재_per_cap)
        data_자본재_div = data_자본재_div.assign(market_weight=data_자본재_div['size_FIF_wisefn']/data_자본재_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_자본재_pbr['1/pbr']*data_자본재_pbr['market_weight'])
        mu_inv_per=np.sum(data_자본재_per['1/per']*data_자본재_per['market_weight'])
        mu_inv_div=np.sum(data_자본재_div['div_yield']*data_자본재_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_자본재_pbr['1/pbr']-mu_inv_pbr)*data_자본재_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_자본재_per['1/per']-mu_inv_per)*data_자본재_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_자본재_div['div_yield']-mu_inv_div)*data_자본재_div['market_weight']))
        
        data_자본재1=(data_자본재_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_자본재1.name= 'pbr_z'
        data_자본재2=(data_자본재_per['1/per']-mu_inv_per)/std_inv_per
        data_자본재2.name= 'per_z'
        data_자본재3=(data_자본재_div['div_yield']-mu_inv_div)/std_inv_div
        data_자본재3.name= 'div_z'
              
        result_자본재 = pd.concat([data_자본재, data_자본재1, data_자본재2, data_자본재3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_자본재 = result_자본재.assign(z_score=np.nanmean(result_자본재.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_자본재[result_자본재['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['자본재','sector_ratio']),:]
        a=a+1
    
    
    #상업서비스와공급품 섹터
    if (np.sum(data['sector']=='상업서비스와공급품')>0):
        data_상업서비스와공급품 = data[data['sector']=='상업서비스와공급품']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_상업서비스와공급품['size_FIF_wisefn']=data_상업서비스와공급품['size_FIF_wisefn']/1000    #size 단위 thousand
        data_상업서비스와공급품['1/pbr']=data_상업서비스와공급품['equity']/data_상업서비스와공급품['size']
        data_상업서비스와공급품['1/per']=data_상업서비스와공급품['ni_12fw']/data_상업서비스와공급품['size']
        data_상업서비스와공급품['div_yield']=data_상업서비스와공급품['cash_div']/data_상업서비스와공급품['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_상업서비스와공급품 = data_상업서비스와공급품.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_상업서비스와공급품_per = data_상업서비스와공급품[data_상업서비스와공급품['1/per'].notnull()]
        data_상업서비스와공급품_pbr = data_상업서비스와공급품[data_상업서비스와공급품['1/pbr'].notnull()]
        data_상업서비스와공급품_div = data_상업서비스와공급품[data_상업서비스와공급품['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_상업서비스와공급품_pbr_cap = np.sum(data_상업서비스와공급품_pbr['size_FIF_wisefn'])
        data_상업서비스와공급품_per_cap = np.sum(data_상업서비스와공급품_per['size_FIF_wisefn'])
        data_상업서비스와공급품_div_cap = np.sum(data_상업서비스와공급품_div['size_FIF_wisefn'])
    
        data_상업서비스와공급품_pbr = data_상업서비스와공급품_pbr.assign(market_weight=data_상업서비스와공급품_pbr['size_FIF_wisefn']/data_상업서비스와공급품_pbr_cap)
        data_상업서비스와공급품_per = data_상업서비스와공급품_per.assign(market_weight=data_상업서비스와공급품_per['size_FIF_wisefn']/data_상업서비스와공급품_per_cap)
        data_상업서비스와공급품_div = data_상업서비스와공급품_div.assign(market_weight=data_상업서비스와공급품_div['size_FIF_wisefn']/data_상업서비스와공급품_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_상업서비스와공급품_pbr['1/pbr']*data_상업서비스와공급품_pbr['market_weight'])
        mu_inv_per=np.sum(data_상업서비스와공급품_per['1/per']*data_상업서비스와공급품_per['market_weight'])
        mu_inv_div=np.sum(data_상업서비스와공급품_div['div_yield']*data_상업서비스와공급품_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_상업서비스와공급품_pbr['1/pbr']-mu_inv_pbr)*data_상업서비스와공급품_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_상업서비스와공급품_per['1/per']-mu_inv_per)*data_상업서비스와공급품_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_상업서비스와공급품_div['div_yield']-mu_inv_div)*data_상업서비스와공급품_div['market_weight']))
        
        data_상업서비스와공급품1=(data_상업서비스와공급품_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_상업서비스와공급품1.name= 'pbr_z'
        data_상업서비스와공급품2=(data_상업서비스와공급품_per['1/per']-mu_inv_per)/std_inv_per
        data_상업서비스와공급품2.name= 'per_z'
        data_상업서비스와공급품3=(data_상업서비스와공급품_div['div_yield']-mu_inv_div)/std_inv_div
        data_상업서비스와공급품3.name= 'div_z'
              
        result_상업서비스와공급품 = pd.concat([data_상업서비스와공급품, data_상업서비스와공급품1, data_상업서비스와공급품2, data_상업서비스와공급품3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_상업서비스와공급품 = result_상업서비스와공급품.assign(z_score=np.nanmean(result_상업서비스와공급품.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_상업서비스와공급품[result_상업서비스와공급품['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['상업서비스와공급품','sector_ratio']),:]
        
        
        
        a=a+1
       #운송 섹터
    if (np.sum(data['sector']=='운송')>0):
        data_운송 = data[data['sector']=='운송']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_운송['size_FIF_wisefn']=data_운송['size_FIF_wisefn']/1000    #size 단위 thousand
        data_운송['1/pbr']=data_운송['equity']/data_운송['size']
        data_운송['1/per']=data_운송['ni_12fw']/data_운송['size']
        data_운송['div_yield']=data_운송['cash_div']/data_운송['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_운송 = data_운송.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_운송_per = data_운송[data_운송['1/per'].notnull()]
        data_운송_pbr = data_운송[data_운송['1/pbr'].notnull()]
        data_운송_div = data_운송[data_운송['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_운송_pbr_cap = np.sum(data_운송_pbr['size_FIF_wisefn'])
        data_운송_per_cap = np.sum(data_운송_per['size_FIF_wisefn'])
        data_운송_div_cap = np.sum(data_운송_div['size_FIF_wisefn'])
    
        data_운송_pbr = data_운송_pbr.assign(market_weight=data_운송_pbr['size_FIF_wisefn']/data_운송_pbr_cap)
        data_운송_per = data_운송_per.assign(market_weight=data_운송_per['size_FIF_wisefn']/data_운송_per_cap)
        data_운송_div = data_운송_div.assign(market_weight=data_운송_div['size_FIF_wisefn']/data_운송_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_운송_pbr['1/pbr']*data_운송_pbr['market_weight'])
        mu_inv_per=np.sum(data_운송_per['1/per']*data_운송_per['market_weight'])
        mu_inv_div=np.sum(data_운송_div['div_yield']*data_운송_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_운송_pbr['1/pbr']-mu_inv_pbr)*data_운송_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_운송_per['1/per']-mu_inv_per)*data_운송_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_운송_div['div_yield']-mu_inv_div)*data_운송_div['market_weight']))
        
        data_운송1=(data_운송_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_운송1.name= 'pbr_z'
        data_운송2=(data_운송_per['1/per']-mu_inv_per)/std_inv_per
        data_운송2.name= 'per_z'
        data_운송3=(data_운송_div['div_yield']-mu_inv_div)/std_inv_div
        data_운송3.name= 'div_z'
              
        result_운송 = pd.concat([data_운송, data_운송1, data_운송2, data_운송3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_운송 = result_운송.assign(z_score=np.nanmean(result_운송.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_운송[result_운송['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['운송','sector_ratio']),:]
        a=a+1
    #자동차와부품 섹터
    if (np.sum(data['sector']=='자동차와부품')>0):
        data_자동차와부품 = data[data['sector']=='자동차와부품']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_자동차와부품['size_FIF_wisefn']=data_자동차와부품['size_FIF_wisefn']/1000    #size 단위 thousand
        data_자동차와부품['1/pbr']=data_자동차와부품['equity']/data_자동차와부품['size']
        data_자동차와부품['1/per']=data_자동차와부품['ni_12fw']/data_자동차와부품['size']
        data_자동차와부품['div_yield']=data_자동차와부품['cash_div']/data_자동차와부품['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_자동차와부품 = data_자동차와부품.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_자동차와부품_per = data_자동차와부품[data_자동차와부품['1/per'].notnull()]
        data_자동차와부품_pbr = data_자동차와부품[data_자동차와부품['1/pbr'].notnull()]
        data_자동차와부품_div = data_자동차와부품[data_자동차와부품['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_자동차와부품_pbr_cap = np.sum(data_자동차와부품_pbr['size_FIF_wisefn'])
        data_자동차와부품_per_cap = np.sum(data_자동차와부품_per['size_FIF_wisefn'])
        data_자동차와부품_div_cap = np.sum(data_자동차와부품_div['size_FIF_wisefn'])
    
        data_자동차와부품_pbr = data_자동차와부품_pbr.assign(market_weight=data_자동차와부품_pbr['size_FIF_wisefn']/data_자동차와부품_pbr_cap)
        data_자동차와부품_per = data_자동차와부품_per.assign(market_weight=data_자동차와부품_per['size_FIF_wisefn']/data_자동차와부품_per_cap)
        data_자동차와부품_div = data_자동차와부품_div.assign(market_weight=data_자동차와부품_div['size_FIF_wisefn']/data_자동차와부품_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_자동차와부품_pbr['1/pbr']*data_자동차와부품_pbr['market_weight'])
        mu_inv_per=np.sum(data_자동차와부품_per['1/per']*data_자동차와부품_per['market_weight'])
        mu_inv_div=np.sum(data_자동차와부품_div['div_yield']*data_자동차와부품_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_자동차와부품_pbr['1/pbr']-mu_inv_pbr)*data_자동차와부품_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_자동차와부품_per['1/per']-mu_inv_per)*data_자동차와부품_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_자동차와부품_div['div_yield']-mu_inv_div)*data_자동차와부품_div['market_weight']))
        
        data_자동차와부품1=(data_자동차와부품_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_자동차와부품1.name= 'pbr_z'
        data_자동차와부품2=(data_자동차와부품_per['1/per']-mu_inv_per)/std_inv_per
        data_자동차와부품2.name= 'per_z'
        data_자동차와부품3=(data_자동차와부품_div['div_yield']-mu_inv_div)/std_inv_div
        data_자동차와부품3.name= 'div_z'
              
        result_자동차와부품 = pd.concat([data_자동차와부품, data_자동차와부품1, data_자동차와부품2, data_자동차와부품3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_자동차와부품 = result_자동차와부품.assign(z_score=np.nanmean(result_자동차와부품.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_자동차와부품[result_자동차와부품['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['자동차와부품','sector_ratio']),:]
        a=a+1
    #내구소비재와의류 섹터
    if (np.sum(data['sector']=='내구소비재와의류')>0):
        data_내구소비재와의류 = data[data['sector']=='내구소비재와의류']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_내구소비재와의류['size_FIF_wisefn']=data_내구소비재와의류['size_FIF_wisefn']/1000    #size 단위 thousand
        data_내구소비재와의류['1/pbr']=data_내구소비재와의류['equity']/data_내구소비재와의류['size']
        data_내구소비재와의류['1/per']=data_내구소비재와의류['ni_12fw']/data_내구소비재와의류['size']
        data_내구소비재와의류['div_yield']=data_내구소비재와의류['cash_div']/data_내구소비재와의류['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_내구소비재와의류 = data_내구소비재와의류.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_내구소비재와의류_per = data_내구소비재와의류[data_내구소비재와의류['1/per'].notnull()]
        data_내구소비재와의류_pbr = data_내구소비재와의류[data_내구소비재와의류['1/pbr'].notnull()]
        data_내구소비재와의류_div = data_내구소비재와의류[data_내구소비재와의류['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_내구소비재와의류_pbr_cap = np.sum(data_내구소비재와의류_pbr['size_FIF_wisefn'])
        data_내구소비재와의류_per_cap = np.sum(data_내구소비재와의류_per['size_FIF_wisefn'])
        data_내구소비재와의류_div_cap = np.sum(data_내구소비재와의류_div['size_FIF_wisefn'])
    
        data_내구소비재와의류_pbr = data_내구소비재와의류_pbr.assign(market_weight=data_내구소비재와의류_pbr['size_FIF_wisefn']/data_내구소비재와의류_pbr_cap)
        data_내구소비재와의류_per = data_내구소비재와의류_per.assign(market_weight=data_내구소비재와의류_per['size_FIF_wisefn']/data_내구소비재와의류_per_cap)
        data_내구소비재와의류_div = data_내구소비재와의류_div.assign(market_weight=data_내구소비재와의류_div['size_FIF_wisefn']/data_내구소비재와의류_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_내구소비재와의류_pbr['1/pbr']*data_내구소비재와의류_pbr['market_weight'])
        mu_inv_per=np.sum(data_내구소비재와의류_per['1/per']*data_내구소비재와의류_per['market_weight'])
        mu_inv_div=np.sum(data_내구소비재와의류_div['div_yield']*data_내구소비재와의류_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_내구소비재와의류_pbr['1/pbr']-mu_inv_pbr)*data_내구소비재와의류_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_내구소비재와의류_per['1/per']-mu_inv_per)*data_내구소비재와의류_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_내구소비재와의류_div['div_yield']-mu_inv_div)*data_내구소비재와의류_div['market_weight']))
        
        data_내구소비재와의류1=(data_내구소비재와의류_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_내구소비재와의류1.name= 'pbr_z'
        data_내구소비재와의류2=(data_내구소비재와의류_per['1/per']-mu_inv_per)/std_inv_per
        data_내구소비재와의류2.name= 'per_z'
        data_내구소비재와의류3=(data_내구소비재와의류_div['div_yield']-mu_inv_div)/std_inv_div
        data_내구소비재와의류3.name= 'div_z'
              
        result_내구소비재와의류 = pd.concat([data_내구소비재와의류, data_내구소비재와의류1, data_내구소비재와의류2, data_내구소비재와의류3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_내구소비재와의류 = result_내구소비재와의류.assign(z_score=np.nanmean(result_내구소비재와의류.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_내구소비재와의류[result_내구소비재와의류['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['내구소비재와의류','sector_ratio']),:]
        a=a+1
    

    #호텔_레스토랑_레저 섹터
    if (np.sum(data['sector']=='호텔,레스토랑,레저등')>0):
        data_호텔_레스토랑_레저 = data[data['sector']=='호텔,레스토랑,레저등']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_호텔_레스토랑_레저['size_FIF_wisefn']=data_호텔_레스토랑_레저['size_FIF_wisefn']/1000    #size 단위 thousand
        data_호텔_레스토랑_레저['1/pbr']=data_호텔_레스토랑_레저['equity']/data_호텔_레스토랑_레저['size']
        data_호텔_레스토랑_레저['1/per']=data_호텔_레스토랑_레저['ni_12fw']/data_호텔_레스토랑_레저['size']
        data_호텔_레스토랑_레저['div_yield']=data_호텔_레스토랑_레저['cash_div']/data_호텔_레스토랑_레저['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_호텔_레스토랑_레저 = data_호텔_레스토랑_레저.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_호텔_레스토랑_레저_per = data_호텔_레스토랑_레저[data_호텔_레스토랑_레저['1/per'].notnull()]
        data_호텔_레스토랑_레저_pbr = data_호텔_레스토랑_레저[data_호텔_레스토랑_레저['1/pbr'].notnull()]
        data_호텔_레스토랑_레저_div = data_호텔_레스토랑_레저[data_호텔_레스토랑_레저['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_호텔_레스토랑_레저_pbr_cap = np.sum(data_호텔_레스토랑_레저_pbr['size_FIF_wisefn'])
        data_호텔_레스토랑_레저_per_cap = np.sum(data_호텔_레스토랑_레저_per['size_FIF_wisefn'])
        data_호텔_레스토랑_레저_div_cap = np.sum(data_호텔_레스토랑_레저_div['size_FIF_wisefn'])
    
        data_호텔_레스토랑_레저_pbr = data_호텔_레스토랑_레저_pbr.assign(market_weight=data_호텔_레스토랑_레저_pbr['size_FIF_wisefn']/data_호텔_레스토랑_레저_pbr_cap)
        data_호텔_레스토랑_레저_per = data_호텔_레스토랑_레저_per.assign(market_weight=data_호텔_레스토랑_레저_per['size_FIF_wisefn']/data_호텔_레스토랑_레저_per_cap)
        data_호텔_레스토랑_레저_div = data_호텔_레스토랑_레저_div.assign(market_weight=data_호텔_레스토랑_레저_div['size_FIF_wisefn']/data_호텔_레스토랑_레저_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_호텔_레스토랑_레저_pbr['1/pbr']*data_호텔_레스토랑_레저_pbr['market_weight'])
        mu_inv_per=np.sum(data_호텔_레스토랑_레저_per['1/per']*data_호텔_레스토랑_레저_per['market_weight'])
        mu_inv_div=np.sum(data_호텔_레스토랑_레저_div['div_yield']*data_호텔_레스토랑_레저_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_호텔_레스토랑_레저_pbr['1/pbr']-mu_inv_pbr)*data_호텔_레스토랑_레저_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_호텔_레스토랑_레저_per['1/per']-mu_inv_per)*data_호텔_레스토랑_레저_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_호텔_레스토랑_레저_div['div_yield']-mu_inv_div)*data_호텔_레스토랑_레저_div['market_weight']))
        
        data_호텔_레스토랑_레저1=(data_호텔_레스토랑_레저_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_호텔_레스토랑_레저1.name= 'pbr_z'
        data_호텔_레스토랑_레저2=(data_호텔_레스토랑_레저_per['1/per']-mu_inv_per)/std_inv_per
        data_호텔_레스토랑_레저2.name= 'per_z'
        data_호텔_레스토랑_레저3=(data_호텔_레스토랑_레저_div['div_yield']-mu_inv_div)/std_inv_div
        data_호텔_레스토랑_레저3.name= 'div_z'
              
        result_호텔_레스토랑_레저 = pd.concat([data_호텔_레스토랑_레저, data_호텔_레스토랑_레저1, data_호텔_레스토랑_레저2, data_호텔_레스토랑_레저3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_호텔_레스토랑_레저 = result_호텔_레스토랑_레저.assign(z_score=np.nanmean(result_호텔_레스토랑_레저.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_호텔_레스토랑_레저[result_호텔_레스토랑_레저['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['호텔,레스토랑,레저등','sector_ratio']),:]
        a=a+1
    #미디어 섹터
    if (np.sum(data['sector']=='미디어')>0):
        data_미디어 = data[data['sector']=='미디어']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_미디어['size_FIF_wisefn']=data_미디어['size_FIF_wisefn']/1000    #size 단위 thousand
        data_미디어['1/pbr']=data_미디어['equity']/data_미디어['size']
        data_미디어['1/per']=data_미디어['ni_12fw']/data_미디어['size']
        data_미디어['div_yield']=data_미디어['cash_div']/data_미디어['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_미디어 = data_미디어.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_미디어_per = data_미디어[data_미디어['1/per'].notnull()]
        data_미디어_pbr = data_미디어[data_미디어['1/pbr'].notnull()]
        data_미디어_div = data_미디어[data_미디어['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_미디어_pbr_cap = np.sum(data_미디어_pbr['size_FIF_wisefn'])
        data_미디어_per_cap = np.sum(data_미디어_per['size_FIF_wisefn'])
        data_미디어_div_cap = np.sum(data_미디어_div['size_FIF_wisefn'])
    
        data_미디어_pbr = data_미디어_pbr.assign(market_weight=data_미디어_pbr['size_FIF_wisefn']/data_미디어_pbr_cap)
        data_미디어_per = data_미디어_per.assign(market_weight=data_미디어_per['size_FIF_wisefn']/data_미디어_per_cap)
        data_미디어_div = data_미디어_div.assign(market_weight=data_미디어_div['size_FIF_wisefn']/data_미디어_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_미디어_pbr['1/pbr']*data_미디어_pbr['market_weight'])
        mu_inv_per=np.sum(data_미디어_per['1/per']*data_미디어_per['market_weight'])
        mu_inv_div=np.sum(data_미디어_div['div_yield']*data_미디어_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_미디어_pbr['1/pbr']-mu_inv_pbr)*data_미디어_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_미디어_per['1/per']-mu_inv_per)*data_미디어_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_미디어_div['div_yield']-mu_inv_div)*data_미디어_div['market_weight']))
        
        data_미디어1=(data_미디어_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_미디어1.name= 'pbr_z'
        data_미디어2=(data_미디어_per['1/per']-mu_inv_per)/std_inv_per
        data_미디어2.name= 'per_z'
        data_미디어3=(data_미디어_div['div_yield']-mu_inv_div)/std_inv_div
        data_미디어3.name= 'div_z'
              
        result_미디어 = pd.concat([data_미디어, data_미디어1, data_미디어2, data_미디어3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_미디어 = result_미디어.assign(z_score=np.nanmean(result_미디어.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_미디어[result_미디어['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['미디어','sector_ratio']),:]
        
        a=a+1
    #소매(유통) 섹터
    if (np.sum(data['sector']=='소매(유통)')>0):
        data_소매_유통 = data[data['sector']=='소매(유통)']
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_소매_유통['size_FIF_wisefn']=data_소매_유통['size_FIF_wisefn']/1000    #size 단위 thousand
        data_소매_유통['1/pbr']=data_소매_유통['equity']/data_소매_유통['size']
        data_소매_유통['1/per']=data_소매_유통['ni_12fw']/data_소매_유통['size']
        data_소매_유통['div_yield']=data_소매_유통['cash_div']/data_소매_유통['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_소매_유통 = data_소매_유통.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_소매_유통_per = data_소매_유통[data_소매_유통['1/per'].notnull()]
        data_소매_유통_pbr = data_소매_유통[data_소매_유통['1/pbr'].notnull()]
        data_소매_유통_div = data_소매_유통[data_소매_유통['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_소매_유통_pbr_cap = np.sum(data_소매_유통_pbr['size_FIF_wisefn'])
        data_소매_유통_per_cap = np.sum(data_소매_유통_per['size_FIF_wisefn'])
        data_소매_유통_div_cap = np.sum(data_소매_유통_div['size_FIF_wisefn'])
    
        data_소매_유통_pbr = data_소매_유통_pbr.assign(market_weight=data_소매_유통_pbr['size_FIF_wisefn']/data_소매_유통_pbr_cap)
        data_소매_유통_per = data_소매_유통_per.assign(market_weight=data_소매_유통_per['size_FIF_wisefn']/data_소매_유통_per_cap)
        data_소매_유통_div = data_소매_유통_div.assign(market_weight=data_소매_유통_div['size_FIF_wisefn']/data_소매_유통_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_소매_유통_pbr['1/pbr']*data_소매_유통_pbr['market_weight'])
        mu_inv_per=np.sum(data_소매_유통_per['1/per']*data_소매_유통_per['market_weight'])
        mu_inv_div=np.sum(data_소매_유통_div['div_yield']*data_소매_유통_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_소매_유통_pbr['1/pbr']-mu_inv_pbr)*data_소매_유통_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_소매_유통_per['1/per']-mu_inv_per)*data_소매_유통_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_소매_유통_div['div_yield']-mu_inv_div)*data_소매_유통_div['market_weight']))
        
        data_소매_유통1=(data_소매_유통_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_소매_유통1.name= 'pbr_z'
        data_소매_유통2=(data_소매_유통_per['1/per']-mu_inv_per)/std_inv_per
        data_소매_유통2.name= 'per_z'
        data_소매_유통3=(data_소매_유통_div['div_yield']-mu_inv_div)/std_inv_div
        data_소매_유통3.name= 'div_z'
              
        result_소매_유통 = pd.concat([data_소매_유통, data_소매_유통1, data_소매_유통2, data_소매_유통3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_소매_유통 = result_소매_유통.assign(z_score=np.nanmean(result_소매_유통.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_소매_유통[result_소매_유통['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['소매(유통)','sector_ratio']),:]
        
        a=a+1
        
     #교육서비스 섹터
    if (np.sum(data['sector']=='교육서비스')>0):
        data_교육서비스 = data[data['sector']=="교육서비스"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_교육서비스['size_FIF_wisefn']=data_교육서비스['size_FIF_wisefn']/1000    #size 단위 thousand
        data_교육서비스['1/pbr']=data_교육서비스['equity']/data_교육서비스['size']
        data_교육서비스['1/per']=data_교육서비스['ni_12fw']/data_교육서비스['size']
        data_교육서비스['div_yield']=data_교육서비스['cash_div']/data_교육서비스['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_교육서비스 = data_교육서비스.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_교육서비스_per = data_교육서비스[data_교육서비스['1/per'].notnull()]
        data_교육서비스_pbr = data_교육서비스[data_교육서비스['1/pbr'].notnull()]
        data_교육서비스_div = data_교육서비스[data_교육서비스['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_교육서비스_pbr_cap = np.sum(data_교육서비스_pbr['size_FIF_wisefn'])
        data_교육서비스_per_cap = np.sum(data_교육서비스_per['size_FIF_wisefn'])
        data_교육서비스_div_cap = np.sum(data_교육서비스_div['size_FIF_wisefn'])
    
        data_교육서비스_pbr = data_교육서비스_pbr.assign(market_weight=data_교육서비스_pbr['size_FIF_wisefn']/data_교육서비스_pbr_cap)
        data_교육서비스_per = data_교육서비스_per.assign(market_weight=data_교육서비스_per['size_FIF_wisefn']/data_교육서비스_per_cap)
        data_교육서비스_div = data_교육서비스_div.assign(market_weight=data_교육서비스_div['size_FIF_wisefn']/data_교육서비스_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_교육서비스_pbr['1/pbr']*data_교육서비스_pbr['market_weight'])
        mu_inv_per=np.sum(data_교육서비스_per['1/per']*data_교육서비스_per['market_weight'])
        mu_inv_div=np.sum(data_교육서비스_div['div_yield']*data_교육서비스_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_교육서비스_pbr['1/pbr']-mu_inv_pbr)*data_교육서비스_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_교육서비스_per['1/per']-mu_inv_per)*data_교육서비스_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_교육서비스_div['div_yield']-mu_inv_div)*data_교육서비스_div['market_weight']))
        
        data_교육서비스1=(data_교육서비스_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_교육서비스1.name= 'pbr_z'
        data_교육서비스2=(data_교육서비스_per['1/per']-mu_inv_per)/std_inv_per
        data_교육서비스2.name= 'per_z'
        data_교육서비스3=(data_교육서비스_div['div_yield']-mu_inv_div)/std_inv_div
        data_교육서비스3.name= 'div_z'
              
        result_교육서비스 = pd.concat([data_교육서비스, data_교육서비스1, data_교육서비스2, data_교육서비스3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_교육서비스 = result_교육서비스.assign(z_score=np.nanmean(result_교육서비스.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_교육서비스[result_교육서비스['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['교육서비스','sector_ratio']),:]
        
        a=a+1
    
     #식품과기본식료품소매 섹터
    if (np.sum(data['sector']=='식품과기본식료품소매')>0):
        data_식품과기본식료품소매 = data[data['sector']=="식품과기본식료품소매"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_식품과기본식료품소매['size_FIF_wisefn']=data_식품과기본식료품소매['size_FIF_wisefn']/1000    #size 단위 thousand
        data_식품과기본식료품소매['1/pbr']=data_식품과기본식료품소매['equity']/data_식품과기본식료품소매['size']
        data_식품과기본식료품소매['1/per']=data_식품과기본식료품소매['ni_12fw']/data_식품과기본식료품소매['size']
        data_식품과기본식료품소매['div_yield']=data_식품과기본식료품소매['cash_div']/data_식품과기본식료품소매['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_식품과기본식료품소매 = data_식품과기본식료품소매.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_식품과기본식료품소매_per = data_식품과기본식료품소매[data_식품과기본식료품소매['1/per'].notnull()]
        data_식품과기본식료품소매_pbr = data_식품과기본식료품소매[data_식품과기본식료품소매['1/pbr'].notnull()]
        data_식품과기본식료품소매_div = data_식품과기본식료품소매[data_식품과기본식료품소매['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_식품과기본식료품소매_pbr_cap = np.sum(data_식품과기본식료품소매_pbr['size_FIF_wisefn'])
        data_식품과기본식료품소매_per_cap = np.sum(data_식품과기본식료품소매_per['size_FIF_wisefn'])
        data_식품과기본식료품소매_div_cap = np.sum(data_식품과기본식료품소매_div['size_FIF_wisefn'])
    
        data_식품과기본식료품소매_pbr = data_식품과기본식료품소매_pbr.assign(market_weight=data_식품과기본식료품소매_pbr['size_FIF_wisefn']/data_식품과기본식료품소매_pbr_cap)
        data_식품과기본식료품소매_per = data_식품과기본식료품소매_per.assign(market_weight=data_식품과기본식료품소매_per['size_FIF_wisefn']/data_식품과기본식료품소매_per_cap)
        data_식품과기본식료품소매_div = data_식품과기본식료품소매_div.assign(market_weight=data_식품과기본식료품소매_div['size_FIF_wisefn']/data_식품과기본식료품소매_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_식품과기본식료품소매_pbr['1/pbr']*data_식품과기본식료품소매_pbr['market_weight'])
        mu_inv_per=np.sum(data_식품과기본식료품소매_per['1/per']*data_식품과기본식료품소매_per['market_weight'])
        mu_inv_div=np.sum(data_식품과기본식료품소매_div['div_yield']*data_식품과기본식료품소매_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_식품과기본식료품소매_pbr['1/pbr']-mu_inv_pbr)*data_식품과기본식료품소매_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_식품과기본식료품소매_per['1/per']-mu_inv_per)*data_식품과기본식료품소매_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_식품과기본식료품소매_div['div_yield']-mu_inv_div)*data_식품과기본식료품소매_div['market_weight']))
        
        data_식품과기본식료품소매1=(data_식품과기본식료품소매_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_식품과기본식료품소매1.name= 'pbr_z'
        data_식품과기본식료품소매2=(data_식품과기본식료품소매_per['1/per']-mu_inv_per)/std_inv_per
        data_식품과기본식료품소매2.name= 'per_z'
        data_식품과기본식료품소매3=(data_식품과기본식료품소매_div['div_yield']-mu_inv_div)/std_inv_div
        data_식품과기본식료품소매3.name= 'div_z'
              
        result_식품과기본식료품소매 = pd.concat([data_식품과기본식료품소매, data_식품과기본식료품소매1, data_식품과기본식료품소매2, data_식품과기본식료품소매3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_식품과기본식료품소매 = result_식품과기본식료품소매.assign(z_score=np.nanmean(result_식품과기본식료품소매.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_식품과기본식료품소매[result_식품과기본식료품소매['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['식품과기본식료품소매','sector_ratio']),:]
        
        a=a+1
    
     #식품,음료,담배 섹터
    if (np.sum(data['sector']=='식품,음료,담배')>0):
        data_식품_음료_담배 = data[data['sector']=="식품,음료,담배"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_식품_음료_담배['size_FIF_wisefn']=data_식품_음료_담배['size_FIF_wisefn']/1000    #size 단위 thousand
        data_식품_음료_담배['1/pbr']=data_식품_음료_담배['equity']/data_식품_음료_담배['size']
        data_식품_음료_담배['1/per']=data_식품_음료_담배['ni_12fw']/data_식품_음료_담배['size']
        data_식품_음료_담배['div_yield']=data_식품_음료_담배['cash_div']/data_식품_음료_담배['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_식품_음료_담배 = data_식품_음료_담배.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_식품_음료_담배_per = data_식품_음료_담배[data_식품_음료_담배['1/per'].notnull()]
        data_식품_음료_담배_pbr = data_식품_음료_담배[data_식품_음료_담배['1/pbr'].notnull()]
        data_식품_음료_담배_div = data_식품_음료_담배[data_식품_음료_담배['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_식품_음료_담배_pbr_cap = np.sum(data_식품_음료_담배_pbr['size_FIF_wisefn'])
        data_식품_음료_담배_per_cap = np.sum(data_식품_음료_담배_per['size_FIF_wisefn'])
        data_식품_음료_담배_div_cap = np.sum(data_식품_음료_담배_div['size_FIF_wisefn'])
    
        data_식품_음료_담배_pbr = data_식품_음료_담배_pbr.assign(market_weight=data_식품_음료_담배_pbr['size_FIF_wisefn']/data_식품_음료_담배_pbr_cap)
        data_식품_음료_담배_per = data_식품_음료_담배_per.assign(market_weight=data_식품_음료_담배_per['size_FIF_wisefn']/data_식품_음료_담배_per_cap)
        data_식품_음료_담배_div = data_식품_음료_담배_div.assign(market_weight=data_식품_음료_담배_div['size_FIF_wisefn']/data_식품_음료_담배_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_식품_음료_담배_pbr['1/pbr']*data_식품_음료_담배_pbr['market_weight'])
        mu_inv_per=np.sum(data_식품_음료_담배_per['1/per']*data_식품_음료_담배_per['market_weight'])
        mu_inv_div=np.sum(data_식품_음료_담배_div['div_yield']*data_식품_음료_담배_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_식품_음료_담배_pbr['1/pbr']-mu_inv_pbr)*data_식품_음료_담배_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_식품_음료_담배_per['1/per']-mu_inv_per)*data_식품_음료_담배_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_식품_음료_담배_div['div_yield']-mu_inv_div)*data_식품_음료_담배_div['market_weight']))
        
        data_식품_음료_담배1=(data_식품_음료_담배_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_식품_음료_담배1.name= 'pbr_z'
        data_식품_음료_담배2=(data_식품_음료_담배_per['1/per']-mu_inv_per)/std_inv_per
        data_식품_음료_담배2.name= 'per_z'
        data_식품_음료_담배3=(data_식품_음료_담배_div['div_yield']-mu_inv_div)/std_inv_div
        data_식품_음료_담배3.name= 'div_z'
              
        result_식품_음료_담배 = pd.concat([data_식품_음료_담배, data_식품_음료_담배1, data_식품_음료_담배2, data_식품_음료_담배3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_식품_음료_담배 = result_식품_음료_담배.assign(z_score=np.nanmean(result_식품_음료_담배.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_식품_음료_담배[result_식품_음료_담배['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['식품,음료,담배','sector_ratio']),:]
        a=a+1
    
     #가정용품과개인용품 섹터
    if (np.sum(data['sector']=='가정용품과개인용품')>0):
        data_가정용품과개인용품 = data[data['sector']=="가정용품과개인용품"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_가정용품과개인용품['size_FIF_wisefn']=data_가정용품과개인용품['size_FIF_wisefn']/1000    #size 단위 thousand
        data_가정용품과개인용품['1/pbr']=data_가정용품과개인용품['equity']/data_가정용품과개인용품['size']
        data_가정용품과개인용품['1/per']=data_가정용품과개인용품['ni_12fw']/data_가정용품과개인용품['size']
        data_가정용품과개인용품['div_yield']=data_가정용품과개인용품['cash_div']/data_가정용품과개인용품['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_가정용품과개인용품 = data_가정용품과개인용품.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_가정용품과개인용품_per = data_가정용품과개인용품[data_가정용품과개인용품['1/per'].notnull()]
        data_가정용품과개인용품_pbr = data_가정용품과개인용품[data_가정용품과개인용품['1/pbr'].notnull()]
        data_가정용품과개인용품_div = data_가정용품과개인용품[data_가정용품과개인용품['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_가정용품과개인용품_pbr_cap = np.sum(data_가정용품과개인용품_pbr['size_FIF_wisefn'])
        data_가정용품과개인용품_per_cap = np.sum(data_가정용품과개인용품_per['size_FIF_wisefn'])
        data_가정용품과개인용품_div_cap = np.sum(data_가정용품과개인용품_div['size_FIF_wisefn'])
    
        data_가정용품과개인용품_pbr = data_가정용품과개인용품_pbr.assign(market_weight=data_가정용품과개인용품_pbr['size_FIF_wisefn']/data_가정용품과개인용품_pbr_cap)
        data_가정용품과개인용품_per = data_가정용품과개인용품_per.assign(market_weight=data_가정용품과개인용품_per['size_FIF_wisefn']/data_가정용품과개인용품_per_cap)
        data_가정용품과개인용품_div = data_가정용품과개인용품_div.assign(market_weight=data_가정용품과개인용품_div['size_FIF_wisefn']/data_가정용품과개인용품_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_가정용품과개인용품_pbr['1/pbr']*data_가정용품과개인용품_pbr['market_weight'])
        mu_inv_per=np.sum(data_가정용품과개인용품_per['1/per']*data_가정용품과개인용품_per['market_weight'])
        mu_inv_div=np.sum(data_가정용품과개인용품_div['div_yield']*data_가정용품과개인용품_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_가정용품과개인용품_pbr['1/pbr']-mu_inv_pbr)*data_가정용품과개인용품_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_가정용품과개인용품_per['1/per']-mu_inv_per)*data_가정용품과개인용품_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_가정용품과개인용품_div['div_yield']-mu_inv_div)*data_가정용품과개인용품_div['market_weight']))
        
        data_가정용품과개인용품1=(data_가정용품과개인용품_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_가정용품과개인용품1.name= 'pbr_z'
        data_가정용품과개인용품2=(data_가정용품과개인용품_per['1/per']-mu_inv_per)/std_inv_per
        data_가정용품과개인용품2.name= 'per_z'
        data_가정용품과개인용품3=(data_가정용품과개인용품_div['div_yield']-mu_inv_div)/std_inv_div
        data_가정용품과개인용품3.name= 'div_z'
              
        result_가정용품과개인용품 = pd.concat([data_가정용품과개인용품, data_가정용품과개인용품1, data_가정용품과개인용품2, data_가정용품과개인용품3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_가정용품과개인용품 = result_가정용품과개인용품.assign(z_score=np.nanmean(result_가정용품과개인용품.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_가정용품과개인용품[result_가정용품과개인용품['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['가정용품과개인용품','sector_ratio']),:]
        
        a=a+1
    
     #건강관리장비와서비스 섹터
    if (np.sum(data['sector']=='건강관리장비와서비스')>0):
        data_건강관리장비와서비스 = data[data['sector']=="건강관리장비와서비스"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_건강관리장비와서비스['size_FIF_wisefn']=data_건강관리장비와서비스['size_FIF_wisefn']/1000    #size 단위 thousand
        data_건강관리장비와서비스['1/pbr']=data_건강관리장비와서비스['equity']/data_건강관리장비와서비스['size']
        data_건강관리장비와서비스['1/per']=data_건강관리장비와서비스['ni_12fw']/data_건강관리장비와서비스['size']
        data_건강관리장비와서비스['div_yield']=data_건강관리장비와서비스['cash_div']/data_건강관리장비와서비스['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_건강관리장비와서비스 = data_건강관리장비와서비스.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_건강관리장비와서비스_per = data_건강관리장비와서비스[data_건강관리장비와서비스['1/per'].notnull()]
        data_건강관리장비와서비스_pbr = data_건강관리장비와서비스[data_건강관리장비와서비스['1/pbr'].notnull()]
        data_건강관리장비와서비스_div = data_건강관리장비와서비스[data_건강관리장비와서비스['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_건강관리장비와서비스_pbr_cap = np.sum(data_건강관리장비와서비스_pbr['size_FIF_wisefn'])
        data_건강관리장비와서비스_per_cap = np.sum(data_건강관리장비와서비스_per['size_FIF_wisefn'])
        data_건강관리장비와서비스_div_cap = np.sum(data_건강관리장비와서비스_div['size_FIF_wisefn'])
    
        data_건강관리장비와서비스_pbr = data_건강관리장비와서비스_pbr.assign(market_weight=data_건강관리장비와서비스_pbr['size_FIF_wisefn']/data_건강관리장비와서비스_pbr_cap)
        data_건강관리장비와서비스_per = data_건강관리장비와서비스_per.assign(market_weight=data_건강관리장비와서비스_per['size_FIF_wisefn']/data_건강관리장비와서비스_per_cap)
        data_건강관리장비와서비스_div = data_건강관리장비와서비스_div.assign(market_weight=data_건강관리장비와서비스_div['size_FIF_wisefn']/data_건강관리장비와서비스_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_건강관리장비와서비스_pbr['1/pbr']*data_건강관리장비와서비스_pbr['market_weight'])
        mu_inv_per=np.sum(data_건강관리장비와서비스_per['1/per']*data_건강관리장비와서비스_per['market_weight'])
        mu_inv_div=np.sum(data_건강관리장비와서비스_div['div_yield']*data_건강관리장비와서비스_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_건강관리장비와서비스_pbr['1/pbr']-mu_inv_pbr)*data_건강관리장비와서비스_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_건강관리장비와서비스_per['1/per']-mu_inv_per)*data_건강관리장비와서비스_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_건강관리장비와서비스_div['div_yield']-mu_inv_div)*data_건강관리장비와서비스_div['market_weight']))
        
        data_건강관리장비와서비스1=(data_건강관리장비와서비스_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_건강관리장비와서비스1.name= 'pbr_z'
        data_건강관리장비와서비스2=(data_건강관리장비와서비스_per['1/per']-mu_inv_per)/std_inv_per
        data_건강관리장비와서비스2.name= 'per_z'
        data_건강관리장비와서비스3=(data_건강관리장비와서비스_div['div_yield']-mu_inv_div)/std_inv_div
        data_건강관리장비와서비스3.name= 'div_z'
              
        result_건강관리장비와서비스 = pd.concat([data_건강관리장비와서비스, data_건강관리장비와서비스1, data_건강관리장비와서비스2, data_건강관리장비와서비스3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_건강관리장비와서비스 = result_건강관리장비와서비스.assign(z_score=np.nanmean(result_건강관리장비와서비스.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_건강관리장비와서비스[result_건강관리장비와서비스['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['건강관리장비와서비스','sector_ratio']),:]
        
        a=a+1
    
     #제약과생물공학 섹터
    if (np.sum(data['sector']=='제약과생물공학')>0):
        data_제약과생물공학 = data[data['sector']=="제약과생물공학"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_제약과생물공학['size_FIF_wisefn']=data_제약과생물공학['size_FIF_wisefn']/1000    #size 단위 thousand
        data_제약과생물공학['1/pbr']=data_제약과생물공학['equity']/data_제약과생물공학['size']
        data_제약과생물공학['1/per']=data_제약과생물공학['ni_12fw']/data_제약과생물공학['size']
        data_제약과생물공학['div_yield']=data_제약과생물공학['cash_div']/data_제약과생물공학['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_제약과생물공학 = data_제약과생물공학.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_제약과생물공학_per = data_제약과생물공학[data_제약과생물공학['1/per'].notnull()]
        data_제약과생물공학_pbr = data_제약과생물공학[data_제약과생물공학['1/pbr'].notnull()]
        data_제약과생물공학_div = data_제약과생물공학[data_제약과생물공학['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_제약과생물공학_pbr_cap = np.sum(data_제약과생물공학_pbr['size_FIF_wisefn'])
        data_제약과생물공학_per_cap = np.sum(data_제약과생물공학_per['size_FIF_wisefn'])
        data_제약과생물공학_div_cap = np.sum(data_제약과생물공학_div['size_FIF_wisefn'])
    
        data_제약과생물공학_pbr = data_제약과생물공학_pbr.assign(market_weight=data_제약과생물공학_pbr['size_FIF_wisefn']/data_제약과생물공학_pbr_cap)
        data_제약과생물공학_per = data_제약과생물공학_per.assign(market_weight=data_제약과생물공학_per['size_FIF_wisefn']/data_제약과생물공학_per_cap)
        data_제약과생물공학_div = data_제약과생물공학_div.assign(market_weight=data_제약과생물공학_div['size_FIF_wisefn']/data_제약과생물공학_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_제약과생물공학_pbr['1/pbr']*data_제약과생물공학_pbr['market_weight'])
        mu_inv_per=np.sum(data_제약과생물공학_per['1/per']*data_제약과생물공학_per['market_weight'])
        mu_inv_div=np.sum(data_제약과생물공학_div['div_yield']*data_제약과생물공학_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_제약과생물공학_pbr['1/pbr']-mu_inv_pbr)*data_제약과생물공학_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_제약과생물공학_per['1/per']-mu_inv_per)*data_제약과생물공학_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_제약과생물공학_div['div_yield']-mu_inv_div)*data_제약과생물공학_div['market_weight']))
        
        data_제약과생물공학1=(data_제약과생물공학_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_제약과생물공학1.name= 'pbr_z'
        data_제약과생물공학2=(data_제약과생물공학_per['1/per']-mu_inv_per)/std_inv_per
        data_제약과생물공학2.name= 'per_z'
        data_제약과생물공학3=(data_제약과생물공학_div['div_yield']-mu_inv_div)/std_inv_div
        data_제약과생물공학3.name= 'div_z'
              
        result_제약과생물공학 = pd.concat([data_제약과생물공학, data_제약과생물공학1, data_제약과생물공학2, data_제약과생물공학3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_제약과생물공학 = result_제약과생물공학.assign(z_score=np.nanmean(result_제약과생물공학.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_제약과생물공학[result_제약과생물공학['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['제약과생물공학','sector_ratio']),:]
        
        a=a+1
   
     #은행 섹터
    if (np.sum(data['sector']=='은행')>0):
        data_은행 = data[data['sector']=="은행"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_은행['size_FIF_wisefn']=data_은행['size_FIF_wisefn']/1000    #size 단위 thousand
        data_은행['1/pbr']=data_은행['equity']/data_은행['size']
        data_은행['1/per']=data_은행['ni_12fw']/data_은행['size']
        data_은행['div_yield']=data_은행['cash_div']/data_은행['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_은행 = data_은행.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_은행_per = data_은행[data_은행['1/per'].notnull()]
        data_은행_pbr = data_은행[data_은행['1/pbr'].notnull()]
        data_은행_div = data_은행[data_은행['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_은행_pbr_cap = np.sum(data_은행_pbr['size_FIF_wisefn'])
        data_은행_per_cap = np.sum(data_은행_per['size_FIF_wisefn'])
        data_은행_div_cap = np.sum(data_은행_div['size_FIF_wisefn'])
    
        data_은행_pbr = data_은행_pbr.assign(market_weight=data_은행_pbr['size_FIF_wisefn']/data_은행_pbr_cap)
        data_은행_per = data_은행_per.assign(market_weight=data_은행_per['size_FIF_wisefn']/data_은행_per_cap)
        data_은행_div = data_은행_div.assign(market_weight=data_은행_div['size_FIF_wisefn']/data_은행_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_은행_pbr['1/pbr']*data_은행_pbr['market_weight'])
        mu_inv_per=np.sum(data_은행_per['1/per']*data_은행_per['market_weight'])
        mu_inv_div=np.sum(data_은행_div['div_yield']*data_은행_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_은행_pbr['1/pbr']-mu_inv_pbr)*data_은행_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_은행_per['1/per']-mu_inv_per)*data_은행_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_은행_div['div_yield']-mu_inv_div)*data_은행_div['market_weight']))
        
        data_은행1=(data_은행_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_은행1.name= 'pbr_z'
        data_은행2=(data_은행_per['1/per']-mu_inv_per)/std_inv_per
        data_은행2.name= 'per_z'
        data_은행3=(data_은행_div['div_yield']-mu_inv_div)/std_inv_div
        data_은행3.name= 'div_z'
              
        result_은행 = pd.concat([data_은행, data_은행1, data_은행2, data_은행3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_은행 = result_은행.assign(z_score=np.nanmean(result_은행.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_은행[result_은행['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['은행','sector_ratio']),:]
        
        a=a+1
    
     #증권 섹터
    if (np.sum(data['sector']=='증권')>0):
        data_증권 = data[data['sector']=="증권"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_증권['size_FIF_wisefn']=data_증권['size_FIF_wisefn']/1000    #size 단위 thousand
        data_증권['1/pbr']=data_증권['equity']/data_증권['size']
        data_증권['1/per']=data_증권['ni_12fw']/data_증권['size']
        data_증권['div_yield']=data_증권['cash_div']/data_증권['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_증권 = data_증권.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_증권_per = data_증권[data_증권['1/per'].notnull()]
        data_증권_pbr = data_증권[data_증권['1/pbr'].notnull()]
        data_증권_div = data_증권[data_증권['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_증권_pbr_cap = np.sum(data_증권_pbr['size_FIF_wisefn'])
        data_증권_per_cap = np.sum(data_증권_per['size_FIF_wisefn'])
        data_증권_div_cap = np.sum(data_증권_div['size_FIF_wisefn'])
    
        data_증권_pbr = data_증권_pbr.assign(market_weight=data_증권_pbr['size_FIF_wisefn']/data_증권_pbr_cap)
        data_증권_per = data_증권_per.assign(market_weight=data_증권_per['size_FIF_wisefn']/data_증권_per_cap)
        data_증권_div = data_증권_div.assign(market_weight=data_증권_div['size_FIF_wisefn']/data_증권_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_증권_pbr['1/pbr']*data_증권_pbr['market_weight'])
        mu_inv_per=np.sum(data_증권_per['1/per']*data_증권_per['market_weight'])
        mu_inv_div=np.sum(data_증권_div['div_yield']*data_증권_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_증권_pbr['1/pbr']-mu_inv_pbr)*data_증권_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_증권_per['1/per']-mu_inv_per)*data_증권_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_증권_div['div_yield']-mu_inv_div)*data_증권_div['market_weight']))
        
        data_증권1=(data_증권_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_증권1.name= 'pbr_z'
        data_증권2=(data_증권_per['1/per']-mu_inv_per)/std_inv_per
        data_증권2.name= 'per_z'
        data_증권3=(data_증권_div['div_yield']-mu_inv_div)/std_inv_div
        data_증권3.name= 'div_z'
              
        result_증권 = pd.concat([data_증권, data_증권1, data_증권2, data_증권3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_증권 = result_증권.assign(z_score=np.nanmean(result_증권.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_증권[result_증권['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['증권','sector_ratio']),:]
        
        a=a+1
    
     #다각화된금융 섹터
    if (np.sum(data['sector']=='다각화된금융')>0):
        data_다각화된금융 = data[data['sector']=="다각화된금융"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_다각화된금융['size_FIF_wisefn']=data_다각화된금융['size_FIF_wisefn']/1000    #size 단위 thousand
        data_다각화된금융['1/pbr']=data_다각화된금융['equity']/data_다각화된금융['size']
        data_다각화된금융['1/per']=data_다각화된금융['ni_12fw']/data_다각화된금융['size']
        data_다각화된금융['div_yield']=data_다각화된금융['cash_div']/data_다각화된금융['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_다각화된금융 = data_다각화된금융.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_다각화된금융_per = data_다각화된금융[data_다각화된금융['1/per'].notnull()]
        data_다각화된금융_pbr = data_다각화된금융[data_다각화된금융['1/pbr'].notnull()]
        data_다각화된금융_div = data_다각화된금융[data_다각화된금융['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_다각화된금융_pbr_cap = np.sum(data_다각화된금융_pbr['size_FIF_wisefn'])
        data_다각화된금융_per_cap = np.sum(data_다각화된금융_per['size_FIF_wisefn'])
        data_다각화된금융_div_cap = np.sum(data_다각화된금융_div['size_FIF_wisefn'])
    
        data_다각화된금융_pbr = data_다각화된금융_pbr.assign(market_weight=data_다각화된금융_pbr['size_FIF_wisefn']/data_다각화된금융_pbr_cap)
        data_다각화된금융_per = data_다각화된금융_per.assign(market_weight=data_다각화된금융_per['size_FIF_wisefn']/data_다각화된금융_per_cap)
        data_다각화된금융_div = data_다각화된금융_div.assign(market_weight=data_다각화된금융_div['size_FIF_wisefn']/data_다각화된금융_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_다각화된금융_pbr['1/pbr']*data_다각화된금융_pbr['market_weight'])
        mu_inv_per=np.sum(data_다각화된금융_per['1/per']*data_다각화된금융_per['market_weight'])
        mu_inv_div=np.sum(data_다각화된금융_div['div_yield']*data_다각화된금융_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_다각화된금융_pbr['1/pbr']-mu_inv_pbr)*data_다각화된금융_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_다각화된금융_per['1/per']-mu_inv_per)*data_다각화된금융_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_다각화된금융_div['div_yield']-mu_inv_div)*data_다각화된금융_div['market_weight']))
        
        data_다각화된금융1=(data_다각화된금융_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_다각화된금융1.name= 'pbr_z'
        data_다각화된금융2=(data_다각화된금융_per['1/per']-mu_inv_per)/std_inv_per
        data_다각화된금융2.name= 'per_z'
        data_다각화된금융3=(data_다각화된금융_div['div_yield']-mu_inv_div)/std_inv_div
        data_다각화된금융3.name= 'div_z'
              
        result_다각화된금융 = pd.concat([data_다각화된금융, data_다각화된금융1, data_다각화된금융2, data_다각화된금융3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_다각화된금융 = result_다각화된금융.assign(z_score=np.nanmean(result_다각화된금융.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_다각화된금융[result_다각화된금융['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['다각화된금융','sector_ratio']),:]
        
        a=a+1
    
     #보험 섹터
    if (np.sum(data['sector']=='보험')>0):
        data_보험 = data[data['sector']=="보험"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_보험['size_FIF_wisefn']=data_보험['size_FIF_wisefn']/1000    #size 단위 thousand
        data_보험['1/pbr']=data_보험['equity']/data_보험['size']
        data_보험['1/per']=data_보험['ni_12fw']/data_보험['size']
        data_보험['div_yield']=data_보험['cash_div']/data_보험['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_보험 = data_보험.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_보험_per = data_보험[data_보험['1/per'].notnull()]
        data_보험_pbr = data_보험[data_보험['1/pbr'].notnull()]
        data_보험_div = data_보험[data_보험['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_보험_pbr_cap = np.sum(data_보험_pbr['size_FIF_wisefn'])
        data_보험_per_cap = np.sum(data_보험_per['size_FIF_wisefn'])
        data_보험_div_cap = np.sum(data_보험_div['size_FIF_wisefn'])
    
        data_보험_pbr = data_보험_pbr.assign(market_weight=data_보험_pbr['size_FIF_wisefn']/data_보험_pbr_cap)
        data_보험_per = data_보험_per.assign(market_weight=data_보험_per['size_FIF_wisefn']/data_보험_per_cap)
        data_보험_div = data_보험_div.assign(market_weight=data_보험_div['size_FIF_wisefn']/data_보험_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_보험_pbr['1/pbr']*data_보험_pbr['market_weight'])
        mu_inv_per=np.sum(data_보험_per['1/per']*data_보험_per['market_weight'])
        mu_inv_div=np.sum(data_보험_div['div_yield']*data_보험_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_보험_pbr['1/pbr']-mu_inv_pbr)*data_보험_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_보험_per['1/per']-mu_inv_per)*data_보험_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_보험_div['div_yield']-mu_inv_div)*data_보험_div['market_weight']))
        
        data_보험1=(data_보험_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_보험1.name= 'pbr_z'
        data_보험2=(data_보험_per['1/per']-mu_inv_per)/std_inv_per
        data_보험2.name= 'per_z'
        data_보험3=(data_보험_div['div_yield']-mu_inv_div)/std_inv_div
        data_보험3.name= 'div_z'
              
        result_보험 = pd.concat([data_보험, data_보험1, data_보험2, data_보험3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_보험 = result_보험.assign(z_score=np.nanmean(result_보험.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_보험[result_보험['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['보험','sector_ratio']),:]
        
        a=a+1
   
     #부동산 섹터
    if (np.sum(data['sector']=='부동산')>0):
        data_부동산 = data[data['sector']=="부동산"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_부동산['size_FIF_wisefn']=data_부동산['size_FIF_wisefn']/1000    #size 단위 thousand
        data_부동산['1/pbr']=data_부동산['equity']/data_부동산['size']
        data_부동산['1/per']=data_부동산['ni_12fw']/data_부동산['size']
        data_부동산['div_yield']=data_부동산['cash_div']/data_부동산['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_부동산 = data_부동산.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_부동산_per = data_부동산[data_부동산['1/per'].notnull()]
        data_부동산_pbr = data_부동산[data_부동산['1/pbr'].notnull()]
        data_부동산_div = data_부동산[data_부동산['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_부동산_pbr_cap = np.sum(data_부동산_pbr['size_FIF_wisefn'])
        data_부동산_per_cap = np.sum(data_부동산_per['size_FIF_wisefn'])
        data_부동산_div_cap = np.sum(data_부동산_div['size_FIF_wisefn'])
    
        data_부동산_pbr = data_부동산_pbr.assign(market_weight=data_부동산_pbr['size_FIF_wisefn']/data_부동산_pbr_cap)
        data_부동산_per = data_부동산_per.assign(market_weight=data_부동산_per['size_FIF_wisefn']/data_부동산_per_cap)
        data_부동산_div = data_부동산_div.assign(market_weight=data_부동산_div['size_FIF_wisefn']/data_부동산_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_부동산_pbr['1/pbr']*data_부동산_pbr['market_weight'])
        mu_inv_per=np.sum(data_부동산_per['1/per']*data_부동산_per['market_weight'])
        mu_inv_div=np.sum(data_부동산_div['div_yield']*data_부동산_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_부동산_pbr['1/pbr']-mu_inv_pbr)*data_부동산_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_부동산_per['1/per']-mu_inv_per)*data_부동산_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_부동산_div['div_yield']-mu_inv_div)*data_부동산_div['market_weight']))
        
        data_부동산1=(data_부동산_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_부동산1.name= 'pbr_z'
        data_부동산2=(data_부동산_per['1/per']-mu_inv_per)/std_inv_per
        data_부동산2.name= 'per_z'
        data_부동산3=(data_부동산_div['div_yield']-mu_inv_div)/std_inv_div
        data_부동산3.name= 'div_z'
              
        result_부동산 = pd.concat([data_부동산, data_부동산1, data_부동산2, data_부동산3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_부동산 = result_부동산.assign(z_score=np.nanmean(result_부동산.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_부동산[result_부동산['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['부동산','sector_ratio']),:]
        
        a=a+1
    
     #기타금융서비스 섹터
    if (np.sum(data['sector']=='기타금융서비스')>0):
        data_기타금융서비스 = data[data['sector']=="기타금융서비스"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_기타금융서비스['size_FIF_wisefn']=data_기타금융서비스['size_FIF_wisefn']/1000    #size 단위 thousand
        data_기타금융서비스['1/pbr']=data_기타금융서비스['equity']/data_기타금융서비스['size']
        data_기타금융서비스['1/per']=data_기타금융서비스['ni_12fw']/data_기타금융서비스['size']
        data_기타금융서비스['div_yield']=data_기타금융서비스['cash_div']/data_기타금융서비스['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_기타금융서비스 = data_기타금융서비스.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_기타금융서비스_per = data_기타금융서비스[data_기타금융서비스['1/per'].notnull()]
        data_기타금융서비스_pbr = data_기타금융서비스[data_기타금융서비스['1/pbr'].notnull()]
        data_기타금융서비스_div = data_기타금융서비스[data_기타금융서비스['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_기타금융서비스_pbr_cap = np.sum(data_기타금융서비스_pbr['size_FIF_wisefn'])
        data_기타금융서비스_per_cap = np.sum(data_기타금융서비스_per['size_FIF_wisefn'])
        data_기타금융서비스_div_cap = np.sum(data_기타금융서비스_div['size_FIF_wisefn'])
    
        data_기타금융서비스_pbr = data_기타금융서비스_pbr.assign(market_weight=data_기타금융서비스_pbr['size_FIF_wisefn']/data_기타금융서비스_pbr_cap)
        data_기타금융서비스_per = data_기타금융서비스_per.assign(market_weight=data_기타금융서비스_per['size_FIF_wisefn']/data_기타금융서비스_per_cap)
        data_기타금융서비스_div = data_기타금융서비스_div.assign(market_weight=data_기타금융서비스_div['size_FIF_wisefn']/data_기타금융서비스_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_기타금융서비스_pbr['1/pbr']*data_기타금융서비스_pbr['market_weight'])
        mu_inv_per=np.sum(data_기타금융서비스_per['1/per']*data_기타금융서비스_per['market_weight'])
        mu_inv_div=np.sum(data_기타금융서비스_div['div_yield']*data_기타금융서비스_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_기타금융서비스_pbr['1/pbr']-mu_inv_pbr)*data_기타금융서비스_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_기타금융서비스_per['1/per']-mu_inv_per)*data_기타금융서비스_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_기타금융서비스_div['div_yield']-mu_inv_div)*data_기타금융서비스_div['market_weight']))
        
        data_기타금융서비스1=(data_기타금융서비스_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_기타금융서비스1.name= 'pbr_z'
        data_기타금융서비스2=(data_기타금융서비스_per['1/per']-mu_inv_per)/std_inv_per
        data_기타금융서비스2.name= 'per_z'
        data_기타금융서비스3=(data_기타금융서비스_div['div_yield']-mu_inv_div)/std_inv_div
        data_기타금융서비스3.name= 'div_z'
              
        result_기타금융서비스 = pd.concat([data_기타금융서비스, data_기타금융서비스1, data_기타금융서비스2, data_기타금융서비스3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_기타금융서비스 = result_기타금융서비스.assign(z_score=np.nanmean(result_기타금융서비스.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_기타금융서비스[result_기타금융서비스['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['기타금융서비스','sector_ratio']),:]
        
        a=a+1
    
     #소프트웨어와서비스 섹터
    if (np.sum(data['sector']=='소프트웨어와서비스')>0):
        data_소프트웨어와서비스 = data[data['sector']=="소프트웨어와서비스"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_소프트웨어와서비스['size_FIF_wisefn']=data_소프트웨어와서비스['size_FIF_wisefn']/1000    #size 단위 thousand
        data_소프트웨어와서비스['1/pbr']=data_소프트웨어와서비스['equity']/data_소프트웨어와서비스['size']
        data_소프트웨어와서비스['1/per']=data_소프트웨어와서비스['ni_12fw']/data_소프트웨어와서비스['size']
        data_소프트웨어와서비스['div_yield']=data_소프트웨어와서비스['cash_div']/data_소프트웨어와서비스['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_소프트웨어와서비스 = data_소프트웨어와서비스.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_소프트웨어와서비스_per = data_소프트웨어와서비스[data_소프트웨어와서비스['1/per'].notnull()]
        data_소프트웨어와서비스_pbr = data_소프트웨어와서비스[data_소프트웨어와서비스['1/pbr'].notnull()]
        data_소프트웨어와서비스_div = data_소프트웨어와서비스[data_소프트웨어와서비스['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_소프트웨어와서비스_pbr_cap = np.sum(data_소프트웨어와서비스_pbr['size_FIF_wisefn'])
        data_소프트웨어와서비스_per_cap = np.sum(data_소프트웨어와서비스_per['size_FIF_wisefn'])
        data_소프트웨어와서비스_div_cap = np.sum(data_소프트웨어와서비스_div['size_FIF_wisefn'])
    
        data_소프트웨어와서비스_pbr = data_소프트웨어와서비스_pbr.assign(market_weight=data_소프트웨어와서비스_pbr['size_FIF_wisefn']/data_소프트웨어와서비스_pbr_cap)
        data_소프트웨어와서비스_per = data_소프트웨어와서비스_per.assign(market_weight=data_소프트웨어와서비스_per['size_FIF_wisefn']/data_소프트웨어와서비스_per_cap)
        data_소프트웨어와서비스_div = data_소프트웨어와서비스_div.assign(market_weight=data_소프트웨어와서비스_div['size_FIF_wisefn']/data_소프트웨어와서비스_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_소프트웨어와서비스_pbr['1/pbr']*data_소프트웨어와서비스_pbr['market_weight'])
        mu_inv_per=np.sum(data_소프트웨어와서비스_per['1/per']*data_소프트웨어와서비스_per['market_weight'])
        mu_inv_div=np.sum(data_소프트웨어와서비스_div['div_yield']*data_소프트웨어와서비스_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_소프트웨어와서비스_pbr['1/pbr']-mu_inv_pbr)*data_소프트웨어와서비스_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_소프트웨어와서비스_per['1/per']-mu_inv_per)*data_소프트웨어와서비스_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_소프트웨어와서비스_div['div_yield']-mu_inv_div)*data_소프트웨어와서비스_div['market_weight']))
        
        data_소프트웨어와서비스1=(data_소프트웨어와서비스_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_소프트웨어와서비스1.name= 'pbr_z'
        data_소프트웨어와서비스2=(data_소프트웨어와서비스_per['1/per']-mu_inv_per)/std_inv_per
        data_소프트웨어와서비스2.name= 'per_z'
        data_소프트웨어와서비스3=(data_소프트웨어와서비스_div['div_yield']-mu_inv_div)/std_inv_div
        data_소프트웨어와서비스3.name= 'div_z'
              
        result_소프트웨어와서비스 = pd.concat([data_소프트웨어와서비스, data_소프트웨어와서비스1, data_소프트웨어와서비스2, data_소프트웨어와서비스3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_소프트웨어와서비스 = result_소프트웨어와서비스.assign(z_score=np.nanmean(result_소프트웨어와서비스.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_소프트웨어와서비스[result_소프트웨어와서비스['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['소프트웨어와서비스','sector_ratio']),:]
        
        a=a+1
    
     #기술하드웨어와장비 섹터
    if (np.sum(data['sector']=='기술하드웨어와장비')>0):
        data_기술하드웨어와장비 = data[data['sector']=="기술하드웨어와장비"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_기술하드웨어와장비['size_FIF_wisefn']=data_기술하드웨어와장비['size_FIF_wisefn']/1000    #size 단위 thousand
        data_기술하드웨어와장비['1/pbr']=data_기술하드웨어와장비['equity']/data_기술하드웨어와장비['size']
        data_기술하드웨어와장비['1/per']=data_기술하드웨어와장비['ni_12fw']/data_기술하드웨어와장비['size']
        data_기술하드웨어와장비['div_yield']=data_기술하드웨어와장비['cash_div']/data_기술하드웨어와장비['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_기술하드웨어와장비 = data_기술하드웨어와장비.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_기술하드웨어와장비_per = data_기술하드웨어와장비[data_기술하드웨어와장비['1/per'].notnull()]
        data_기술하드웨어와장비_pbr = data_기술하드웨어와장비[data_기술하드웨어와장비['1/pbr'].notnull()]
        data_기술하드웨어와장비_div = data_기술하드웨어와장비[data_기술하드웨어와장비['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_기술하드웨어와장비_pbr_cap = np.sum(data_기술하드웨어와장비_pbr['size_FIF_wisefn'])
        data_기술하드웨어와장비_per_cap = np.sum(data_기술하드웨어와장비_per['size_FIF_wisefn'])
        data_기술하드웨어와장비_div_cap = np.sum(data_기술하드웨어와장비_div['size_FIF_wisefn'])
    
        data_기술하드웨어와장비_pbr = data_기술하드웨어와장비_pbr.assign(market_weight=data_기술하드웨어와장비_pbr['size_FIF_wisefn']/data_기술하드웨어와장비_pbr_cap)
        data_기술하드웨어와장비_per = data_기술하드웨어와장비_per.assign(market_weight=data_기술하드웨어와장비_per['size_FIF_wisefn']/data_기술하드웨어와장비_per_cap)
        data_기술하드웨어와장비_div = data_기술하드웨어와장비_div.assign(market_weight=data_기술하드웨어와장비_div['size_FIF_wisefn']/data_기술하드웨어와장비_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_기술하드웨어와장비_pbr['1/pbr']*data_기술하드웨어와장비_pbr['market_weight'])
        mu_inv_per=np.sum(data_기술하드웨어와장비_per['1/per']*data_기술하드웨어와장비_per['market_weight'])
        mu_inv_div=np.sum(data_기술하드웨어와장비_div['div_yield']*data_기술하드웨어와장비_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_기술하드웨어와장비_pbr['1/pbr']-mu_inv_pbr)*data_기술하드웨어와장비_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_기술하드웨어와장비_per['1/per']-mu_inv_per)*data_기술하드웨어와장비_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_기술하드웨어와장비_div['div_yield']-mu_inv_div)*data_기술하드웨어와장비_div['market_weight']))
        
        data_기술하드웨어와장비1=(data_기술하드웨어와장비_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_기술하드웨어와장비1.name= 'pbr_z'
        data_기술하드웨어와장비2=(data_기술하드웨어와장비_per['1/per']-mu_inv_per)/std_inv_per
        data_기술하드웨어와장비2.name= 'per_z'
        data_기술하드웨어와장비3=(data_기술하드웨어와장비_div['div_yield']-mu_inv_div)/std_inv_div
        data_기술하드웨어와장비3.name= 'div_z'
              
        result_기술하드웨어와장비 = pd.concat([data_기술하드웨어와장비, data_기술하드웨어와장비1, data_기술하드웨어와장비2, data_기술하드웨어와장비3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_기술하드웨어와장비 = result_기술하드웨어와장비.assign(z_score=np.nanmean(result_기술하드웨어와장비.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_기술하드웨어와장비[result_기술하드웨어와장비['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['기술하드웨어와장비','sector_ratio']),:]
        
        a=a+1
    
     #반도체와반도체장비 섹터
    if (np.sum(data['sector']=='반도체와반도체장비')>0):
        data_반도체와반도체장비 = data[data['sector']=="반도체와반도체장비"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_반도체와반도체장비['size_FIF_wisefn']=data_반도체와반도체장비['size_FIF_wisefn']/1000    #size 단위 thousand
        data_반도체와반도체장비['1/pbr']=data_반도체와반도체장비['equity']/data_반도체와반도체장비['size']
        data_반도체와반도체장비['1/per']=data_반도체와반도체장비['ni_12fw']/data_반도체와반도체장비['size']
        data_반도체와반도체장비['div_yield']=data_반도체와반도체장비['cash_div']/data_반도체와반도체장비['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_반도체와반도체장비 = data_반도체와반도체장비.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_반도체와반도체장비_per = data_반도체와반도체장비[data_반도체와반도체장비['1/per'].notnull()]
        data_반도체와반도체장비_pbr = data_반도체와반도체장비[data_반도체와반도체장비['1/pbr'].notnull()]
        data_반도체와반도체장비_div = data_반도체와반도체장비[data_반도체와반도체장비['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_반도체와반도체장비_pbr_cap = np.sum(data_반도체와반도체장비_pbr['size_FIF_wisefn'])
        data_반도체와반도체장비_per_cap = np.sum(data_반도체와반도체장비_per['size_FIF_wisefn'])
        data_반도체와반도체장비_div_cap = np.sum(data_반도체와반도체장비_div['size_FIF_wisefn'])
    
        data_반도체와반도체장비_pbr = data_반도체와반도체장비_pbr.assign(market_weight=data_반도체와반도체장비_pbr['size_FIF_wisefn']/data_반도체와반도체장비_pbr_cap)
        data_반도체와반도체장비_per = data_반도체와반도체장비_per.assign(market_weight=data_반도체와반도체장비_per['size_FIF_wisefn']/data_반도체와반도체장비_per_cap)
        data_반도체와반도체장비_div = data_반도체와반도체장비_div.assign(market_weight=data_반도체와반도체장비_div['size_FIF_wisefn']/data_반도체와반도체장비_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_반도체와반도체장비_pbr['1/pbr']*data_반도체와반도체장비_pbr['market_weight'])
        mu_inv_per=np.sum(data_반도체와반도체장비_per['1/per']*data_반도체와반도체장비_per['market_weight'])
        mu_inv_div=np.sum(data_반도체와반도체장비_div['div_yield']*data_반도체와반도체장비_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_반도체와반도체장비_pbr['1/pbr']-mu_inv_pbr)*data_반도체와반도체장비_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_반도체와반도체장비_per['1/per']-mu_inv_per)*data_반도체와반도체장비_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_반도체와반도체장비_div['div_yield']-mu_inv_div)*data_반도체와반도체장비_div['market_weight']))
        
        data_반도체와반도체장비1=(data_반도체와반도체장비_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_반도체와반도체장비1.name= 'pbr_z'
        data_반도체와반도체장비2=(data_반도체와반도체장비_per['1/per']-mu_inv_per)/std_inv_per
        data_반도체와반도체장비2.name= 'per_z'
        data_반도체와반도체장비3=(data_반도체와반도체장비_div['div_yield']-mu_inv_div)/std_inv_div
        data_반도체와반도체장비3.name= 'div_z'
              
        result_반도체와반도체장비 = pd.concat([data_반도체와반도체장비, data_반도체와반도체장비1, data_반도체와반도체장비2, data_반도체와반도체장비3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_반도체와반도체장비 = result_반도체와반도체장비.assign(z_score=np.nanmean(result_반도체와반도체장비.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_반도체와반도체장비[result_반도체와반도체장비['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['반도체와반도체장비','sector_ratio']),:]
        
        a=a+1
    
     #전자와 전기제품 섹터
    if (np.sum(data['sector']=='전자와 전기제품')>0):
        data_전자와_전기제품 = data[data['sector']=="전자와 전기제품"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_전자와_전기제품['size_FIF_wisefn']=data_전자와_전기제품['size_FIF_wisefn']/1000    #size 단위 thousand
        data_전자와_전기제품['1/pbr']=data_전자와_전기제품['equity']/data_전자와_전기제품['size']
        data_전자와_전기제품['1/per']=data_전자와_전기제품['ni_12fw']/data_전자와_전기제품['size']
        data_전자와_전기제품['div_yield']=data_전자와_전기제품['cash_div']/data_전자와_전기제품['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_전자와_전기제품 = data_전자와_전기제품.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_전자와_전기제품_per = data_전자와_전기제품[data_전자와_전기제품['1/per'].notnull()]
        data_전자와_전기제품_pbr = data_전자와_전기제품[data_전자와_전기제품['1/pbr'].notnull()]
        data_전자와_전기제품_div = data_전자와_전기제품[data_전자와_전기제품['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_전자와_전기제품_pbr_cap = np.sum(data_전자와_전기제품_pbr['size_FIF_wisefn'])
        data_전자와_전기제품_per_cap = np.sum(data_전자와_전기제품_per['size_FIF_wisefn'])
        data_전자와_전기제품_div_cap = np.sum(data_전자와_전기제품_div['size_FIF_wisefn'])
    
        data_전자와_전기제품_pbr = data_전자와_전기제품_pbr.assign(market_weight=data_전자와_전기제품_pbr['size_FIF_wisefn']/data_전자와_전기제품_pbr_cap)
        data_전자와_전기제품_per = data_전자와_전기제품_per.assign(market_weight=data_전자와_전기제품_per['size_FIF_wisefn']/data_전자와_전기제품_per_cap)
        data_전자와_전기제품_div = data_전자와_전기제품_div.assign(market_weight=data_전자와_전기제품_div['size_FIF_wisefn']/data_전자와_전기제품_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_전자와_전기제품_pbr['1/pbr']*data_전자와_전기제품_pbr['market_weight'])
        mu_inv_per=np.sum(data_전자와_전기제품_per['1/per']*data_전자와_전기제품_per['market_weight'])
        mu_inv_div=np.sum(data_전자와_전기제품_div['div_yield']*data_전자와_전기제품_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_전자와_전기제품_pbr['1/pbr']-mu_inv_pbr)*data_전자와_전기제품_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_전자와_전기제품_per['1/per']-mu_inv_per)*data_전자와_전기제품_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_전자와_전기제품_div['div_yield']-mu_inv_div)*data_전자와_전기제품_div['market_weight']))
        
        data_전자와_전기제품1=(data_전자와_전기제품_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_전자와_전기제품1.name= 'pbr_z'
        data_전자와_전기제품2=(data_전자와_전기제품_per['1/per']-mu_inv_per)/std_inv_per
        data_전자와_전기제품2.name= 'per_z'
        data_전자와_전기제품3=(data_전자와_전기제품_div['div_yield']-mu_inv_div)/std_inv_div
        data_전자와_전기제품3.name= 'div_z'
              
        result_전자와_전기제품 = pd.concat([data_전자와_전기제품, data_전자와_전기제품1, data_전자와_전기제품2, data_전자와_전기제품3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_전자와_전기제품 = result_전자와_전기제품.assign(z_score=np.nanmean(result_전자와_전기제품.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_전자와_전기제품[result_전자와_전기제품['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['전자와 전기제품','sector_ratio']),:]
        
        a=a+1
    
     #디스플레이 섹터
    if (np.sum(data['sector']=='디스플레이')>0):
        data_디스플레이 = data[data['sector']=="디스플레이"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_디스플레이['size_FIF_wisefn']=data_디스플레이['size_FIF_wisefn']/1000    #size 단위 thousand
        data_디스플레이['1/pbr']=data_디스플레이['equity']/data_디스플레이['size']
        data_디스플레이['1/per']=data_디스플레이['ni_12fw']/data_디스플레이['size']
        data_디스플레이['div_yield']=data_디스플레이['cash_div']/data_디스플레이['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_디스플레이 = data_디스플레이.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_디스플레이_per = data_디스플레이[data_디스플레이['1/per'].notnull()]
        data_디스플레이_pbr = data_디스플레이[data_디스플레이['1/pbr'].notnull()]
        data_디스플레이_div = data_디스플레이[data_디스플레이['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_디스플레이_pbr_cap = np.sum(data_디스플레이_pbr['size_FIF_wisefn'])
        data_디스플레이_per_cap = np.sum(data_디스플레이_per['size_FIF_wisefn'])
        data_디스플레이_div_cap = np.sum(data_디스플레이_div['size_FIF_wisefn'])
    
        data_디스플레이_pbr = data_디스플레이_pbr.assign(market_weight=data_디스플레이_pbr['size_FIF_wisefn']/data_디스플레이_pbr_cap)
        data_디스플레이_per = data_디스플레이_per.assign(market_weight=data_디스플레이_per['size_FIF_wisefn']/data_디스플레이_per_cap)
        data_디스플레이_div = data_디스플레이_div.assign(market_weight=data_디스플레이_div['size_FIF_wisefn']/data_디스플레이_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_디스플레이_pbr['1/pbr']*data_디스플레이_pbr['market_weight'])
        mu_inv_per=np.sum(data_디스플레이_per['1/per']*data_디스플레이_per['market_weight'])
        mu_inv_div=np.sum(data_디스플레이_div['div_yield']*data_디스플레이_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_디스플레이_pbr['1/pbr']-mu_inv_pbr)*data_디스플레이_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_디스플레이_per['1/per']-mu_inv_per)*data_디스플레이_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_디스플레이_div['div_yield']-mu_inv_div)*data_디스플레이_div['market_weight']))
        
        data_디스플레이1=(data_디스플레이_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_디스플레이1.name= 'pbr_z'
        data_디스플레이2=(data_디스플레이_per['1/per']-mu_inv_per)/std_inv_per
        data_디스플레이2.name= 'per_z'
        data_디스플레이3=(data_디스플레이_div['div_yield']-mu_inv_div)/std_inv_div
        data_디스플레이3.name= 'div_z'
              
        result_디스플레이 = pd.concat([data_디스플레이, data_디스플레이1, data_디스플레이2, data_디스플레이3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_디스플레이 = result_디스플레이.assign(z_score=np.nanmean(result_디스플레이.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_디스플레이[result_디스플레이['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['디스플레이','sector_ratio']),:]
        
        a=a+1
    
     #통신서비스 섹터
    if (np.sum(data['sector']=='통신서비스')>0):
        data_통신서비스 = data[data['sector']=="통신서비스"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_통신서비스['size_FIF_wisefn']=data_통신서비스['size_FIF_wisefn']/1000    #size 단위 thousand
        data_통신서비스['1/pbr']=data_통신서비스['equity']/data_통신서비스['size']
        data_통신서비스['1/per']=data_통신서비스['ni_12fw']/data_통신서비스['size']
        data_통신서비스['div_yield']=data_통신서비스['cash_div']/data_통신서비스['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_통신서비스 = data_통신서비스.replace([np.inf, -np.inf],np.nan)  
        
        # Null 값 제거
        data_통신서비스_per = data_통신서비스[data_통신서비스['1/per'].notnull()]
        data_통신서비스_pbr = data_통신서비스[data_통신서비스['1/pbr'].notnull()]
        data_통신서비스_div = data_통신서비스[data_통신서비스['div_yield'].notnull()]
    
    
        # 시가총액비중 구함 
        data_통신서비스_pbr_cap = np.sum(data_통신서비스_pbr['size_FIF_wisefn'])
        data_통신서비스_per_cap = np.sum(data_통신서비스_per['size_FIF_wisefn'])
        data_통신서비스_div_cap = np.sum(data_통신서비스_div['size_FIF_wisefn'])
    
        data_통신서비스_pbr = data_통신서비스_pbr.assign(market_weight=data_통신서비스_pbr['size_FIF_wisefn']/data_통신서비스_pbr_cap)
        data_통신서비스_per = data_통신서비스_per.assign(market_weight=data_통신서비스_per['size_FIF_wisefn']/data_통신서비스_per_cap)
        data_통신서비스_div = data_통신서비스_div.assign(market_weight=data_통신서비스_div['size_FIF_wisefn']/data_통신서비스_div_cap)
        
        # 시총가중 평균 
        mu_inv_pbr=np.sum(data_통신서비스_pbr['1/pbr']*data_통신서비스_pbr['market_weight'])
        mu_inv_per=np.sum(data_통신서비스_per['1/per']*data_통신서비스_per['market_weight'])
        mu_inv_div=np.sum(data_통신서비스_div['div_yield']*data_통신서비스_div['market_weight'])
        
        # 시총 가중 표준편자
        std_inv_pbr=np.sqrt(np.sum(np.square(data_통신서비스_pbr['1/pbr']-mu_inv_pbr)*data_통신서비스_pbr['market_weight']))
        std_inv_per=np.sqrt(np.sum(np.square(data_통신서비스_per['1/per']-mu_inv_per)*data_통신서비스_per['market_weight']))
        std_inv_div=np.sqrt(np.sum(np.square(data_통신서비스_div['div_yield']-mu_inv_div)*data_통신서비스_div['market_weight']))
        
        data_통신서비스1=(data_통신서비스_pbr['1/pbr']-mu_inv_pbr)/std_inv_pbr
        data_통신서비스1.name= 'pbr_z'
        data_통신서비스2=(data_통신서비스_per['1/per']-mu_inv_per)/std_inv_per
        data_통신서비스2.name= 'per_z'
        data_통신서비스3=(data_통신서비스_div['div_yield']-mu_inv_div)/std_inv_div
        data_통신서비스3.name= 'div_z'
              
        result_통신서비스 = pd.concat([data_통신서비스, data_통신서비스1, data_통신서비스2, data_통신서비스3], axis = 1)
        
        # np.nanmean : nan 값 포함해서 평균 내기!!
        result_통신서비스 = result_통신서비스.assign(z_score=np.nanmean(result_통신서비스.iloc[:,[15,16,17]],axis=1))
    #    result_temp = result
    
        
        # z_score > 0 인것이 가치주라고 msci에서 하고있음
        locals()['result_{}'.format(a)] =result_통신서비스[result_통신서비스['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['통신서비스','sector_ratio']),:]
        
        a=a+1
    
     #유틸리티 섹터
    if (np.sum(data['sector']=='유틸리티')>0):
        data_유틸리티 = data[data['sector']=="유틸리티"]
        # per, pbr, div_yield 구할때는 전체 시가총액을 사용,
        # 시총비중 구할떄는 free-float
        data_유틸리티['size_FIF_wisefn']=data_유틸리티['size_FIF_wisefn']/1000    #size 단위 thousand
        data_유틸리티['1/pbr']=data_유틸리티['equity']/data_유틸리티['size']
        data_유틸리티['1/per']=data_유틸리티['ni_12fw']/data_유틸리티['size']
        data_유틸리티['div_yield']=data_유틸리티['cash_div']/data_유틸리티['size']
        
        # inf, -inf 값들을 NAN 값으로 변경 (그래야 한번에 제거 가능)
        data_유틸리티 = data_유틸리티.replace([np.inf, -np.inf],np.nan)  
        
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
        locals()['result_{}'.format(a)] =result_유틸리티[result_유틸리티['z_score'].notnull()]
        locals()['result_{}'.format(a)] = locals()['result_{}'.format(a)].sort_values(['z_score'],ascending=False)
        # iloc 안에 int 가 있어야함.. float은 안되는군..
        locals()['result_{}'.format(a)] =locals()['result_{}'.format(a)].iloc[0:int(data_sector_ratio.loc['유틸리티','sector_ratio']),:]
        
        a=a+1
    
    
    
    for y in range(2,a):    
        result_1 = pd.concat([result_1,locals()['result_{}'.format(y)]],axis=0,join='inner')
   
    
    result = result_1
    
    #상위 65%로 결정하면 삼성전자가 n=64,65,66일때 모두 포함이 된다.
#    z_score1_max=np.percentile(result['z_score'],50)
#    result =result[result['z_score']>z_score1_max]
    result=result.assign(rnk=result['z_score'].rank(method='first',ascending=False)) 
    
#    result = pd.concat([result,pd.DataFrame(result_temp.loc[390,:]).transpose()],axis=0)

    #중복 rows 1개 빼고 다 제거 
    result = result.drop_duplicates()
    result2 = result[result['rnk']<16] 
    
    result = pd.concat([result1,result2])

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
    market_cap교육서비스al=np.sum(result['size_FIF_wisefn'])
    result=result.assign(market_weight2=result['size_FIF_wisefn']/market_cap교육서비스al)          
    
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
sector_data_temp = sector_data.set_index([0],drop=False)
#초기값 설정
#매번 포함되는 섹터가 다르기 때문에 기존에 10개가 아니다. 
sector_length = len(sector_data_temp[0][sector_data_temp[0].notnull()])
sector_data_count = sector_data_temp.iloc[0:sector_length,1]
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

