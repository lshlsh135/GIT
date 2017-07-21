# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 08:45:03 2017

@author: SH-NoteBook
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:37:56 2017

@author: SH-NoteBook
"""


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
turnover_result = pd.DataFrame(np.ones((1,65)))
#result_for_turnover = pd.DataFrame(np.zeros((200,1)))
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
    data=data.assign(size_rank=data['size'].rank(method='first',ascending=False)) 
#    result_temp = data
#    samsung = pd.DataFrame(data.loc[390,:]).transpose()
    #만약 삼전의 재무재표중 없는게 있다면 14 column을 맞추기 위해 0을 넣어버림
#    if (np.isnan(result_temp.loc[390]['equity']))|(np.isnan(result_temp.loc[390]['ni_12fw']))|(np.isnan(result_temp.loc[390]['cash_div'])):
#        samsung = pd.DataFrame(result_temp.loc[390,:]).transpose()
#        samsung['1/pbr'] = 0
#        samsung['1/per'] = 0
#        samsung['div_yield'] = 0
#        samsung['pbr_z'] = 0
#        samsung['per_z'] = 0
#        samsung['div_z'] = 0
#        samsung['z_score'] = 0
    
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
    
    #전체 몇개를 뽑을것인가
    total_number = 50
    #시가총액 상위 몇개를 기본적으로 깔고 갈것인가
    size_rank_min = 25
    result1 = result[result['size_rank']<size_rank_min+1]
    result2 = result[result['size_rank']>size_rank_min]
    
    
    # np.nanmean : nan 값 포함해서 평균 내기!!
    result2 = result2.assign(z_score=np.nanmean(result2.iloc[:,[16,17,18]],axis=1))
#    result_temp = result

    
    # z_score > 0 인것이 가치주라고 msci에서 하고있음
    result2 =result2[result2['z_score'].notnull()]
    
    #상위 65%로 결정하면 삼성전자가 n=64,65,66일때 모두 포함이 된다.
#    z_score1_max=np.percentile(result['z_score'],50)
#    result =result[result['z_score']>z_score1_max]
    result2=result2.assign(rnk=result2['z_score'].rank(method='first',ascending=False)) 
    
#    result = pd.concat([result,pd.DataFrame(result_temp.loc[390,:]).transpose()],axis=0)
    #어쨋든 삼전이 탈락했다면
#    if np.sum(result['name']=="삼성전자")!=1:
#        samsung['1/pbr'] = 0
#        samsung['1/per'] = 0
#        samsung['div_yield'] = 0
#        samsung['pbr_z'] = 0
#        samsung['per_z'] = 0
#        samsung['div_z'] = 0
#        samsung['z_score'] = 0
#        result = pd.concat([result,samsung],axis=0)
        
    #중복 rows 1개 빼고 다 제거 
    result2 = result2[result2['rnk']<total_number-size_rank_min+1] 
    result1['z_score'] = 0
    result1['rnk'] = 0
    result = pd.concat([result1,result2])
    result = result.drop_duplicates()
    
####################################################################################
#    # 각 종목별 정확한 turnover 계산
#    if n>3:
#        new_stock_count = len(result['return'])
#        result = result.assign(new_weight=1/len(result['return']))
#        result_for_turnover = pd.concat([result_for_turnover,rtn_sum[n-3]],axis=1,join='inner')
#        result_for_turnover = pd.concat([result_for_turnover,result['return'],result['new_weight']],axis=1)
#        result_for_turnover = result_for_turnover.replace([np.nan],0)
#        result_for_turnover = result_for_turnover.assign(before_weight=result_for_turnover.iloc[:,1]/np.sum(result_for_turnover.iloc[:,1]))
#        turnover_result.iloc[:,n-3] = np.sum(np.abs(result_for_turnover['new_weight']-result_for_turnover['before_weight']))
#        result_for_turnover = result_for_turnover.iloc[:,2][result_for_turnover.iloc[:,2]>0]
#    if n==3:
#        result_for_turnover = result['return']
#    result = pd.concat([result,rtn_sum[n-3]],axis=1,join='inner',ignore_index=True) #수익률 매칭
#####################################################################################
  
                       
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
    
    #유동비율 시가총액 -> 결과는 극혐
#    market_capital=np.sum(result['size_FIF_wisefn'])
#    result=result.assign(market_weight2=result['size_FIF_wisefn']/market_capital)


    #연말현금배당수익률 저장
    if (n>4)&((n-4)%4==2):
        result_cash_temp= pd.concat([result['name'],cash_div_rtn_sum[(n+2)/4-2]],axis=1)
        result_cash_temp=result_cash_temp[result_cash_temp['name'].notnull()]
        result_cash[[z,z+1]] = result_cash_temp.iloc[:,[0,1]].reset_index(drop=True)
        z=z+2
        
    
#    #동일가중(월별 수익률까지 묶어서)
#    return_data.iloc[0,n-3]=np.mean(result['return'])
#
#    #월별 수익률 구하기
#    result = result.assign(gross_return_2 = result['return_month1']*result['return_month2'])
#    
#    #아래처럼 구하면 누적수익률이 달라짐
##    return_month_data[[3*(n-3),3*(n-3)+1,3*(n-3)+2]]=pd.DataFrame(np.mean(result[['return_month1','return_month2','return_month3']])).transpose()
#    #forward yield같은 느낌으로 구함
#    return_month_data[3*(n-3)] = np.mean(result['return_month1'])
#    return_month_data[3*(n-3)+1] = np.mean(result['gross_return_2'])/return_month_data[3*(n-3)]
#    return_month_data[3*(n-3)+2] = np.mean(result['return'])/np.mean(result['gross_return_2'])
#    
    #real 시가총액가중
    market_capital=np.sum(result['size'])
    result=result.assign(market_weight2=result['size']/market_capital) 
    #시총가중 분기별 수익률
    return_data.iloc[0,n-3]=np.sum(result['return']*result['market_weight2'])
    #시총가중 월별수익률
    
    result = result.assign(gross_return_2 = result['return_month1']*result['return_month2'])
    return_month_data[3*(n-3)] = np.sum(result['return_month1']*result['market_weight2'])
    return_month_data[3*(n-3)+1] = np.sum(result['gross_return_2']*result['market_weight2'])/return_month_data[3*(n-3)]
    return_month_data[3*(n-3)+2] = np.sum(result['return']*result['market_weight2'])/np.sum(result['gross_return_2']*result['market_weight2'])
# 






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
##################################################################################
# 옛날방식 - 바뀐 종목 % 를 turnover로 생각
#turnvoer에 1% 곱해서 거래비용 계산하기
#첫기에는 거래비용이 100%이다
turnover_temp = pd.DataFrame(np.ones((1,1)))
turnover_quarter = pd.DataFrame(turnover_quarter).transpose().reset_index(drop=True)
turnover_quarter = pd.concat([turnover_temp,turnover_quarter],axis=1)
turnover_quarter = turnover_quarter * 0.01
return_diff = return_data - np.tile(turnover_quarter,(5,1))
return_transaction_cost_final=np.product(return_diff,axis=1)
#################################################################################

# 제대로된 turnover
#turnover_result = turnover_result * 0.0015
#return_diff = return_data - np.tile(turnover_result,(5,1))
#return_transaction_cost_final=np.product(return_diff,axis=1)

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
#128 column은 앞쪽에 소형주가 포함이 안되서 후반부로 온거
group_data_temp = group_data.set_index([128],drop=False)
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

