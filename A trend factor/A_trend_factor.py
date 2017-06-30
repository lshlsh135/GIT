# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:35:31 2017

@author: SH-NoteBook
"""

import pandas as pd
import numpy as np



#rtn_monthly = pd.read_excel('A_trend_factor.xlsm',sheetname='월별수익률1')
#monthly_date = pd.read_excel('A_trend_factor.xlsm',sheetname='월말날짜1',header=None)
#rtn_daily1.to_pickle('rtn_daily1')
#rtn_monthly.to_pickle('rtn_monthly')
monthly_date = pd.read_pickle('monthly_date')
rtn_monthly = pd.read_pickle('rtn_monthly')
rtn_daily = pd.read_pickle('rtn_daily')
higher_return_final = pd.DataFrame(np.zeros((1,134)))
lower_return_final = pd.DataFrame(np.zeros((1,134)))

for n in range(12,146):
    
    beta_3_temp = pd.DataFrame(np.zeros((1,12)))
    for i in range(0,12):
        #df.columns.get_loc() 이걸 하면 column 위치를 알 수 있다!!
        rebalancing_date_column=rtn_daily.columns.get_loc(monthly_date.iloc[0,n-i])
        
        #3일 moving_average를 마지막 가격으로 나누워준 값
        # iloc을 이용해서 column 범위를 구하면 원하는 column +1을 해줘야하네
        ma_3=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-2:rebalancing_date_column+1],axis=1)/3/rtn_daily.iloc[:,rebalancing_date_column])
        ma_3=ma_3[ma_3[0].notnull()]
        future_rtn = pd.DataFrame(rtn_monthly.iloc[:,n-i+1])
    #    future_rtn = pd.DataFrame(rtn_daily.loc[:,monthly_date.iloc[0,n+1-i]])
        future_rtn = future_rtn[future_rtn[monthly_date.iloc[0,n+1-i]].notnull()]
        ma_3_temp = pd.concat([ma_3,future_rtn],axis=1)
        ma_3_temp = ma_3_temp.assign(product=ma_3_temp.iloc[:,0]*ma_3_temp.iloc[:,1])
        ma_3_temp = ma_3_temp[ma_3_temp['product'].notnull()]
        
        beta_3 = pd.DataFrame(np.dot(np.linalg.inv(np.dot(ma_3.T,ma_3)),np.dot(ma_3_temp[0].T,ma_3_temp.iloc[:,1])))
        beta_3_temp.iloc[0,i]=beta_3.iloc[0,0]
    
    beta_3=np.average(beta_3_temp)    
    return_3=beta_3 * ma_3
    
    beta_5_temp = pd.DataFrame(np.zeros((1,12)))
    for i in range(0,12):
        #df.columns.get_loc() 이걸 하면 column 위치를 알 수 있다!!
        rebalancing_date_column=rtn_daily.columns.get_loc(monthly_date.iloc[0,n-i])
        
        #3일 moving_average를 마지막 가격으로 나누워준 값
        # iloc을 이용해서 column 범위를 구하면 원하는 column +1을 해줘야하네
        ma_5=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-4:rebalancing_date_column+1],axis=1)/5/rtn_daily.iloc[:,rebalancing_date_column])
        ma_5=ma_5[ma_5[0].notnull()]
        future_rtn = pd.DataFrame(rtn_monthly.iloc[:,n-i+1])
    #    future_rtn = pd.DataFrame(rtn_daily.loc[:,monthly_date.iloc[0,n+1-i]])
        future_rtn = future_rtn[future_rtn[monthly_date.iloc[0,n+1-i]].notnull()]
        ma_5_temp = pd.concat([ma_5,future_rtn],axis=1)
        ma_5_temp = ma_5_temp.assign(product=ma_5_temp.iloc[:,0]*ma_5_temp.iloc[:,1])
        ma_5_temp = ma_5_temp[ma_5_temp['product'].notnull()]
        
        beta_5 = pd.DataFrame(np.dot(np.linalg.inv(np.dot(ma_5.T,ma_5)),np.dot(ma_5_temp[0].T,ma_5_temp.iloc[:,1])))
        beta_5_temp.iloc[0,i]=beta_5.iloc[0,0]
    
    beta_5=np.average(beta_5_temp)    
    return_5=beta_5 * ma_5
    
    beta_10_temp = pd.DataFrame(np.zeros((1,12)))
    for i in range(0,12):
        #df.columns.get_loc() 이걸 하면 column 위치를 알 수 있다!!
        rebalancing_date_column=rtn_daily.columns.get_loc(monthly_date.iloc[0,n-i])
        
        #3일 moving_average를 마지막 가격으로 나누워준 값
        # iloc을 이용해서 column 범위를 구하면 원하는 column +1을 해줘야하네
        ma_10=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-9:rebalancing_date_column+1],axis=1)/10/rtn_daily.iloc[:,rebalancing_date_column])
        ma_10=ma_10[ma_10[0].notnull()]
        future_rtn = pd.DataFrame(rtn_monthly.iloc[:,n-i+1])
    #    future_rtn = pd.DataFrame(rtn_daily.loc[:,monthly_date.iloc[0,n+1-i]])
        future_rtn = future_rtn[future_rtn[monthly_date.iloc[0,n+1-i]].notnull()]
        ma_10_temp = pd.concat([ma_10,future_rtn],axis=1)
        ma_10_temp = ma_10_temp.assign(product=ma_10_temp.iloc[:,0]*ma_10_temp.iloc[:,1])
        ma_10_temp = ma_10_temp[ma_10_temp['product'].notnull()]
        
        beta_10 = pd.DataFrame(np.dot(np.linalg.inv(np.dot(ma_10.T,ma_10)),np.dot(ma_10_temp[0].T,ma_10_temp.iloc[:,1])))
        beta_10_temp.iloc[0,i]=beta_10.iloc[0,0]
    
    beta_10=np.average(beta_10_temp)    
    return_10=beta_10 * ma_10
    
    beta_20_temp = pd.DataFrame(np.zeros((1,12)))
    for i in range(0,12):
        #df.columns.get_loc() 이걸 하면 column 위치를 알 수 있다!!
        rebalancing_date_column=rtn_daily.columns.get_loc(monthly_date.iloc[0,n-i])
        
        #3일 moving_average를 마지막 가격으로 나누워준 값
        # iloc을 이용해서 column 범위를 구하면 원하는 column +1을 해줘야하네
        ma_20=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-19:rebalancing_date_column+1],axis=1)/20/rtn_daily.iloc[:,rebalancing_date_column])
        ma_20=ma_20[ma_20[0].notnull()]
        future_rtn = pd.DataFrame(rtn_monthly.iloc[:,n-i+1])
    #    future_rtn = pd.DataFrame(rtn_daily.loc[:,monthly_date.iloc[0,n+1-i]])
        future_rtn = future_rtn[future_rtn[monthly_date.iloc[0,n+1-i]].notnull()]
        ma_20_temp = pd.concat([ma_20,future_rtn],axis=1)
        ma_20_temp = ma_20_temp.assign(product=ma_20_temp.iloc[:,0]*ma_20_temp.iloc[:,1])
        ma_20_temp = ma_20_temp[ma_20_temp['product'].notnull()]
        
        beta_20 = pd.DataFrame(np.dot(np.linalg.inv(np.dot(ma_20.T,ma_20)),np.dot(ma_20_temp[0].T,ma_20_temp.iloc[:,1])))
        beta_20_temp.iloc[0,i]=beta_20.iloc[0,0]
    
    beta_20=np.average(beta_20_temp)    
    return_20=beta_20 * ma_20
    
    beta_50_temp = pd.DataFrame(np.zeros((1,12)))
    for i in range(0,12):
        #df.columns.get_loc() 이걸 하면 column 위치를 알 수 있다!!
        rebalancing_date_column=rtn_daily.columns.get_loc(monthly_date.iloc[0,n-i])
        
        #3일 moving_average를 마지막 가격으로 나누워준 값
        # iloc을 이용해서 column 범위를 구하면 원하는 column +1을 해줘야하네
        ma_50=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-49:rebalancing_date_column+1],axis=1)/50/rtn_daily.iloc[:,rebalancing_date_column])
        ma_50=ma_50[ma_50[0].notnull()]
        future_rtn = pd.DataFrame(rtn_monthly.iloc[:,n-i+1])
    #    future_rtn = pd.DataFrame(rtn_daily.loc[:,monthly_date.iloc[0,n+1-i]])
        future_rtn = future_rtn[future_rtn[monthly_date.iloc[0,n+1-i]].notnull()]
        ma_50_temp = pd.concat([ma_50,future_rtn],axis=1)
        ma_50_temp = ma_50_temp.assign(product=ma_50_temp.iloc[:,0]*ma_50_temp.iloc[:,1])
        ma_50_temp = ma_50_temp[ma_50_temp['product'].notnull()]
        
        beta_50 = pd.DataFrame(np.dot(np.linalg.inv(np.dot(ma_50.T,ma_50)),np.dot(ma_50_temp[0].T,ma_50_temp.iloc[:,1])))
        beta_50_temp.iloc[0,i]=beta_50.iloc[0,0]
    
    beta_50=np.average(beta_50_temp)    
    return_50=beta_50 * ma_50
    
    beta_100_temp = pd.DataFrame(np.zeros((1,12)))
    for i in range(0,12):
        #df.columns.get_loc() 이걸 하면 column 위치를 알 수 있다!!
        rebalancing_date_column=rtn_daily.columns.get_loc(monthly_date.iloc[0,n-i])
        
        #3일 moving_average를 마지막 가격으로 나누워준 값
        # iloc을 이용해서 column 범위를 구하면 원하는 column +1을 해줘야하네
        ma_100=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-99:rebalancing_date_column+1],axis=1)/100/rtn_daily.iloc[:,rebalancing_date_column])
        ma_100=ma_100[ma_100[0].notnull()]
        future_rtn = pd.DataFrame(rtn_monthly.iloc[:,n-i+1])
    #    future_rtn = pd.DataFrame(rtn_daily.loc[:,monthly_date.iloc[0,n+1-i]])
        future_rtn = future_rtn[future_rtn[monthly_date.iloc[0,n+1-i]].notnull()]
        ma_100_temp = pd.concat([ma_100,future_rtn],axis=1)
        ma_100_temp = ma_100_temp.assign(product=ma_100_temp.iloc[:,0]*ma_100_temp.iloc[:,1])
        ma_100_temp = ma_100_temp[ma_100_temp['product'].notnull()]
        
        beta_100 = pd.DataFrame(np.dot(np.linalg.inv(np.dot(ma_100.T,ma_100)),np.dot(ma_100_temp[0].T,ma_100_temp.iloc[:,1])))
        beta_100_temp.iloc[0,i]=beta_100.iloc[0,0]
    
    beta_100=np.average(beta_100_temp)    
    return_100=beta_100 * ma_100
    
    beta_200_temp = pd.DataFrame(np.zeros((1,12)))
    for i in range(0,12):
        #df.columns.get_loc() 이걸 하면 column 위치를 알 수 있다!!
        rebalancing_date_column=rtn_daily.columns.get_loc(monthly_date.iloc[0,n-i])
        
        #3일 moving_average를 마지막 가격으로 나누워준 값
        # iloc을 이용해서 column 범위를 구하면 원하는 column +1을 해줘야하네
        ma_200=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-199:rebalancing_date_column+1],axis=1)/200/rtn_daily.iloc[:,rebalancing_date_column])
        ma_200=ma_200[ma_200[0].notnull()]
        future_rtn = pd.DataFrame(rtn_monthly.iloc[:,n-i+1])
    #    future_rtn = pd.DataFrame(rtn_daily.loc[:,monthly_date.iloc[0,n+1-i]])
        future_rtn = future_rtn[future_rtn[monthly_date.iloc[0,n+1-i]].notnull()]
        ma_200_temp = pd.concat([ma_200,future_rtn],axis=1)
        ma_200_temp = ma_200_temp.assign(product=ma_200_temp.iloc[:,0]*ma_200_temp.iloc[:,1])
        ma_200_temp = ma_200_temp[ma_200_temp['product'].notnull()]
        
        beta_200 = pd.DataFrame(np.dot(np.linalg.inv(np.dot(ma_200.T,ma_200)),np.dot(ma_200_temp[0].T,ma_200_temp.iloc[:,1])))
        beta_200_temp.iloc[0,i]=beta_200.iloc[0,0]
    
    beta_200=np.average(beta_200_temp)    
    return_200=beta_200 * ma_200
    
    beta_400_temp = pd.DataFrame(np.zeros((1,12)))
    for i in range(0,12):
        #df.columns.get_loc() 이걸 하면 column 위치를 알 수 있다!!
        rebalancing_date_column=rtn_daily.columns.get_loc(monthly_date.iloc[0,n-i])
        
        #3일 moving_average를 마지막 가격으로 나누워준 값
        # iloc을 이용해서 column 범위를 구하면 원하는 column +1을 해줘야하네
        ma_400=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-399:rebalancing_date_column+1],axis=1)/400/rtn_daily.iloc[:,rebalancing_date_column])
        ma_400=ma_400[ma_400[0].notnull()]
        future_rtn = pd.DataFrame(rtn_monthly.iloc[:,n-i+1])
    #    future_rtn = pd.DataFrame(rtn_daily.loc[:,monthly_date.iloc[0,n+1-i]])
        future_rtn = future_rtn[future_rtn[monthly_date.iloc[0,n+1-i]].notnull()]
        ma_400_temp = pd.concat([ma_400,future_rtn],axis=1)
        ma_400_temp = ma_400_temp.assign(product=ma_400_temp.iloc[:,0]*ma_400_temp.iloc[:,1])
        ma_400_temp = ma_400_temp[ma_400_temp['product'].notnull()]
        
        beta_400 = pd.DataFrame(np.dot(np.linalg.inv(np.dot(ma_400.T,ma_400)),np.dot(ma_400_temp[0].T,ma_400_temp.iloc[:,1])))
        beta_400_temp.iloc[0,i]=beta_400.iloc[0,0]
    
    beta_400=np.average(beta_400_temp)    
    return_400=beta_400 * ma_400
    
    beta_600_temp = pd.DataFrame(np.zeros((1,12)))
    for i in range(0,12):
        #df.columns.get_loc() 이걸 하면 column 위치를 알 수 있다!!
        rebalancing_date_column=rtn_daily.columns.get_loc(monthly_date.iloc[0,n-i])
        
        #3일 moving_average를 마지막 가격으로 나누워준 값
        # iloc을 이용해서 column 범위를 구하면 원하는 column +1을 해줘야하네
        ma_600=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-599:rebalancing_date_column+1],axis=1)/600/rtn_daily.iloc[:,rebalancing_date_column])
        ma_600=ma_600[ma_600[0].notnull()]
        future_rtn = pd.DataFrame(rtn_monthly.iloc[:,n-i+1])
    #    future_rtn = pd.DataFrame(rtn_daily.loc[:,monthly_date.iloc[0,n+1-i]])
        future_rtn = future_rtn[future_rtn[monthly_date.iloc[0,n+1-i]].notnull()]
        ma_600_temp = pd.concat([ma_600,future_rtn],axis=1)
        ma_600_temp = ma_600_temp.assign(product=ma_600_temp.iloc[:,0]*ma_600_temp.iloc[:,1])
        ma_600_temp = ma_600_temp[ma_600_temp['product'].notnull()]
        
        beta_600 = pd.DataFrame(np.dot(np.linalg.inv(np.dot(ma_600.T,ma_600)),np.dot(ma_600_temp[0].T,ma_600_temp.iloc[:,1])))
        beta_600_temp.iloc[0,i]=beta_600.iloc[0,0]
    
    beta_600=np.average(beta_600_temp)    
    return_600=beta_600 * ma_600
    
    beta_800_temp = pd.DataFrame(np.zeros((1,12)))
    for i in range(0,12):
        #df.columns.get_loc() 이걸 하면 column 위치를 알 수 있다!!
        rebalancing_date_column=rtn_daily.columns.get_loc(monthly_date.iloc[0,n-i])
        
        #3일 moving_average를 마지막 가격으로 나누워준 값
        # iloc을 이용해서 column 범위를 구하면 원하는 column +1을 해줘야하네
        ma_800=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-799:rebalancing_date_column+1],axis=1)/800/rtn_daily.iloc[:,rebalancing_date_column])
        ma_800=ma_800[ma_800[0].notnull()]
        future_rtn = pd.DataFrame(rtn_monthly.iloc[:,n-i+1])
    #    future_rtn = pd.DataFrame(rtn_daily.loc[:,monthly_date.iloc[0,n+1-i]])
        future_rtn = future_rtn[future_rtn[monthly_date.iloc[0,n+1-i]].notnull()]
        ma_800_temp = pd.concat([ma_800,future_rtn],axis=1)
        ma_800_temp = ma_800_temp.assign(product=ma_800_temp.iloc[:,0]*ma_800_temp.iloc[:,1])
        ma_800_temp = ma_800_temp[ma_800_temp['product'].notnull()]
        
        beta_800 = pd.DataFrame(np.dot(np.linalg.inv(np.dot(ma_800.T,ma_800)),np.dot(ma_800_temp[0].T,ma_800_temp.iloc[:,1])))
        beta_800_temp.iloc[0,i]=beta_800.iloc[0,0]
    
    beta_800=np.average(beta_800_temp)    
    return_800=beta_800 * ma_800
    
    beta_1000_temp = pd.DataFrame(np.zeros((1,12)))
    for i in range(0,12):
        #df.columns.get_loc() 이걸 하면 column 위치를 알 수 있다!!
        rebalancing_date_column=rtn_daily.columns.get_loc(monthly_date.iloc[0,n-i])
        
        #3일 moving_average를 마지막 가격으로 나누워준 값
        # iloc을 이용해서 column 범위를 구하면 원하는 column +1을 해줘야하네
        ma_1000=pd.DataFrame(np.sum(rtn_daily.iloc[:,rebalancing_date_column-999:rebalancing_date_column+1],axis=1)/1000/rtn_daily.iloc[:,rebalancing_date_column])
        ma_1000=ma_1000[ma_1000[0].notnull()]
        future_rtn = pd.DataFrame(rtn_monthly.iloc[:,n-i+1])
    #    future_rtn = pd.DataFrame(rtn_daily.loc[:,monthly_date.iloc[0,n+1-i]])
        future_rtn = future_rtn[future_rtn[monthly_date.iloc[0,n+1-i]].notnull()]
        ma_1000_temp = pd.concat([ma_1000,future_rtn],axis=1)
        ma_1000_temp = ma_1000_temp.assign(product=ma_1000_temp.iloc[:,0]*ma_1000_temp.iloc[:,1])
        ma_1000_temp = ma_1000_temp[ma_1000_temp['product'].notnull()]
        
        beta_1000 = pd.DataFrame(np.dot(np.linalg.inv(np.dot(ma_1000.T,ma_1000)),np.dot(ma_1000_temp[0].T,ma_1000_temp.iloc[:,1])))
        beta_1000_temp.iloc[0,i]=beta_1000.iloc[0,0]
    
    beta_1000=np.average(beta_1000_temp)    
    return_1000=beta_1000 * ma_1000
        
    final_return=return_3+return_5+return_10+return_20+return_50+return_100+return_200+return_400+return_600+return_800+return_1000
    
    final_return= pd.concat([final_return,pd.DataFrame(rtn_monthly.iloc[:,n+1])],axis=1, join_axes=[final_return.index])
    final_return = final_return[final_return.iloc[:,1].notnull()]
    final_return = final_return.assign(rnk=final_return.iloc[:,0].rank(method='first',ascending=False))
    rtn_min=np.percentile(final_return['rnk'],20)
    rtn_max=np.percentile(final_return['rnk'],80)
    higher_return = final_return[final_return['rnk']<rtn_min]
    lower_return = final_return[final_return['rnk']>rtn_max]
    higher_return_final[n-12] = np.average(higher_return.iloc[:,1])
    lower_return_final[n-12] = np.average(lower_return.iloc[:,1])
    
    