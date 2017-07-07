# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:47:32 2017

@author: SH-NoteBook
"""
#==============================================================================
# 5개의 그룹으로 나눈 경우, 직전 11개월 수익률
#==============================================================================
import pandas as pd
import numpy as np


return_data = np.zeros((5,195))
return_data = pd.DataFrame(return_data)
rtn_month = pd.read_excel('wics 10섹터별 수익률.xlsm',sheetname='월별수익률1',header=None)
adj_price_52week_price_ratio = pd.read_excel('wics 10섹터별 수익률.xlsm',sheetname='수정주가52주최고수정주가1',header=None)

for i in range(1,6):
    locals()['data_name_{}'.format(i)] = pd.DataFrame(np.zeros((200,250)))

turnover = pd.DataFrame(np.zeros((5,1)))
name = rtn_month.loc[:,0]
for n in range(1,196):
    
    data = pd.concat([name,rtn_month.loc[:,n:n+10]],axis=1,join='inner',ignore_index=True)
    
    temp_return=data.loc[:,0:12]
    for k in range(2,12):
        temp_return[1]=temp_return[1]*temp_return[k]
        
    gross_return=temp_return[[0,1]]
    gross_return=gross_return[gross_return[1].notnull()]
    data_size= len(gross_return)     # Row count
    
    gross_return=gross_return.assign(rnk=np.floor(gross_return[1].rank(method='first')/(data_size/5+1/5))) 
    
    data_1=gross_return.query('5>rnk>3')   # 4
    data_2=gross_return.query('4>rnk>2')   # 3
    data_3=gross_return.query('3>rnk>1')   # 2
    data_4=gross_return.query('2>rnk>0')   # 1
    data_5=gross_return.query('1>rnk>-1')  # 0
    
    data_1=pd.concat([data_1,rtn_month[n+12]],axis=1,join='inner',ignore_index=True)    # 각각 수익률 매칭
    data_2=pd.concat([data_2,rtn_month[n+12]],axis=1,join='inner',ignore_index=True)
    data_3=pd.concat([data_3,rtn_month[n+12]],axis=1,join='inner',ignore_index=True)
    data_4=pd.concat([data_4,rtn_month[n+12]],axis=1,join='inner',ignore_index=True)
    data_5=pd.concat([data_5,rtn_month[n+12]],axis=1,join='inner',ignore_index=True)
    
    data_1=data_1[data_1[3].notnull()]
    data_2=data_2[data_2[3].notnull()]
    data_3=data_3[data_3[3].notnull()]
    data_4=data_4[data_4[3].notnull()]
    data_5=data_5[data_5[3].notnull()]
    
    for i in range(1,6):
        locals()['data_name_{}'.format(i)][n-1] = locals()['data_{}'.format(i)][0].reset_index(drop=True)
    
    return_data.iloc[0,n-1]=np.mean(data_1[3])    # 각각  누적수익률 기록
    return_data.iloc[1,n-1]=np.mean(data_2[3])
    return_data.iloc[2,n-1]=np.mean(data_3[3])
    return_data.iloc[3,n-1]=np.mean(data_4[3])
    return_data.iloc[4,n-1]=np.mean(data_5[3])

    if n == 195:
        pass
    
return_final=np.product(return_data,axis=1)



#==============================================================================
# 2개의 그룹으로 나눈 경우(5개,5개) 직전 11개월 수익률
#==============================================================================
import pandas as pd
import numpy as np


return_data = np.zeros((2,195))
return_data = pd.DataFrame(return_data)
rtn_month = pd.read_excel('wics 10섹터별 수익률.xlsm',sheetname='월별수익률1',header=None)

for i in range(1,3):
    locals()['data_name_{}'.format(i)] = pd.DataFrame(np.zeros((200,250)))

turnover = pd.DataFrame(np.zeros((5,1)))
name = rtn_month.loc[:,0]
for n in range(1,196):
    
    data = pd.concat([name,rtn_month.loc[:,n:n+10]],axis=1,join='inner',ignore_index=True)
    
    temp_return=data.loc[:,0:12]
    for k in range(2,12):
        temp_return[1]=temp_return[1]*temp_return[k]
        
    gross_return=temp_return[[0,1]]
    gross_return=gross_return[gross_return[1].notnull()]
    data_size= len(gross_return)     # Row count
    
    gross_return=gross_return.assign(rnk=np.floor(gross_return[1].rank(method='first')))
    
    data_1=gross_return.query('rnk>5')   # 4
    data_2=gross_return.query('rnk<6')   # 3

    
    data_1=pd.concat([data_1,rtn_month[n+12]],axis=1,join='inner',ignore_index=True)    # 각각 수익률 매칭
    data_2=pd.concat([data_2,rtn_month[n+12]],axis=1,join='inner',ignore_index=True)

    data_1=data_1[data_1[3].notnull()]
    data_2=data_2[data_2[3].notnull()]

    for i in range(1,3):
        locals()['data_name_{}'.format(i)][n-1] = locals()['data_{}'.format(i)][0].reset_index(drop=True)
    
    return_data.iloc[0,n-1]=np.mean(data_1[3])    # 각각  누적수익률 기록
    return_data.iloc[1,n-1]=np.mean(data_2[3])


    if n == 195:
        pass
    
return_final=np.product(return_data,axis=1)

aa=np.product(np.mean(rtn_month.loc[:,1:207],axis=0))

#==============================================================================
# 수정주가 / 52주 최고 수정주가 모멘텀
#==============================================================================
import pandas as pd
import numpy as np


return_data = np.zeros((5,65))
return_data = pd.DataFrame(return_data)
rtn_quarter = pd.read_excel('wics 10섹터별 수익률.xlsm',sheetname='분기수익률1',header=None)
adj_price_52week_price_ratio = pd.read_excel('wics 10섹터별 수익률.xlsm',sheetname='수정주가52주최고수정주가1',header=None)

for i in range(1,6):
    locals()['data_name_{}'.format(i)] = pd.DataFrame(np.zeros((200,250)))

turnover = pd.DataFrame(np.zeros((5,1)))
name = rtn_quarter.loc[:,0]
for n in range(1,66):
    
    gross_return = pd.concat([name,adj_price_52week_price_ratio.loc[:,n]],axis=1,join='inner',ignore_index=True)
    
    data_size= len(gross_return)  

        

    
    gross_return=gross_return.assign(rnk=np.floor(gross_return[1].rank(method='first',ascending=False)/(data_size/5+1/5))) 
    
    data_1=gross_return.query('5>rnk>3')   # 4
    data_2=gross_return.query('4>rnk>2')   # 3
    data_3=gross_return.query('3>rnk>1')   # 2
    data_4=gross_return.query('2>rnk>0')   # 1
    data_5=gross_return.query('1>rnk>-1')  # 0
    
    data_1=pd.concat([data_1,rtn_quarter[n]],axis=1,join='inner',ignore_index=True)    # 각각 수익률 매칭
    data_2=pd.concat([data_2,rtn_quarter[n]],axis=1,join='inner',ignore_index=True)
    data_3=pd.concat([data_3,rtn_quarter[n]],axis=1,join='inner',ignore_index=True)
    data_4=pd.concat([data_4,rtn_quarter[n]],axis=1,join='inner',ignore_index=True)
    data_5=pd.concat([data_5,rtn_quarter[n]],axis=1,join='inner',ignore_index=True)
    
    data_1=data_1[data_1[3].notnull()]
    data_2=data_2[data_2[3].notnull()]
    data_3=data_3[data_3[3].notnull()]
    data_4=data_4[data_4[3].notnull()]
    data_5=data_5[data_5[3].notnull()]
    
    for i in range(1,6):
        locals()['data_name_{}'.format(i)][n-1] = locals()['data_{}'.format(i)][0].reset_index(drop=True)
    
    return_data.iloc[0,n-1]=np.mean(data_1[3])    # 각각  누적수익률 기록
    return_data.iloc[1,n-1]=np.mean(data_2[3])
    return_data.iloc[2,n-1]=np.mean(data_3[3])
    return_data.iloc[3,n-1]=np.mean(data_4[3])
    return_data.iloc[4,n-1]=np.mean(data_5[3])

    if n == 65:
        pass
    
return_final=np.product(return_data,axis=1)

aa=np.product(np.mean(rtn_month.loc[:,1:207],axis=0))


#==============================================================================
# 2개의 그룹으로 나눈 경우(모멘텀 양 vs 음) 직전 11개월 수익률
#==============================================================================
import pandas as pd
import numpy as np


return_data = np.zeros((2,195))
return_data = pd.DataFrame(return_data)
rtn_month = pd.read_excel('wics 10섹터별 수익률.xlsm',sheetname='월별수익률1',header=None)

for i in range(1,3):
    locals()['data_name_{}'.format(i)] = pd.DataFrame(np.zeros((200,250)))

turnover = pd.DataFrame(np.zeros((5,1)))
name = rtn_month.loc[:,0]
for n in range(1,196):
    
    data = pd.concat([name,rtn_month.loc[:,n:n+10]],axis=1,join='inner',ignore_index=True)
    
    temp_return=data.loc[:,0:12]
    for k in range(2,12):
        temp_return[1]=temp_return[1]*temp_return[k]
        
    gross_return=temp_return[[0,1]]
    gross_return=gross_return[gross_return[1].notnull()]
    data_size= len(gross_return)     # Row count
    
    
    
    
    data_1=gross_return[gross_return[1]>=1]
    data_2=gross_return[gross_return[1]<1]   # 3

    
    data_1=pd.concat([data_1,rtn_month[n+12]],axis=1,join='inner',ignore_index=True)    # 각각 수익률 매칭
    data_2=pd.concat([data_2,rtn_month[n+12]],axis=1,join='inner',ignore_index=True)

    data_1=data_1[data_1[2].notnull()]
    data_2=data_2[data_2[2].notnull()]

    for i in range(1,3):
        locals()['data_name_{}'.format(i)][n-1] = locals()['data_{}'.format(i)][0].reset_index(drop=True)
    
    return_data.iloc[0,n-1]=np.mean(data_1[2])    # 각각  누적수익률 기록
    return_data.iloc[1,n-1]=np.mean(data_2[2])


    if n == 195:
        pass
    
return_data = return_data.replace([np.nan],1)   
return_final=np.product(return_data,axis=1)

aa=np.product(np.mean(rtn_month.loc[:,1:207],axis=0))


#==============================================================================
# 2개의 그룹으로 나눈 경우(개수 바꿀수 있음) 직전 11개월 수익률
#==============================================================================
import pandas as pd
import numpy as np


return_data = np.zeros((2,195))
return_data = pd.DataFrame(return_data)
rtn_month = pd.read_excel('wics 10섹터별 수익률.xlsm',sheetname='월별수익률1',header=None)

for i in range(1,3):
    locals()['data_name_{}'.format(i)] = pd.DataFrame(np.zeros((200,250)))

turnover = pd.DataFrame(np.zeros((5,1)))
name = rtn_month.loc[:,0]
for n in range(1,196):
    
    data = pd.concat([name,rtn_month.loc[:,n:n+10]],axis=1,join='inner',ignore_index=True)
    
    temp_return=data.loc[:,0:12]
    for k in range(2,12):
        temp_return[1]=temp_return[1]*temp_return[k]
        
    gross_return=temp_return[[0,1]]
    gross_return=gross_return[gross_return[1].notnull()]
    data_size= len(gross_return)     # Row count
    
    gross_return=gross_return.assign(rnk=np.floor(gross_return[1].rank(method='first')))
    
    data_1=gross_return.query('rnk>8')   # 4
    data_2=gross_return.query('rnk<9')   # 3

    
    data_1=pd.concat([data_1,rtn_month[n+12]],axis=1,join='inner',ignore_index=True)    # 각각 수익률 매칭
    data_2=pd.concat([data_2,rtn_month[n+12]],axis=1,join='inner',ignore_index=True)

    data_1=data_1[data_1[3].notnull()]
    data_2=data_2[data_2[3].notnull()]

    for i in range(1,3):
        locals()['data_name_{}'.format(i)][n-1] = locals()['data_{}'.format(i)][0].reset_index(drop=True)
    
    return_data.iloc[0,n-1]=np.mean(data_1[3])    # 각각  누적수익률 기록
    return_data.iloc[1,n-1]=np.mean(data_2[3])


    if n == 195:
        pass
    
return_final=np.product(return_data,axis=1)

aa=np.product(np.mean(rtn_month.loc[:,1:207],axis=0))