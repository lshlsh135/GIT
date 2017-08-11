# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 08:36:33 2017

1. oracle 에서 받아오는 db를 이용해서 backtesting_per 과 똑같은 결과를 얻자.
2. 직접 숫자를 입력하는 코드는 없어야한다. (input에 따른 자동화)
3. class를 활용할 수 있는 방법을 고려해본다.

@author: SH-NoteBook
"""

import pandas as pd
import numpy as np
import cx_Oracle



#이거 두개 반드시 선언!
cx0=cx_Oracle.makedsn("localhost",1521,"xe")
connection = cx_Oracle.connect("lshlsh135","2tkdgns2",cx0) #이게 실행이 안될때가 있는데
#그때는 services에 들어가서 oracle listner를 실행시켜줘야함


#DATA를 가져온다!!
kospi = pd.read_sql("""select * from kospi_ex""",con=connection)
kosdaq = pd.read_sql("""select * from kosdaq_ex""",con=connection)
rebalancing_date = pd.read_sql("""select * from rebalancing_date""",con=connection)

raw_data = pd.concat([kospi,kosdaq],axis=0,ignore_index=True).drop_duplicates()   #왜인지 모르겠는데 db에 중복된 정보가 들어가있네 ..? 
col_length = len(rebalancing_date)-1 #rebalancing_date의 길이는 66이다. range로 이렇게 하면 0부터 65까지 66개의 i 가 만들어진다. -1을 해준건 실제 수익률은 -1개가 생성되기 때문.




group_goal = 5  #몇개의 그룹으로 나눌지 설정한다.

for i in range(group_goal):
    locals()['data_{}'.format(i)] = pd.DataFrame(np.zeros((200,col_length)))

for i in range(group_goal):
    locals()['data_name_{}'.format(i)] = pd.DataFrame(np.zeros((200,col_length)))
    

return_data = pd.DataFrame(np.zeros((group_goal,col_length)))
return_final = pd.DataFrame(np.zeros((group_goal,1)))


for n in range(col_length): 
    first_data = raw_data[raw_data['TRD_DATE']==rebalancing_date.iloc[n,0]] # rebalanging할 날짜에 들어있는 모든 db data를 받아온다.
    target_data = raw_data[raw_data['TRD_DATE']==rebalancing_date.iloc[n+1,0]]
    target_data = target_data.loc[:,['TRD_DATE','GICODE','ADJ_PRC']]
    first_data = first_data[(first_data['CAP_SIZE']==1)|(first_data['CAP_SIZE']==2)|(first_data['CAP_SIZE']==3)]
    #first_data = first_data[first_data['EQUITY'].notnull()] # 처음 받아온 전체 data 에서 equity가 없는 종목은 제외한다 -> equity가 null값이라는건 저 당시에 데이타가 존재하지 않는다는 것.
    first_data['1/PER'] = first_data['NI'] / first_data['MARKET_CAP'] # 1/PER를 구한다.
    first_data = first_data[first_data['1/PER'].notnull()] # 1/PER이 NULL인걸 제외한다. EQUITY 가 NULL인걸 미리 제거하지 않아도 여기서 제거됨.
    data_length = len(first_data) # 몇개의 종목이 rebalanging_date때 존재했는지 본다.
    
    
    
    first_data=first_data.assign(rnk=np.floor(first_data['1/PER'].rank(method='first')/(data_length/group_goal+1/group_goal)))          # rnk가 클수록 저평가
    
    sum_data = pd.merge(target_data,first_data,on='GICODE') # 3개월치 수익률을 구하기 위해 3개월 후 존재하는 data에 현재 data를 붙임
    sum_data['3M_RETURN'] = sum_data['ADJ_PRC_x']/sum_data['ADJ_PRC_y']
   
    #숫자가 클수록 1/PER 가 큰거니 저평가
    for i in range(group_goal):
        locals()['data_{}'.format(i)] = sum_data[sum_data['rnk']==i]
       
    for i in range(group_goal):
        locals()['data_name_{}'.format(i)][n] = locals()['data_{}'.format(i)]['CO_NM'].reset_index(drop=True)
        
    for i in range(group_goal):
        return_data.iloc[i,n] = np.mean(locals()['data_{}'.format(i)]['3M_RETURN'])

return_final=np.product(return_data,axis=1)
    



    
    
    
    
    