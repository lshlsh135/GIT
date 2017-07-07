# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 15:24:56 2017



https://wikidocs.net/28



@author: SH-NoteBook
"""

result = 0

def adder(num):
    global result
    result += num    # 코드를 더 짧게 쓸수 있겠구만
#    result = result + num
    return result

print(adder(3))
print(adder(4))

#==============================================================================
# 
#==============================================================================
class calculator:
    def __init__(self):     #  _ 이거 두개씩 ;;
        self.result = 0
        
    def adder(self, num):
        self.result += num
        return self.result
    
cal1 = calculator()
cal1.adder(1)
cal2 = calculator()
cal2.adder(2)
