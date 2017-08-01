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
# =============================================================================
# self는 첫번째 input으로 객체를 받아온다.
# =============================================================================
class Service:
    secret = "이상훈은 존나 똑똑하다"
    def setname(self, name):
        self.name = name
    def sum(self, a, b):
        result = a + b
        print("%s님 %s + %s = %s입니다." %(self.name,a,b,result))
    

SangHoon = Service()
SangHoon.setname("이상훈")
SangHoon.sum(3,6)
# =============================================================================
# __init__은 class를 선언할 때 항상 받아오는 값이다.
# =============================================================================
class Service:
    secret = "이상훈은 존나 똑똑하다"
    def __init__(self, name):
        self.name = name
    def sum(self, a, b):
        result = a + b
        print("%s님 %s + %s = %s입니다." %(self.name,a,b,result))
    
SangHoon = Service("이상훈")  # __init__을 통해서 class 선언할 때 바로 해줌
SangHoon.sum(3,6)
# =============================================================================
# 
# =============================================================================
class Fourcal:
    def __init__(self,a,b):
        self.a = a
        self.b = b
    def sum(self):
        result = self.a + self.b
        return result
    def mul(self):
        result = self.a * self.b
        return result
        
cal = Fourcal(3,7)
cal.sum()        
cal.mul()
# =============================================================================
# 
# =============================================================================
class HousePark():
    def __init__(self,middlename,lastname):
        self.middlename = middlename
        self.lastname = lastname
    def fullname(self):
        self.fullname = self.middlename + self.lastname
        return self.fullname
    
pey = HousePark("상훈","이")
pey.fullname()    
        
        
        
        
        
        


