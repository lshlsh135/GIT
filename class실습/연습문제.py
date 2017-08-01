# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 10:29:37 2017

https://wikidocs.net/3465

@author: SH-NoteBook
"""

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def setx(self,x):
        self.x = x
    def sety(self,y):
        self.y = y
    def get(self):
        return (self.x,self.y)
    def move(self,dx,dy):
        self.x +=  dx      #이런 코드에 익숙하지 않음
        self.y +=  dy
        return (self.x,self.y)
    
problem1 = Point(3,6)
aa=problem1.get()  #tuple 로 결과 받기
problem1.setx(4)
problem1.sety(7)
problem1.get() 
problem1.move(-4,-7)
problem1.get() 
