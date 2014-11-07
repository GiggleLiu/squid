#!/usr/bin/python
#Author: GiggleLiu

from physics import *
from scipy.interpolate import interp1d
from setting.local import *
class Param(object):
    '''parameter class'''
    def __init__(self,label,value=0.):
        self.value=value
        self.label=label
        self.tp='default'
        self.covarients={}

    def set(self,value,func=None):
        '''set value of param'''
        if func==None:
            self.value=value
            #refresh covarients
            for cov in self.covarients.keys():
                cov.set(self.value,func=self.covarients[cov])
        else:
            self.value=func(value)

    def addcov(self,target,func):
        '''add covarients'''
        self.covarients[target]=func

class Current(Param):
    '''current parameter.'''
    def __init__(self,label,value=0.):
        super(Current,self).__init__(label,value)
        self.tp='I'

class Volt(Param):
    '''voltage parameter.'''
    def __init__(self,label,value=0.):
        super(Volt,self).__init__(label,value)
        self.tp='Volt'

class Phi(Param):
    '''phase parameter.'''
    def __init__(self,label,value=0.):
        super(Phi,self).__init__(label,value)
        self.tp='Phi'
