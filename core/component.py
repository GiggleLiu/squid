#!/usr/bin/python
#Author: GiggleLiu

from physics import *
from scipy.interpolate import interp1d
from setting.local import *
from core.param import *
class Component(object):
    '''Electric component base class.
    units are:
        Phi_0/2/pi=1.
        '''
    def __init__(self):
        self.tp='Default'
        self.params={}

    def U(self):
        '''return the energy conserved in this component'''
        pdb.set_trace()

    def register_param(self,param):
        '''register a parameter.'''
        self.params[param.label]=param


class JJ(Component):
    '''Josephson junction class.'''
    def __init__(self,Ic):
        super(JJ,self).__init__()
        self.tp='JJ'
        self.Ic=Ic
        self.V(0)
        self.hasR=False
        self.hasC=False

        #volt=Volt('V')
        #self.register_param(volt)
        phi=Phi('Phi')
        current=Current('I')
        current.addcov(phi,func=lambda I:arcsin(I/self.Ic))
        phi.addcov(phi,func=lambda phi:self.Ic*sin(phi))
        self.register_param(phi)
        self.register_param(current)

    def U(self):
        '''return the energy conserved in this JJ'''
        u=0.
        u+=self.Ic*(1-cos(self.phase()))/2  #take Phi0/(2pi) as 1/2
        if self.hasR:
            u+=self.R**self.I()/2

    def Kinetic(self):
        '''return the kinetic energy of this JJ.'''
        return self.V()**2/8*self.C

    def Mass(self):
        '''get the mass matrix.'''
        return 1./4*self.C()

    def C(self,c=None):
        '''set and get the capacitance.'''
        if c==None:
            return self._C
        elif c<1e-5:
            self._C=0.
            self.hasC=False
        else:
            self._C=c
            self.hasC=True

    def R(self,r=None):
        '''set and get the resistance.'''
        if r==None:
            return self._R
        elif r<1e-5:
            self._R=0.
            self.hasR=False
        else:
            self._R=r
            self.hasR=True

    def V(self,q=None):
        '''set and get the voltage.'''
        if q==None:
            return self._V
        else:
            self._V=q

    def I(self,value=None):
        '''return the current going through this junction'''
        if value==None:
            return self.params['I'].val()
        else:
            self.params['I'].set(value)

    def Phi(self,p=None):
        '''set and get the phase.'''
        if p==None:
            return self.params['Phi'].val()
        else:
            self.params['Phi'].set(p)

class Inductance(Component):
    '''Inductance class'''
    def __init__(self,L):
        super(Inductance,self).__init__()
        self.tp='Inductance'
        self.L=L
        self.I(0.)

        phi=Phi('Phi')
        current=Current('I')
        current.addcov(phi,func=lambda I:self.L*I)
        phi.addcov(current,func=lambda phi:phi/self.L)
        self.register_param(phi)
        self.register_param(current)

    def I(self,i=None):
        '''set and get the charge.'''
        if i==None:
            return self._I
        else:
            self._I=i

    def U(self):
        '''return the energy reserved in this JJ'''
        return self.L*self.I()**2/2

    def Phi(self,p=None):
        '''get the induced flux.'''
        if p==None:
            return self.params['Phi'].val()
        else:
            self.params['Phi'].set(p)

class Capacitance(Component):
    '''Capacitance class'''
    def __init__(self,C):
        super(Capacitance,self).__init__()
        self.tp='Capacitance'
        self.C=C
        self.V=0.

    def U(self):
        '''return the energy reserved in this JJ'''
        return self.C*self.V**2/2


