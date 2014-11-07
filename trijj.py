#!/usr/bin/python
#Author: GiggleLiu

from core.physics import *
from scipy.interpolate import interp1d
from setting.local import *
from core.component import *
from core.param import *

class SQUID(object):
    '''A SQUID'''
    def __init__(self):
        self.name='SQUID'
        self.components={}
    
    def register(self,name,c):
        '''register a component'''
        self.components[name]=c

    def U(self):
        '''The total energy'''
        u=0.
        for c in self.components.values():
            u+=c.U()
        return u

class TriJJ(SQUID):
    '''3JJ looped SQUID'''
    def __init__(self):
        super(TriJJ,self).__init__()
        #set first JJ
        j1=JJ(SQUID_IC1)
        j1.C(SQUID_C1+SQUID_CS)
        self.register('J1',j1)
        j2=JJ(SQUID_IC2)
        j2.C(SQUID_C2+SQUID_CS)
        self.register('J2',j2)
        j3=JJ(SQUID_IC3)
        j3.C(SQUID_C3+SQUID_CS)
        self.register('J3',j3)
        self.register('L',Inductance(SQUID_L))
        self.Phi(0.)

        self.bQ=self.components['L'].L*(1/self.components['J1'].Ic+1/self.components['J2'].Ic+1/self.components['J3'].Ic)
        self.aQ=2*self.components['J3'].Ic/(self.components['J1'].Ic+self.components['J2'].Ic)
        self.kQ=(self.components['J1'].Ic-self.components['J2'].Ic)/(self.components['J1'].Ic+self.components['J2'].Ic)
        self.sQ=2*SQUID_CS/(SQUID_C1+SQUID_C2)

    def initUmesh(self):
        self.Utsa=array([array([1.,0,0,0]),self.aQ/(1+2*self.aQ)*array([1.,-1,-1,-1]),array([-2*self.aQ,-1,-1,2*self.aQ])/2/(1+2*self.aQ),array([0.,1,-1,0])/2]) #make phi_0=Phi
        self.Utsa_inv=inv(self.Utsa[-3:,-3:]) #with Phi0 Truncated.
        self.Utsa_inv4=inv(self.Utsa) #with Phi0 Considered.

    def phi2tsa(self,inmesh):
        '''transform the basic to another basis (0,t,s,a).
        the original inmesh is define as (Phi,phi1,phi2,phi3) type list,
        or matrix on this basis.'''
        dim=len(inmesh)
        if ndim(inmesh)==1:
            return dot(self.Utsa[-dim:,-dim:],inmesh)
        else:
            return dot(self.Utsa_inv.T,dot(inmesh,self.Utsa_inv))

    def tsa2phi(self,inmesh):
        '''transform basis (0,t,s,a) to (Phi,phi1,phi2,phi3).'''
        #Phi0 term is ignored!
        if ndim(inmesh)==1:
            #phi_0=(self.Phi()-self.Utsa_inv[0,1:])/self.Utsa[0,0]
            return dot(self.Utsa_inv,inmesh)
        else:
            return dot(self.Utsa.T,dot(inmesh,self.Utsa))

    def I(self,i=None):
        '''get of set circuit current.'''
        if i==None:
            return self._I
        else:
            self._I=i
            for c in self.components.values():
                c.I(i)

    def Phi(self,p=None):
        '''get of set circuit flux.'''
        if p==None:
            return self._Phi
        else:
            self._Phi=p
            for c in self.components.values():
                c.Phi(p)

    def Kinetic(self,getmat=True):
        '''get the kinetic energy.'''
        kn=[]
        for c in self.components.values():
            if c.tp == 'JJ':
                kn.append(c.Kinetic(getmat=getmat))
        if getmat:
            return diag(kn)
        else:
            return sum(kn)

    def U(self):
        '''get the potential energy.'''
        for c in self.components.values():
            pass

    def initMeff(self):
        '''get effective mass'''
        self.Meff_origin=self.Kinetic(getmat=True)*2
        self.Meff=self.phi2tsa(self.Meff_origin)

    def getpotentialmesh(self):
        '''get the potential mesh.'''
        #first: U of JJs as a function of phi_a, phi_s and phi_t
        #phi_t has 4 kinds in k space
        #phi_s has 2
        #phi_a has 5(4)]
        N=1
        Ns=5
        Na=3
        phimesh=zeros([N,Ns,Na],dtype='complex128')
        indoffset=array([Ns/2,Na/2])
        phi_t=linspace(0,2*pi,N)
        for i in xrange(3):
            Ic=self.components['J'+str(i+1)].Ic
            cvals=self.Utsa_inv4[i+1]
            ind1=array((cvals[-2:]).round(),dtype='int32')
            _ind1=-ind1+indoffset
            ind1+=indoffset

            phimesh[:,ind1[0],ind1[1]]=Ic*exp(1j*(phi_t*cvals[1]+cvals[0]*self.Phi()))
            phimesh[:,_ind1[0],_ind1[1]]=Ic*exp(-1j*(phi_t*cvals[1]+cvals[0]*self.Phi()))
        phimesh/=2
        pdb.set_trace()

if __name__=='__main__':
    tj=TriJJ()
    tmat0=tj.Kinetic()
    tmat=tj.phi2tsa(tmat0)
    tj.Phi(pi/2)
    tj.getpotentialmesh()
    print tj.sQ,tj.kQ,tj.aQ
    pdb.set_trace()






















