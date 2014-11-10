#!/usr/bin/python
#Author: GiggleLiu

from core.physics import *
from scipy.interpolate import interp1d
from setting.local import *
from core.component import *
from core.param import *
from core.boson import NBoson,Boson
import os,time

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
        #register components(JJs and Inductance)
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

        #initialize Phi0 and other basic parameters
        self.Phi(0.)            #note that phiQ=self.Phi()*2
        self.bQ=2*self.components['L'].L/(1/self.components['J1'].Ic+1/self.components['J2'].Ic+1/self.components['J3'].Ic)
        self.aQ=2*self.components['J3'].Ic/(self.components['J1'].Ic+self.components['J2'].Ic)
        self.kQ=(self.components['J1'].Ic-self.components['J2'].Ic)/(self.components['J1'].Ic+self.components['J2'].Ic)
        self.sQ=2*SQUID_CS/(SQUID_C1+SQUID_C2)
        self.Ej=(self.components['J1'].Ic+self.components['J2'].Ic)/4
        self.Ec=1./(SQUID_C1+SQUID_C2)

        #get U matrix self.Usta which makes linear combination of phases(phiQ, phi1, phi2, and phi3), and effective-mass matrix self.Meff
        self.initUmat()
        self.initMeff()

        #attach a Boson Model to this SQUID
        self.model=NBoson()
        m=1./self.Meff_inv[0,0]
        oscil=Boson(MODEL_NT+1,m=m,w=sqrt(self.Ej*(1+2*self.aQ)**2*(1-self.kQ**2)/(2*self.aQ*self.bQ*(1+2*self.aQ-self.kQ**2))*2/m))
        js=Boson(2*MODEL_NS+1,offset=-MODEL_NS)
        ja=Boson(2*MODEL_NA+1,offset=-MODEL_NA)
        self.model.pushboson(oscil)
        self.model.pushboson(js)
        self.model.pushboson(ja)

        #decide where to store data
        self.setfolder(MODEL_FOLDER)

    def setfolder(self,folder):
        '''set folder to store data'''
        try:
            os.mkdir(folder)
        except:
            pass
        self.folder=folder
        
    def show_params(self):
        '''show parameters'''
        words='''
        aQ: %s
        bQ: %s
        sQ: %s
        kQ: %s
        Ej: %s
        Ec: %s
        '''%(self.aQ,self.bQ,self.sQ,self.kQ,self.Ej,self.Ec)
        words2='''
        Ic1: %s
        Ic2: %s
        Ic3: %s
        C1: %s
        C2: %s
        C3: %s
        Loop Inductance: %s
        '''%(self.components['J1'].Ic,self.components['J2'].Ic,self.components['J3'].Ic,self.components['J1'].C(),self.components['J2'].C(),self.components['J3'].C(),self.components['L'].L)
        print words,words2

    def initH(self):
        '''initialize hamiltonian'''
        #set JJ terms
        self.model.emptyH()
        bosons=self.model.bosons
        oscil=bosons[0]
        for i in xrange(3):
            jj=self.components['J'+str(i+1)]
            gl=self.Utsa_inv4[i+1]
            for k in [-1,1]:
                glist=k*gl   #decouple a JJ to phi(2*self.Phi),t,s,a channel
                matlist=[]
                matlist.append(oscil.mat_expx(l=1j*glist[1]))
                for j in [1,2]:
                    boson=self.model.bosons[j]
                    matlist.append(boson.mat_c(int(round(glist[1+j])),withfactor=False))
                self.model.pushmats(matlist,factor=-jj.Ic/4.*exp(1j*glist[0]*self.Phi()*2))

        #set oscillator term
        self.model.pushmats([oscil.mat_n(),bosons[1].I,bosons[2].I],factor=oscil.w)

        #set kinetic terms
        for i in xrange(3):
            for j in xrange(3):
                ml=[bosons[k].I for k in xrange(3)]
                if i!=0:
                    ml[i]=dot(ml[i],bosons[i].mat_n())
                    if j==0:
                        ml[j]=dot(ml[j],bosons[j].mat_p())
                if j!=0:
                    ml[j]=dot(ml[j],bosons[j].mat_n())
                    if i==0:
                        ml[i]=dot(ml[i],bosons[i].mat_p())
                if i!=0 or j!=0:
                    self.model.pushmats(ml,factor=self.Meff_inv[i,j]/2.)
        return self.model.H

    def initHlist(self):
        '''initialize H(phi)'''
        self.philist=linspace(PHIMIN,PHIMAX,NPHI)
        def geth(phi):
            self.Phi(phi)
            return self.initH()
        self.hlist=array([geth(phi) for phi in self.philist])

    def initeklist(self,fast=True,append=False):
        '''fast won't generate vkmesh!'''
        eklfile=self.folder+'/ekl_'+str(PHIMAX-PHIMIN)+'.npy'
        vklfile=self.folder+'/vkl_'+str(PHIMAX-PHIMIN)+'.npy'
        if append:
            self.philist=linspace(PHIMIN,PHIMAX,NPHI)
            self.eklmesh=load(eklfile)
            try:
                self.vklmesh=load(vklfile)
            except:
                pass
            return self.eklmesh
        if fast:
            self.eklmesh=ndarray([len(self.philist),self.model.ndim],dtype='float64')
            for i in xrange(len(self.philist)):
                self.eklmesh[i,:]=eigvalsh(self.hlist.take(i,axis=0))
        else:
            self.eklmesh,self.vklmesh=eigh(self.hlist)
        save(eklfile,self.eklmesh)
        if not fast:
            save(vklfile,self.vklmesh)
        return self.eklmesh

    def plotband(self,withfigure=True):
        '''plot the band, and show results.'''
        nphi=len(self.philist)
        if withfigure:
            figure()

        #adjust the band top of the lowest band to zero energy
        plotek=self.eklmesh-self.eklmesh[NPHI/2].min() #+self.model.bosons[0].w/2+(self.components['J1'].Ic+self.components['J2'].Ic+self.components['J3'].Ic)/2
        pl=self.philist/pi
        plot(pl,plotek)
        #we will plot lowest nband bands
        minval=plotek.min()
        maxval=sort(plotek[nphi/2])[LOWEST_NBAND]
        dval=maxval-minval
        minval-=dval*0.1
        maxval+=dval*0.1
        ylim([minval,maxval])
        xlim(pl.min(),pl.max())
        axhline(y=0.,color='#777777',ls='--')
        if withfigure:
            show()

    def initMeff(self):
        '''get effective mass'''
        self.Meff_origin=self.getMass()
        self.Meff=self.phi2tsa(self.Meff_origin)
        self.Meff_inv=inv(self.Meff)

    def initUmat(self):
        '''initialize the Umatrix that make linear combination of JJs'''
        self.Utsa=array([array([1.,0,0,0]),self.aQ/(1+2*self.aQ)*array([1.,-1,-1,-1]),array([-2*self.aQ,-1,-1,2*self.aQ])/2/(1+2*self.aQ),array([0.,1,-1,0])/2]) #make phi_0=Phi
        self.Utsa_inv=inv(self.Utsa[-3:,-3:]) #with Phi0 Truncated.
        self.Utsa_inv4=inv(self.Utsa) #with Phi0 Considered.

    def phi2tsa(self,inmesh):
        '''transform the original basic to basis (0,t,s,a).
        the original inmesh is define as (Phi,phi1,phi2,phi3) type list,
        or matrix on this basis.'''
        dim=len(inmesh)
        if ndim(inmesh)==1:
            return dot(self.Utsa[-dim:,-dim:],inmesh)
        else:
            Utsa_inv=self.Utsa_inv[-dim:,-dim:]
            return dot(Utsa_inv.T,dot(inmesh,Utsa_inv))

    def tsa2phi(self,inmesh):
        '''transform basis (0,t,s,a) to (Phi,phi1,phi2,phi3).'''
        dim=len(inmesh)
        if ndim(inmesh)==1:
            return dot(self.Utsa_inv[-dim:],inmesh)
        else:
            Utsa=self.Utsa[-dim:,-dim:]
            return dot(Utsa.T,dot(inmesh,Utsa))

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

    def Kinetic(self):
        '''get the kinetic energy.'''
        kn=[]
        for c in self.components.values():
            if c.tp == 'JJ':
                kn.append(c.Kinetic())
        return sum(kn)

    def getMass(self):
        '''get the masses of Josephson junctions.'''
        kn=[]
        for c in self.components.values():
            if c.tp == 'JJ':
                kn.append(c.Mass())
        return diag(kn)

    def U(self):
        '''get the potential energy.'''
        for c in self.components.values():
            pass

def job_plotband():
    '''show results.'''
    tj=TriJJ()
    tj.show_params()
    if APPENDEK:
        tj.initeklist(append=True)
    else:
        tj.initHlist()
        tj.initeklist()
    tj.plotband()

def job_testboson():
    '''test boson operators.'''
    bs=Boson(5,offset=-2)
    print 'n-matrix: ',bs.mat_n()
    print 'x-matrix: ',bs.mat_x()
    print 'p-matrix: ',bs.mat_p()
    print 'exp(l*x)-matrix: ',bs.mat_expx(l=1.)

if __name__=='__main__':
    #job_testboson()
    job_plotband()
