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

        self.bQ=2*self.components['L'].L/(1/self.components['J1'].Ic+1/self.components['J2'].Ic+1/self.components['J3'].Ic)
        self.aQ=2*self.components['J3'].Ic/(self.components['J1'].Ic+self.components['J2'].Ic)
        self.kQ=(self.components['J1'].Ic-self.components['J2'].Ic)/(self.components['J1'].Ic+self.components['J2'].Ic)
        self.sQ=2*SQUID_CS/(SQUID_C1+SQUID_C2)
        self.Ej=(self.components['J1'].Ic+self.components['J2'].Ic)/4
        self.Ec=1./(SQUID_C1+SQUID_C2)
        self.initUmat()
        self.initMeff()

        self.model=NBoson()
        m=1./self.Meff_inv[0,0]
        oscil=Boson(3,m=m,w=sqrt(self.Ej*(1+2*self.aQ)**2*(1-self.kQ**2)/(2*self.aQ*self.bQ*(1+2*self.aQ-self.kQ**2))*2/m))
        Ns,Na=10,5
        js=Boson(2*Ns+1,offset=-Ns)
        ja=Boson(2*Na+1,offset=-Na)
        self.model.pushboson(oscil)
        self.model.pushboson(js)
        self.model.pushboson(ja)
        self.setfolder(MODEL_FOLDER)

    def setfolder(self,folder):
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
        print words

    def initH(self):
        '''initialize hamiltonian'''
        #set JJ terms
        self.model.emptyH()
        bosons=self.model.bosons
        oscil=bosons[0]
        for i in xrange(3):
            if DEBUG:
                print 'for jj - ',i
            jj=self.components['J'+str(i+1)]
            gl=self.Utsa_inv4[i+1]
            for k in [-1,1]:
                glist=k*gl   #decouple a JJ to phi(2*self.Phi),t,s,a channel
                if DEBUG:
                    print 'with sign - ',k
                    print glist
                    print 'we get the first boson operator exp(%sx)'%(1j*glist[1])
                matlist=[]
                matlist.append(oscil.mat_expx(l=1j*glist[1]))
                for j in [1,2]:
                    if DEBUG:
                        print 'we get component of ',j+1,'-th boson - ',glist[1+j]
                    boson=self.model.bosons[j]
                    matlist.append(boson.mat_c(int(round(glist[1+j])),withfactor=False))
                self.model.pushmats(matlist,factor=-jj.Ic/4.*exp(1j*glist[0]*self.Phi()*2))
                if DEBUG:
                    print 'with factor - ',-jj.Ic/4.*exp(1j*glist[0]*self.Phi()*2)
                    pdb.set_trace()

        #set oscillator term
        self.model.pushmats([oscil.mat_n(),bosons[1].I,bosons[2].I],factor=oscil.w)
        #print self.model.H.max()

        #set kinetic terms
        #for JJs
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
        '''plot the band structure.'''
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

    def initUmat(self):
        '''initialize the Umatrix that make linear combination of JJs'''
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
        dim=len(inmesh)
        if ndim(inmesh)==1:
            #phi_0=(self.Phi()-self.Utsa_inv[0,1:])/self.Utsa[0,0]
            return dot(self.Utsa_inv,inmesh)
        else:
            return dot(self.Utsa[-dim:,-dim:].T,dot(inmesh,self.Utsa[-dim:,-dim:]))

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
        self.Meff_inv=inv(self.Meff)

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

def job_plotband():
    tj=TriJJ()
    tj.show_params()
    if APPENDEK:
        tj.initeklist(append=True)
    else:
        tj.initHlist()
        tj.initeklist()
    tj.plotband()
    pdb.set_trace()

def job_testboson():
    bs=Boson(5,offset=-2)
    print bs.mat_n()
    print bs.mat_x()
    print bs.mat_p()
    print bs.mat_expx(l=1.)
    pdb.set_trace()

if __name__=='__main__':
    #job_testboson()
    job_plotband()
