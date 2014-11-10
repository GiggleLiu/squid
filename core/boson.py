from numpy import *
from core.physics import *
from setting.local import *
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as spl
from scipy.misc import factorial

from itertools import combinations
import time,pickle

class NBoson(object):
    '''
        Boson Model for:
        1. Hamiltonian generation with interaction
        2. Thats all lol
    '''
    def __init__(self,N=[]):
        super(NBoson,self).__init__()
        self.N=array(N)
        self.nbose=len(self.N)
        self.bosons=[]
        for i in xrange(self.nbose):
            self.bosons.append(Boson(self.N[i]))
        self.ndim=self.N.prod()

    def pushboson(self,boson):
        '''push a boson to bosons'''
        self.bosons.append(boson)
        self.N=array(list(self.N)+[boson.ndim])
        self.nbose+=1
        self.ndim*=boson.ndim

    def emptyH(self):
        self.H=zeros([self.ndim,self.ndim],dtype='complex128')

    def pushmats(self,mats,factor=1.):
        '''push a n body operator to the hamiltonian'''
        res=1.
        for i in xrange(self.nbose):
            res=kron(res,mats[i])
        self.H+=res*factor

    def pushc(self,nlist,factor):
        '''the c matrix for N boson.'''
        res=1.
        for i in xrange(nbose):
            res=kron(res,self.boselist[i].mat_c(nlist[i]))
        self.H+=res


class Boson(object):
    '''The Boson item Model.'''
    def __init__(self,ndim,m=1.,w=1.,offset=0):
        super(Boson,self).__init__()
        self.ndim=ndim
        self.m=m
        self.w=w
        self.circle=False
        self.offset(offset)

    def initbasicmats(self):
        '''initialize basic matrices'''
        self.c=diag(sqrt(arange(1+self.offset(),self.ndim+self.offset()),dtype='complex128'),k=1)
        self.cdag=diag(sqrt(arange(1+self.offset(),self.ndim+self.offset(),dtype='complex128')),k=-1)
        if self.circle:
            self.c[-1,0]=sqrt(self.offset()+0j)
            self.c[0,-1]=sqrt(self.offset()+0j)
            print self.c
            print self.cdag
        self.I=identity(self.ndim)

    def offset(self,i=None):
        '''offset the basis for i'''
        if i!=None:
            self._offset=i
            self.initbasicmats()
        else:
            return self._offset

    def mat_c(self,n,withfactor=True):
        '''get the c^n matrix of the i-th bose.'''
        if withfactor:
            if n==-1:
                return self.cdag 
            elif n==1:
                return self.c
            elif n<0:
                return dot(self.cdag,self.mat_c(n+1))
            elif n>0:
                return dot(self.c,self.mat_c(n-1))
            else:
                return identity(self.ndim)
        else:
            return diag(ones(self.ndim-abs(n)),k=-n)

    def mat_expx(self,l):
        '''get a matrix expression of exp(a+a^dag)'''
        lamb=l/sqrt(self.m*self.w*2)
        def eval_expx(lamb,i,j):
            '''get the i,j element of exp(lamb(a+a^dag))'''
            res=0.
            r=min(i,j)+1
            for m in xrange(r):
                res+=lamb**(i+j-2*m)*sqrt(factorial(j)*factorial(i))/factorial(m)/factorial(i-m)/factorial(j-m)
            return exp(lamb**2/2)*res #it is a pending issue whether '-' sign make sense here.

        return array([[eval_expx(lamb,i+self.offset(),j+self.offset()) for j in xrange(self.ndim)] for i in xrange(self.ndim)])

    def mat_n(self,conpensate=True):
        '''get a matrix description ni type hamiltonian'''
        res=dot(self.cdag,self.c)
        if conpensate==True:
            res[0,0]=self.offset()
        return res

    def mat_x(self):
        '''get the x operator'''
        return (self.c+self.cdag)/sqrt(2*self.m*self.w)

    def mat_p(self):
        '''get the p operator'''
        return 1j*(self.cdag-self.c)*sqrt(self.m*self.w/2)


if __name__=='__main__':
    bs=Boson(4)
    pdb.set_trace()
