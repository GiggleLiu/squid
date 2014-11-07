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
    def __init__(self):
        super(SupercellModel,self).__init__(structure)
        self.N=array(MODEL_N)
        self.nbose=len(N)
        self.ndim=self.N.prod()

    def pushmats(self,mats):
        '''push a n body operator to the hamiltonian'''
        res=1.
        for i in xrange(nbose):
            res=kron(res,mats[i])
        self.H+=res

    def pushc(self,nlist,factor):
        '''the c matrix for N boson.'''
        res=1.
        for i in xrange(nbose):
            res=kron(res,self.boselist[i].mat_c(nlist[i]))
        self.H+=res


class Boson(object):
    '''The Boson item Model.'''
    def __init__(self,ndim):
        super(Boson,self).__init__()
        self.ndim=ndim
        self.c=diag(sqrt(arange(1,self.ndim)),k=1)
        self.cdag=diag(sqrt(arange(1,self.ndim)),k=-1)

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

    def mat_expaa(self,lamb):
        '''get a matrix expression of exp(a+a^dag)'''
        def eval_expaa(lamb,i,j):
            '''get the i,j element of exp(lamb(a+a^dag))'''
            res=0.
            r=min(i,j)+1
            for m in xrange(r):
                res+=lamb**(i+j-2*m)*sqrt(factorial(j)*factorial(i))/factorial(m)/factorial(i-m)/factorial(j-m)
            return exp(lamb**2/2)*res #it is a pending issue whether '-' sign make sense here.

        return array([[eval_expaa(lamb,i,j) for j in xrange(self.ndim+1)] for i in xrange(self.ndim+1)])

    def mat_n(self):
        '''get a matrix description ni type hamiltonian'''
        return dot(self.cdag,self.c)


if __name__=='__main__':
    bs=Boson(4)
    pdb.set_trace()
