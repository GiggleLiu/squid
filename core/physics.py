'''
Author: Giggle Leo
Date : 8 September 2014
Description : physics library

'''

from numpy import *
from numpy.dual import *
from numpy.linalg import *
try:
    from matplotlib.pyplot import *
    from mpl_toolkits.mplot3d import Axes3D
except:
    print 'import error'
import scipy
from scipy.integrate import quad
from scipy.sparse import coo_matrix
from mpi4py import MPI
import pdb
 
# pauli spin

sx = array([[0, 1],[ 1, 0]])
sy = array([[0, -1j],[1j, 0]])
sz = array([[1, 0],[0, -1]])

# sigma functions

sigma_x = array([[0, 1],[1, 0]])
sigma_y = array([[0, -1j],[1j, 0]])
sigma_z = array([[1, 0],[0, -1]])

# standard basis

spin_up = array([[1],[0]])
spin_down = array([[0],[1]])

# spin ladder operators

sigma_one_plus = sigma_x + 1j * sigma_y
sigma_two_plus = sigma_x + 1j * sigma_y

#print 'example 1\n========='
#print 's_{z} |+1/2, +1/2> = frac{hbar}{2} ( sigma_{1z} tensor 1_{2} + 1_{1} tensor sigma_{2z} ) |+1/2, +1/2>'
#print mat(kron(sigma_z, identity(2))) * mat(kron(spin_up, spin_up))

#print 'example 2\n========='
#print 's_{+} |-1/2, -1/2>'
#print (mat(kron(sigma_one_plus, identity(2))) * mat(kron(spin_down, spin_down)) + mat(kron(identity(2), sigma_one_plus)) * mat(kron(spin_down, spin_down))) / 2

#print 'example 3\n========='
#print 's_{+} |+1/2, -1/2>'
#print (mat(kron(sigma_one_plus, identity(2))) * mat(kron(spin_up, spin_down)) + mat(kron(identity(2), sigma_one_plus)) * mat(kron(spin_up, spin_down))) / 2
 
#print 'example 4\n========='
#print 's_{+} |-1/2, +1/2>'
#print (mat(kron(sigma_one_plus, identity(2))) * mat(kron(spin_down, spin_up)) + mat(kron(identity(2), sigma_one_plus)) * mat(kron(spin_down, spin_up))) / 2

def commute(m1,m2):
    '''calculate commutation'''
    return dot(m1,m2)-dot(m2,m1)

Lx=array([[0,1,0],[1,0,1],[0,1,0]],dtype='complex128')/sqrt(2.0)
Lz=array([[1,0,0],[0,0,0],[0,0,-1]],dtype='complex128')
Ly=-1j*commute(Lz,Lx)

s=[identity(2),sx,sy,sz]
Gmat=array([kron(sz,sx),kron(identity(2),sy),kron(sx,sx),kron(sy,sx),kron(identity(2),sz)])

def getsocmat(tp='p'):
    s=[sx,sy,sz]
    L=[Lx,Ly,Lz]
    if tp=='p':
        LpS=zeros([6,6],dtype='complex128')
        Ump=array([[-sqrt(2.)/2,sqrt(2.)*1j/2,0.],[0.,0.,1.],[sqrt(2.)/2,sqrt(2.)*1j/2,0.]])
        UmpH=conj(transpose(Ump))
        for i in xrange(3):
            LpS+=kron(s[i],dot(dot(UmpH,L[i]),Ump))
        return LpS
    if tp=='sp3':
        LpS=zeros([8,8],dtype='complex128')
        pmask=arange(8)%4!=0
        pmask=kron(pmask,pmask).reshape([8,8])
        Ump=array([[-sqrt(2.)/2,sqrt(2.)*1j/2,0.],[0.,0.,1.],[sqrt(2.)/2,sqrt(2.)*1j/2,0.]])
        UmpH=conj(transpose(Ump))
        for i in xrange(3):
            LpS[pmask]+=kron(s[i],dot(dot(UmpH,L[i]),Ump))
        return LpS
def pauli(index,nl,nr):
    '''get (super)pauli matrix'''
    if index==0:
        return identity(nl*nr*2)
    elif index==1:
        return kron(kron(identity(nl),sx),identity(nr))
    elif index==2:
        return kron(kron(identity(nl),sy),identity(nr))
    elif index==3:
        return kron(kron(identity(nl),sz),identity(nr))

def pf(m) :
    mat=copy(m)
    ndim = shape(mat)[0]
    t1=1.0
    for j in range(ndim/2) :
        t1 *= mat[0,1]
        #print mat
        if j <ndim/2-1 :
            ndimr=shape(mat)[0]
            for i in range(2,ndimr) :
                if mat[0,1] != 0.0 :
                    tv=mat[1,:]*mat[i,0]/mat[1,0]
                    #print tv
                    mat[i,: ] -= tv
                    tv=mat[:,1]*mat[0,i]/mat[0,1]
                    #print tv
                    mat[:,i ] -= tv
                else :
                    print 'need to pivot'
                    raise Exception
            mat=mat[2:,2:]
    return t1
 
def reciprocal2D(a1,a2):
    s=cross(a1,a2)
    b1=2*pi*array([a2[1],-a2[0]])/s
    b2=2*pi*array([-a1[1],a1[0]])/s
    return (b1,b2)

def toreciprocal(a):
    '''get the reciprocal lattice vectors'''
    res=2*pi*transpose(inv(a))
    return res
 
def decompose(imodel,q):
    (qn1,qn2)=solve(concatenate([matrix(imodel.b1).T,matrix(imodel.b2).T],1),matrix(q).T)
    qn1=round(qn1*imodel.N1)
    qn2=round(qn2*imodel.N2)
    return (qn1,qn2)
def plotdecompose(imodel,q):
    (qn1,qn2)=solve(concatenate([matrix(imodel.b1).T,matrix(imodel.b2).T],1),matrix(q).T)
    qn1=round(qn1*imodel.plotN1)
    qn2=round(qn2*imodel.plotN2)
    return (qn1,qn2)

def fstat(energy,T=0):
    '''fermi statistics'''
    if rank(energy)==0:
        return fermi(energy,T)
    elif rank(energy)==1:
        res=ndarray(energy.shape,dtype='float64')
        dlen=len(energy)
        for i in xrange(dlen):
             res[i]=fermi(energy[i],T)
        return res
    else:
        print 'rank error @fermi'

def rotate(vector,theta):
    try:
        rotatematrix=array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]],dtype='float64',order='F')
        return dot(rotatematrix,vector)
    except:
        print 'Rotate Error: vector dimension mismatch! it should have two rows but got ',len(vector),'.'
        return

def sv(vector,n):
    '''return the mirror of vector to n'''
    n=n/norm(n)
    return vector-2*dot(vector,n)*n

def Srot(theta,n=array([0,0,1.])):
    '''rotation matrix that will rotate the spin wave function by theta,
    equivalently, it will rotate the orientation of s matrix by -theta when acting on s matrix like U$^\dag$sU'''
    sigman=vec2s(n)
    return identity(2)*cos(theta/2)-1j*sin(theta/2)*sigman

def Ssv(n):
    '''return the spin imaging operation about the plane perpendicular to axis n'''
    return -1j*vec2s(n)

def genmesh(func,mesh,meshdim=2,dtype='float64',shape=1):
    if meshdim==2 and shape!=1:
        res=ndarray([mesh.shape[0],mesh.shape[1],shape],dtype=dtype)
        for i in xrange(mesh.shape[0]):
            for j in xrange(mesh.shape[1]):
                res[i,j,:]=func(mesh[i,j,:])
    elif meshdim==2 and shape==1:
        res=ndarray([mesh.shape[0],mesh.shape[1]],dtype=dtype)
        for i in xrange(mesh.shape[0]):
            for j in xrange(mesh.shape[1]):
               res[i,j]=func(mesh[i,j,:])
    elif meshdim==1 and shape==1:
        res=ndarray([mesh.shape[0]],dtype=dtype)
        for i in xrange(mesh.shape[0]):
            res[i,j]=func(mesh[i][j])

def matrixsqrt(A):
    '''Calculate the Matrix Square of A Hermian Matrix'''
    A=matrix(A,dtype='complex128')
    ndim=A.shape[0]
    evalue,evect=eig(A)
    evalue=sqrt(evalue)
    B=zeros([ndim,ndim],dtype='complex128')
    for i in range(ndim):
        B=B+evalue[i]*matrix(evect[i]).T*matrix(evect[i])
    return B

def gauss_random(w=1):
    '''Generate a random number which is Gaussian distributed with variance w^2'''
    r=0
    for i in range(12):
        r=r+random.random()-0.5
    return r*w

def cquad(func, a, b, **kwargs):
    def real_func(x):
        return func(x).real
    def imag_func(x):
        return func(x).imag
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])
 
def bisect(func, low, high,Nmax=50,eps=1e-4):
    'Find root of continuous function where f(low) and f(high) have opposite signs'
    flow=func(low)
    fhigh=func(high)
    assert flow*fhigh<0 #should not have the same sign
    for i in range(Nmax):
        midpoint = (low + high) / 2.0
        fmid=func(midpoint)
        if flow*fmid>0:
            low = midpoint
            flow=fmid
        else:
            high = midpoint
            fhigh=fmid
        if abs(fmid)<=eps:
            break
    return midpoint

def argkv(kv):
    '''return the angle of k'''
    return angle(complex(kv[0],kv[1]))

def fillmesh_mpi(inputmesh,outputmesh,func,islist=False,useMPI=True):
    '''fill the return mesh byacting func on inputmesh using mpi'''
    try:
        comm=MPI.COMM_WORLD
        size=comm.Get_size()
        rank=comm.Get_rank()
    except:
        size=1
        rank=0
        useMPI=False
    N1=inputmesh.shape[0]
    N2=1
    if not islist:
        N2=inputmesh.shape[1]
    blocksize=N1/size
    block=copy(outputmesh[:blocksize])
    if N1%size!=0:
        print 'error, mpi dimension mismatch'
        pdb.set_trace()
    for i in xrange(N1):
        if i/blocksize==rank:
            if islist:
                q=inputmesh[i]
                block[i%blocksize]=func(q)
            else:
                for j in xrange(N2):
                    q=inputmesh[i,j]
                    block[i%blocksize,j]=func(q)
    if useMPI:
        blocklist=comm.gather(block,root=0)
    if rank==0:
        #gather the mesh
        outputmesh=concatenate(blocklist,axis=0)
    #broadcast mesh
    if useMPI:
        outputmesh=comm.bcast(outputmesh,root=0)
    return outputmesh

def mydiff(mesh,axis=0):
    '''differentiation in the periodic meshgrid, returns x(i+1)-x(i)'''
    #newshape=list(mesh.shape)
    #newshape[axis]=1
    #mesh2=concatenate([mesh,mesh.take(0,axis=axis).reshape(newshape)],axis=axis)
    #return diff(mesh2,axis=axis)
    mesh2=roll(mesh,-1,axis=axis)
    return mesh2-mesh

def connectify(mat,axis=0,maxrange=10):
    N=mat.shape[axis]
    #pmat=roll(mat,1,axis=axis)
    nmat=roll(mat,-1,axis=axis)

    def connect(i,cj,nj):
        cj,nj=(nj,cj) if cj>nj else (cj,nj+1)
        if axis==0:
            mat[i,cj:nj]=True
        else:
            mat[cj:nj,i]=True

    for i in xrange(N):
        pending=None
        cjlist=where(mat.take(i,axis=axis))[0]
        #pjlist=where(pmat.take(i,axis=axis))
        njlist=where(nmat.take(i,axis=axis))[0]

        #connectednj=[]
        for cj in cjlist:
            connected=False
            if len(njlist)!=0:
                jindex=abs(njlist-cj).argmin()
                nj=njlist[jindex]
                if abs(nj-cj)<=maxrange:
                    connect(i,cj,nj)
                    njlist=delete(njlist,jindex)
                    connected=True
            if not connected:
                #connect this two points
                if pending!=None:
                    connect(i,pending,cj)
                    pending=None
                else:
                    pending=cj
        njlist=list(njlist)

        while len(njlist)>=2:
            pnj=njlist.pop()
            cnj=njlist.pop()
            connect((i+1)%N,cnj,pnj)
    return mat

def append_tofile(value,f,last=False):
    if (type(value)==complex) or (type(value)==complex128):
        string=str(value.real)+' '+str(value.imag)+' '
    else:
        string=str(value)+' '
    if last==True:
        string+='\n'
    f.write(string)

def s2vec(s):
    '''return a 4 dimensional vector, corresponding to s0,sx,sy,sz component.'''
    res=array([trace(s),trace(dot(sx,s)),trace(dot(sy,s)),trace(dot(sz,s))])/2
    return res

def vec2s(n):
    '''get sigma_n'''
    n=n/norm(n)
    if len(n)==3:
        return sx*n[0]+sy*n[1]+sz*n[2]
    elif len(n)==4:
        return identity(2)*n[0]+sx*n[1]+sy*n[2]+sz*n[3]
    else:
        print 'Dimension Error'
        pdb.set_trace()

def plotspin(slist,onball=True):
    '''plot a spin on the bloch ball'''
    slist=array(slist).reshape([-1,2,2])
    nlist=[]
    for s in slist:
        nlist.append(s2vec(s).real[1:])
    plotvec(nlist,onball=onball)

def plotvec(nlist,onball=True):
    fig = figure()
    ax = fig.add_subplot(111, projection='3d',adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #draw a sphere
    u, v = mgrid[0:2*pi:40j, 0:pi:20j]
    x=cos(u)*sin(v)
    y=sin(u)*sin(v)
    z=cos(v)
    ax.plot_wireframe(x, y, z,  rstride=1, cstride=1, color='#AAAAAA')
    ax.set_aspect("equal")

    color=linspace(0,1,len(nlist))
    basecolor=array([1.,0,0.])
    for i in xrange(len(nlist)):
        n=nlist[i]
        if onball:
            n=n/norm(n)
        ax.scatter(n[0:1],n[1:2],n[2:3],linewidth=3,color=color[i]*basecolor)
    show()

def subtrace(A,ndim,subdim):
    '''subtrace means trace on specific dimention, the traced matrix dimention is ndim,
    and subdimention(dimension of matrix in the right hand side in kronecker product) is subdim'''
    superdim=A.shape[0]/ndim/subdim
    res=zeros([A.shape[0]/ndim,A.shape[1]/ndim],dtype=A.dtype)
    for i in xrange(superdim):
        for j in xrange(superdim):
            subblock=A[i*ndim*subdim:(i+1)*ndim*subdim,j*ndim*subdim:(j+1)*ndim*subdim]
            for k in xrange(subdim):
                for l in xrange(subdim):
                    res[i*subdim+k,j*subdim+l]=subblock[k::subdim,l::subdim].trace()
    return res

def subtract(A,ndim,subdim):
    '''subtract sub matrix from kronicker product(the matrix must be decomposible), the result will take an arbituary constant factor A11.'''
    dim=A.shape[0]
    ddim=ndim*subdim
    superdim=dim/ddim
    #decide a none zero subdim index
    nzindex=argmax(abs(A))
    nzi,nzj=nzindex/dim,nzindex%dim
    supernzi,midnzi,lowernzi=nzi/ndim/subdim,nzi%(ndim*subdim)/subdim,nzi%subdim
    supernzj,midnzj,lowernzj=nzj/ndim/subdim,nzj%(ndim*subdim)/subdim,nzj%subdim
    subblock=A[supernzi*ddim:(supernzi+1)*ddim,supernzj*ddim:(supernzj+1)*ddim]
    return subblock[lowernzi::subdim,lowernzj::subdim]

def sort_natural(a,ev=None,periodic=True):
    '''sort a list of array by continuity.'''
    gatefactor=5
    N,ndim=a.shape
    dar1=roll(a,-1,axis=0)-a
    dal1=a-roll(a,1,axis=0)
    gate=abs(dar1).mean()*gatefactor
    minmask=(dar1>0)&(dal1<0) #a local maximum
    maxmask=(dar1<0)&(dal1>0) #a local minimum
    crossinds=[]
    rg=arange(N)
    for i in rg:
        maxmaski=maxmask[i]
        minmaski=minmask[i]
        if any(maxmaski) and any(minmaski):
            ai=a[i]
            maxlist=[]
            for b in xrange(ndim):
                if maxmaski[b]:
                    maxlist.append(b)
                elif minmaski[b] and maxlist!=[]:
                    nmax=len(maxlist)
                    for mi in xrange(nmax):
                        if abs(ai[b]-ai[maxlist[mi]])<gate:
                            crossinds.append([i,maxlist.pop(mi),b])
                            break

    #now exchange them
    nchange=len(crossinds)
    for i in xrange(nchange):
        startindex,b1,b2=crossinds[nchange-i-1] #swich from last change!!! otherwise, it will change band order
        temp=copy(a[startindex:,b1])
        a[startindex:,b1]=a[startindex:,b2]
        a[startindex:,b2]=temp
        print 'switching ',b1,b2,'from',startindex
        if ev!=None:
            temp=copy(ev[startindex:,b1])
            ev[startindex:,:,b1]=ev[startindex:,:,b2]
            ev[startindex:,:,b2]=temp
    return crossinds

def pol(lst):
    '''parity of list'''
    n=len(lst)
    perms=0
    for i in xrange(n):
        cvalue=lst[i]
        for j in xrange(i):
            if lst[j]>cvalue:
                perms+=1
    return 1-2*perms%2

def testBit(int_type, offset):
    mask = 1 << offset
    return 1 if (int_type & mask) else 0
def clearBit(int_type, offset):
    mask = ~(1 << offset)
    return(int_type & mask)
def setBit(int_type, offset):
    mask = 1 << offset
    return(int_type | mask)
def toggleBit(int_type, offset):
    mask = 1 << offset
    return(int_type ^ mask)

def bitLenCount(int_type,offset=0):
    length = 0
    count = 0
    int_type >>= offset
    while (int_type):
        count += (int_type & 1)
        length += 1
        int_type >>= 1
    return (length, count)

def parityOf(int_type,offset=0):
    '''return 1 for even parity, -1 for odd parity'''
    parity = 0
    int_type >>= offset
    while (int_type):
        parity = ~parity
        int_type = int_type & (int_type - 1)
    return(parity*2+1)

def bitCount(int_type):
    count = 0
    while(int_type):
        int_type &= int_type - 1
        count += 1
    return(count)

def reshape_sparse(a, shape):
    """Reshape the sparse matrix `a`.
    Returns a coo_matrix with shape `shape`.
    """
    if not hasattr(shape, '__len__') or len(shape) != 2:
        raise ValueError('`shape` must be a sequence of two integers')

    c = a.tocoo()
    nrows, ncols = c.shape
    size = nrows * ncols

    new_size =  shape[0] * shape[1]
    if new_size != size:
        raise ValueError('total size of new array must be unchanged')

    flat_indices = ncols * c.row + c.col
    new_row, new_col = divmod(flat_indices, shape[1])

    b = coo_matrix((c.data, (new_row, new_col)), shape=shape)
    return b

def c2ind(c,N):
    '''transform c into index,
    N is the space config.'''
    n=len(c)
    cc=c[0]
    for i in xrange(n-1):
        cc=cc*N[i+1]+c[i+1]
    return cc

def ind2c(ind,N):
    '''index2l in the 2D case.'''
    dim=len(N)
    indl=array(dim,dtype='int32')
    for i in xrange(dim):
        indl[-1-i]=index%N[-1-i]
        index=index/N[-1-i]
    return indl


if __name__=='__main__':
    #test spin operation
    ma=random.random([2,2])
    mb=random.random([3,3])
    mc=kron(ma,mb)
    print mc
    print subtract(mc,ndim=2,subdim=3)
    pdb.set_trace()
    s=vec2s(random.random([3]))
    U=Ssv([1,0,0])
    s1=dot(dot(conj(transpose(U)),s),U)
    plotspin([s,s1])

