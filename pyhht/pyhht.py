#!/usr/bin/env python

from numpy import zeros, vstack, array, sin, linspace,hstack, real, imag,diff
from numpy.random import normal
from scipy.interpolate import splrep,splev
from scipy.io.wavfile import read
from scipy.signal import hilbert
from scipy import angle, unwrap
from math import pi
from matplotlib.pyplot import plot,show,imshow,colorbar,subplot,grid

def envlps(f):
    x=zeros((len(f),),dtype=int)
    y=x.copy()
    
    for i in range(1,len(f)-1):
        if (f[i]>f[i-1])&(f[i]>f[i+1]):
            x[i]=1
        if (f[i]<f[i-1])&(f[i]<f[i+1]):
            y[i]=1
    
    x=(x>0).nonzero()
    y=(y>0).nonzero()
    y=y[0]
    x=x[0]
    x=hstack((0,x,len(f)-1))
    y=hstack((0,y,len(f)-1))
    
    t=splrep(x,f[x])
    # numpy.arange
    top=splev(array(range(len(f))),t)
    t=splrep(y,f[y])
    bot=splev(array(range(len(f))),t)
    
    return top,bot

def sift(t):
    top,bot=envlps(t)
    c=t-(top+bot)/2
    return c

def localmean(t):
    top,bot=envlps(t)
    return (top+bot)/2
    

def checkimf(t):
    
    xtrm=zeros((len(t),),dtype=int)
    zcross=xtrm.copy()
    
    for i in range(1,len(t)-1):
        if (t[i]>t[i-1])&(t[i]>t[i+1]):
            xtrm[i]=1
        if (t[i]<t[i-1])&(t[i]<t[i+1]):
            xtrm[i]=1
        if t[i-1]*t[i+1]<0:
            zcross[i]=1
            
    
    a=(xtrm>0).nonzero()
    b=(zcross>0).nonzero()
    
    return abs(len(a[0])-len(b[0]))
    
def siftrun(t,n):
    d=checkimf(t)
    iters=0
    while d>1:
        t=sift(t)
        iters=iters+1
        d=checkimf(t)
        if iters==n:
            break
    return t
    
def emd(f,n):
    imfs=zeros((1,len(f)),dtype=float)
    for i in range(n):
        t=f-sum(imfs)
        t=siftrun(t,100)
        print i
        imfs=vstack((imfs,t))
    imfs = imfs[1:n+1,:]
    imfs=vstack((imfs,f-sum(imfs)))
    return imfs

def demoinit():
    Fs,f=read('sare.wav')
    f=f[30000:31001]
    return Fs,f
    
def plothilbert(imfs):
    for i in range(imfs.shape[0]):
        h=hilbert(imfs[i,:])
        plot(real(h),imag(h))
    show()
    
def symmetrydemo():
    a=sin(linspace(-5*pi,5*pi,10000))
    b=a+2
    c=a-0.5
    ah,bh,ch=hilbert(a),hilbert(b),hilbert(c)
    ph_a,ph_b,ph_c=unwrap(angle(ah)),unwrap(angle(bh)),unwrap(angle(ch))
    omega_a=diff(ph_a)
    omega_b=diff(ph_b)
    omega_c=diff(ph_c)
    subplot(211),plot(ph_a),plot(ph_b),plot(ph_c)
    subplot(212),plot(omega_a),plot(omega_b),plot(omega_c)
    grid()
    show()
    return a,b,c
    

def getinstfreq(imfs):
    omega=zeros((imfs.shape[0],imfs.shape[1]-1),dtype=float)
    for i in range(imfs.shape[0]):
        h=hilbert(imfs[i,:])
        theta=unwrap(angle(h))
        omega[i,:]=diff(theta)
        
    return omega