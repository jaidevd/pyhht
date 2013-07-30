# -*- coding: utf-8 -*-
#References
#
#
# [1] N. E. Huang et al., "The empirical mode decomposition and the
# Hilbert spectrum for non-linear and non stationary time series analysis",
# Proc. Royal Soc. London A, Vol. 454, pp. 903-995, 1998
#
# [2] G. Rilling, P. Flandrin and P. Gonçalves
# "On Empirical Mode Decomposition and its algorithms",
# IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing
# NSIP-03, Grado (I), June 2003
#
# [3] G. Rilling, P. Flandrin, P. Gonçalves and J. M. Lilly.,
# "Bivariate Empirical Mode Decomposition",
# Signal Processing Letters (submitted)
#
# [4] N. E. Huang et al., "A confidence limit for the Empirical Mode
# Decomposition and Hilbert spectral analysis",
# Proc. Royal Soc. London A, Vol. 459, pp. 2317-2345, 2003
#
# [5] R. Deering and J. F. Kaiser, "The use of a masking signal to improve 
# empirical mode decomposition", ICASSP 2005

import warnings
import numpy as np
from math import pi
from matplotlib.mlab import find
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev


class EMD(object):
    
    def __init__(self,x,t=None,sd=0.05,sd2=0.5,tol=0.05,MODE_COMPLEX=None,
                 ndirs=4,display_sifting=False,sdt=None,sd2t=None,r=None,
                 imf=None,k=1,nbit=0,NbIt=0,FIXE=0,MAXITERATIONS=2000,
                 FIXE_H=0,MAXMODES=0,INTERP='spline',mask=0):
        
        self.sd              = sd
        self.sd2             = sd2
        self.tol             = tol
        self.display_sifting = display_sifting
        self.MAXITERATIONS   = MAXITERATIONS
        self.FIXE_H          = FIXE_H
        self.ndirs           = ndirs
        self.complex_version = 2
        self.nbit            = nbit
        self.Nbit            = NbIt
        self.MAXMODES        = MAXMODES
        self.k               = k
        self.mask            = mask
                
        if x.ndim > 1:
            if 1 not in x.shape:
                raise TypeError("x must have only one row or one column.")
        if x.shape[0]>1:
            x = x.ravel()
        if not np.all(np.isfinite(x)):
            raise TypeError("All elements of x must be finite.")
        self.x = x
        self.ner = self.nzr = len(self.x)
        self.r = self.x.copy()
        
        if t is None:
            self.t = np.arange(np.max(x.shape))
        else:
            if t.shape != self.x.shape:
                raise TypeError("t must have the same dimensions as x.")
            if t.ndim > 1:
                if 1 not in t.shape:
                    raise TypeError("t must have only one column or one row.")
            if not np.isreal(t):
                raise TypeError("t must be a real vector.")
            if t.shape[0]>1:
                t = t.ravel()
            self.t = t
        
        if INTERP not in ['linear','cubic','spline']:
            raise TypeError("INTERP should be one of 'interp','cubic' or " +
                            "'spline'")
        self.INTERP = INTERP
        
        if sdt is None:
            self.sdt = sd*np.ones((len(self.x),))
        else:
            self.sdt = sdt
        
        if sd2t is None:
            self.sd2t = sd2*np.ones((len(self.x),))
        else:
            self.sd2t = sd2t
        
        if FIXE:
            self.MAXITERATIONS = FIXE
            if self.FIXE_H:
                raise TypeError("Cannot use both FIXE and FIXE_H modes")
        self.FIXE = FIXE
        
        if MODE_COMPLEX is None:
            MODE_COMPLEX = not(np.all(np.isreal(self.x)*self.complex_version))
        self.MODE_COMPLEX = MODE_COMPLEX
    
        self.imf = []
        self.nbits = []
        
        """ Masking not enabled because depends on the emd() method."""
        #if np.any(mask):
        #    if mask.shape != x.shape:
        #        raise TypeError("Masking signal must have the same dimensions" +
        #                        "as the input signal x.")
        #    if mask.shape[0]>1:
        #        mask = mask.ravel()
        #    imf1 = emd(x+mask, opts)

    def io(self):
        n = len(self.imf)
        s = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    s += np.abs(np.sum(self.imf[i]*np.conj(self.imf[j]))/np.sum(self.x**2))
        return 0.5*s

    def boundary_conditions(self):
        
        """ Generates mirrored extrema beyond the singal limits. """
    
        NBSYM = 2
        
        indmin, indmax = self.extr()[:2]
        
        lmin = indmin[:NBSYM]
        lmax = indmax[:NBSYM]
        rmin = indmin[len(indmin)-NBSYM:]
        rmax = indmax[len(indmax)-NBSYM:]
        
        lmin_extended = -1*lmin[::-1]
        lmax_extended = -1*lmax[::-1]
        rmin_extended = (len(self.x)-rmin)[::-1] - 1 + len(self.x)
        rmax_extended = (len(self.x)-rmax)[::-1] - 1 + len(self.x)
        
        tmin = np.concatenate((lmin_extended,indmin,rmin_extended))
        tmax = np.concatenate((lmax_extended,indmax,rmax_extended))
        
        zmin = self.x[indmin]
        zmax = self.x[indmax]
        
        zmin_left = self.x[lmin][::-1]
        zmax_left = self.x[lmax][::-1]
        zmin_right = self.x[rmin][::-1]
        zmax_right = self.x[rmax][::-1]
        
        zmin = np.concatenate((zmin_left, zmin, zmin_right))
        zmax = np.concatenate((zmax_left, zmax, zmax_right))
        
        return tmin, tmax, zmin, zmax

    
    def extr(self, x):
    
        """ Extracts the indices of the extrema and zero crossings. """
        
        m = len(x)

        x1 = x[:m-1]
        x2 = x[1:m]
        indzer = find(x1*x2<0)
        if np.any(x==0):
            iz = find(x==0)
            indz = [];
            if np.any(np.diff(iz)==1):
                zer = x == 0
                dz = np.diff([0,zer,0])
                debz = find(dz == 1)
                finz = find(dz == -1)-1
                indz = np.round((debz+finz)/2)
            else:
                indz = iz
            indzer = np.sort([indzer,indz])
    
        d = np.diff(x)
    
        n = len(d)
        d1 = d[:n-1]
        d2 = d[1:n]
        indmin = set(find(d1*d2<0)).intersection(find(d1<0))
        indmin = np.array(list(indmin)) + 1
        indmax = set(find(d1*d2<0)).intersection(find(d1>0))
        indmax = np.array(list(indmax)) + 1
    
        if np.any(d==0):
            imax = []
            imin = []
            bad = (d==0)
            dd = np.diff([0,bad,0])
            debs = find(dd == 1)
            fins = find(dd == -1)
            if debs[0] == 1:
                if len(debs) > 1:
                    debs = debs[2:]
                    fins = fins[2:]
                else:
                    debs = []
                    fins = []
            if len(debs) > 0:
                if fins(len(fins)-1) == m:
                    if len(debs)>1:
                        debs = debs[:len(debs)-1]
                        fins = fins[:len(fins)-1]
                    else:
                        debs = []
                        fins = []
            lc = len(debs)
            if lc > 0:
                for k in range(lc):
                    if d[debs[k]-1]>0:
                        if d[fins[k]] < 0:
                            imax = [imax,np.round((fins[k]+debs[k])/2)]
                    else:
                        if d[fins[k]]>- 0 :
                            imin = [imin, np.round((fins[k]+debs[k])/2)]
            
            if len(imax)>0:
                indmax = np.sort([indmax,imax])
            if len(imin)>0:
                indmin = np.sort([indmin, imin])
        return indmin, indmax, indzer    
    

    def stop_EMD(self):
        
        """ Tests if there are enough extrema (3) to continue sifting. """
        
        if self.MODE_COMPLEX:
            ner = []
            for k in range(self.ndirs):
                phi = k*pi/self.ndirs
                indmin, indmax = self.extr(np.real(np.exp(1j*phi)*self.r))[:2]
                ner.append(len(indmin)+len(indmax))
            stop = np.any(ner<3)
        else:
            indmin, indmax = self.extr(self.r)[:2]
            ner = len(indmin) + len(indmax)
            stop = ner < 3
        return stop

    def mean_and_amplitude(self, m):
        
        """ Computes the mean of the envelopes and the mode amplitudes."""
        
        if self.MODE_COMPLEX:
            if self.MODE_COMPLEX == 1:
                nem = []
                nzm = []
                envmin = np.zeros((self.ndirs,len(self.t)))
                envmax = np.zeros((self.ndirs,len(self.t)))
                for k in range(self.ndirs):
                    phi = k*pi/self.ndirs
                    y = np.real(np.exp(-1j*phi)*m)
                    indmin, indmax, indzer = self.extr(y)
                    nem.append(len(indmin)+len(indmax))
                    nzm.append(len(indzer))
                    tmin, tmax, zmin, zmax  = self.boundary_conditions()
                    
                    f = splrep(tmin,zmin)
                    spl = splev(self.t,f)
                    envmin[k,:] = spl
                    
                    f = splrep(tmax,zmax)
                    spl = splev(self.t,f)
                    envmax[k,:] = spl
                
                envmoy = np.mean((envmin+envmax)/2,axis=0)
                amp = np.mean(abs(envmax-envmin),axis=0)/2
            
            elif self.MODE_COMPLEX == 2:
                nem = []
                nzm = []
                envmin = np.zeros((self.ndirs,len(self.t)))
                envmax = np.zeros((self.ndirs,len(self.t)))
                for k in range(self.ndirs):
                    phi = k*pi/self.ndirs
                    indmin, indmax, indzer = self.extr(y)
                    nem.append(len(indmin)+len(indmax))
                    nzm.append(len(indzer))
                    tmin, tmax, zmin, zmax = self.boundary_conditions()
                    f = splrep(tmin, zmin)
                    spl = splev(self.t,f)
                    envmin[k,:] = np.exp(1j*phi)*spl
                    
                    f = splrep(tmax, zmax)
                    spl = splev(self.t,f)
                    envmax[k,:] = np.exp(1j*phi)*spl
                
                envmoy = np.mean((envmin+envmax),axis=0)
                amp = np.mean(abs(envmax-envmin),axis=0)/2
    
            else:
                indmin, indmax, indzer = self.extr(m)
                nem = len(indmin)+len(indmax);
                nzm = len(indzer);
                tmin,tmax,mmin,mmax = self.boundary_conditions();
                
                f = splrep(tmin, mmax)
                envmin = splev(self.t,f)
                
                f = splrep(tmax, mmax)
                envmax = splev(self.t,f);
                
                envmoy = (envmin+envmax)/2;
                amp = np.mean(abs(envmax-envmin),axis=0)/2
        
        return envmoy, nem, nzm, amp

    def stop_sifting(self, m):
        try:
            envmoy, nem, nzm, amp = self.mean_and_amplitude(m)
            sx = np.abs(envmoy)/amp
            s = np.mean(sx)
            stop = not((np.mean(sx>self.sd)>self.tol | np.any(sx>self.sd2)) \
                   and np.all(nem>2))
            if not self.MODE_COMPLEX:
                stop = stop and not(np.abs(nzm-nem)>1)
        except:
            stop = 1
            envmoy = np.zeros((len(m),))
            s = np.nan
        return stop, envmoy, s
    
    def stop_sifting_fixe(self):
        moyenne = self.mean_and_amplitude()
        stop = 0
        return stop, moyenne
    
    def stop_sifting_fixe_h(self, m):
        try:
            moyenne, nem, nzm = self.mean_and_amplitude(m)[:3]
            
            if np.all(abs(nzm-nem)>1):
                stop = 0
                stop_count = 0
            else:
                stop_count += 1
                stop = (stop_count == self.FIXE_H)
        except:
            moyenne = np.zeros((len(m)))
            stop = 1
        
        return stop, moyenne, stop_count

    
    def emd(self):
        if self.display_sifting:
            fig_h = plt.figure()
        
        A = not(self.stop_EMD())
        B = (self.k<self.MAXMODES+1 or self.MAXMODES==0)
        C = not(np.any(self.mask))
        
        while (A and B and C):
            
            # current mode
            m = self.r
            
            # mode at previous iteration
            mp = m.copy()
            
            # computing mean and stopping criterion
            if self.FIXE:
                stop_sift, moyenne = self.stop_sifting_fixe()
            elif self.FIXE_H:
                stop_count = 0
                stop_sift, moyenne = self.stop_sifting_fixe_h()
            else:
                stop_sift, moyenne = self.stop_sifting(m)[:2]
            
            # in case current mode is small enough to cause spurious extrema
            if np.max(np.abs(m)) < (1e-10)*np.max(np.abs(self.x)):
                if not stop_sift:
                    warnings.warn("EMD Warning: Amplitude too small, stopping.")
                else:
                    print "Force stopping EMD: amplitude too small."
                return
            
            # SIFTING LOOP:
            while not(stop_sift) and (self.nbit<self.MAXITERATIONS):
                
                if (not(self.MODE_COMPLEX) and (self.nbit>self.MAXITERATIONS/5) \
                    and self.nbit%np.floor(self.MAXITERATIONS/10)==0 and \
                    not(self.FIXE) and self.nbit > 100):
                    print "Mode "+str(self.k) + ", Iteration " + str(self.nbit)
                    im, iM = self.extr(m)
                    print str(np.sum(m[im]>0)) + " minima > 0; " + \
                          str(np.sum(m[im]<0)) + " maxima < 0."
                
                # Sifting
                m = m - moyenne
                
                # Computing mean and stopping criterion
                if self.FIXE:
                    stop_sift, moyenne = self.stop_sifting_fixe()
                elif self.FIXE_H:
                    stop_sift, moyenne, stop_count = self.stop_sifting_fixe_h()
                else:
                    stop_sift, moyenne, s = self.stop_sifting()
                
                # Display
                if self.display_sifting and self.MODE_COMPLEX:
                    indmin, indmax = self.extr(m)
                    tmin, tmax, mmin, mmax = self.boundary_conditions()
                    
                    f = splrep(tmin,mmin)
                    envminp = splev(self.t,f)
                    f = splrep(tmax,mmax)
                    envmaxp = splev(self.t,f)
                    
                    envmoyp = (envminp+envmaxp)/2;
                    
                    if self.FIXE or self.FIXE_H:
                        self.display_emd_fixe(mp, envminp, envmaxp, envmoyp)
                    else:
                        sxp = 2*(np.abs(envmoyp))/np.abs(envmaxp-envminp)
                        sp = np.mean(sxp)
                        self.display_emd(mp, envminp, envmaxp, envmoyp, sp, sxp)
                
                mp = m
                self.nbit += 1
                self.NbIt += 1
                
                if (self.nbit==(self.MAXITERATIONS-1)) and not(self.FIXE) and (self.nbit>100):
                    warnings.warn("Emd:warning, Forced stop of sifting - "+
                                  "too many iterations")
            
            self.imf.append(m)
            if self.display_sifting:
                print "mode "+str(self.k)+ " stored"
            
            self.nbits.append(self.nbit)
            self.k += 1
            
            self.r = self.r - m
            ort = self.io()
            
            self.ort = ort
            return self.imf

if __name__ == "__main__":
    x = np.random.random((1000,))
    plt.subplot(311),plt.plot(x)
    emd = EMD(x)
    imf = emd.emd()[0]
    plt.subplot(312),plt.plot(imf)
    plt.subplot(313),plt.plot(x-imf)
    plt.show()