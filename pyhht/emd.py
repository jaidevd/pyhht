"""
To do:
    Tests / Examples (same in some literature)
    Instantaneous frequencies
    Hilbert-Huang Transform
"""
import numpy as np
from numpy import pi
from scipy.interpolate import splrep, splev
from scipy.signal import argrelmax, argrelmin
import matplotlib.pyplot as plt
from matplotlib.mlab import find
import warnings


__all__ = 'EMD'


def emd(data, extrapolation='mirror', nimfs=12, shifting_distance=0.2):
    """
    Perform a Empirical Mode Decomposition on a data set.

    This function will return an array of all the Imperical Mode Functions as
    defined in [1]_, which can be used for further Hilbert Spectral Analysis.

    The EMD uses a spline interpolation function to approcimate the upper and
    lower envelopes of the signal, this routine implements a extrapolation
    routine as described in [2]_ as well as the standard spline routine.
    The extrapolation method removes the artifacts introduced by the spline fit
    at the ends of the data set, by making the dataset a continuious circle.

    Parameters
    ----------
    data : array_like
            Signal Data
    extrapolation : str, optional
            Sets the extrapolation method for edge effects.
            Options: None
                     'mirror'
            Default: 'mirror'
    nimfs : int, optional
            Sets the maximum number of IMFs to be found
            Default : 12
    shifiting_distance : float, optional
            Sets the minimum variance between IMF iterations.
            Default : 0.2

    Returns
    -------
    IMFs : ndarray
            An array of shape (len(data),N) where N is the number of found IMFs

    Notes
    -----

    References
    ----------
    .. [1] Huang H. et al. 1998 'The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis.'
    Procedings of the Royal Society 454, 903-995

    .. [2] Zhao J., Huang D. 2001 'Mirror extending and circular spline function for empirical mode decomposition method'
    Journal of Zhejiang University (Science) V.2, No.3,P247-252

    .. [3] Rato R.T., Ortigueira M.D., Batista A.G 2008 'On the HHT, its problems, and some solutions.'
    Mechanical Systems and Signal Processing 22 1374-1394


    """

    #Set up signals array and IMFs array based on type of extrapolation
    # No extrapolation and 'extend' use signals array which is len(data)
    # Mirror extrapolation (Zhao 2001) uses a signal array len(2*data)
    if not(extrapolation):
        base = len(data)
        signals = np.zeros([base, 2])
        nimfs = range(nimfs) # Max number of IMFs
        IMFs = np.zeros([base, len(nimfs)])
        ncomp = 0
        residual = data
        signals[:, 0] = data
        #DON'T do spline fitting with periodic bounds
        inter_per = 0

    elif extrapolation == 'mirror':
        #Set up base
        base = len(data)
        nimfs = range(nimfs) # Max number of IMFs
        IMFs = np.zeros([base, len(nimfs)])
        ncomp = 0
        residual = data
        #Signals is 2*base
        signals = np.zeros([base*2, 2])
        #Mirror Dataset
        signals[0:base / 2, 0] = data[::-1][base / 2:]
        signals[base / 2:base + base / 2, 0] = data
        signals[base + base / 2:base * 2, 0] = data[::-1][0:base / 2]
        # Redfine base as len(signals) for IMFs
        base = len(signals)
        data_length = len(data) # Data length is used in recovering input data
        #DO spline fitting with periodic bounds
        inter_per = 1

    else:
        raise Exception(
        "Please Specifiy extrapolation keyword as None or 'mirror'")

    for j in nimfs:
#       Extract at most nimfs IMFs no more IMFs to be found when Finish is True
        k = 0
        sd = 1.
        finish = False

        while sd > shifting_distance and not(finish):
            min_env = np.zeros(base)
            max_env = min_env.copy()

            min_env = np.logical_and(
                                np.r_[True, signals[1:,0] > signals[:-1,0]],
                                np.r_[signals[:-1,0] > signals[1:,0], True])
            max_env = np.logical_and(
                                np.r_[True, signals[1:,0] < signals[:-1,0]],
                                np.r_[signals[:-1,0] < signals[1:,0], True])
            max_env[0] = max_env[-1] = False
            min_env = min_env.nonzero()[0]
            max_env = max_env.nonzero()[0]

            #Cubic Spline by default
            order_max = 3
            order_min = 3

            if len(min_env) < 2 or len(max_env) < 2:
                #If this IMF has become a straight line
                finish = True
            else:
                if len(min_env) < 4:
                    order_min = 1 #Do linear interpolation if not enough points

                if len(max_env) < 4:
                    order_max = 1 #Do linear interpolation if not enough points

#==============================================================================
# Mirror Method requires per flag = 1 No extrapolation requires per flag = 0
# This is set in intial setup at top of function.
#==============================================================================
                t = interpolate.splrep(min_env, signals[min_env,0],
                                       k=order_min, per=inter_per)
                top = interpolate.splev(
                                    np.arange(len(signals[:,0])), t)

                b = interpolate.splrep(max_env, signals[max_env,0],
                                       k=order_max, per=inter_per)
                bot = interpolate.splev(
                                    np.arange(len(signals[:,0])), b)

            #Calculate the Mean and remove from the data set.
            mean = (top + bot)/2
            signals[:,1] = signals[:,0] - mean

            #Calculate the shifting distance which is a measure of
            #simulartity to previous IMF
            if k > 0:
                sd = (np.sum((np.abs(signals[:,0] - signals[:,1])**2))
                             / (np.sum(signals[:,0]**2)))

            #Set new iteration as previous and loop
            signals = signals[:,::-1]
            k += 1

        if finish:
            #If IMF is a straight line we are done here.
            IMFs[:,j]= residual
            ncomp += 1
            break

        if not(extrapolation):
            IMFs[:,j] = signals[:,0]
            residual = residual - IMFs[:,j]#For j==0 residual is initially data
            signals[:,0] = residual
            ncomp += 1

        elif extrapolation == 'mirror':
            IMFs[:,j] = signals[data_length / 2:data_length
                                                           + data_length / 2,0]
            residual = residual - IMFs[:,j]#For j==0 residual is initially data

            #Mirror case requires IMF subtraction from data range then
            # re-mirroring for each IMF
            signals[0:data_length / 2,0] = residual[::-1][data_length / 2:]
            signals[data_length / 2:data_length + data_length / 2,0] = residual
            signals[data_length
                + data_length / 2:,0] = residual[::-1][0:data_length / 2]
            ncomp += 1

        else:
            raise Exception(
                "Please Specifiy extrapolation keyword as None or 'mirror'")

    return IMFs[:,0:ncomp]


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

    def boundary_conditions(self, NBYSUM=2):

        """ Generates mirrored extrema beyond the singal limits. """


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
        # FIXME: This doesn't have to be a method here.
        m = x.shape[0]


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
            indzer = np.sort(np.hstack([indzer,indz]))

        indmax = argrelmax(x)[0]
        indmin = argrelmin(x)[0]

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
        # FIXME: needs the rest of the parameters to work on!!!
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


    def decompose(self):
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
                stop_sift, moyenne, _ = self.stop_sifting(m)

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
            return np.array(self.imf)



if __name__ == "__main__":
    Fs = 10000.0
    ts = np.linspace(0, 1, Fs)
    f1, f2 = 5, 10
    y1 = np.sin(2*np.pi*f1*ts)
    y2 = np.sin(2*np.pi*f2*ts)
    y = y1 + y2
    x = y
    x += np.linspace(0, 1, x.shape[0])

    plt.plot(x)
    plt.show()
    emd = EMD(x)
