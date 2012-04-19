"""
To do:
    Tests / Examples (same in some literature)
    Instanous freequncuies 
    Hilbert-Huang Transform
"""
import numpy as np
from scipy import interpolate

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