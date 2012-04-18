import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

__all__ = 'EMD'

def EMD(data, inter = 'Default'):
    """
    OMG, HERE IS SOME EMD ON EMD AWESOME SAUCE ACTION GOING ON!
    """
        
    if inter == 'Default' or inter == 'extrap':
        base = len(data) 
        nimfs = range(12) # Max number of IMFs
        IMFs = np.zeros([base,len(nimfs)])    
        ncomp = 0
        residual = 0 
        signals = np.zeros([base,2])
        signals[:,0] = data
    elif inter == 'Mirror':
        base = len(data) 
        nimfs = range(12) # Max number of IMFs
        IMFs = np.zeros([base,len(nimfs)])    
        ncomp = 0
        residual = 0 
        signals = np.zeros([base*3,2])
        signals[0:base,0] = data[::-1]
        signals[base:base*2,0] = data
        signals[base*2:base*3,0] = data[::-1]
        base = len(signals)
        data_length = len(data) 
    else:
        raise Exception("Bite me bitch")
            
    for j in nimfs:
    
        k = 0
        sd = 1.
        finish = False
                    
        while sd > 0.2 and not(finish):
                    
                x = np.zeros(base)
                y = x.copy()
                
            
                for i in range(1,len(signals[:,0])-1):
                    if (signals[i,0] > signals[i-1,0] and signals[i,0] >= signals[i+1,0]) or (signals[i,0] >= signals[i-1,0] and signals[i,0] > signals[i+1,0]):
                        x[i]=1
                    if signals[i,0] < signals[i-1,0] and signals[i,0] <= signals[i+1,0] or (signals[i,0] <= signals[i-1,0] and signals[i,0] < signals[i+1,0]):
                        y[i]=1
                xc = x.nonzero()[0] 
                yc = y.nonzero()[0]
                
                if len(xc) < 2 or len(yc) < 2:
                    print "escape"
                    finish = True
                else:
                                    
                    if len(xc) < 4:
                        t = interpolate.splrep(xc,signals[xc,0], k=1)
                        top = interpolate.splev(np.arange(len(signals[:,0])),t)
                    else:
                        if inter == 'extrap':
                            x0 = x[:xc[0]+1][::-1]
                            x1 = x[xc[-1]:][::-1]
                            xn = np.hstack((x0,x,x1)).nonzero()[0]
                            sx = np.hstack((signals[:xc[0]+1,0][::-1],signals[:,0],signals[xc[-1]:,0][::-1]))
                            t = interpolate.splrep(xn,sx[xn])
                            top = interpolate.splev(np.arange(xc[0]+1,len(signals[:,0])+xc[0]+1),t)
                        else:
                            t = interpolate.splrep(xc,signals[xc,0])
                            top = interpolate.splev(np.arange(len(signals[:,0])),t)

                    if len(yc) < 4:
                        b = interpolate.splrep(yc,signals[yc,0], k=1)
                        bot = interpolate.splev(np.arange(len(signals[:,0])),b)
                    else:
                        if inter == 'extrap':
                            y0 = y[:yc[0]+1][::-1]
                            y1 = y[yc[-1]:][::-1]
                            yn = np.hstack((y0,y,y1)).nonzero()[0]
                            sy = np.hstack((signals[:yc[0]+1,0][::-1],signals[:,0],signals[yc[-1]:,0][::-1]))
                            b = interpolate.splrep(yn,sy[yn])
                            bot = interpolate.splev(np.arange(yc[0]+1,len(signals[:,0])+yc[0]+1),b)
                        else:
                             b = interpolate.splrep(yc,signals[yc,0])
                             bot = interpolate.splev(np.arange(len(signals[:,0])),b)

    
#                    plt.plot(signals[: ,0])
#                    plt.plot(top,'bo')
#                    plt.plot(bot,'ro')
#                    plt.show()

                    mean = (top + bot)/2
                    signals[:,1] = signals[:,0] - mean
                
                    if k > 0:
                        sd = np.sum((np.abs(signals[:,0] - signals[:,1])**2))/(np.sum(signals[:,0]**2))
                
                    signals = signals[:,::-1]
                    k += 1

        if inter == 'Default' or inter == 'extrap':
            if finish:
                IMFs[:,j]= residual
                ncomp += 1
                break
            elif j == 0:
                IMFs[:,j] = signals[:,0]
                residual = data - IMFs[:,j]
                signals[:,0] = residual
                ncomp = 1 
            else:
                IMFs[:,j] = signals[:,0]
                residual = residual - IMFs[:,j]
                signals[:,0] = residual
                ncomp += 1
        elif inter == 'Mirror':
            if finish:
                IMFs[:,j]= residual
                ncomp += 1
                break
            elif j == 0:
                IMFs[:,j] = signals[data_length:data_length*2,0]
                residual = data - IMFs[:,j]
                signals[0:data_length,0] = residual[::-1]
                signals[data_length:data_length*2,0] = residual
                signals[data_length*2:data_length*3,0] = residual[::-1]
                ncomp = 1 
            else:
                IMFs[:,j] = signals[data_length:data_length*2,0]
                residual = residual - IMFs[:,j]
                signals[0:data_length,0] = residual[::-1]
                signals[data_length:data_length*2,0] = residual
                signals[data_length*2:data_length*3,0] = residual[::-1]
                ncomp += 1
        else:
            raise Exception("Bite me bitch")
            
    return IMFs[:,0:ncomp]

if __name__ == "__main__": 
    import matplotlib.ticker as tick
    
    # Signal creation
#    base = np.linspace(0,250,500)
#    a = 100 * np.sin(base) 
#    b = 50 * np.cos(base/5)
#    c = 100 * np.sin(base/10)
#    d = 50 * np.cos(base)
#    data = 2*base + a + b + c + d #+ 50*np.random.random(base)

    # Real data
    int_data = np.load('/home/stuart/Documents/Ellerman_data/IBIS/Int_red130.npy')
    area_data = np.load('/home/stuart/Documents/Ellerman_data/IBIS/Area_red130.npy')
#    int_data = np.load('/home/nabobalis/Dropbox/Int_red130.npy')
#    area_data = np.load('/home/nabobalis/Dropbox/Area_red130.npy')    
    ind = 3801
    fin = np.isfinite(int_data[ind,:])
    int_data = int_data[ind,:][fin]    
    area_data = area_data[ind,:][fin]
    time = (np.arange(0,len(int_data))*26.9)/60.
    
    # Calls our EMD!    
    imfs_area = EMD(area_data,inter = 'Mirror')
    imfs_int = EMD(int_data,inter = 'Mirror')
    print "-------------------------------------"
    print "Area:",imfs_area.shape
    print "Intensity:",imfs_int.shape     
    print "-------------------------------------"
    
    # Sexy image plotting
    plt.figure()
    ax1 = plt.subplot(111)
    ax1.yaxis.set_major_locator(tick.MaxNLocator(7,symmetric=True))
    ax1.plot(time,imfs_area[:,0],'x--',color='k',label="Area")
    ax1.plot(time,imfs_area[:,2:],'x--',color='k',label="Area")
    pltA, = ax1.plot(time,imfs_area[:,1],'o-',color='r',label="Area")
#    pltDA, = ax1.plot(time,area_data-imfs_area[:,-1],'o--',color='r')
#    ax2 = ax1.twinx()
#    ax2= plt.subplot(111)
#    ax2.yaxis.set_major_locator(tick.MaxNLocator(7,symmetric=True))
#    ax2.plot(time,imfs_int[:,0],'x--',color='k',label="Intensity")
#    ax2.plot(time,imfs_int[:,2:],'x--',color='k',label="Intensity")
#    pltI, = ax2.plot(time,imfs_int[:,1],'o-',color='b',label="Intensity")
#    pltDI, = ax2.plot(time,int_data-imfs_int[:,-1],'o--',color='b')
#    plt.title("Detrended Data EB $%i$"%ind)
    ax1.set_ylabel("Area (pixels)")
#    ax2.set_ylabel("Intensity (A.U.)")
    ax1.set_xlabel("Time [Minutes]")
#    leg = ax2.legend([pltDA,pltDI],["Area","Intensity"],loc='best', fancybox=True)
#    leg.get_frame().set_alpha(0.5)
#    plt.show()
    
    plt.figure()
    n= len(time)
    #Create blank array
    fft_a = np.zeros([n],dtype=np.complex64)
    result_a = np.fft.fft(imfs_area[:,1])
    xa = np.linspace(0, len(result_a)/2., n/2)
    xa /= n * 26.9
    xa = 1./xa
    xa /= 60.
    # Swap halfs
    fft_a[0:n/2] = result_a[n/2:n]
    fft_a[n/2:n] = result_a[0:n/2]
    fft_a = np.absolute(fft_a[n/2:])
    
    fft_i = np.zeros([n],dtype=np.complex64)
    result_i = np.fft.fft(imfs_int[:,1])
    xi = np.linspace(0, len(result_i)/2., n/2)
    xi /= n * 26.9
    xi = 1./xi
    xi /= 60.
    # Swap halfs
    fft_i[0:n/2] = result_i[n/2:n]
    fft_i[n/2:n] = result_i[0:n/2]
    fft_i = np.absolute(fft_i[n/2:])
    ax1 = plt.subplot(111)
#    plt.tight_layout(pad=5)
    pltA, = plt.plot(xa,fft_a**2,'ro-',label="Area")
    ax2 = ax1.twinx()
    pltI, = plt.plot(xi,fft_i**2,'bo-',label="Intensity")
    ax1.set_xlabel("Period [Minutes]")
    ax1.set_ylabel("Power [A.U.]")
    ax2.set_ylabel("Power [A.U.]")
    leg = ax2.legend([pltA,pltI],["Area","Intensity"],loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.show()