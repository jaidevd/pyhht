import numpy as np
import matplotlib.pyplot as plt
from flandrin_emd import extr

def boundary_conditions(x, t=None, z=None, NBSYM=2):
    
    """ Generates mirrored extrema beyond the singal limits. """

    if not t:
        t = np.arange(len(x))
    if not z:
        z = x
    
    indmin, indmax = extr(x)[:2]
    
    lmin = indmin[:NBSYM]
    lmax = indmax[:NBSYM]
    rmin = indmin[len(indmin)-NBSYM:]
    rmax = indmax[len(indmax)-NBSYM:]
    
    lmin_extended = -1*lmin[::-1]
    lmax_extended = -1*lmax[::-1]
    rmin_extended = (len(x)-rmin)[::-1] - 1 + len(x)
    rmax_extended = (len(x)-rmax)[::-1] - 1 + len(x)
    
    tmin = np.concatenate((lmin_extended,indmin,rmin_extended))
    tmax = np.concatenate((lmax_extended,indmax,rmax_extended))
    
    zmin = x[indmin]
    zmax = x[indmax]
    
    zmin_left = x[lmin][::-1]
    zmax_left = x[lmax][::-1]
    zmin_right = x[rmin][::-1]
    zmax_right = x[rmax][::-1]
    
    zmin = np.concatenate((zmin_left, zmin, zmin_right))
    zmax = np.concatenate((zmax_left, zmax, zmax_right))
    
    return tmin, tmax, zmin, zmax

if __name__ == "__main__":
    x = np.random.randn(100)
    indmin, indmax = extr(x)[:2]
    plt.plot(x)
    tmin, tmax, zmin, zmax = boundary_conditions(x)
    plt.plot(tmin, zmin, 'r.')
    plt.plot(tmax, zmax, 'g.')
    plt.show()