#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:49:41 2021

@author: mireland
"""
#Additional Imports
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import numpy as np
import astropy.constants as c
import astropy.units as u
from scipy.interpolate import RectBivariateSpline
import astropy.io.fits as pyfits

#Global variable inputs from adiabat.fits
ff = pyfits.open('adiabats.fits')
hh = ff[0].header
#density logarithm
rhol = hh['CRVAL1'] + hh['CDELT1']*np.arange(hh['NAXIS1'])
#entropy
ent = hh['CRVAL2'] + hh['CDELT2']*np.arange(hh['NAXIS2'])

#Load in the tables
T_tab = ff[0].data
P_tab = ff[1].data
Gamma1_tab = ff[2].data

#Turn these into cubic interpolation functions.
T_func = RectBivariateSpline(ent, rhol, np.log10(T_tab))
P_func = RectBivariateSpline(ent, rhol, np.log10(P_tab))
Gamma1_func = RectBivariateSpline(ent, rhol, Gamma1_tab)

#Define key functions
def find_derivatives(r_in_rsun, M_rho, entropy, rho_c):
    """Given an interior mass M, a density rho, find the derivatives of M and rho for a constant
    entropy.
    
    Derivatives are in units of solar radii. 
    
    Parameters
    ----------
    r_in_rsun: radius to compute derivatives in solar units
    M_rho: numpy array-like, including M in solar units, 
         logarithm base 10 of density in cgs units.
    entropy: the entropy.
        
    Returns
    -------
    derivatives: Derivatives of M in solar units, density normalised by central density, 
    r_in_rsun, as a numpy array-like variable.
    """
    M_in_Msun, rho = M_rho
    rho *= rho_c #Convert to cgs units
    rhol = np.log10(rho.to(u.g/u.cm**3).value)
    
    #Mass continuity
    dM_in_Msundr = 4*np.pi*r_in_rsun**2*float(rho*c.R_sun**3/c.M_sun)
    
    #Hydrostatic Equilibrium
    if r_in_rsun==0:
        drhodr=0
    else:
        #Find the density derivative.
        drhodr = -c.G*M_in_Msun*u.M_sun*rho**2/(r_in_rsun*u.R_sun)**2
        drhodr /= 10**(P_func(entropy, rhol)[0,0]) *u.dyn/u.cm**2 * Gamma1_func(entropy, rhol)[0,0]
        #Convert to normalised units (central density per unit solar radius)
        drhodr = float(drhodr*u.R_sun/rho_c)
        
    #print("{:.3f} {:.3f} {:.3f} {:.3e} {:.4f} {:.3f}".format(r_in_rsun, M_in_Msun, rhol, T_func(entropy, rhol)[0,0], dM_in_Msundr, drhodr))
    #import pdb; pdb.set_trace()

    return np.array([dM_in_Msundr, drhodr])
    
def near_vacuum(r_in_rsun, M_rho,  entropy, rho_c):
    """Determine a surface condition by the surface pressure becoming low.
    
    We'll use 1e-8 of central density as a surface condition.
    """
    return M_rho[1] -1e-8

near_vacuum.terminal = True
near_vacuum.direction = -1

def convective_star(rho_c, T_c):
    """Assuming a fully convective star, compute the structure using an equation of
    state and the first two equations of stellar structure.
    
    Parameters
    ----------
    rho_c: Central density, including units from astropy.units
    T_c: Central temperature, including units from astropy.units
    """
    #Compute the star central entropy 
    Tdiff = lambda ent, Ttarg, rhol : T_func(ent, rhol)[0,0] - Ttarg
    result = root_scalar(Tdiff, bracket=[12,22], 
                args=(np.log10(T_c.to(u.K).value),np.log10(rho_c.to(u.g/u.cm**3).value),), xtol=1e-6)
    if result.converged:
        entropy = result.root
        print("Modelling a star with entropy: {:.1f} k_B/baryon".format(entropy))
    else:
        raise(UserWarning("Could not find central entropy."))
    
    #Start the problem at the star center.
    y0 = [0, 1] 
    
    #Don't go past 100 R_sun!
    rspan = [0,100] 
    
    #Solve the initial value problem!
    result = solve_ivp(find_derivatives, rspan, y0, events=[near_vacuum], rtol=1e-4, \
                       args=(entropy, rho_c), method='RK23')
    
    #Extract the results
    r_in_Rsun = result.t
    M_in_Msun = result.y[0]
    rho = result.y[1]*rho_c
    rhol = np.log10(rho.to(u.g/u.cm**3).value)
    P = 10**(P_func(entropy, rhol, grid=False))*u.dyn/u.cm**2
    T = 10**(T_func(entropy, rhol, grid=False))*u.K
    return r_in_Rsun, M_in_Msun, P, rho, T
    
def mass_difference(rho_c_cgs, T_c, M_in_Msun_target):
    """Target function to solve for the central density.
    """
    r_in_Rsun, M_in_Msun, P, rho, T = convective_star(rho_c_cgs*u.g/u.cm**3, T_c)
    return M_in_Msun[-1] - M_in_Msun_target
    
def convective_star_fixed_mass(T_c, M_in_Msun_target):
    """Given a central temperature and a total mass, find P, T and rho
    as a function of R for a star
    
    Parameters
    ----------
    T_c: Temperature, including astropy units.
    M_in_Msun_target: (M/M_sun) target mass.
    """
    result = root_scalar(mass_difference, bracket=[1,300], args=(T_c, M_in_Msun_target,), xtol=1e-6)
    r_in_Rsun, M_in_Msun, P, rho, T = convective_star(result.root*u.g/u.cm**3, T_c)
    return r_in_Rsun, M_in_Msun, P, rho, T

if __name__=="__main__":
    #The example from week in 2019
    r_in_Rsun, M_in_Msun, P, rho, T = convective_star(1*u.g/u.cm**3, 3*u.MK)
    print("2019 example stellar radius: {:.3f}".format(r_in_Rsun[-1]))
    print("2019 example mass: {:.3f}".format(M_in_Msun[-1]))
    
    plt.figure(1)
    plt.clf()
    plt.loglog(rho, T)
    plt.xlabel('Density (g/cm^3)')
    plt.ylabel('Temperature (K)')
    #The 0.28 M_sun example from 2021
    r_in_Rsun, M_in_Msun, P, rho, T = convective_star_fixed_mass(8e6*u.K, 0.28)
    print("Radius of star solved for with 0.28 M_sun: {:.3f}".format(r_in_Rsun[-1]))
    plt.plot(rho, T)
    