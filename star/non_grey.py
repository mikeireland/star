import numpy as np
import matplotlib.pyplot as plt
import matrix_operators as m
import opac as o
import astropy.io.fits as pyfits
import astropy.units as u
import astropy.constants as c
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

#log(g) in cgs units
logg = 3.0

#Temperature in K. NB: As we're not dealing with convection yet, we can't 
#go much less than 10,000K.
Teff=10000

#Number of iterations
Niter = 1

#Number of tau grid points
Ntau = 21

#Number of frequency grid points
Nnu = 1000
#-------------------------------
#Gravity in physical units
grav = (10**logg) * u.cm/u.s**2

#Make a tau grid that is finer towards the surface.
tau_ross_grid = 10*np.linspace(0,1,Ntau)**2

#Create an initial T(tau)
Ttau = Teff*(3/4*(tau_ross_grid + 2/3))**(1/4)

#Some variables we'll fill in each layer.
kappaR_bar_tau = np.empty(Ntau)
nHI_tau = np.empty(Ntau)
nHII_tau = np.empty(Ntau)
nHm_tau = np.empty(Ntau)
ne_tau = np.empty(Ntau)

#Preliminaries - we need to be able to interpolate in our tables for 
#log(P) and T.
eos_fits = pyfits.open('rho_Ui_mu_ns_ne.fits')
h = eos_fits[0].header
nHI_tab = eos_fits['ns'].data[:,:,0]
nHII_tab = eos_fits['ns'].data[:,:,1]
nHm_tab = eos_fits['ns'].data[:,:,2]
ne_tab = eos_fits['n_e'].data
kappa_tab = pyfits.getdata('Ross_Planck_opac.fits')
Ts_grid = h['CRVAL1'] + np.arange(h['NAXIS1'])*h['CDELT1']
Plog10_grid = h['CRVAL2'] + np.arange(h['NAXIS2'])*h['CDELT2']

nHI = RegularGridInterpolator((Plog10_grid, Ts_grid), nHI_tab)
nHII = RegularGridInterpolator((Plog10_grid, Ts_grid), nHII_tab)
nHm = RegularGridInterpolator((Plog10_grid, Ts_grid), nHm_tab)
n_e = RegularGridInterpolator((Plog10_grid, Ts_grid), ne_tab)
kappa_Ross = RegularGridInterpolator((Plog10_grid, Ts_grid), kappa_tab)

def dPdtau(tau, P):
	"""
	Find the hydostatic equilibrium derivative for solve_ivp
	"""
	T = np.interp(tau, tau_ross_grid, Ttau)
	return grav/kappa_Ross([np.log10(P[0]), T])
	

#Now the iterative loop...
for i in range(Niter):
	#Solve equation of hydrostatic equilibrium, giving us our thermodynamic constants
	#dP/dtau = g/kappaR_bar
	res = solve_ivp(dPdtau, [0,np.max(tau_ross_grid)], [10**Plog10_grid[0]], t_eval = tau_ross_grid)
	Ptau = res.y[0]
	
	#Find tau_nu(tau) so we can get T(tau) and the source function
	
	#Solve for S, J, B, H, f and g for all wavelengths!
	
	#Make the integrals.
	
	#Do the Unsold-Lucy iteration!

	