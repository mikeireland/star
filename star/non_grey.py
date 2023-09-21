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
Teff=9000

#Number of iterations
Niter = 1

#Number of tau grid points
Ntau = 21

#Number of frequency grid points
Nnu = 500
nu_max = 3e8/30e-9 #30nm as a frequency
#-------------------------------
#Gravity in physical units
grav = (10**logg) * u.cm/u.s**2

#Make a tau grid that is finer towards the surface.
tau_ross_grid = 10*np.linspace(0,1,Ntau)**2

#Create an initial T(tau)
Ttau = Teff*(3/4*(tau_ross_grid + 2/3))**(1/4)

#Some frequencies
dnu = nu_max/Nnu
nu = dnu*np.arange(Nnu) + dnu/2

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
rho_tab = eos_fits[0].data
nHI_tab = eos_fits['ns'].data[:,:,0]
nHII_tab = eos_fits['ns'].data[:,:,1]
nHm_tab = eos_fits['ns'].data[:,:,2]
ne_tab = eos_fits['n_e'].data
kappa_tab = pyfits.getdata('Ross_Planck_opac.fits')
Ts_grid = h['CRVAL1'] + np.arange(h['NAXIS1'])*h['CDELT1']
Plog10_grid = h['CRVAL2'] + np.arange(h['NAXIS2'])*h['CDELT2']

#Density should have been log(10), because log(rho) is a roughly linear function 
#of log(P) but not of P.
rho_log10 = RegularGridInterpolator((Plog10_grid, Ts_grid), np.log10(rho_tab))
nHI = RegularGridInterpolator((Plog10_grid, Ts_grid), nHI_tab)
nHII = RegularGridInterpolator((Plog10_grid, Ts_grid), nHII_tab)
nHm = RegularGridInterpolator((Plog10_grid, Ts_grid), nHm_tab)
n_e = RegularGridInterpolator((Plog10_grid, Ts_grid), ne_tab)
kappa_bar_Ross = RegularGridInterpolator((Plog10_grid, Ts_grid), kappa_tab)
kappa_bar_nu = np.empty((Nnu,Ntau))

def dPdtau(tau, P):
	"""
	Find the hydostatic equilibrium derivative for solve_ivp
	"""
	T = np.interp(tau, tau_ross_grid, Ttau)
	return grav/kappa_bar_Ross([np.log10(P[0]), T])
	
#Now the iterative loop...
for i in range(Niter):
	#Solve equation of hydrostatic equilibrium, giving us our thermodynamic constants
	#dP/dtau = g/kappaR_bar
	res = solve_ivp(dPdtau, [0,np.max(tau_ross_grid)], [10**Plog10_grid[8]], t_eval = tau_ross_grid)
	Ptau = res.y[0]
	logP_T_tau = np.array([np.log10(Ptau), Ttau]).T
	nHItau = nHI(logP_T_tau)
	nHIItau = nHII(logP_T_tau)
	nHmtau = nHm(logP_T_tau)
	n_etau = n_e(logP_T_tau)
	rhotau = 10**rho_log10(logP_T_tau)
	
	#Find tau_nu(tau) so we can get T(tau) and the source function
	for i in range(Ntau):
		#kappa is the cross-section per unit volume, or 1/mean_free_path.
		#We get this by number_density * cross_section_per_atom
		kappa = nHItau[i] * o.Hbf(nu,Ttau[i])  + nHIItau[i] * n_etau[i] * o.Hff(nu,Ttau[i]) + \
					    nHmtau[i] * o.Hmbf(nu,Ttau[i]) + nHItau[i]  * n_etau[i] * o.Hmff(nu,Ttau[i])
		#Now divide by density to get mass-weighted opacity (cm^2/g)
		kappa_bar_nu[:,i] = kappa/rhotau[i]

	#Loop over kappa to get tau_nu(tau_ross)	

	#Solve for S, J, B, H, f and g for all wavelengths!
	#Initially, do LTE, where we ignore sigma, and "solve" for S as S=B
	
	#Make the integrals.
	
	#Do the Unsold-Lucy iteration!

	