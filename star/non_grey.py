import numpy as np
import matplotlib.pyplot as plt
import matrix_operators as m
import opac as o
import astropy.io.fits as pyfits
import astropy.units as u
import astropy.constants as c
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import cumtrapz
plt.ion()

#log(g) in cgs units
logg = 4.3 #About a main sequence A0 star. 

#Temperature in K. NB: As we're not dealing with convection yet, we can't 
#go much less than 10,000K.
Teff=10000

#Number of iterations
Niter = 40

#Number of tau grid points
Ntau = 50

#Number of frequency grid points
Nnu = 500
nu_max = 3e8/30e-9 #30nm as a frequency
#-------------------------------
#Our target (Eddington) flux
Htarg = (c.sigma_sb*(Teff*u.K)**4/4/np.pi).cgs.value

#Gravity in physical units
grav = (10**logg) #* u.cm/u.s**2

#Make a tau grid that is finer towards the surface.
tau_ross_grid = 10*np.linspace(0,1,Ntau)**5

#Create an initial T(tau)
Ttau0 = Teff*(3/4*(tau_ross_grid + 2/3))**(1/4)
Ttau = Ttau0.copy()

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

#Find the 2D arrays we need for the final Unsold-Lucy computation
kappa_bar_nu = np.empty((Nnu,Ntau))
tau_nu = np.empty((Nnu,Ntau))
Bnu = np.empty((Nnu,Ntau))
Jnu = np.empty((Nnu,Ntau))
Hnu = np.empty((Nnu,Ntau))
Knu = np.empty((Nnu,Ntau))

def dPdtau(tau, P):
	"""
	Find the hydostatic equilibrium derivative for solve_ivp
	"""
	T = np.interp(tau, tau_ross_grid, Ttau)
	return grav/kappa_bar_Ross([np.log10(P[0]), T])

h_c2 = (c.h/c.c**2).cgs.value
h_kB = (c.h/c.k_B).cgs.value
def calc_Bnu(T, nu):
	"""
	Find the Planck function Bnu in cgs units
	"""
	return 2*h_c2*nu**3 / (np.exp(h_kB*nu/T) - 1)
	
plt.clf()
#Now the iterative loop...
DeltaT = np.ones(Ntau)
accelerated = True
for i in range(Niter):
	#Solve equation of hydrostatic equilibrium, giving us our thermodynamic constants
	#dP/dtau = g/kappaR_bar
	res = solve_ivp(dPdtau, [0,np.max(tau_ross_grid)], [10**Plog10_grid[10]], t_eval = tau_ross_grid)
	Ptau = res.y[0]
	
	#Create 1-dimensional functions of key variables. We end each of these in 
	#"tau" to indicate they are a function of Rosseland optical depth tau
	logP_T_tau = np.array([np.log10(Ptau), Ttau]).T
	nHItau = nHI(logP_T_tau)
	nHIItau = nHII(logP_T_tau)
	nHmtau = nHm(logP_T_tau)
	n_etau = n_e(logP_T_tau)
	rhotau = 10**rho_log10(logP_T_tau)
	kappa_bartau = kappa_bar_Ross(logP_T_tau)
	
	#Find tau_nu(tau) so we can get T(tau) and the source function
	for i in range(Ntau):
		#kappa is the cross-section per unit volume, or 1/mean_free_path.
		#We get this by number_density * cross_section_per_atom
		kappa = nHItau[i] * o.Hbf(nu,Ttau[i])  + nHIItau[i] * n_etau[i] * o.Hff(nu,Ttau[i]) + \
						nHmtau[i] * o.Hmbf(nu,Ttau[i]) + nHItau[i]	* n_etau[i] * o.Hmff(nu,Ttau[i])
		#Now divide by density to get mass-weighted opacity (cm^2/g)
		kappa_bar_nu[:,i] = kappa/rhotau[i]

	#Loop over kappa to get tau_nu(tau_ross)
	for j in range(Nnu):
		#FIXME: The following line needs (kappa_bar_nu + sigma_bar_nu), once
		#sigma is added in!
		chi_nu_rat = kappa_bar_nu[j]/kappa_bartau
		
		#Lets keep this simple and just do a trapezoidal integration
		tau_nu[j] = np.concatenate(([0],cumtrapz(chi_nu_rat, tau_ross_grid)))
		
		#FIXME: This is just an LTE (no scattering) calculation
		Bnu[j] = calc_Bnu(Ttau, nu[j])
		S = Bnu[j]
		
		#Now calculate J, H and K!
		Lambda = m.lambda_matrix(tau_nu[j])
		Jnu[j] = np.dot(Lambda, S)
		Hnu[j] = np.dot(m.phi_matrix(tau_nu[j]),S)
		Knu[j] = np.dot(m.X_matrix(tau_nu[j]),S)
		
	#Make the wavelength integrals as sums (as we have an evenly spaced)
	#frequency distribution
	B = np.sum(Bnu, axis=0)*dnu
	J = np.sum(Jnu, axis=0)*dnu
	H = np.sum(Hnu, axis=0)*dnu
	K = np.sum(Knu, axis=0)*dnu
	f = K/J
	g0 = H[0]/J[0]
	kappa_J = np.sum(Jnu*kappa_bar_nu, axis=0)*dnu/J
	kappa_B = np.sum(Bnu*kappa_bar_nu, axis=0)*dnu/B
	
	#FIXME: this next line should include sigma
	chi_H = np.sum(Hnu*kappa_bar_nu, axis=0)*dnu/H
	
	DeltaH = Htarg - H
	#Do the Unsold-Lucy iteration! Use a simple trapezoidal integration 
	#for the integral
	integral = np.concatenate(([0],cumtrapz(chi_H/kappa_B*DeltaH, tau_ross_grid)))
	RHS = kappa_J/kappa_B*J - B + kappa_J/kappa_B/f *(integral + f[0]*DeltaH[0]/g0)
	last_DeltaT = DeltaT.copy()
	DeltaT = RHS*np.pi/4/(c.sigma_sb*(Ttau*u.K)**3).cgs.value
	r = DeltaT/last_DeltaT
	if (np.max(r) < 1.2) and not accelerated:
		accelerated = True
		print("Accelerating the increment!!!")
		DeltaT *= 1/(1-np.minimum(r,0.9))
	else:
		accelerated = False
	Ttau += DeltaT
	print("This DeltaT: ")
	print(DeltaT)
	plt.plot(tau_ross_grid, Ttau)
	plt.xlabel('Rosseland tau')
	plt.ylabel('Temperature (K)')
	plt.pause(.01)