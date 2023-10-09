import astropy.units as u
import astropy.constants as c
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
plt.ion()

"""
This is based on:
https://iopscience.iop.org/article/10.3847/1538-4357/ac9b40/pdf

NB there are other resources for non-stellar atmosphere opacities, 
e.g.
https://chiantipy.readthedocs.io/en/latest/
https://chianti-atomic.github.io/api/ChiantiPy.core.html#id91
http://spiff.rit.edu/classes/phys370/lectures/statstar/statstar_python3.py

"""

#From OCR online, from https://articles.adsabs.harvard.edu/pdf/1988A%26A...193..189J
#A to F in columns, n=2 to 6 in rows
Hmff_table = np.array(
[[2483.3460 ,          285.8270   ,      -2054.2910    ,   2827.7760     ,   -1341.5370     ,       208.9520],
[-3449.8890 ,         -1158.3820  ,      8746.5230     ,   -11485.6320   ,    5303.6090     ,      -812.9390],
[2200.0400  ,          2427.7190  ,       -13651.1050  ,    16755.5240   ,     -7510.4940   ,      1132.7380],
[-696.2710  ,       -1841.4000    ,       8624.9700    ,    -10051.5300  ,     4400.0670    ,      -655.0200],
[88.2830    ,         444.5170    ,      -1863.8640    ,     2095.2880   ,    -901.7880     ,       132.9850]])

Hff_const = np.sqrt(32*np.pi)/3/np.sqrt(3)*(c.e.esu**6/c.c/c.h/(c.k_B*c.m_e**3*u.K)**(1/2)/(1*u.Hz)**3).to(u.cm**5).value
h_kB_cgs = (c.h/c.k_B).cgs.value
H_excitation_T = (13.595*u.eV/c.k_B).cgs.value

def Hmbf(nu, T):
	"""Compute the Hydrogen minus bound-free cross sections in cgs units as a
	function of temperature in K. Computed per atom. Compute using:
	https://ui.adsabs.harvard.edu/abs/1988A%26A...193..189J/abstract
	
	Parameters
	----------
	nu: Frequency or a list (numpy array) of frequencies.
	"""
	Cn = [152.519, 49.534, -118.858, 92.536,-34.194,4.982]
	alpha = np.zeros_like(nu)
	wave_um = c.c.si.value/nu * 1e6
	for n in range(1,7):
		alpha += Cn[n-1] * np.abs(1/wave_um - 1/1.6419)**((n-1)/2)
	alpha *= (wave_um<=1.6419) * 1e-18 * wave_um**3 * np.abs(1/wave_um - 1/1.6419)**(3/2)
	#Return the cross-section, corrected for stimulated emission so it can be directly
	#used as an opacity.
	return alpha * (1-np.exp(-h_kB_cgs*nu/T))
	

def Hmff(nu, T):
	"""Compute the Hydrogen minus bound-free cross sections in cgs units as a
	function of temperature in K. Computed per H atom per unit (cgs) electron
	density. Compute using:
	https://ui.adsabs.harvard.edu/abs/1988A%26A...193..189J/abstract
	
	Parameters
	----------
	nu: Frequency or a list (numpy array) of frequencies.
	"""
	alpha = np.zeros_like(nu)
	wave_um = np.maximum(c.c.si.value/nu * 1e6,0.3645)
	for n in range(2,7):
		row = n-2
		coeff = 1e-29 * (5040/T)**((n+1)/2)
		for i, exponent in enumerate([2,0,-1,-2,-3,-4]):
			alpha += coeff*wave_um**exponent * Hmff_table[row,i]
	#alpha is now in units of cross section per unit electron pressure
	#We want to multiply by the ratio of electron pressure to electron 
	#density, which is just k_B T
	return alpha * c.k_B.cgs.value * T 
	
def Hbf(nu, T):
	"""Compute the Hydrogen bound-free cross sections in cgs units as a
	function of temperature in K. Computed per atom. Computed using:
	https://articles.adsabs.harvard.edu/pdf/1970SAOSR.309.....K
	
	Parameters
	----------
	nu: Frequency or a list (numpy array) of frequencies.
	"""
	alpha = np.zeros_like(nu)
	ABC = np.array([[.9916,2.719e13,-2.268e30],
		[1.105,-2.375e14,4.077e28],
		[1.101,-9.863e13,1.035e28],
		[1.101,-5.765e13,4.593e27],
		[1.102,-3.909e13,2.371e27],
		[1.0986,-2.704e13,1.229e27]])
	for n in range(1,7):
		Boltzmann_fact = n**2*np.exp(-H_excitation_T*(1-1/n**2)/T)
		alpha += 2.815e29/n**5/nu**3*(ABC[n-1,0] + (ABC[n-1,1] + ABC[n-1,2]/nu)/nu) * (nu>3.28805e15/n**2) * Boltzmann_fact
	#FIXME : add higher values of n
	#FIXME : Add in the partition function U, which is implicitly taken to be 2.0 above.
	return alpha * (1-np.exp(-h_kB_cgs*nu/T))
	

def Hff(nu, T):
	"""Compute the Hydrogen free-free cross sections in cgs units as a
	function of temperature in K. Computed per atom per unit (cgs) electron
	density
	
	Parameters
	----------
	nu: Frequency or a list (numpy array) of frequencies.
	"""
	#Approximate a Gaunt factor of 1.0!
	#FIXME : Remove the approximation
	return Hff_const /nu**3/np.sqrt(T)
	
if __name__=="__main__":
	#Lets compute a rosseland mean opacity!
	if True:
		#Create a grid of frequencies from 30 nm to 30 microns.
		dnu = 1e13
		plt.clf()
		nu = dnu*np.arange(1000) + dnu/2
		f = pyfits.open('rho_Ui_mu_ns_ne.fits')
		h = f[0].header
		natoms = f['ns'].data.shape[2]//3
		Ts = h['CRVAL1'] + np.arange(h['NAXIS1'])*h['CDELT1']
		Ps_log10 = h['CRVAL2'] + np.arange(h['NAXIS2'])*h['CDELT2']
		kappa_bar_Planck = np.zeros_like(f[0].data)
		kappa_bar_Ross = np.zeros_like(f[0].data)
		for i, P_log10 in enumerate(Ps_log10):
			for j, T in enumerate(Ts):
				nHI = f['ns'].data[i,j,0]
				nHII = f['ns'].data[i,j,1]
				nHm = f['ns'].data[i,j,2]
				ne = f['n_e'].data[i,j]
				#Compute the volume-weighted absorption coefficient
				kappa = nHI * Hbf(nu,T)  + nHII * ne * Hff(nu,T) + \
					    nHm * Hmbf(nu,T) + nHI  * ne * Hmff(nu,T)
				#Now compute the Rosseland and Planck means.
				Bnu = nu**3/(np.exp(h_kB_cgs*nu/T)-1)
				dBnu = nu**4 * np.exp(h_kB_cgs*nu/T)/(np.exp(h_kB_cgs*nu/T)-1)**2
				kappa_bar_Planck[i,j] = np.sum(kappa*Bnu)/np.sum(Bnu)/f[0].data[i,j]
				kappa_bar_Ross[i,j] = 1/(np.sum(dBnu/kappa)/np.sum(dBnu))/f[0].data[i,j]
#				if (i==30): #This is log_10(P)=3.5 - similar to solar photosphere.
#					plt.loglog(3e8/nu, kappa/f[0].data[i,j])
#					plt.pause(.5)
#					print(T)
#					if (j==20):
#						import pdb; pdb.set_trace()
		hdu1 = pyfits.PrimaryHDU(kappa_bar_Ross)
		hdu1.header['CRVAL1'] = Ts[0]
		hdu1.header['CDELT1'] = Ts[1]-Ts[0]
		hdu1.header['CTYPE1'] = 'Temperature [K]'
		hdu1.header['CRVAL2'] = Ps_log10[0]
		hdu1.header['CDELT2'] = Ps_log10[1]-Ps_log10[0]
		hdu1.header['CTYPE2'] = 'log10(pressure) [dyne/cm^2]'
		hdu1.header['EXTNAME'] = 'kappa_Ross [cm**2/g]'
		hdu2 = pyfits.ImageHDU(kappa_bar_Planck)
		hdu2.header['EXTNAME'] = 'kappa_Planck [cm**2/g]'
		hdulist = pyfits.HDUList([hdu1, hdu2])
		hdulist.writeto('Ross_Planck_opac.fits', overwrite=True)