import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expn
import astropy.units as u
import astropy.constants as c
from scipy.integrate import cumtrapz

def lambda_matrix(tau_grid):
	"""
	Compute the Lambda operator as a matrix based on a tau grid.
	
	Parameters
	----------
	tau_grid: (n_tau) numpy.array
		A grid of optical depth points, starting at 0 and monotonically increasing.
	
	Returns
	-------
	lambda_mat: (n_tau) numpy.array
		A matrix defined so that to obtain vector of mean intensities J from a vector 
		of source functions S we simply compute: J=numpy.dot(lambda_mat, S)
	"""
	#Fill our final result matrix with zeros
	lambda_mat = np.zeros( (len(tau_grid), len(tau_grid)) )
	
	#Create a delta-tau grid
	delta_tau = tau_grid[1:] - tau_grid[:-1]
	
	#For simplicity and readability, just go through one layer at a time.
	for j in range(len(tau_grid)):
		#Create E2 and E3 vectors
		E2_vect = expn(2,np.abs(tau_grid - tau_grid[j]))
		E3_vect = expn(3,np.abs(tau_grid - tau_grid[j]))
		
		#Add the contribution from the i'th layer, for upwards going rays
		lambda_mat[j,j:-1] +=  E2_vect[j:-1] - (E3_vect[j:-1] - E3_vect[j+1:])/delta_tau[j:] 
		
		#Add the contribution from the i+1'th layer, for upwards going rays
		lambda_mat[j,j+1:]  += -E2_vect[j+1:]  + (E3_vect[j:-1] - E3_vect[j+1:])/delta_tau[j:]
		
		#Add the contribution from the i'th layer, for downwards going rays
		lambda_mat[j,1:j+1] +=  E2_vect[1:j+1] - (E3_vect[1:j+1] - E3_vect[:j])/delta_tau[:j] 
		
		#Add the contribution from the i-1'th layer, for downwards going rays
		lambda_mat[j,:j]  += -E2_vect[:j]  + (E3_vect[1:j+1] - E3_vect[:j])/delta_tau[:j] 
		
		#Add the contribution from the lower boundary condition
		lambda_mat[j,-1] +=  E3_vect[-1]/delta_tau[-1] + E2_vect[-1]
		lambda_mat[j,-2] += -E3_vect[-1]/delta_tau[-1]
	return 0.5*lambda_mat
	
def X_matrix(tau_grid):
	"""
	Compute the X operator as a matrix based on a tau grid.
	This is cut and paste from the Lambda matrix! 
	Just E2->E4 and E3->E5
	
	Parameters
	----------
	tau_grid: (n_tau) numpy.array
		A grid of optical depth points, starting at 0 and monotonically increasing.
	
	Returns
	-------
	X_mat: (n_tau) numpy.array
		A matrix defined so that to obtain vector of K from a vector 
		of source functions S we simply compute: K=numpy.dot(X_mat, S)
	"""
	#Fill our final result matrix with zeros
	X_mat = np.zeros( (len(tau_grid), len(tau_grid)) )
	
	#Create a delta-tau grid
	delta_tau = tau_grid[1:] - tau_grid[:-1]
	
	#For simplicity and readability, just go through one layer at a time.
	for j in range(len(tau_grid)):
		#Create E4 and E5 vectors
		E4_vect = expn(4,np.abs(tau_grid - tau_grid[j]))
		E5_vect = expn(5,np.abs(tau_grid - tau_grid[j]))
		
		#Add the contribution from the i'th layer, for upwards going rays
		X_mat[j,j:-1] +=  E4_vect[j:-1] - (E5_vect[j:-1] - E5_vect[j+1:])/delta_tau[j:] 
		
		#Add the contribution from the i+1'th layer, for upwards going rays
		X_mat[j,j+1:]  += -E4_vect[j+1:]  + (E5_vect[j:-1] - E5_vect[j+1:])/delta_tau[j:]
		
		#Add the contribution from the i'th layer, for downwards going rays
		X_mat[j,1:j+1] +=  E4_vect[1:j+1] - (E5_vect[1:j+1] - E5_vect[:j])/delta_tau[:j] 
		
		#Add the contribution from the i-1'th layer, for downwards going rays
		X_mat[j,:j]  += -E4_vect[:j]  + (E5_vect[1:j+1] - E5_vect[:j])/delta_tau[:j] 
		
		#Add the contribution from the lower boundary condition
		X_mat[j,-1] +=  E5_vect[-1]/delta_tau[-1] + E4_vect[-1]
		X_mat[j,-2] += -E5_vect[-1]/delta_tau[-1]
	return 0.5*X_mat
	
def phi_matrix(tau_grid):
	"""
	Compute the phi operator as a matrix based on a tau grid.
	
	This differs from the Lambda matrix because of a sign difference.
	
	Parameters
	----------
	tau_grid: (n_tau) numpy.array
		A grid of optical depth points, starting at 0 and monotonically increasing.
	
	Returns
	-------
	phi_mat: (n_tau) numpy.array
		A matrix defined so that to obtain vector of Eddington fluxes H from a vector 
		of source functions S we simply compute: H=numpy.dot(phi_mat, S)
	"""
	#Fill our final result matrix with zeros
	phi_mat = np.zeros( (len(tau_grid), len(tau_grid)) )
	
	#Create a delta-tau grid
	delta_tau = tau_grid[1:] - tau_grid[:-1]
	
	#For simplicity and readability, just go through one layer at a time.
	for j in range(len(tau_grid)):
		#Create E3 and E4 vectors
		E3_vect = expn(3,np.abs(tau_grid - tau_grid[j]))
		E4_vect = expn(4,np.abs(tau_grid - tau_grid[j]))
		
		#Add the contribution from the i'th layer, for upwards going rays
		phi_mat[j,j:-1] +=  E3_vect[j:-1] - (E4_vect[j:-1] - E4_vect[j+1:])/delta_tau[j:] 
		
		#Add the contribution from the i+1'th layer, for upwards going rays
		phi_mat[j,j+1:]  += -E3_vect[j+1:]  + (E4_vect[j:-1] - E4_vect[j+1:])/delta_tau[j:]
		
		#Add the contribution from the i'th layer, for downwards going rays
		phi_mat[j,1:j+1] -=  E3_vect[1:j+1] - (E4_vect[1:j+1] - E4_vect[:j])/delta_tau[:j] 
		
		#Add the contribution from the i-1'th layer, for downwards going rays
		phi_mat[j,:j]  -= -E3_vect[:j]  + (E4_vect[1:j+1] - E4_vect[:j])/delta_tau[:j] 
		
		#Add the contribution from the lower boundary condition
		phi_mat[j,-1] +=  E4_vect[-1]/delta_tau[-1] + E3_vect[-1]
		phi_mat[j,-2] += -E4_vect[-1]/delta_tau[-1]
	return 0.5*phi_mat

def calc_Bnu(T, nu):
	return ( 2*c.h*nu**3/c.c**2 / (np.exp(c.h*nu/c.k_B/T) - 1) ).to(u.erg/u.cm**2)
	
if __name__=="__main__":
	#Create a couple of tau grids. The second is random.
	tau_grids = [np.array([0,1,2]), \
		np.concatenate(([0],np.cumsum(np.random.random(5))))]

	#Create the Lambda matrices, and compute J, H and K
	for tau_grid in tau_grids:
		ntau = len(tau_grid)
		lambda_mat = lambda_matrix(tau_grid)
		phi_mat = phi_matrix(tau_grid)
		X_mat = X_matrix(tau_grid)
		
		Jconst = np.dot(lambda_mat, np.ones(ntau))
		Jlin = np.dot(lambda_mat, 1 + tau_grid)
		print("\nJ for constant S:")
		print(Jconst)
		print("J for S = 1 + tau:")
		print(Jlin)

		Hconst = np.dot(phi_mat, np.ones(ntau))
		Hlin = np.dot(phi_mat, 1 + tau_grid)
		print("H for constant S:")
		print(Jconst)
		print("H for S = 1 + tau:")
		print(Hlin)
		
		Kconst = np.dot(X_mat, np.ones(ntau))
		Klin = np.dot(X_mat, 1 + tau_grid)
		print("K for constant S:")
		print(Kconst)
		print("K for S = 1 + tau:")
		print(Klin)
		
	print("\nEddington-Barbier for J: {:.4f}".format((1 + 1/2)/2))
	print("Eddington-Barbier for H: {:.4f}".format((1 + 2/3)/4))

	
		