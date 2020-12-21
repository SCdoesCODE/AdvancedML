"""

Finding the posterior distribution is often not computationally feasible. EM is one solution to this, but not for most
models. To calculate the posterior distribution for models with a small dimension of 2 we can simply use Bayes rule
(by using our prior belief of the distribution), but as the dimensions grow calculating the posterior becomes
very costly. One tool to tackle this is Markov Chain Monte Carlo MCMC. But moving on to deep learning tasks
with large datasets and millions of parameters, sampling from MCMC is often too slow. Instead we rely on
Variational Inference. Instead of computing the distribution, or approximating the real posterior by sampling from it, 
we choose an approximated posterior distribution and try to make it resemble the real posterior as close as possible.
The drawback of this method is that our approximation can be really off. 

We don't know the real posterior so we start by looking at some distributions Q* which are easy to work with
and want to find the distribution Q(theta) which is the closest to the actual posterior P(theta|data). This closeness
is measured using Kullback-Leibner divergence. But how do we compete closeness if we don't know the posterior?
This is why we use KL-divergence - where the posterior is not needed for calculating the KL-divergence. It is not
a real metric. If we rewrite the KL-divergence algorithm we arrive at the realization that 
the formula can be rewritten into the summation of two terms, one of them being the formula for ELBO which
can be written in expectation (over theta) form. The other realization is that the KL-divergence range is always 
positive. Thus to minimize the KL-divergence we have to maximize the ELBO. This is why we don't need the posterior
as the ELBO can be computed without it, it only contains Q(thete) and P(theta,data) 

RECAP : We want to approximate the posterior P(theta|data) =(approx)= Q(theta)

Notation
 - the subscript in expected value notation refers to the distribution we are taking the expectation with respect to


"""

import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from math import exp, pi, sqrt

#np.random.seed(101)


def generate_data(N, mu=0.0, sigma=1.0):
	data = np.random.normal(mu, sigma, N)
	return(data)    

#Approximation of posterior
def calc_q(data,muN, lamdaN, aN, bN, mu, tau):

	#gaussian distribution with mean and precision given by muN and lamdaN
	q_mu = sqrt(1/(2*pi*lamdaN)) * np.exp(-0.5 * np.dot(lamdaN, np.transpose((mu-muN)**2)))
	#gamma distribution with params aN and bN
	q_tau = (1.0/gamma(aN)) * bN**aN * tau**(aN-1) * np.exp(-bN*tau)
	return q_mu*q_tau
	
# Exact posterior
def calc_p(muT, lambdaT, aT, bT, mu, tau):
	#pdf for normal-gamma
	p = (bT**aT)*sqrt(lambdaT) / (gamma(aT)*sqrt(2*pi)) * tau**(aT-0.5) \
		* np.exp(-bT*tau) * np.exp(-0.5*lambdaT*np.dot(tau,((mu-muT)**2).T))
	
	return p
	

def update_bN_lamdaN(b0,data, N, mu, lamda0,mu0,aN,lamdaN,bN,muN):

	E_mu = muN
	E_mu2 = 1.0 / lamdaN + muN**2
	E_tau = aN / bN
	lamdaN = (lamda0 + N) * E_tau
	bN = b0 - (sum(data) + lamda0*mu0)*E_mu \
		+ 0.5*(sum(data**2) + lamda0*mu0**2 + (lamda0+N)*E_mu2)
	
	return bN, lamdaN


def vi():
	max_iter = 100
	conv_limit = 0.001
	N = 30
	mu0 = a0 = b0 = lamda0 = 0
	aN = a0 + (N + 1) / 2
	data = generate_data(N)
	x_mean = data.mean()
	muN = (lamda0*mu0 + N*x_mean)/(lamda0+N)
	lamdaN = bN = 0.1 #this is what we will approximate so starting out with a guess
	mu 	= np.linspace(-1, 1, 100)
	tau = np.linspace( -1, 2, 100)

	#params for exact posterior

	muT = (lamda0 * mu0 + N * x_mean) / (lamda0 + N)
	lamdaT = lamda0 + N
	aT = a0 + N/2
	bT = b0 + 0.5*sum((data-x_mean)**2) + (lamda0*N*(x_mean-mu0)**2)/(2*(lamda0+N))

	p = calc_p(muT, lamdaT, aT, bT, mu[:,None], tau[:,None])

	bOld = 0.1
	lamdaOld = 0.1

	for i in range(max_iter):
		bN,lamdaN = update_bN_lamdaN(b0,data, N, mu, lamda0,mu0,aN,lamdaN,bN,muN)
		q = calc_q(data,muN, lamdaN, aN, bN, mu[:,None], tau[:,None])

		if (abs(lamdaN - lamdaOld) < conv_limit) and (abs(bN - bOld) < conv_limit):
			break

		plot(p,q,mu,tau,i)
		lamdaOld = lamdaN
		bOld = bN
	
		plt.savefig(str(N) + "points_"+str(i)+ "_iter",bbox_inches='tight')
		
			
	



def plot(p, q,mu,tau,i):
	m, t = np.meshgrid(mu, tau)
	plt.figure()
	p_c = plt.contour(m, t, p,colors="r")
	q_c = plt.contour(m, t, q, colors="b")
	plt.xlabel('mu')
	plt.ylabel('tau')
	plt.axis("equal")
	plt.title("Iteration " + str(i))
	
	p_c.collections[0].set_label("p")
	q_c.collections[1].set_label("q")

	plt.legend(loc='upper left')
	


vi()
