# running multiple chains of rad model MCMC on cluster
import pymc as mc
import gefry3
import pickle as pickle
import numpy as np
import multiprocessing
from multiprocessing import Pool

np.random.seed(1546)

num_cores = 3;
num_chain = 10;

#--- load random seeds ---#

#--- Initialize Random Seeds and Stuff ---#
x = np.random.uniform(0,246.615,num_chain)
y = np.random.uniform(0,176.333,num_chain)
I = np.random.uniform(5e8, 5e10,num_chain)
s = np.random.randint(0,2000000000,size=(num_chain,1));

chain_parms = []
for i in range(num_chain):
	chain_parms.append((x[i],y[i],I[i],s[i][0],i))

def multipleMCMC(parameters):
	if __name__ == '__main__':
		my_pool   = Pool(processes=num_cores) # depends on available cores
		result = my_pool.map(radiation_model_sampler, chain_parms)
		my_pool.close() # not optimal! but easy
		my_pool.join()
		print(result)
		return 0
		#return cleaned;
	else:
		return 0

#--- radiation model sampler---#
def radiation_model_sampler(init_values):

	np.random.seed(init_values[3])

	x = mc.Uniform('x', lower=0, upper=246.615, value=init_values[0])
	y = mc.Uniform('y', lower=0, upper=176.333, value=init_values[1])
	I = mc.Uniform('I', lower=5e8, upper=5e10, value=init_values[2])

	n_obs = 10  #num observations per detector
	obs   = np.loadtxt('obs.dat')

	#--- Setup Detectors ---#
	dA = 0.005806  # m^2
	r_d = np.loadtxt('det.dat') #jk different order of the stuff
	detectors = [gefry3.Detector(i, 0.62, dA, 5) for i in r_d]  # for each detector location, dwell time = 5.0, epsilon = 0.62 (efficiency)

	#--- Setup Domain ---#
	with open('./revised_geo.pkl') as f:
		geo = pickle.load(f)

	buildings = [gefry3.Solid(s.exterior.coords) for s in geo]
	bbox      = [(0, 0), (250, 0), (250, 178), (0, 178), (0, 0)]
	domain    = gefry3.Domain(bbox, buildings) #geo is a list of shapely polygons

	sigmas    = np.loadtxt('cross_sec.dat') #these go in the interstitial materials component?
	materials = [gefry3.Material(1,s) for s in sigmas] #this is a bit of a hack, these are the building cross sections

	interstitial_material = gefry3.Material(1,0) #cross section of the air the gammas are moving through (but we don't have this?)
	# I don't think this is correct, but let's try it
	#domain = gefry2.Domain(geo, sigma)

	#--- Setup Source ---#
	true_source = gefry3.Source((158,98),3.214e9)

	#--- Initalize Radiation Model ---#
	#P = gefry2.Problem(domain, detectors, S, 300)
	P = gefry3.SimpleProblem(domain, interstitial_material, materials, true_source, detectors)

	#print(P((158,98),3.214e9))
	background_count = 300

	def radiation_model(x, y, I, detectors, n_obs):
		sensor_response = P((x, y), I) # .astype(np.float64)
		return(sensor_response)

	#Likelihood
	@mc.stochastic(observed=True)
	def rad_counts(value=obs, x=x, y=y, I=I):
		output = np.rint(P((x, y), I)) + background_count
		LogLik = 0
		for i in range(0,n_obs):
			LogLik = LogLik + (np.sum(obs[:,i]*np.log(output[i])) - n_obs * output[i])
			return LogLik

	model = mc.MCMC([x,y,I,rad_counts])
	model.use_step_method(mc.AdaptiveMetropolis,[x,y,I],shrink_if_necessary=True)
	#model.sample(iter=10**4+3000, burn=3000)
	model.sample(iter=20, burn=0, verbose = 0) #testing purposes

	# Chains
	samples_x = model.trace('x')[:]
	samples_y = model.trace('y')[:]
	samples_I = model.trace('I')[:]
	chain     = np.vstack((samples_x,samples_y,samples_I)).T
	file_name = 'radiation_chain%d.txt' % (init_values[4])
	np.savetxt(file_name,chain, delimiter=',')
	return 0

multipleMCMC(chain_parms)
