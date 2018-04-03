# running multiple chains of rad model MCMC on cluster
import pymc as mc
import gefry3
import pickle as pickle
import numpy as np

np.random.seed(1546)

file_prefix = "small_sample_85-85"

r_d    = [(25,25),(25,125),(150,125)]
det_filename = file_prefix + "_det"
np.savetxt(det_filename,r_d, delimiter=' ')

n_obs        = 5
background   = 300
dA           = 0.005806  # m^2

#--- Setup Detectors ---#
detectors = [gefry3.Detector(i, 0.62, dA, 5) for i in r_d]

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

#--- Setup Source ---#
true_source = gefry3.Source((158,98),3.214e9)

#--- Initalize Radiation Model ---#
P = gefry3.SimpleProblem(domain, interstitial_material, materials, true_source, detectors)

mean_response = P((85,85),3.214e9) + background
#     file_name = 'radiation_chain%d.txt' % (init_values[4])
# 	np.savetxt(file_name,chain, delimiter=',')
#
obs = []
for i in range(n_obs):
	obs.append(np.random.poisson(mean_response))

obs_filename = file_prefix + "_obs"
np.savetxt(obs_filename,obs, delimiter=' ')

#
# 	def radiation_model(x, y, I, detectors, n_obs):
# 		sensor_response = P((x, y), I) # .astype(np.float64)
# 		return(sensor_response)
#
# 	#Likelihood
# 	@mc.stochastic(observed=True)
# 	def rad_counts(value=obs, x=x, y=y, I=I):
# 		output = np.rint(P((x, y), I)) + background_count
# 		LogLik = 0
# 		for i in range(0,n_obs):
# 			LogLik = LogLik + (np.sum(obs[:,i]*np.log(output[i])) - n_obs * output[i])
# 			return LogLik
#
# 	model = mc.MCMC([x,y,I,rad_counts])
# 	model.use_step_method(mc.AdaptiveMetropolis,[x,y,I],shrink_if_necessary=True)
# 	#model.sample(iter=10**4+3000, burn=3000)
# 	model.sample(iter=20, burn=0, verbose = 0) #testing purposes
#
# 	# Chains
# 	samples_x = model.trace('x')[:]
# 	samples_y = model.trace('y')[:]
# 	samples_I = model.trace('I')[:]
# 	chain     = np.vstack((samples_x,samples_y,samples_I)).T
# 	file_name = 'narrow_prior_radiation_chain%d.txt' % (init_values[4])
#     file_name = 'radiation_chain%d.txt' % (init_values[4])
# 	np.savetxt(file_name,chain, delimiter=',')
# 	return 0
#
# multipleMCMC(chain_parms)
