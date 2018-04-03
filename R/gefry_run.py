import pymc as mc
import matplotlib.pyplot as plt
import gefry3
import pickle as pickle
import numpy as np
from shapely import geometry as g
from shapely.ops import cascaded_union

def run_point(sx,sy,sI,dx,dy,geometry,cross_sections):
  x = sx
  y = sy
  I = sI
  dA = 0.005806  # m^2
  #detector inputs: location (x,y), epsilon = 0.62, area, dwell = 5.0 secs
  detectors = [gefry3.Detector((dx,dy), 0.62, dA, 5) for i in range(2)]

  #--- Setup Domain ---#
  print("1")
  with open(geometry) as f:
    geo   = pickle.load(f)  #geo is a list of shapely polygons
  print("2")
  buildings = [gefry3.Solid(s.exterior.coords) for s in geo]
  bbox      = [(0, 0), (250, 0), (250, 178), (0, 178), (0, 0)]
  domain    = gefry3.Domain(bbox, buildings)
  sigmas    = np.loadtxt(cross_sections)
  materials = [gefry3.Material(1,s) for s in sigmas] #this is a bit of a hack, these are the building cross sections

  #Interstitial material is just air. Don't think this was in gefry2.
  interstitial_material = gefry3.Material(1,0) #cross section of the air
  # I don't think this is correct, but let's assume the air doesn't attenuate rads

  #--- Setup Source ---#
  true_source = gefry3.Source((158,98),3.214e9) #not really used for anything

  #--- Initalize Radiation Model ---#
  P = gefry3.SimpleProblem(domain, interstitial_material, materials, true_source, detectors)
  return P((x,y),I)

def run_model(x,y,I,detectors,geometry,cross_sections):
    r_d       = np.loadtxt(detectors)
    #--- Setup Detectors ---#
    dA = 0.005806  # m^2
    #detector inputs: location (x,y), epsilon = 0.62, area, dwell = 5.0 secs
    detectors = [gefry3.Detector(i, 0.62, dA, 5) for i in r_d]

    #--- Setup Domain ---#
    with open(geometry) as f:
        geo   = pickle.load(f)  #geo is a list of shapely polygons
    buildings = [gefry3.Solid(s.exterior.coords) for s in geo]
    bbox      = [(0, 0), (250, 0), (250, 178), (0, 178), (0, 0)]
    domain    = gefry3.Domain(bbox, buildings)
    sigmas    = np.loadtxt(cross_sections)
    materials = [gefry3.Material(1,s) for s in sigmas] #this is a bit of a hack, these are the building cross sections

    #Interstitial material is just air. Don't think this was in gefry2.
    interstitial_material = gefry3.Material(1,0) #cross section of the air
    # I don't think this is correct, but let's assume the air doesn't attenuate rads

    #--- Setup Source ---#
    true_source = gefry3.Source((158,98),3.214e9) #not really used for anything

    #--- Initalize Radiation Model ---#
    P = gefry3.SimpleProblem(domain, interstitial_material, materials, true_source, detectors)
    return P((x,y),I)


def abc(prior,background,observations,detectors,geometry,cross_sections):
    obs       = np.loadtxt(observations)
    n_obs     = obs.shape[0]
    n_sen     = obs.shape[1]
    r_d       = np.loadtxt(detectors)

    #--- Setup Detectors ---#
    dA = 0.005806  # m^2
    #detector inputs: location (x,y), epsilon = 0.62, area, dwell = 5.0 secs
    detectors = [gefry3.Detector(i, 0.62, dA, 5) for i in r_d]

    #--- Setup Domain ---#
    with open(geometry) as f:
        geo   = pickle.load(f)  #geo is a list of shapely polygons
    buildings = [gefry3.Solid(s.exterior.coords) for s in geo]
    bbox      = [(0, 0), (250, 0), (250, 178), (0, 178), (0, 0)]
    domain    = gefry3.Domain(bbox, buildings)
    sigmas    = np.loadtxt(cross_sections)
    materials = [gefry3.Material(1,s) for s in sigmas] #this is a bit of a hack, these are the building cross sections

    #Interstitial material is just air. Don't think this was in gefry2.
    interstitial_material = gefry3.Material(1,0) #cross section of the air
    # I don't think this is correct, but let's assume the air doesn't attenuate rads

    #--- Setup Source ---#
    true_source = gefry3.Source((158,98),3.214e9) #not really used for anything

    #--- Initalize Radiation Model ---#
    P = gefry3.SimpleProblem(domain, interstitial_material, materials, true_source, detectors)

    #--- ABC run ---#
    N      = prior.shape[0]
    result = np.zeros((N,n_sen))
    for i in range(N):
        if ((i % 1000) == 0): print(i/N)
        result[i,] = np.random.poisson(np.rint(P((prior[i,0],prior[i,1]),prior[i,2])+background))
    return result
