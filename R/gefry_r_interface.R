# Interface between R and gefry3 radiation model
# Filename: gefry_R_interface.R
# Written by: Isaac Michaud
# Created on: 10-27-17
# Edited on:  01-23-28
# Description: This creates R callable functions for gefry3. I added the ability to pass detector locations to the python code without having to manually create a new det.dat file. At some point in the future it may be interesting to package these functions together in a R package. It takes on average 3 seconds to run Katie's base problem which means that it may still be a little slow for doing complete MCMC in R using DRAM
# many of these functions are outdated because they rely of using the gefry2 files instead of the gefry3 yaml files....
library(reticulate)

gefry <- import_from_path('gefry_run','.')

run_gefry <- function(x,
                      y,
                      I,
                      detectors      = 'det.dat',
                      geometry       = 'revised_geo.pkl',
                      cross_sections = 'cross_sec.dat'
                      ) {
  if (is.matrix(detectors)) {
  	write(t(detectors),file='temp_detectors.dat',ncolumns=2)
  	result <- gefry$run_model(x,y,I,'temp_detectors.dat',geometry,cross_sections)
  } else {
  	result <- gefry$run_model(x,y,I,detectors,geometry,cross_sections)
  }
  return(result)
}

# run_gefry <- function(x,
                      # y,
                      # I,
                      # observations   = 'obs.dat',
                      # detectors      = 'det.dat',
                      # geometry       = 'revised_geo.pkl',
                      # cross_sections = 'cross_sec.dat'
                      # ) {
  # gefry$run_model(x,y,I,observations,detectors,geometry,cross_sections)
# }

abc_gefry <- function(prior,
                      background     = 300,
                      observations   = 'obs.dat',
                      detectors      = 'det.dat',
                      geometry       = 'revised_geo.pkl',
                      cross_sections = 'cross_sec.dat') {
  gefry$abc(prior,background,observations,detectors,geometry,cross_sections)
}

run_point_gefry <- function(sx,
                            sy,
                            sI,
                            dx,
                            dy,
                            geometry       = 'revised_geo.pkl',
                            cross_sections = 'cross_sec.dat') {
  return(gefry$run_point(sx,sy,sI,dx,dy,geometry,cross_sections))
}

run_point_gefry2 <- function(sx,
                             sy,
                             sI,
                             dx,
                             dy,
                             geometry       = 'revised_geo.pkl',
                             cross_sections = 'cross_sec.dat') {
  result <- run_gefry(sx,sy,sI,detectors = matrix(c(dx,dy,0,0),ncol=2,byrow=TRUE))
  return(result[1])
}


gefry2 <- import_from_path('run_gefry_mcmc','.')

run_gefry_mcmc <- function(deck,num_samples) {
  gefry2$run_mcmc(deck,num_samples)
}


