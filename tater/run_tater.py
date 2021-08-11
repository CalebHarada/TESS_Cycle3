""" Script to run tater """


import tater


# choose a planet and initialize
tic_id = 230127302
transit_fitter = tater.TransitFitter(tic_id)

# download data
transit_fitter.download_data()

# find planets
planets = transit_fitter.find_planets()

# do transit fits
transit_fitter.fit_transits()
