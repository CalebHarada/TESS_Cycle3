""" Script to run tater """


import tater


# choose a planet and initialize
tic_id = 43647325 # WASP-35 b
transit_fitter = tater.TransitFitter(tic_id,
                                     auto_params=True # automatically find stellar params on MAST
                                     )

# download data and show plot
transit_fitter.download_data(show_plot=True)

# find planets
planets = transit_fitter.find_planets(show_plots=False)

# do transit fits
transit_fitter.fit_transits(show_plots=False)
