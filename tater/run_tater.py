""" Script to run tater """


import tater


# choose a planet
tic_id = 43647325  # WASP-35 b
# tic_id = 230127302  # TOI-1246

# initialize TATER class
transit_fitter = tater.TransitFitter(tic_id,
                                     auto_params=True  # automatically find stellar params on MAST  (default: false)
                                     )

# download data and show plot
transit_fitter.download_data(window_size=3.0,
                             n_sectors=None,  # number of TESS sectors to load (default: all)
                             show_plot=True  # option to show light curve (default: false)
                             )

# find planets
planets = transit_fitter.find_planets(max_iterations=7,  # maximum number of search iterations (default: 7)
                                      tce_threshold=12.0,  # Minimum SDE that counts as a TCE (default: 8.0)
                                      show_plots=False  # option to show periodogram and transit model (default: false)
                                      )

# do transit fits
transit_fitter.fit_transits(show_plots=False,  # show MCMC plots (default: false)
                            save_results=True  # save all results to PDF/txt files (default: true)
                            )
