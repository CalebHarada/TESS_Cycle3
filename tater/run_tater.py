""" Script to run tater """
import matplotlib
matplotlib.use('agg')
import tater
import sys


# choose a planet
#tic_id = 43647325  # WASP-35 b
# tic_id = 230127302  # TOI-1246

#tic_ids = [1003831, 1129033, 1528696, 2527981, 2670610, 2760710, 4616072, 4646810, 5109298, 5772442]
#tic_ids = [309402106, 149301575, 300013921, 306472057, 389924075, 244161191, 384549882, 238004786, 34068865, 206609630]
#tic_ids = [300013921, 306472057, 389924075, 244161191, 384549882, 238004786, 34068865, 206609630]

if len(sys.argv) > 1:
	tic_ids = sys.argv[1:]
	try:
		tic_ids = [int(elt) for elt in tic_ids]
	except ValueError:
		print('All input TIC IDs must be integers')
		sys.exit()
else:
	# fill in with whatever list of tic_ids you'd like
	#tic_ids = [309402106, 149301575, 300013921, 306472057, 389924075, 244161191, 384549882, 238004786, 34068865, 206609630]
	print('Must provide one or more input TIC IDs as additional arguments')
	sys.exit()

for i,tic_id in enumerate(tic_ids):
	print('TIC ' + str(tic_id) + ', ' + str(i+1) + ' of ' + str(len(tic_ids)))
	# initialize TATER class
	transit_fitter = tater.TransitFitter(tic_id,
		auto_params=True,  # automatically find stellar params on MAST  (default: false)
		ask_user=False, # do not ask user for input (default: false)
		assume_solar=True # fill in solar values if not found on MAST (default: false)
		)
	#if transit_fitter.missing != []:
	#	continue

	# download data and show plot
	transit_fitter.download_data(window_size=3.0,
		n_sectors=None,  # number of TESS sectors to load (default: all)
		#show_plot=True  # option to show light curve (default: false)
		show_plot=False  # option to show light curve (default: false)
		)

	# find planets
	planets = transit_fitter.find_planets(max_iterations=7,  # maximum number of search iterations (default: 7)
		#tce_threshold=12.0,  # Minimum SDE that counts as a TCE (default: 8.0)
		tce_threshold=8.0,  # Minimum SDE that counts as a TCE (default: 8.0)
		#period_min=10., # Minimum TLS period to explore
		show_plots=False  # option to show periodogram and transit model (default: false)
		)

	
	if len(transit_fitter.planet_candidates) >= 1:
		# do transit fits
		transit_fitter.fit_transits(show_plots=False,  # show MCMC plots (default: false)
			save_results=True  # save all results to PDF/txt files (default: true)
			)
