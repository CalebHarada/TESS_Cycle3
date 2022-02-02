""" Script to run tater """
import matplotlib
matplotlib.use('agg')
import sys
import tater


# choose a planet
#tic_ids = [43647325]  # WASP-35 b

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

	# download data and show plot
	transit_fitter.download_data(window_size=3.0,
		n_sectors=1,  # number of TESS sectors to load (default: all)
		show_plot=False  # option to show light curve (default: false)
		)

	# find planets
	planets = transit_fitter.find_planets(max_iterations=7,  # maximum number of search iterations (default: 7)
		tce_threshold=8.0,  # Minimum SDE that counts as a TCE (default: 8.0)
		period_min=0.8, # Minimum TLS period to explore
		show_plots=False,  # option to show periodogram and transit model (default: false)
		save_results = True  # save all results to PDF/txt files (default: true)
		)

	
	if len(transit_fitter.TCEs) >= 1:
		# do vetting
		transit_fitter.vet_TCEs(
			save_results=True  # save all results to PDF/txt files (default: true)
		)
