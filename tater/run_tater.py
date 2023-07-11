""" Script to run tater """
import matplotlib
matplotlib.use('agg')
import sys
import tater
import datetime as dt
import os

inject_recover = False
verbose = False

# choose a planet
#tic_ids = [43647325]  # WASP-35 b

def unix_time(dt0):
	epoch = dt.datetime.utcfromtimestamp(0)
	delta = dt0 - epoch
	return delta.total_seconds()

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
	save_to_path = "{}/BLS_full_run_outputs/{}".format(os.getcwd(), tic_id)
	if not os.path.isdir(save_to_path):
		os.mkdir(save_to_path)
	time_audit_file = "{}/time_audit_{}.txt".format(save_to_path, tic_id)
	with open(time_audit_file,'w') as f:
		f.write( 'Time audit of ' + str(tic_id) + '\n')

	print('TIC ' + str(tic_id) + ', ' + str(i+1) + ' of ' + str(len(tic_ids)))
	# initialize TATER class
	before1 = unix_time(dt.datetime.now())
	'''
	transit_fitter = tater.TransitFitter(
		tic_id,
		auto_params=True,  # automatically find stellar params on MAST  (default: false)
		ask_user=False, # do not ask user for input (default: false)
		assume_solar=True # fill in solar values if not found on MAST (default: false)
		)
	'''
	import pandas as pd
	import numpy as np
	stellar_params_file = 'updated_stellar_properties_08jun2023.csv'
	stellar_params_table = pd.read_csv(stellar_params_file)
	stellar_params_row = stellar_params_table[stellar_params_table.TIC == tic_id].head(1) # first match only
	stellar_params = pd.DataFrame(columns=['R_star', 'M_star', 'Teff', 'logg', '[Fe/H]'], index=range(1))
	stellar_params.R_star = float(stellar_params_row.adopted_rad)
	stellar_params.M_star = float(stellar_params_row.adopted_mass)
	stellar_params.Teff = float(stellar_params_row.adopted_teff)
	stellar_params.logg = float(stellar_params_row.adopted_logg)
	stellar_params['[Fe/H]'] = float(stellar_params_row.MH)
	# [Fe/H] is the least important, so we can fill in Solar value if otherwise undefined
	if np.isnan(float(stellar_params['[Fe/H]'])):
		stellar_params['[Fe/H]'] = 0. # Solar value of 'Fe_H' used instead: 0.
	if np.isnan(float(stellar_params.R_star)) or np.isnan(float(stellar_params.M_star)) or \
		np.isnan(float(stellar_params.Teff)) or np.isnan(float(stellar_params.logg)):
		print('Skipping TIC ' + str(tic_id) + ': missing stellar parameters in ' + str(stellar_params_file))
		continue # only run system if all stellar parameters present
	transit_fitter = tater.TransitFitter(
		tic_id,
		auto_params=False,  # automatically find stellar params on MAST  (default: false)
		ask_user=False, # do not ask user for input (default: false)
		assume_solar=False, # fill in solar values if not found on MAST (default: false)
		preloaded_stellar_params=True,
		stellar_params=stellar_params
		)
	after1 = unix_time(dt.datetime.now())
	print()
	print()
	print('TATER class initialization: ', after1 - before1, 's')
	with open(time_audit_file,'a') as f:
		f.write( 'TATER class initialization: ' + str(after1 - before1) + ' s\n')
	print()
	print()

	# download data and show plot
	before1 = unix_time(dt.datetime.now())
	nsec_found = transit_fitter.download_data(
		window_size=3.0,
		#n_sectors=1,  # number of TESS sectors to load (default: all)
		show_plot=False  # option to show light curve (default: false)
		)
	after1 = unix_time(dt.datetime.now())
	print()
	print()
	print('Data download ', after1 - before1, 's')
	with open(time_audit_file,'a') as f:
		f.write( 'Data download: ' + str(after1 - before1) + ' s\n')
	print()
	print()

	if nsec_found == 0:
		if verbose == True:
			print('No sectors of data found for TIC'+str(tic_id)+'. Skipping to next target.')

	# search for planets and save results
	if not inject_recover:
		# find planets
		before1 = unix_time(dt.datetime.now())
		planets = transit_fitter.find_planets(
			mode='turbo', # 'tls' for TLS only, 'turbo' for BLS only, 'combo' for BLS and then TLS if BLS fails
			max_iterations=7,  # maximum number of search iterations (default: 7)
			tce_threshold=8.0,  # Minimum SDE that counts as a TCE (default: 8.0)
			#period_min=0.8, # Minimum TLS period to explore
			period_min=0.4, # Minimum TLS period to explore
			show_plots=False,  # option to show periodogram and transit model (default: false)
			)
		after1 = unix_time(dt.datetime.now())
		print()
		print()
		print('transit_fitter.find_planets() ', after1 - before1, 's')
		with open(time_audit_file,'a') as f:
			f.write( 'transit_fitter.find_planets(): ' + str(after1 - before1) + ' s\n')
		print()
		print()
		if not len(transit_fitter.TCEs) >= 1:
			continue # No TCEs found, move to next TIC ID

		#Run MCMC fits
		before1 = unix_time(dt.datetime.now())
		planets = transit_fitter.run_mcmc(
				show_plots=False,  # option to show periodogram and transit model (default: false)
				save_results = True  # save all results to PDF/txt files (default: true)
				)
		after1 = unix_time(dt.datetime.now())
		print()
		print()
		print('transit_fitter.run_mcmc() ', after1 - before1, 's')
		with open(time_audit_file,'a') as f:
			f.write( 'transit_fitter.run_mcmc(): ' + str(after1 - before1) + ' s\n')
		print()
		print()

		before1 = unix_time(dt.datetime.now())
		if len(transit_fitter.TCEs) >= 1:
			# do vetting
			transit_fitter.vet_TCEs(
				save_results=True  # save all results to PDF/txt files (default: true)
			)
		after1 = unix_time(dt.datetime.now())
		print()
		print()
		print('transit_fitter.vet_TCEs() ', after1 - before1, 's')
		with open(time_audit_file,'a') as f:
			f.write( 'transit_fitter.vet_TCEs(): ' + str(after1 - before1) + ' s\n')
		print()
		print()

	# or run injection and recovery
	else:
		transit_fitter._explore_(
			time=transit_fitter.time_raw, # Light curve time series
			flux=transit_fitter.f_raw, # Light curve flux
			flux_err=transit_fitter.ferr_raw, # Light curve flux error
			mstar=transit_fitter.M_star.value, # Stellar mass (Solar masses)
			rstar=transit_fitter.R_star.value, # Stellar radius (Solar radii)
			N=50, # number of injections (default: 25)
			raw_flux=True # option to inject signal before or after flattening (default: True)
			)
