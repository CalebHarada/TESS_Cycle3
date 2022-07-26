'''
Run TATER for all stars in the TOI Imaging paper
CDD
20 May 2022
'''

import tater
import pandas as pd
import numpy as np
from os.path import exists

#Default local_settings
verbose = True

#Directories & Files
tfile = '/Users/courtney/Documents/data/toi_paper_data/triceratops_tess_lightcurves/targets_exofop_toi_properties.csv'
taterdir = '/Users/courtney/Documents/GitHub/TESS_Cycle3/tater/'

#Read list of TOIs in paper
tois = pd.read_csv(tfile)

#Get list of unique TIC IDs
tic_ids = np.unique(tois.TIC)
print('Found '+str(len(tic_ids))+' TIC IDs.')

for i,tic_id in enumerate(tic_ids):
    tic_id = int(tic_id)
    print('TIC ' + str(tic_id) + ', ' + str(i+1) + ' of ' + str(len(tic_ids)))

    #Check if TATER has already been run. If so, skip this target
    tateroutput = taterdir + 'outputs/'+str(tic_id)+'/tater_report_'+str(tic_id)+'*.txt'
    if exists(tateroutput):
        continue

	# initialize TATER class
    transit_fitter = tater.TransitFitter(
        tic_id,
        auto_params=True,  # automatically find stellar params on MAST  (default: false)
        ask_user=False, # do not ask user for input (default: false)
        assume_solar=True # fill in solar values if not found on MAST (default: false)
        )

    # load data
    nsec_found = transit_fitter.use_local_data(
    )

    if nsec_found == 0:
        if verbose == True:
            print('No sectors of data found for TIC'+str(tic_id)+'. Skipping to next target.')
        continue

    # find planets
    planets = transit_fitter.find_planets(
        max_iterations=4,  # maximum number of search iterations (default: 7)
        tce_threshold=8.0,  # Minimum SDE that counts as a TCE (default: 8.0)
        period_min=0.8, # Minimum TLS period to explore
        show_plots=False,  # option to show periodogram and transit model (default: false)
        )

    #Run MCMC fits
    planets = transit_fitter.run_mcmc(
        show_plots=False,  # option to show periodogram and transit model (default: false)
        save_results = True  # save all results to PDF/txt files (default: true)
        )


    if len(transit_fitter.TCEs) >= 1:
        # do vetting
        transit_fitter.vet_TCEs(
            save_results=True  # save all results to PDF/txt files (default: true)
            )
