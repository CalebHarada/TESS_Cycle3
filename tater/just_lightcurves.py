#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrapper to use TATER to download and process TESS light curves.
Produces output files suitable for use with TRICERATOPS.
Developed by CDD on 11 May 2022.
"""

import matplotlib
import sys
import pandas as pd
import numpy as np
import tater

verbose = True

basedir = '/Users/courtney//Documents/data/toi_paper_data/'
outdir = basedir+'tess_lightcurves/'

targetlistfile = '/Users/courtney/Documents/data/toi_paper_data/paper_targets.csv'

#Load list of TOIs and remove duplicates
targets = pd.read_csv(targetlistfile)
targets = targets.drop_duplicates('TIC')
tic_ids = targets['TIC']
print('Number of targets: ', len(tic_ids))
have_data = np.zeros(len(tic_ids))

for i,tic_id in enumerate(tic_ids):
    if i < 455:
        continue #that's how far we made it before

    if tic_id == 29918916:
        print('skipping TIC 29918916 because of buffer issue.')
        continue
    if tic_id == 219776325:
        print('skipping TIC 219776325 because of buffer issue.')
        continue

    print('TIC ' + str(tic_id) + ', ' + str(i+1) + ' of ' + str(len(tic_ids)))
    # initialize TATER class
    transit_fitter = tater.TransitFitter(
        tic_id,
        auto_params=False,  # automatically find stellar params on MAST  (default: false)
        ask_user=True, # do not ask user for input (default: false)
        assume_solar=True # fill in solar values if not found on MAST (default: false)
        )
    # download data and show plot
    nsec_found = transit_fitter.download_data(
        window_size=3.0,
        #n_sectors=1,  # number of TESS sectors to load (default: all)
        show_plot=False  # option to show light curve (default: false)
        )
    print('sectors found: ', nsec_found)
    if  nsec_found == 0:
        if verbose == True:
        		print('No sectors of data found for TIC'+str(tic_id)+'. Skipping to next target.')
    else:
        lc_df = pd.DataFrame({'time': transit_fitter.time, 'flux': transit_fitter.f, 'err': transit_fitter.f_err})
        lc_df.to_csv(outdir+str(tic_id)+'_lightcurve.csv',index=0)
        have_data[i] = 1

print('Found data for '+str(np.sum(have_data))+ 'targets.')
