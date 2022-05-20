'''
Run MCMC for a single target without first searching the lightcurve for transits.
CDD
20 May 2022
'''

import tater
import pandas as pd
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#Default settings
verbose = True
show_plots = False

#Directories & Files
tfile = '/Users/courtney/Documents/data/toi_paper_data/triceratops_tess_lightcurves/triceratops_inputs.csv'
taterdir = '/Users/courtney/Documents/GitHub/TESS_Cycle3/tater/'
outdir = '/Users/courtney/Documents/data/toi_paper_data/tater_mcmc_fits/'

#Read list of TOIs in paper
tois = pd.read_csv(tfile)

#Change column names
tois['per'] = tois.triceratops_per
tois['T0'] = tois.triceratops_t0
tois['rp_rs']= np.sqrt(tois.triceratops_depth)

#Loop through all TOIs! :)
for ii in np.arange(len(tois)):
    toi = tois.iloc[ii]
    tic_id = int(toi.TIC)

    print('Fitting TOI ', toi.TIC, '(TIC '+str(tic_id)+')')

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

    if nsec_found <= 0:
        print('No lightcurve found for TOI ', toi.TIC, '(TIC '+str(tic_id)+')')
        print('Skipping.')
        continue

    if toi.per <= 0:
        print('Period is nonpositive found for TOI ', toi.TIC, '(TIC '+str(tic_id)+')')
        print('Skipping.')
        continue

    #Find semimajor axis
    arstar = transit_fitter._P_to_a_(toi.per)

    # initialize MCMC parameters
    theta_0 = dict(per=toi.per,
                   t0=toi.T0,
                   rp_rs=toi.rp_rs,
                   a_rs=max(1, arstar),
                   b=0.1
                   )

    #Not currently masking out other planets in the system
    #intransit = np.zeros(len(self.time_raw), dtype=bool)
    #time, flux, flux_err = transit_fitter._get_intransit_flux_(TCE, msk=intransit)

    time = transit_fitter.time
    flux = transit_fitter.f
    flux_err = transit_fitter.f_err


    # run the MCMC (option to show plots)
    planet_fit, walker_fig, corner_fig, best_fig, best_full_fig = transit_fitter._execute_mcmc_(
        theta_0, time, flux, flux_err, show_plots=show_plots)

    planet_fit.to_csv(outdir+str(tic_id)+'_'+str(int(toi.TOI*100))+'_planet_fit.csv')
    print(planet_fit)

    # save figures to candidate object dictionary
    transit_fitter.fit_results = planet_fit
    transit_fitter.mcmc_fig = walker_fig
    transit_fitter.corner_fig = corner_fig
    transit_fitter.result_fig = best_fig
    transit_fitter.result_full_fig = best_full_fig

    plt.close()

    # use PdfPages to make a new PDF document
    with PdfPages(outdir+str(tic_id)+'_'+str(int(toi.TOI*100))+'_planet_fit.pdf') as pdf:

        # save all the figures to the PDF
        if transit_fitter.corner_fig is not None:
            pdf.savefig(transit_fitter.corner_fig)
        if transit_fitter.result_fig is not None:
            pdf.savefig(transit_fitter.result_fig)
        if transit_fitter.result_full_fig is not None:
            pdf.savefig(transit_fitter.result_full_fig)
        if transit_fitter.mcmc_fig is not None:
            pdf.savefig(transit_fitter.mcmc_fig)
