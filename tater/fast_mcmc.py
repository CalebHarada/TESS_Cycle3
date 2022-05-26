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
tfile = '/Users/courtney/Documents/data/toi_paper_data/triceratops_tess_lightcurves/targets_exofop_toi_properties.csv'
taterdir = '/Users/courtney/Documents/GitHub/TESS_Cycle3/tater/'
outdir = '/Users/courtney/Documents/data/toi_paper_data/tater_mcmc_fits/'

def trim_to_transits(lc, toi,nfactor=3.):
    #Trim down to just flux near transit
    t0 = toi.exofop_t0
    period = toi.exofop_per
    dur = toi.exofop_duration
    tdiff = np.mod(lc.time - t0-(period/2.), period)-period/2.
    wnear = np.where(np.abs(tdiff) < (dur*nfactor))
    lc = lc.iloc[wnear]
    return lc

#Read list of TOIs in paper
tois = pd.read_csv(tfile)

#Change column names
tois['per'] = tois.exofop_per
tois['T0'] = tois.exofop_t0
tois['rp_rs']= np.sqrt(tois.exofop_depth)

#Fill in missing stellar parameters
# solar logg is from Smalley et al. (2005)
values = {"R_star": 1, "M_star": 1, "logg": 4.4374, "[Fe/H]": 0}
tois = tois.fillna(value=values)

#Loop through all TOIs! :)
for ii in np.arange(len(tois)):
#for ii in [3]:
    toi = tois.iloc[ii]
    tic_id = int(toi.TIC)

    print(ii, 'Fitting TOI ', toi.TIC, '(TIC '+str(tic_id)+')')

    #Set output file base
    outbase = outdir+str(tic_id)+'_'+str(int(toi.TOI*100))
    print('   ', outbase)

#   #continue progress
    if ii < 0:
        continue

    if tic_id == 355867695: #temporary workaround
        print('skipping TIC 355867695')
        continue

    #Stellar parameters
    wantcol = ['Teff','logg','R_star',
    'M_star','[Fe/H]']
    stellar_params = toi[wantcol]

    # initialize TATER class
    transit_fitter = tater.TransitFitter(
        tic_id,
        auto_params=False,  # automatically find stellar params on MAST  (default: false)
        ask_user=False, # do not ask user for input (default: false)
        assume_solar=False, # fill in solar values if not found on MAST (default: false)
        preloaded_stellar_params=True, #stellar parameters already downloaded
        stellar_params=stellar_params #dataframe containing stellar parameters
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

    #Normalize each transit & trim lightcurve to near transit
    time, flux, flux_err = transit_fitter._renorm_flux_(toi)

    #Save the trimmed and renormalized lightcurve used by TATER
    tlc = pd.DataFrame({'time': time, 'flux': flux, 'error': flux_err})
    tlc.to_csv(outbase+'_trimmed_lc.csv',index=False)

    # run the MCMC (option to show plots)
    planet_fit, walker_fig, corner_fig, best_fig, best_full_fig = transit_fitter._execute_mcmc_(
        theta_0, time, flux, flux_err, show_plots=show_plots, outbase=outbase)

    planet_fit.to_csv(outbase+'_planet_fit.csv')
    print(planet_fit)

    # save figures to candidate object dictionary
    transit_fitter.fit_results = planet_fit
    transit_fitter.mcmc_fig = walker_fig
    transit_fitter.corner_fig = corner_fig
    transit_fitter.result_fig = best_fig
    transit_fitter.result_full_fig = best_full_fig

    plt.close()

    # use PdfPages to make a new PDF document
    with PdfPages(outbase+'_planet_fit.pdf') as pdf:

        # save all the figures to the PDF
        if transit_fitter.corner_fig is not None:
            pdf.savefig(transit_fitter.corner_fig)
        if transit_fitter.result_fig is not None:
            pdf.savefig(transit_fitter.result_fig)
        if transit_fitter.result_full_fig is not None:
            pdf.savefig(transit_fitter.result_full_fig)
        if transit_fitter.mcmc_fig is not None:
            pdf.savefig(transit_fitter.mcmc_fig)
