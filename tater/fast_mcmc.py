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


def renorm_flux(lc, toi):
    """Helper function to extract and re-normalize in-transit flux
    to save compute time for MCMC fit. Nearly identical to Caleb's _get_intransit_flux_

    @param tce_dict: TCE results dictionary
    @type tce_dict: dict


    @return new_t, new_f, new_ferr: normalized in-transit time, flux, and uncert

    #CDD TO DO: Compute transit times. Change from self to lc.

    """

    # get the individual transit times and transit duration to create mask

    #CDD DO NEXT
    transit_times = tce_dict.transit_times #REPLACE WITH: Find transit times from period, t0, and lc time stamps
    #END CDD DO NEXT
    duration = toi.exofop_duration

    # linear function we will use to re-normalize the flux
    linear_func = lambda x, m, b: m * x + b

    # initialize new arrays for time, flux and uncert
    new_t = np.array([])
    new_f = np.array([])
    new_ferr = np.array([])

    # create previous transit mask if none provided
    if len(msk) == 0:
        msk = np.zeros(len(self.time_raw), dtype=bool)

    # normalize each transit individually
    for t0 in transit_times:

        # consider data within 3 x the transit duration of the central transit time
        transit_msk = (self.time_raw[~msk] > t0 - 3 * duration) & (self.time_raw[~msk] < t0 + 3 * duration)

        # we will normalize by the out-of-transit flux between
        # 3x and 1.5x the transit duration away from the central transit time
        pre_msk = (self.time_raw[~msk] > t0 - 3 * duration) & (self.time_raw[~msk] < t0 - 1.5 * duration)
        post_msk = (self.time_raw[~msk] < t0 + 3 * duration) & (self.time_raw[~msk] > t0 + 1.5 * duration)
        trend_msk = pre_msk | post_msk

        # require at least N_min data points during, before, and after transit
        cadence = 2. / 1440   # 2-min cadence
        N_min_sides = 0.8 * (1.5 * duration / cadence)   # 80% complete data
        N_min_total = 0.7 * (6 * duration / cadence)   # 70% complete data
        if (np.sum(pre_msk) > N_min_sides) & (np.sum(post_msk) > N_min_sides) & (np.sum(transit_msk) > N_min_total):

            # fit linear model ("trend") to out-of-transit data
            slope, intercept, _, _, _ = linregress(self.time_raw[~msk][trend_msk], self.f_raw[~msk][trend_msk])
            y_trend = linear_func(self.time_raw[~msk][transit_msk], slope, intercept)

            # normalize the transit by the linear trend; save to array
            new_f = np.concatenate((new_f, self.f_raw[~msk][transit_msk] / y_trend))

            # save transit time and uncert arrays
            new_t = np.concatenate((new_t, self.time_raw[~msk][transit_msk]))
            new_ferr = np.concatenate((new_ferr, self.ferr_raw[~msk][transit_msk] / self.f_raw[~msk][transit_msk]))

    # clean arrays
    new_t, new_f, new_ferr = cleaned_array(new_t, new_f, new_ferr)

    return new_t, new_f, new_ferr

#Read list of TOIs in paper
tois = pd.read_csv(tfile)

#Change column names
tois['per'] = tois.exofop_per
tois['T0'] = tois.exofop_t0
tois['rp_rs']= np.sqrt(tois.exofop_depth)

#Loop through all TOIs! :)
for ii in np.arange(len(tois)):
    toi = tois.iloc[ii]
    tic_id = int(toi.TIC)

    print(ii, 'Fitting TOI ', toi.TIC, '(TIC '+str(tic_id)+')')

#   #continue progress
    if ii < 0:
        continue

    if tic_id == 355867695: #temporary workaround
        print('skipping TIC 355867695')
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

    #Trim down to just flux near transit
    lc = pd.DataFrame({'time': transit_fitter.time,
        'flux': transit_fitter.f,
        'err': transit_fitter.f_err,})
    lc = trim_to_transits(lc, toi)
    time = np.array(lc.time)
    flux = np.array(lc.flux)
    flux_err = np.array(lc.err)


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
