#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tess trAnsiT fittER (TATER)

TATER downloads TESS photometry and implements an MCMC
ensemble sampler to fit a BATMAN transit model to the data.
"""

# Futures
# [...]

# Built-in/Generic Imports
import os, sys
# [...]

# Libs
import batman, emcee, corner
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize
from scipy.stats import linregress
from scipy.signal import find_peaks
from scipy.special import erfcinv
from lightkurve import search_lightcurve
from astropy.timeseries import BoxLeastSquares
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
from wotan import flatten
from tqdm import tqdm


# [...]

# Own modules
# [...]

__author__ = "Caleb K. Harada"
__copyright__ = " "
__credits__ = [" ", ]
__license__ = " "
__version__ = "1.0.0"
__maintainer__ = "Caleb K. Harada"
__email__ = "charada@berkeley.edu"
__status__ = " "

# </header>
# Code begins here.


class TransitFitter(object):
    """blah blah"""


    def __init__(self, tic_id):
        """Initialization"""

        if (not isinstance(tic_id, int)) | (tic_id <= 0):
            raise ValueError("TIC ID ({}) must be a positive integer.".format(tic_id))

        # initialize some variables
        self.tic_id = "TIC {}".format(tic_id)
        self.labels = ["$T_0$", "$r_p/R_*$", "$a/R_*$", "$b$"]

        # initialize BLS arrays
        self.period_grid = None
        self.bls_rs = np.array([])
        self.bls_durs = np.array([])
        self.bls_t0s = np.array([])
        self.bls_depths = np.array([])
        self.TCEs = None
        self.planet_candidates = []

        # initialize stellar params
        self.R_star = None
        self.M_star = None
        self.logg = None
        self.Teff = None
        self.Fe_H = None
        self.u1 = None
        self.u2 = None
        
        # initialize data arrays
        self.lc = None
        self.time_raw = None
        self.f_raw = None
        self.ferr_raw = None
        self.time = np.array([])
        self.f500 = np.array([])
        self.f1000 = np.array([])
        self.f2000 = np.array([])
        self.f_err = np.array([])

        # initialize MCMC options
        self.ndim = len(self.labels)
        self.nwalkers = 15
        self.nsteps = 2500
        self.nburn = 1500



    def download_data(self, plot=False):
        """Function to download and flatten raw light curve data"""

        # get stellar params
        print("   retrieving stellar parameters...")
        self._get_stellar_params_()

        # download
        print("   acquiring TESS data...")
        search_result = search_lightcurve(self.tic_id, mission="TESS", author="SPOC")
        self.lc = search_result.download_all(quality_bitmask="default")
        self.time_raw, self.f_raw, self.ferr_raw = np.concatenate([(sector.remove_nans().time.value,
                                                                    sector.remove_nans().flux.value,
                                                                    sector.remove_nans().flux_err.value
                                                                    ) for sector in self.lc], axis=1)

        if plot:
            self.lc.plot()

        # flatten
        for lc in tqdm(self.lc, desc="   flattening light curve"):
            
            time = lc.remove_nans().time.value
            flux = lc.remove_nans().flux.value
            flux_err = lc.remove_nans().flux_err.value
            
            self.time = np.concatenate((self.time, time))
            self.f_err = np.concatenate((self.f_err, flux_err/flux))
            
            # window lengths in minutes: 500, 1000, 2000
            for w in [500, 1000, 2000]:
                window = w / 1440.
                flatten_lc, _ = flatten(
                    time,                  # Array of time values
                    flux,                  # Array of flux values
                    method='trim_mean',
                    window_length=window,  # The length of the filter window in units of ``time``
                    edge_cutoff=window/2,  # length (in units of time) to be cut off each edge.
                    return_trend=True,     # Return trend and flattened light curve
                    proportiontocut=0.023  # Cut 2.3% off both ends (clip above 2-sigma)
                    )
                
                if w==500:
                    self.f500 = np.concatenate((self.f500, flatten_lc))
                elif w==1000:
                    self.f1000 = np.concatenate((self.f1000, flatten_lc))
                elif w==2000:
                    self.f2000 = np.concatenate((self.f2000, flatten_lc))
        
        print("   done.")

        return None



    def find_planets(self, method="bls"):
        """Funtion to identify planets"""

        if not method in ["bls", "tls"]:
            raise ValueError("Transit finding 'method' ('{}') must be either 'bls' or 'tls'.".format(method))

        if method == "bls":
            TCEs = self._bls_search_()

        print("   vetting TCEs...") # see Zink+2020

        # previous planet check
        TCEs = self._vet_previous_planets_(TCEs)

        # alias check
        TCEs = self._vet_alias_periods_(TCEs)


        for TCE in TCEs:
            if TCE["FP"] == False:
                self.planet_candidates.append(TCE)

        print("   vetting recovered {} planet candidates from {} TCEs.".format(len(self.planet_candidates), len(self.TCEs)))

        return self.planet_candidates


        












    def _vet_previous_planets_(self, TCE_list):
        """previous planet check defined by Zink+2020"""

        for i in tqdm(range(len(TCE_list)), desc="   checking previous signals"):
            for j in range(0, i):
        
                P_A, P_B = np.sort((TCE_list[i]["per"], TCE_list[j]["per"]))
                delta_P = (P_B - P_A) / P_A
                sigma_P = np.sqrt(2) * erfcinv(np.abs(delta_P - round(delta_P)))

                delta_t0 = np.abs(TCE_list[i]["t0"] - TCE_list[j]["t0"]) / TCE_list[i]["dur"]

                delta_SE1 = np.abs(TCE_list[i]["t0"] - TCE_list[j]["t0"] + TCE_list[i]["per"]/2) / TCE_list[i]["dur"]

                delta_SE2 = np.abs(TCE_list[i]["t0"] - TCE_list[j]["t0"] - TCE_list[i]["per"]/2) / TCE_list[i]["dur"]

                if (sigma_P > 2.5) & (TCE_list[j]["FP"] == True):
                    TCE_list[i]["FP"] = True
                    break

                if (sigma_P > 2.5) & (delta_t0 < 1):
                    TCE_list[i]["FP"] = True
                    break

                elif (sigma_P > 2.5) & ((delta_SE1 < 1) | (delta_SE2 < 1)):
                    TCE_list[i]["FP"] = True
                    break

        return TCE_list



    def _vet_alias_periods_(self, TCE_list):
        """period alias check defined by Zink+2020"""

        for TCE in tqdm(TCE_list, desc="   checking alias periods"):

            if TCE["FP"] == False:

                # choose smoothing filter based on transit duration
                time = self.time[np.isfinite(self.f2000)]
                flux = self.f2000[np.isfinite(self.f2000)]
                flux_err = self.f_err[np.isfinite(self.f2000)]

                if (TCE["dur"] * 1440) < 200:

                    if (TCE["dur"] * 1440) < 50:
                        time = self.time[np.isfinite(self.f500)]
                        flux = self.f500[np.isfinite(self.f500)]
                        flux_err = self.f_err[np.isfinite(self.f500)]

                    elif (TCE["dur"] * 1440) < 100:
                        time = self.time[np.isfinite(self.f1000)]
                        flux = self.f1000[np.isfinite(self.f1000)]
                        flux_err = self.f_err[np.isfinite(self.f1000)]

                # anonymous functions
                nll = lambda *args: -self._log_likelihood_min_(*args)
                chi_square = lambda O, E, sigma : np.sum( np.square((O - E) / sigma) )

                chi2_aliases = []
                for alias_per in (TCE["per"] * np.array([2, 3, 1/2, 1/3])):

                    a = self._P_to_a_(alias_per)
                    inc = np.arccos(TCE["b"] / a) * (180 / np.pi)

                    # compute model
                    batman_params = batman.TransitParams()
                    batman_params.t0 = TCE["t0"]                   # time of inferior conjunction
                    batman_params.per = alias_per                  # orbital period
                    batman_params.rp = np.sqrt(TCE["dep"])         # planet radius (in units of stellar radii)
                    batman_params.a = a                            # semi-major axis (in units of stellar radii)
                    batman_params.inc = inc                        # orbital inclination (in degrees)
                    batman_params.ecc = 0.                         # eccentricity
                    batman_params.w = 90.                          # longitude of periastron (in degrees)
                    batman_params.u = [self.u1, self.u2]           # limb darkening coefficients []
                    batman_params.limb_dark = "quadratic"          # limb darkening model

                    transit_model = batman.TransitModel(batman_params, time)      # initializes model
                    f_model = transit_model.light_curve(batman_params)            # calculates light curve
                    chi2_model = chi_square(flux, f_model, flux_err) / (len(time) - 4)         # chi^2 of model

                    chi2_aliases.append(chi2_model)

                    """
                    plt.plot(time, flux, "k.")
                    plt.plot(time, f_model, "r-", label=chi2_model)
                    plt.title(alias_per)
                    plt.legend()
                    plt.show()
                    """

                # check that no alias has a higher chi2 than the signal in question
                if not np.all( TCE["chi2_r"] < np.array(chi2_aliases) ):
                    TCE["FP"] == True

        return TCE_list



    def _vet_individual_transits_(self, TCE_list):
        """individual transit check defined by Zink+2020"""

        for TCE in TCE_list:

            if TCE["FP"] == False:

                time_masked = np.array([])
                f_masked = np.array([])
                ferr_masked = np.array([])

                # mask out-of-transit data and re-normalize each transit event
                for t0 in TCE["transit_times"]:
                    
                    # consider data points more than one and less than 3.5 expected full transit durations away from putative t0
                    trend_msk = ((self.time_raw > t0 + TCE["dur"]) & (self.time_raw < t0 + (3.5*TCE["dur"]))) | \
                                ((self.time_raw < t0 - TCE["dur"]) & (self.time_raw > t0 - (3.5*TCE["dur"])))

                    transit_msk = (self.time_raw > t0 - (3.5*TCE["dur"])) & (self.time_raw < t0 + (3.5*TCE["dur"]))

                    if np.sum(trend_msk) > 100:

                        # fit line to data immediately preceding and following transit
                        slope, intercept, _, _, _ = linregress(self.time_raw[trend_msk], self.f_raw[trend_msk])

                        make_line = lambda x : slope*x + intercept

                        time_masked = np.concatenate((time_masked, self.time_raw[transit_msk]))
                        f_masked = np.concatenate((f_masked, self.f_raw[transit_msk] / make_line(self.time_raw[transit_msk])))
                        ferr_masked = np.concatenate((ferr_masked, self.ferr_raw[transit_msk] / self.f_raw[transit_msk]))

                msk = np.isfinite(f_masked)
                time_masked = time_masked[msk]
                f_masked = f_masked[msk]
                ferr_masked = ferr_masked[msk]

        return TCE_list










    def _bls_search_(self):
        """Function to search for transits using BLS"""

        # max period is the minimum of 100 days, or 1/3 of the total duration of observations
        max_period = np.min([100., (self.time.max() - self.time.min()) / 3])
        self.period_grid = np.logspace(np.log10(0.5), np.log10(max_period), 12000)

        # split period grid into 12 sections
        for period_interval in tqdm(np.split(self.period_grid, 12), desc="   running BLS"):

            min_dur = self._estimate_duration_(period_interval.min()) / 15.
            max_dur = self._estimate_duration_(period_interval.max())
            dur_interval = np.linspace(min_dur, max_dur, 20)

            # choose smoothing filter based on transit duration
            time = self.time[np.isfinite(self.f2000)]
            flux = self.f2000[np.isfinite(self.f2000)]
            flux_err = self.f_err[np.isfinite(self.f2000)]

            if (max_dur * 1440) < 200:

                if (max_dur * 1440) < 50:
                    #print("   (selecting smoothing window of 500)")
                    time = self.time[np.isfinite(self.f500)]
                    flux = self.f500[np.isfinite(self.f500)]
                    flux_err = self.f_err[np.isfinite(self.f500)]

                elif (max_dur * 1440) < 100:
                    #print("   (selecting smoothing window of 1000)")
                    time = self.time[np.isfinite(self.f1000)]
                    flux = self.f1000[np.isfinite(self.f1000)]
                    flux_err = self.f_err[np.isfinite(self.f1000)]

            # do BLS
            bls = BoxLeastSquares(time, flux, dy=flux_err)
            bls_result = bls.power(period_interval, dur_interval)

            self.bls_rs = np.concatenate((self.bls_rs, bls_result.power))
            self.bls_durs = np.concatenate((self.bls_durs, bls_result.duration))
            self.bls_t0s = np.concatenate((self.bls_t0s, bls_result.transit_time))
            self.bls_depths = np.concatenate((self.bls_depths, bls_result.depth))

        # anonymous functions
        nll = lambda *args: -self._log_likelihood_min_(*args)
        chi_square = lambda O, E, sigma : np.sum( np.square((O - E) / sigma) )

        # find significant peaks
        TCEs = []   # empty list to store Treshold Crossing Events (SDE > 6)
        peak_inds = find_peaks(self.bls_rs, prominence=(self.bls_rs.max() - self.bls_rs.min()) / 10)[0]
        for i in tqdm(peak_inds, desc="   identifying TCEs"):

            # signal detection efficiency (SDE)
            SDE = (self.bls_rs[i] - np.mean(self.bls_rs)) / np.std(self.bls_rs)

            if SDE > 6.0:

                bls_stats = bls.compute_stats(self.period_grid[i], self.bls_durs[i], self.bls_t0s[i])

                # check if there are at leaast 3 transits
                if (len(bls_stats["transit_times"]) >= 3):

                    # choose smoothing filter based on transit duration
                    time = self.time[np.isfinite(self.f2000)]
                    flux = self.f2000[np.isfinite(self.f2000)]
                    flux_err = self.f_err[np.isfinite(self.f2000)]

                    if (self.bls_durs[i] * 1440) < 200:

                        if (self.bls_durs[i] * 1440) < 50:
                            #print("   (selecting smoothing window of 500)")
                            time = self.time[np.isfinite(self.f500)]
                            flux = self.f500[np.isfinite(self.f500)]
                            flux_err = self.f_err[np.isfinite(self.f500)]

                        elif (self.bls_durs[i] * 1440) < 100:
                            #print("   (selecting smoothing window of 1000)")
                            time = self.time[np.isfinite(self.f1000)]
                            flux = self.f1000[np.isfinite(self.f1000)]
                            flux_err = self.f_err[np.isfinite(self.f1000)]

                    # find max liklihood model
                    initial = np.array([self.bls_t0s[i], np.sqrt(self.bls_depths[i]), self._P_to_a_(self.period_grid[i]), 0.0])

                    soln = minimize(nll, initial, args=(self.period_grid[i], time, flux, flux_err))

                    t0_ml, rp_ml, a_ml, b_ml = soln.x

                    # compute best model
                    batman_params = batman.TransitParams()
                    inc = np.arccos(b_ml / a_ml) * (180 / np.pi)
                    batman_params.t0 = t0_ml                       # time of inferior conjunction
                    batman_params.per = self.period_grid[i]        # orbital period
                    batman_params.rp = rp_ml                       # planet radius (in units of stellar radii)
                    batman_params.a = a_ml                         # semi-major axis (in units of stellar radii)
                    batman_params.inc = inc                        # orbital inclination (in degrees)
                    batman_params.ecc = 0.                         # eccentricity
                    batman_params.w = 90.                          # longitude of periastron (in degrees)
                    batman_params.u = [self.u1, self.u2]           # limb darkening coefficients []
                    batman_params.limb_dark = "quadratic"          # limb darkening model

                    transit_model = batman.TransitModel(batman_params, time)      # initializes model
                    f_model = transit_model.light_curve(batman_params)            # calculates light curve
                    chi2_model = chi_square(flux, f_model, flux_err) / (len(time) - 4)         # chi^2 of model

                    TCEs.append(dict(sde=SDE,
                                        per=self.period_grid[i],
                                        dur=self.bls_durs[i],
                                        t0=t0_ml,
                                        dep=np.square(rp_ml),
                                        a=a_ml,
                                        b=b_ml,
                                        transit_times=bls_stats["transit_times"],
                                        chi2_r=chi2_model,
                                        FP=False
                                    )
                                )

        #plt.plot(self.period_grid, self.bls_rs)

        # sort TCEs by SDE value
        self.TCEs = sorted(TCEs, key=lambda d: d['sde'], reverse=True)

        print("   BLS recovered {} TCEs (SDE > 6).".format(len(self.TCEs)))

        return self.TCEs



    def _get_stellar_params_(self):
        """Function that loads stellar parameters from MAST and interpolates LD coeffs from table
        """

        # get stellar params from MAST
        tic_table = Catalogs.query_object(self.tic_id, catalog="TIC", radius=0.01)[0]

        self.R_star = tic_table["rad"] * u.R_sun
        if not np.isfinite(self.R_star.value):
            print("   Could not locate valid 'R_star'.")
            self.R_star = self._ask_user_("R_star [R_sun]", limits=(0, 1000)) * u.R_sun

        self.M_star = tic_table["mass"] * u.M_sun
        if not np.isfinite(self.M_star.value):
            print("   Could not locate valid 'M_star'.")
            self.M_star = self._ask_user_("M_star [M_sun]", limits=(0, 1000)) * u.M_sun

        self.logg = tic_table["logg"]
        if not 0 < self.logg < 5:
            print("   Could not locate valid 'logg'.")
            self.logg = self._ask_user_("logg", limits=(0, 5))

        self.Teff = tic_table["Teff"]
        if not 3500 < self.Teff < 50000:
            print("   Could not locate valid 'Teff'.")
            self.Teff = self._ask_user_("Teff [K]", limits=(3500, 50000))

        self.Fe_H = tic_table["MH"]
        if not -5 < self.Fe_H < 1:
            print("   Could not locate valid 'Fe_H'.")
            self.Fe_H = self._ask_user_("Fe_H", limits=(-5, 1))

        # get limb darkening coeffs
        Vizier.ROW_LIMIT = -1
        ld_table = Vizier.get_catalogs("J/A+A/600/A30/table25")[0]
        ld_table = ld_table[ld_table["xi"] == 2.0]
        ld_points = np.array([ld_table["logg"], ld_table["Teff"], ld_table["Z"]]).T
        ld_values = np.array([ld_table["aLSM"], ld_table["bLSM"]]).T
        ld_interpolator = LinearNDInterpolator(ld_points, ld_values)
        self.u1, self.u2 = ld_interpolator(self.logg, self.Teff, self.Fe_H)

        # print star info
        star_info = dict(zip(
            ["TIC ID", "R_star", "M_star", "logg", "Teff", "[Fe/H]", "u1", "u2"],
            [self.tic_id, self.R_star, self.M_star, self.logg, self.Teff, self.Fe_H, self.u1, self.u2]
            )
        )

        print("------------------------------")
        for i in star_info: print("   {}:\t{}".format(i, star_info[i]))
        print("------------------------------")

        return None



    def _ask_user_(self, variable_name, limits=None):
        """Function for user input

        :variable_name: string
        :limits: optional limits on numerical user input
        
        :returns: user's response (float)
        """
        
        while True:
            try:
                response = float(input("Please enter a valid '{}' value: ".format(variable_name)))
                if limits:
                    if (response < limits[0]) | (response > limits[1]):
                        raise ValueError
                break
                
            except ValueError:
                if limits:
                    print("'{}' must be a number between {} and {}.".format(variable_name, limits[0], limits[1]))
                else:
                    print("'{}' must be a number.".format(variable_name))

        return response



    def _estimate_duration_(self, period):
        """Function to compute approx transit duration given period.
        See Eq 19 of Winn (2010) in Exoplanets

        :period: unit days

        :returns: approx duration in days
        """
        
        rho_star = (self.M_star / self.R_star**3).value   # solar units
        P_yr = period / 365.   # years

        T = (13./24) * (P_yr)**(1/3) * (rho_star)**(-1/3)   # days

        return T



    def _P_to_a_(self, per):
        """Function to convert P to a using Kepler's third law"""

        M_star = self.M_star.to(u.kg).value
        R_star = self.R_star.to(u.m).value
        
        a = ((6.67e-11 * M_star) / (4 * np.pi**2) * (per * 86400.)**2)**(1/3)
        
        return a / R_star



    def _log_likelihood_min_(self, theta, period, x, y, yerr):
        """Function that returns log-likelihood with fixed period for likelihood maximization"""

        t0, rp, a, b = theta
            
        inc = np.arccos(b / a) * (180 / np.pi)
        
        params = batman.TransitParams()
        params.t0 = t0                       # time of inferior conjunction
        params.per = period                  # orbital period
        params.rp = rp                       # planet radius (in units of stellar radii)
        params.a = a                         # semi-major axis (in units of stellar radii)
        params.inc = inc                     # orbital inclination (in degrees)
        params.ecc = 0.                      # eccentricity
        params.w = 90.                       # longitude of periastron (in degrees)
        params.u = [self.u1, self.u2]        # limb darkening coefficients []
        params.limb_dark = "quadratic"       # limb darkening model
        
        model = batman.TransitModel(params, x)    # initializes model
        flux_m = model.light_curve(params)        # calculates light curve
            
        return -0.5 * np.sum( (y - flux_m)**2 / yerr**2 + np.log(2*np.pi*yerr**2) )



    def _plot_candidate_event_(self, period, model_params):
        """plot lc and best model from likelihood maximization"""

        t0, rp, a, b = model_params

        inc = np.arccos(b / a) * (180 / np.pi)

        params = batman.TransitParams()
        params.t0 = t0                       # time of inferior conjunction
        params.per = period                  # orbital period
        params.rp = rp                       # planet radius (in units of stellar radii)
        params.a = a                         # semi-major axis (in units of stellar radii)
        params.inc = inc                     # orbital inclination (in degrees)
        params.ecc = 0.                      # eccentricity
        params.w = 90.                       # longitude of periastron (in degrees)
        params.u = [self.u1, self.u2]        # limb darkening coefficients []
        params.limb_dark = "quadratic"       # limb darkening model
        
        model = batman.TransitModel(params, self.time)    # initializes model
        flux_m = model.light_curve(params)        # calculates light curve

        gridspec = dict(wspace=0.0, hspace=0.0, height_ratios=[2, 1])
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw=gridspec)   # set up figure

        # plot folded light curve and models
        ax = axes[0]
        ax.set_title("{} --- period = {} days".format(self.tic_id, period))
        t_fold = (self.time - t0 + 0.5 * period) % period - 0.5 * period
        m = np.isfinite(self.f2000)
        ax.errorbar(t_fold[m], self.f2000[m], yerr=self.f_err[m], fmt='k.', ms=1, alpha=0.1)   # plot data
        ax.plot(t_fold[m], self.f2000[m], 'k.', ms=1, alpha=0.5)

        ax.plot(np.sort(t_fold[m]), flux_m[m][np.argsort(t_fold[m])], 'r-', zorder=100)
        ax.set_ylabel("relative flux")

        ax = axes[1]

        ax.errorbar(t_fold[m], self.f2000[m]-flux_m[m], yerr=self.f_err[m], fmt='k.', ms=1, alpha=0.1)   # plot data
        ax.plot(t_fold[m], self.f2000[m]-flux_m[m], 'k.', ms=1, alpha=0.5)

        ax.axhline(0.0, color="r", zorder=100)
        
        ax.set_ylabel("residuals")
        ax.set_xlabel("phase (days)")
        ax.set_xlim([-period/2, period/2])

        # zoom in 
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.errorbar(t_fold[m], self.f2000[m], yerr=self.f_err[m], fmt='k.', ms=1, alpha=0.1)   # plot data
        ax.plot(t_fold[m], self.f2000[m], 'k.', ms=1, alpha=0.5)

        ax.plot(np.sort(t_fold[m]), flux_m[m][np.argsort(t_fold[m])], 'r-', zorder=100)
        ax.set_ylabel("relative flux")
        ax.set_xlabel("phase (days)")
        ax.set_title("zoom")

        dur = self._estimate_duration_(period)
        ax.set_xlim([-dur, dur])

        plt.show()




















































    def _log_likelihood_(self, theta, init_values, x, y, yerr):
        """Function that returns log-liklihood for MCMC"""

        t0, rp, a, b = theta
            
        inc = np.arccos(b / a) * (180 / np.pi)
        
        params = batman.TransitParams()
        params.t0 = t0                       # time of inferior conjunction
        params.per = init_values["p"]        # orbital period
        params.rp = rp                       # planet radius (in units of stellar radii)
        params.a = a                         # semi-major axis (in units of stellar radii)
        params.inc = inc                     # orbital inclination (in degrees)
        params.ecc = 0.                      # eccentricity
        params.w = 90.                       # longitude of periastron (in degrees)
        params.u = [self.u1, self.u2]        # limb darkening coefficients []
        params.limb_dark = "quadratic"       # limb darkening model
        
        model = batman.TransitModel(params, x)    # initializes model
        flux_m = model.light_curve(params)        # calculates light curve
            
        return -0.5 * np.sum( (y - flux_m)**2 / yerr**2 + np.log(2*np.pi*yerr**2) )



    def _log_prior_(self, theta, init_values):
        """Function that returns log-prior for MCMC"""

        t0, rp, a, b = theta
        
        if (init_values["t0"] - 0.083 < t0 < init_values["t0"] + 0.083) \
        and (0.0 < rp < 1.5 * init_values["rp_rs"]) \
        and (0.5 * init_values["a_rs"] < a < 2.0 * init_values["a_rs"]) \
        and (0.0 < b < 1.0):
            return 0.0
        
        return -np.inf



    def _log_probability_(self, theta, init_values, x, y, yerr):
        """Function that returns posterior log-probability for MCMC"""
        
        lp = self._log_prior_(theta, init_values)
        
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self._log_likelihood_(theta, init_values, x, y, yerr)



    def _run_mcmc_(self, init_values, t, f, ferr, show_plots=False):
        """Function to run MCMC"""

        np.random.seed(420)

        # initial state
        pos = np.array([init_values["t0"], init_values["rp_rs"], init_values["a_rs"], init_values["b"]]) \
                        + 5e-5 * np.random.randn(self.nwalkers, self.ndim)

        # run MCMC
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self._log_probability_, 
                                        args=(init_values, t, f, ferr)
                                        )
        sampler.run_mcmc(pos, self.nsteps, progress=False)

        
        flat_samples = sampler.get_chain(discard=self.nburn, flat=True)

        if show_plots:

            samples = sampler.get_chain()
            self._plot_mcmc_diagnostics_(samples, flat_samples)

        results_dict = dict(zip(["median", "lower", "upper"], np.quantile(flat_samples, [0.5, 0.16, 0.84], axis=0)))

        return results_dict






    def _plot_mcmc_diagnostics_(self, samples, flat_samples):
        """Function for plotting MCMC walkers and posterior distributions in a corner plot"""

        fig, axes = plt.subplots(self.ndim, figsize=(12, 10), sharex=True)

        labels = self.labels

        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[self.nburn:, :, i], alpha=0.3)
            ax.set_xlim(0, len(samples)-self.nburn)
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")

        _ = corner.corner(flat_samples,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True)

        plt.show()





























    def _plot_lc(self):
        """Function for plotting raw light curve"""

        fig, ax = plt.subplots(1, 1, figsize=(10,5))

        ax.axhline(1, c="k", lw=1)
        ax.errorbar(self.time, self.flux, yerr=self.flux_err, fmt="k.", alpha=0.5, elinewidth=1)
        ax.set_xlabel("Time - 245700 (BTJD days)")
        ax.set_ylabel("Normalized Flux")
        ax.set_title(self.tic_id)

        plt.show()





    def _plot_periodogram(self):
        """Function for plotting periodogram and folded light curve"""

        fig, axes = plt.subplots(2, 1, figsize=(10,10))

        # Plot the periodogram
        ax = axes[0]
        ax.axvline(np.log10(self.periodogram_stats["period"]), color="y", lw=5, alpha=0.5)
        for i in range(2,11):
            ax.axvline(np.log10(self.periodogram_stats["period"] / i), color="y", lw=2, ls='--', alpha=0.4)
            ax.axvline(np.log10(self.periodogram_stats["period"] * i), color="y", lw=2, ls='--', alpha=0.4)
        ax.plot(np.log10(self.periodogram.period), self.periodogram.power, "k")
        ax.annotate(
            "period = {0:.4f} d".format(self.periodogram_stats["period"]),
            (0, 1),
            xycoords="axes fraction",
            xytext=(5, -5),
            textcoords="offset points",
            va="top",
            ha="left",
            fontsize=12,
            )
        ax.set_ylabel("bls power")
        ax.set_yticks([])
        ax.set_xlim(np.log10(self.periodogram.period.min()), np.log10(self.periodogram.period.max()))
        ax.set_xlabel("log10(period)")

        # Plot the folded transit
        ax = axes[1]
        t_fold = (self.time - self.periodogram_stats["t0"] + 0.5 * self.periodogram_stats["period"]) % \
            self.periodogram_stats["period"] - 0.5 * self.periodogram_stats["period"]
        m = np.abs(t_fold) < 0.4
        ax.plot(t_fold[m], self.flux[m], "k.")

        # Overplot the phase binned light curve
        bins = np.linspace(-0.41, 0.41, 32)
        denom, _ = np.histogram(t_fold, bins)
        num, _ = np.histogram(t_fold, bins, weights=self.flux)
        denom[num == 0] = 1.0
        ax.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, color="y")

        ax.set_xlim(-0.3, 0.3)
        ax.set_ylabel("relative flux")
        _ = ax.set_xlabel("phase")

        plt.show()





    def _plot_diagnostics(self):
        """Function for plotting MCMC walkers and posterior distributions in a corner plot"""

        fig, axes = plt.subplots(self.ndim, figsize=(12, 10), sharex=True)

        labels = self.labels

        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(self.samples[self.nburn:, :, i], alpha=0.3)
            ax.set_xlim(0, len(self.samples)-self.nburn)
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")

        _ = corner.corner(self.flat_samples,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True)

        plt.show()





    def _plot_best_fit(self):
        """Function for plotting the folded light curve and best-fit model (50 random samples)"""

        gridspec = dict(wspace=0.0, hspace=0.0, height_ratios=[2, 1])
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw=gridspec)   # set up figure

        # plot folded light curve and models
        ax = axes[0]
        ax.set_title(self.tic_id)
        t_fold = (self.time - self.fit_results.loc["$T_0$"]["median"] + 0.5 * self.fit_results.loc["$P$"]["median"]) % \
            self.fit_results.loc["$P$"]["median"] - 0.5 * self.fit_results.loc["$P$"]["median"]
        m = np.abs(t_fold) < 0.4
        ax.errorbar(t_fold[m], self.flux[m], yerr=self.flux_err[m], fmt='k.', ms=1, alpha=0.1)   # plot data
        ax.plot(t_fold[m], self.flux[m], 'k.', ms=1, alpha=0.5)

        inds = np.random.randint(len(self.flat_samples), size=50)   # plot 50 random samples
        for ind in inds:
            
            t0, per, rp, a, b = self.flat_samples[ind]
                
            inc = np.arccos(b / a) * (180 / np.pi)
            
            params = batman.TransitParams()
            params.t0 = t0                       # time of inferior conjunction
            params.per = per                     # orbital period
            params.rp = rp                       # planet radius (in units of stellar radii)
            params.a = a                         # semi-major axis (in units of stellar radii)
            params.inc = inc                     # orbital inclination (in degrees)
            params.ecc = 0.                      # eccentricity
            params.w = 90.                       # longitude of periastron (in degrees)
            params.u = [self.u1, self.u2]        # limb darkening coefficients []
            params.limb_dark = "quadratic"       # limb darkening model
            
            model = batman.TransitModel(params, self.time)    # initializes model
            flux_m = model.light_curve(params)                # calculates light curve
       
            ax.plot(np.sort(t_fold[m]), flux_m[m][np.argsort(t_fold[m])], 'r-', lw=1, alpha=0.5)
            ax.set_ylabel("relative flux")

        # plot residuals for median "best-fit" model
        ax = axes[1]

        t0, per, rp, a, b, _, _, _ = self.fit_results["median"].values

        inc = np.arccos(b / a) * (180 / np.pi)

        params = batman.TransitParams()
        params.t0 = t0                    # time of inferior conjunction
        params.per = per                  # orbital period
        params.rp = rp                    # planet radius (in units of stellar radii)
        params.a = a                      # semi-major axis (in units of stellar radii)
        params.inc = inc                  # orbital inclination (in degrees)
        params.ecc = 0.                   # eccentricity
        params.w = 90.                    # longitude of periastron (in degrees)
        params.u = [self.u1, self.u2]     # limb darkening coefficients []
        params.limb_dark = "quadratic"    # limb darkening model

        model = batman.TransitModel(params, self.time)    # initializes model
        flux_m = model.light_curve(params)                # calculates light curve

        chi2, _ = chisquare(self.flux, flux_m)
        ax.text(0.1, 0.1, "$\\chi^2 = {:.3f}$".format(chi2), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        ax.errorbar(t_fold, self.flux-flux_m, yerr=self.flux_err, fmt='k.', ms=1, alpha=0.1)   # plot data
        ax.plot(t_fold, self.flux-flux_m, 'k.', ms=1, alpha=0.5)

        ax.axhline(0.0, color="r", alpha=0.75)
        
        ax.set_ylabel("median residuals")
        ax.set_xlabel("phase")
        ax.set_xlim([-0.3, 0.3])

        plt.show()





    def _plot_visual(self):
        """Function for plotting visualization of planet"""

        Rp_Rs = self.fit_results.iloc[2]["median"]
        a_Rs = self.fit_results.iloc[3]["median"]
        b = self.fit_results.iloc[4]["median"]
        R_p = self.fit_results.iloc[-2]["median"]


        fig, axes = plt.subplots(3, 1, figsize=(6,8))

        ax = axes[0]
        ax.set_aspect('equal')
        star = plt.Circle((0,0), radius=1.0, alpha=0.5, color="y")
        planet = plt.Circle((0,b), radius=Rp_Rs, alpha=1.0, color="k")
        ax.add_patch(star)
        ax.axhline(b, color="k", ls="--", lw=1)
        ax.plot([0, 0], [0, b], "k--", lw=1)
        ax.plot(0, 0, "k.")
        ax.add_patch(planet)
        ax.set_xlim([-1.2, 1.2])
        ax.axis("off")

        ax = axes[1]
        ax.set_aspect('equal')
        star = plt.Circle((0,0), radius=1.0, alpha=0.5, color="y")
        planet = plt.Circle((a_Rs,0), radius=Rp_Rs, alpha=1.0, color="k")
        ax.add_patch(star)
        ax.add_patch(planet)
        ax.plot([0,a_Rs], [0,0], "k--", lw=1)
        ax.set_xlim([-1.2, a_Rs+Rp_Rs+1.2])
        ax.axis("off")

        ax = axes[2]
        ax.set_aspect('equal')
        jupiter = plt.Circle((1,0), radius=1.0, alpha=0.5, color="r")
        planet = plt.Circle((R_p,0), radius=R_p, color="gray", alpha=0.3, ec='k', lw=2)
        ax.add_patch(jupiter)
        ax.add_patch(planet)
        ax.plot(1, 0, "k.")
        ax.plot([0,1], [0,0], "k--", lw=1)
        ax.text(0.5, 0.1, "$R_J$", size=12)
        ax.set_xlim([-0.2, 2*R_p+0.2])
        if R_p < 1:
            ax.set_xlim([-0.2, 2.2])
        ax.axis("off")

        plt.show()





        




    def _log_likelihood(self, theta, x, y, yerr):
        """Function that returns log-liklihood for MCMC"""

        t0, per, rp, a, b = theta
            
        inc = np.arccos(b / a) * (180 / np.pi)
        
        params = batman.TransitParams()
        params.t0 = t0                       # time of inferior conjunction
        params.per = per                     # orbital period
        params.rp = rp                       # planet radius (in units of stellar radii)
        params.a = a                         # semi-major axis (in units of stellar radii)
        params.inc = inc                     # orbital inclination (in degrees)
        params.ecc = 0.                      # eccentricity
        params.w = 90.                       # longitude of periastron (in degrees)
        params.u = [self.u1, self.u2]        # limb darkening coefficients []
        params.limb_dark = "quadratic"       # limb darkening model
        
        model = batman.TransitModel(params, x)    # initializes model
        flux_m = model.light_curve(params)        # calculates light curve
            
        return -0.5 * np.sum( (y - flux_m)**2 / yerr**2 + np.log(2*np.pi*yerr**2) )





    def _log_prior(self, theta):
        """Function that returns log-prior for MCMC"""

        t0, per, rp, a, b = theta
        
        if (self.periodogram_stats["t0"] * 0.9 < t0 < self.periodogram_stats["t0"] * 1.1) \
        and (self.periodogram_stats["period"] * 0.1 < per < self.periodogram_stats["period"] * 10.0) \
        and (np.sqrt(self.periodogram_stats["depth"]) * 0.1 < rp < np.sqrt(self.periodogram_stats["depth"]) * 10.0) \
        and (self._P_to_a(self.periodogram_stats["period"] * 0.1) < a < self._P_to_a(self.periodogram_stats["period"] * 10.0)) \
        and (0.0 < b < 1.0):
            return 0.0
        
        return -np.inf





    def _log_probability(self, theta, x, y, yerr):
        """Function that returns posterior log-probability for MCMC"""
        
        lp = self._log_prior(theta)
        
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self._log_likelihood(theta, x, y, yerr)











    def find_period(self, method="BLS", show_plot=True):
        """Function to find the orbital period of a planet"""

        period_grid = np.exp(np.linspace(np.log(1), np.log(15), 50000))

        if method == "BLS":

            bls = BoxLeastSquares(self.time, self.flux, dy=self.flux_err)
            self.periodogram = bls.power(period_grid, 0.1, oversample=20)


        max_power = np.argmax(self.periodogram.power)

        self.periodogram_stats = dict(
            period=self.periodogram.period[max_power],
            duration=self.periodogram.duration[max_power],
            t0=self.periodogram.transit_time[max_power],
            depth=self.periodogram.depth[max_power]
            )

        if show_plot:
            self._plot_periodogram()





    def execute_fit(self, show_plots=True):
        """Main function that executes other functions to complete an MCMC fit"""

        print("   Fetching stellar parameters...")
        self._get_stellar_params()

        print("   Downloading data...")
        self.download_data()

        print("   Estimating period...")
        self.find_period()


        print("   Running MCMC...")

        # initial state
        np.random.seed(42)
        pos = np.array([self.periodogram_stats["t0"], 
                        self.periodogram_stats["period"], 
                        np.sqrt(self.periodogram_stats["depth"]), 
                        self._P_to_a(self.periodogram_stats["period"]),
                         0.1]) + 5e-5 * np.random.randn(self.nwalkers, self.ndim)

        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self._log_probability, 
                                        args=(self.time, 
                                              self.flux,
                                              self.flux_err)
                                        )

        sampler.run_mcmc(pos, self.nsteps, progress=True);

        self.samples = sampler.get_chain()
        self.flat_samples = sampler.get_chain(discard=self.nburn, flat=True)

        if show_plots:
            self._plot_diagnostics()

        results_dict = dict(zip(["median", "lower", "upper"], np.quantile(self.flat_samples, [0.5, 0.16, 0.84], axis=0)))

        results_df = DataFrame(results_dict, index=self.labels)
        results_df["(+)"] = results_df["median"] - results_df["lower"]
        results_df["(-)"] = results_df["upper"] - results_df["median"]

        # add convenient units
        results_df = results_df.append(
            DataFrame(
                dict(zip(results_df.columns, 
                    np.array(
                        [(results_df.loc["$r_p/R_*$"].values * self.R_star).to(u.R_earth).value,
                        (results_df.loc["$r_p/R_*$"].values * self.R_star).to(u.R_jupiter).value,
                        (results_df.loc["$a/R_*$"].values * self.R_star).to(u.AU).value]).T
                    )),
                index=["$r_p$", "$r_p$", "$a$"]
                )
            )

        results_df["units"] = ["d", "d", "-", "-", "-", "R_Earth", "R_Jup", "AU"]


        self.fit_results = results_df

        if show_plots:
            self._plot_best_fit()
            self._plot_visual()

        print("   tater done.")

        return self.fit_results

        












