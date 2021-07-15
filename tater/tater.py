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


