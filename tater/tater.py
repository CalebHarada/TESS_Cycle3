#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tess trAnsiT fittER (TATER)

TATER downloads TESS photometry and implements an MCMC
ensemble sampler to fit a BATMAN transit model to the data.
"""

# Futures
# [...]

# Built-in/Generic Imports
import os, sys, datetime
# [...]

# Libs
import batman, emcee, corner
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pandas import DataFrame
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize
from scipy.stats import linregress
from scipy.signal import find_peaks, medfilt
from scipy.special import erfcinv
from lightkurve import search_lightcurve
from transitleastsquares import transitleastsquares, transit_mask, cleaned_array
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
    """Main class"""


    def __init__(self, tic_id):
        """Initialization"""

        if (not isinstance(tic_id, int)) | (tic_id <= 0):
            raise ValueError("TIC ID ({}) must be a positive integer.".format(tic_id))

        # initialize some variables
        self.tic_id = "TIC {}".format(tic_id)
        self.labels = ["$P$", "$T_0$", "$r_p/R_*$", "$a/R_*$", "$b$"]

        # initialize
        self.TCEs = []
        self.planet_candidates = []
        self.lc_figure = None

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
        self.f = np.array([])
        self.f_err = np.array([])
        self.trend = np.array([])

        # initialize MCMC options
        self.ndim = len(self.labels)
        self.nwalkers = 15
        self.nsteps = 10000
        self.nburn = 5000



    def download_data(self, show_plot=False):
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

        fig, axes = plt.subplots(2, 1,
                                figsize=(10, 8),
                                sharex=True,
                                gridspec_kw=dict(wspace=0.0, hspace=0.0, height_ratios=[1, 1])
                            )

        axes[0].set_title(self.tic_id)
        axes[0].set_ylabel("Flux")
        axes[1].set_xlabel("Time (days)")
        axes[1].set_ylabel("Relative flux")

        # flatten !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for lc in tqdm(self.lc, desc="   flattening light curve"):
            
            time = lc.remove_nans().time.value
            flux = lc.remove_nans().flux.value
            flux_err = lc.remove_nans().flux_err.value
            
            self.time = np.concatenate((self.time, time))
            self.f_err = np.concatenate((self.f_err, flux_err/flux))

            flatten_lc, trend_lc = flatten(
                                    time,                 # Array of time values
                                    flux,                 # Array of flux values
                                    method='median',
                                    window_length=1.0,    # The length of the filter window in units of ``time``
                                    edge_cutoff=0.5,      # length (in units of time) to be cut off each edge.
                                    break_tolerance=0.5,  # Split into segments at breaks longer than that
                                    return_trend=True     # Return trend and flattened light curve
                                )
                
            self.f = np.concatenate((self.f, flatten_lc))
            self.trend = np.concatenate((self.trend, trend_lc))

            axes[0].scatter(time, flux, c='k', s=1, alpha=0.2)
            axes[0].plot(time, trend_lc, "b-", lw=2)
            axes[1].scatter(time, flatten_lc, c='k', s=1, alpha=0.2)
        
        self.lc_figure = fig
        if show_plot:
            plt.show()

        print("   done.")

        return None



    def find_planets(self):
        """Funtion to identify planets"""

        TCEs = self._tls_search_()

        print("   vetting TCEs...") # see Zink+2020

        # previous planet check
        TCEs = self._vet_previous_planets_(TCEs)
        
        for TCE in TCEs:
            if TCE.FP == False:
                self.planet_candidates.append(TCE)

        print(" ")
        print("   vetting recovered {} planet candidate(s) from {} TCE(s).".format(len(self.planet_candidates), len(self.TCEs)))

        return self.planet_candidates



    def fit_transits(self, save_results=True):
        """Function to perform MCMC fit"""

        if not len(self.planet_candidates) >= 1:
            raise ValueError("No planet candidates were found.")
            return None

        for i, candidate in enumerate(self.planet_candidates):

            print("   Running MCMC for planet candidate with $P = {:.6f}$ days (SDE={:.6f})".format(candidate.period, candidate.SDE))

            theta_0 = dict(per=candidate.period,
                t0=candidate.T0,
                rp_rs=candidate.rp_rs,
                a_rs=self._P_to_a_(candidate.period),
                b=0.1
                )

            intransit = transit_mask(self.time, candidate.period, 3*candidate.duration, candidate.T0)

            time, flux, flux_err = cleaned_array(self.time[intransit], self.f[intransit], self.f_err[intransit])

            planet_fit, walker_fig, corner_fig, best_fig, best_full_fig = self._execute_mcmc_(
                theta_0, time, flux, flux_err, show_plots=False)

            candidate.fit_results = planet_fit
            candidate.mcmc_fig = walker_fig
            candidate.corner_fig = corner_fig
            candidate.result_fig = best_fig
            candidate.result_full_fig = best_full_fig

            if save_results:
                print("   saving results...")
                self._save_results_pdf_(candidate, i)

        return None

        


    def _save_results_pdf_(self, planet_dict, planet_ind):
        """Function to save figures and fit results to one PDF file"""

        tic_no = self.tic_id[4:]

        save_to_path = "{}/outputs/{}".format(os.getcwd(), tic_no)

        if not os.path.isdir(save_to_path):
            os.mkdir(save_to_path)

        with PdfPages("{}/tater_report_{}_0{}.pdf".format(save_to_path, tic_no, planet_ind+1)) as pdf:

            pdf.savefig( self.lc_figure )
            pdf.savefig( planet_dict.periodogram_fig )
            pdf.savefig( planet_dict.model_fig )
            pdf.savefig( planet_dict.mcmc_fig )
            pdf.savefig( planet_dict.corner_fig )
            pdf.savefig( planet_dict.result_fig )
            pdf.savefig( planet_dict.result_full_fig )

            table_fig, _ = self._render_mpl_table_(planet_dict.fit_results)
            pdf.savefig( table_fig )
            
            d = pdf.infodict()
            d['Title'] = 'TATER Report {}'.format(self.tic_id)
            d['Author'] = 'Caleb K. Harada'
            d['Keywords'] = 'TESS, TLS, TATER'
            d['CreationDate'] = datetime.datetime.today()

        with open("{}/tater_summary_{}_0{}.txt".format(save_to_path, tic_no, planet_ind+1), "w") as text_file:
            text_file.write("{}\n \n".format(self.tic_id))
            for key in planet_dict.keys():
                text_file.write("{} : {} \n".format(key, planet_dict[key]))
            text_file.write("\n LaTeX table : \n")
            text_file.write(planet_dict.fit_results.to_latex())

        return None



    def _tls_search_(self, show_plots=False):
        """Function to run TLS planet search"""

        time = self.time
        flux = self.f
        flux_err = self.f_err

        period_max = np.ptp(time) / 3   # at least 3 transits
        if period_max > 60:
            period_max = 60

        intransit = np.zeros(len(time), dtype="bool")

        for i in range(6):

            time, flux, flux_err = cleaned_array(time[~intransit], flux[~intransit], flux_err[~intransit])

            #plt.figure()
            #plt.scatter(time, flux, c='k', s=1, alpha=0.2)
            #plt.show()

            tls = transitleastsquares(time, flux, flux_err)
            tls_results = tls.power(
                                    R_star=self.R_star.value,
                                    R_star_min=self.R_star.value - 0.3,
                                    R_star_max=self.R_star.value + 0.3,
                                    M_star=self.M_star.value,
                                    M_star_min=self.M_star.value - 0.3,
                                    M_star_max=self.M_star.value + 0.3,
                                    u=[self.u1, self.u2],
                                    period_max=period_max,
                                    period_min=0.8
                                )

            fig1, ax1 = plt.subplots(1, 1, figsize=(12,8))
            ax1.axhline(8.0, ls="--", c="r", alpha=0.6)
            ax1.axvline(tls_results.period, alpha=0.2, lw=6, c="b")
            for i in range(2, 15):
                ax1.axvline(tls_results.period * i, alpha=0.2, lw=1, c="b", ls='--')
                ax1.axvline(tls_results.period / i, alpha=0.2, lw=1, c="b", ls='--')
            ax1.plot(tls_results.periods, tls_results.power, 'k-', lw=1)
            ax1.set_xlim([tls_results.periods.min(), tls_results.periods.max()])
            ax1.set_xlabel("period (days)")
            ax1.set_ylabel("power")
            ax1.set_title("Peak at {:.6f} days".format(tls_results.period))

            fig2, ax2 = plt.subplots(1, 1, figsize=(12,8))
            ax2.scatter(tls_results.folded_phase, tls_results.folded_y, s=1, c='k')
            ax2.plot(tls_results.model_folded_phase, tls_results.model_folded_model)
            ax2.set_title("Preliminary transit model")
            ax2.set_xlim([0.45, 0.55])
            ax2.set_xlabel("phase")
            ax2.set_ylabel("relative flux")

            if show_plots:
                plt.show()

            if tls_results.SDE <= 8.0:
                print("   No TCEs found above SDE=8.0.")
                print(" ")
                break

            print("   TCE found at $P = {:.6f}$ days with SDE of {:.6f}".format(tls_results.period, tls_results.SDE))
            print(" ")

            # add False Positive keyword and plots to result
            tls_results.periodogram_fig = fig1
            tls_results.model_fig = fig2
            tls_results.FP = False

            self.TCEs.append(tls_results)
            intransit = transit_mask(time, tls_results.period, 2.5*tls_results.duration, tls_results.T0)


        return self.TCEs



    def _vet_previous_planets_(self, TCE_list):
        """previous planet check defined by Zink+2020"""

        for i in tqdm(range(len(TCE_list)), desc="   checking previous signals"):
            for j in range(0, i):
        
                P_A, P_B = np.sort((TCE_list[i].period, TCE_list[j].period))
                delta_P = (P_B - P_A) / P_A
                sigma_P = np.sqrt(2) * erfcinv(np.abs(delta_P - round(delta_P)))

                delta_t0 = np.abs(TCE_list[i].T0 - TCE_list[j].T0) / TCE_list[i].duration

                delta_SE1 = np.abs(TCE_list[i].T0 - TCE_list[j].T0 + TCE_list[i].period/2) / TCE_list[i].duration

                delta_SE2 = np.abs(TCE_list[i].T0 - TCE_list[j].T0 - TCE_list[i].period/2) / TCE_list[i].duration

                if (sigma_P > 2.0) & (TCE_list[j].FP == True):
                    TCE_list[i].FP = True
                    break

                if (sigma_P > 2.0) & (delta_t0 < 1):
                    TCE_list[i].FP = True
                    break

                elif (sigma_P > 2.0) & ((delta_SE1 < 1) | (delta_SE2 < 1)):
                    TCE_list[i].FP = True
                    break

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



    def _log_likelihood_(self, theta, x, y, yerr):
        """Function that returns log-liklihood for MCMC"""

        per, t0, rp, a, b = theta
            
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



    def _log_prior_(self, theta, theta_0):
        """Function that returns log-prior for MCMC"""

        per, t0, rp, a, b = theta
        
        if (theta_0["per"] - 0.1 < per < theta_0["per"] + 0.1) \
        and (theta_0["t0"] - 0.083 < t0 < theta_0["t0"] + 0.083) \
        and (0.0 < rp < 1.5 * theta_0["rp_rs"]) \
        and (0.5 * theta_0["a_rs"] < a < 2.0 * theta_0["a_rs"]) \
        and (0.0 < b < 1.0):
            return 0.0
        
        return -np.inf



    def _log_probability_(self, theta, theta_0, x, y, yerr):
        """Function that returns posterior log-probability for MCMC"""
        
        lp = self._log_prior_(theta, theta_0)
        
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self._log_likelihood_(theta, x, y, yerr)



    def _execute_mcmc_(self, theta_0, t, f, f_err, show_plots=False):
        """Function to run MCMC"""

        np.random.seed(420)

        # initial state
        pos = np.array([theta_0["per"], theta_0["t0"], theta_0["rp_rs"], theta_0["a_rs"], theta_0["b"]]) \
                        + 5e-5 * np.random.randn(self.nwalkers, self.ndim)

        # run MCMC
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self._log_probability_, 
                                        args=(theta_0, t, f, f_err)
                                        )
        sampler.run_mcmc(pos, self.nsteps, progress=True)

        
        flat_samples = sampler.get_chain(discard=self.nburn, flat=True)
        samples = sampler.get_chain()

        if show_plots:
            walker_fig, corner_fig = self._plot_mcmc_diagnostics_(samples, flat_samples, show_plot=True)
            best_fig, best_full_fig = self._plot_best_fit_(t, f, f_err, flat_samples, show_plot=True)
        else:
            walker_fig, corner_fig = self._plot_mcmc_diagnostics_(samples, flat_samples)
            best_fig, best_full_fig = self._plot_best_fit_(t, f, f_err, flat_samples)

        # package up fit results
        results_dict = dict(zip(["median", "lower", "upper"], np.quantile(flat_samples, [0.5, 0.16, 0.84], axis=0)))
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

        return results_df, walker_fig, corner_fig, best_fig, best_full_fig



    def _plot_mcmc_diagnostics_(self, samples, flat_samples, show_plot=False):
        """Function for plotting MCMC walkers and posterior distributions in a corner plot"""

        walker_fig, axes = plt.subplots(self.ndim, figsize=(12, 10), sharex=True)

        labels = self.labels

        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[self.nburn:, :, i], alpha=0.3)
            ax.set_xlim(0, len(samples)-self.nburn)
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")

        corner_fig = corner.corner(flat_samples,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True)

        if show_plot:
            plt.show()

        return walker_fig, corner_fig



    def _plot_best_fit_(self, t, f, f_err, flat_samples, show_plot=False):
        """Function for plotting the folded light curve and best-fit model (50 random samples)"""

        gridspec = dict(wspace=0.0, hspace=0.0, height_ratios=[2, 1])
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw=gridspec)   # set up figure

        p_best, t0_best, rp_best, a_best, b_best = np.quantile(flat_samples, 0.5, axis=0)

        # plot folded light curve and models
        ax = axes[0]
        t_fold = (t - t0_best + 0.5 * p_best) % p_best - 0.5 * p_best
        ax.errorbar(t_fold, f, yerr=f_err, fmt='k.', ms=1, alpha=0.1)   # plot data
        ax.plot(t_fold, f, 'k.', ms=1, alpha=0.5)
        ax.set_ylabel("relative flux")
        
        inds = np.random.randint(len(flat_samples), size=100)   # plot 100 random samples
        t_model = np.sort(t_fold)
        for ind in inds:
            
            per, t0, rp, a, b = flat_samples[ind]
                
            inc = np.arccos(b / a) * (180 / np.pi)
            
            params = batman.TransitParams()
            params.t0 = 0                        # time of inferior conjunction
            params.per = per                     # orbital period
            params.rp = rp                       # planet radius (in units of stellar radii)
            params.a = a                         # semi-major axis (in units of stellar radii)
            params.inc = inc                     # orbital inclination (in degrees)
            params.ecc = 0.                      # eccentricity
            params.w = 90.                       # longitude of periastron (in degrees)
            params.u = [self.u1, self.u2]        # limb darkening coefficients []
            params.limb_dark = "quadratic"       # limb darkening model
            
            model = batman.TransitModel(params, t_model)      # initializes model
            flux_model = model.light_curve(params)            # calculates light curve
       
            ax.plot(t_model, flux_model, 'b-', lw=1, alpha=0.5)

        # plot residuals for median "best-fit" model
        ax = axes[1]

        inc = np.arccos(b_best / a_best) * (180 / np.pi)

        params = batman.TransitParams()
        params.t0 = t0_best                # time of inferior conjunction
        params.per = p_best                # orbital period
        params.rp = rp_best                # planet radius (in units of stellar radii)
        params.a = a_best                  # semi-major axis (in units of stellar radii)
        params.inc = inc                   # orbital inclination (in degrees)
        params.ecc = 0.                    # eccentricity
        params.w = 90.                     # longitude of periastron (in degrees)
        params.u = [self.u1, self.u2]      # limb darkening coefficients []
        params.limb_dark = "quadratic"     # limb darkening model

        model = batman.TransitModel(params, t)    # initializes model
        flux_m = model.light_curve(params)        # calculates light curve

        ax.errorbar(t_fold, f-flux_m, yerr=f_err, fmt='k.', ms=1, alpha=0.1)   # plot data
        ax.plot(t_fold, f-flux_m, 'k.', ms=1, alpha=0.5)

        ax.set_xlim([t_fold.min(), t_fold.max()])

        ax.axhline(0.0, color="b", alpha=0.75)
        
        ax.set_ylabel("median residuals")
        ax.set_xlabel("phase (days)")

        # full light curve with model
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        ax2.scatter(self.time, self.f, c='k', s=1, alpha=0.2)
        model = batman.TransitModel(params, self.time)    # initializes model
        flux_m = model.light_curve(params)                # calculates light curve
        ax2.plot(self.time, flux_m, "b-", alpha=0.75)
        ax2.set_ylabel("relative flux")
        ax2.set_xlabel("time (days)")

        if show_plot:
            plt.show()

        return fig, fig2



    def _render_mpl_table_(self, data, col_width=3.0, row_height=0.625,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
        """Function to convert pandas df table to a figure
        see https://stackoverflow.com/questions/19726663/how-to-save-the-pandas-dataframe-series-data-as-a-figure"""

        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')
        mpl_table = ax.table(cellText=data.values, bbox=bbox,
                             rowLabels=data.index, colLabels=data.columns, **kwargs)
        mpl_table.auto_set_font_size(True)

        for k, cell in mpl_table._cells.items():
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

        return ax.get_figure(), ax








