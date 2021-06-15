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
from scipy.stats import chisquare
from lightkurve import search_lightcurve
from astropy.timeseries import BoxLeastSquares
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier

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

        self.tic_id = "TIC {}".format(tic_id)
        self.labels = ["$T_0$", "$P$", "$r_p/R_*$", "$a/R_*$", "$b$"]
        
        # initialize data arrays
        self.time = None
        self.flux = None
        self.flux_err = None

        # initialize MCMC options
        self.ndim = len(self.labels)
        self.nwalkers = 20
        self.nsteps = 3000
        self.nburn = 1500





    def _ask_user(self, variable_name, limits=None):
        """Function for user input"""
        
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

        chi2 = chisquare(self.flux, flux_m)[0]
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





    def _P_to_a(self, per):
        """Function to convert P to a using Kepler's third law"""

        M_star = self.M_star.to(u.kg).value
        R_star = self.R_star.to(u.m).value
        
        a = ((6.67e-11 * M_star) / (4 * np.pi**2) * (per * 86400)**2)**(1/3)
        
        return a / R_star
        




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





    def get_stellar_params(self):
        """Function that loads stellar parameters from MAST and interpolates LD coeffs from table"""

        # get stellar params from MAST
        tic_table = Catalogs.query_object(self.tic_id, catalog="TIC", radius=0.01)[0]

        self.R_star = tic_table["rad"] * u.R_sun
        if not np.isfinite(self.R_star.value):
            print("   Could not find valid 'R_star'.")
            self.R_star = self._ask_user("R_star [R_sun]", limits=(0, 1000)) * u.R_sun

        self.M_star = tic_table["mass"] * u.M_sun
        if not np.isfinite(self.M_star.value):
            print("   Could not find valid 'M_star'.")
            self.M_star = self._ask_user("M_star [M_sun]", limits=(0, 1000)) * u.M_sun

        self.logg = tic_table["logg"]
        if not 0 < self.logg < 5:
            print("   Could not find valid 'logg'.")
            self.logg = self._ask_user("logg", limits=(0, 5))

        self.Teff = tic_table["Teff"]
        if not 3500 < self.Teff < 50000:
            print("   Could not find valid 'Teff'.")
            self.Teff = self._ask_user("Teff [K]", limits=(3500, 50000))

        self.Fe_H = tic_table["MH"]
        if not -5 < self.Fe_H < 1:
            print("   Could not find valid 'Fe_H'.")
            self.Fe_H = self._ask_user("Fe_H", limits=(-5, 1))


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





    def download_data(self, show_plot=True):
        """Function to download raw light curve data"""

        search_result = search_lightcurve(self.tic_id, mission="TESS", author="SPOC")
        lc_all = search_result.download_all().stitch().remove_nans()


        # SHOULD CREATE NEW FUNCTION FOR TRANSIT SEARCH/FLATTENING
        # This function should simply download data as 'self.lc' or something
        # e.g., 'def transit_search(self, ...): ...'

        bls = lc_all.to_periodogram("bls")
        msk = lc_all.create_transit_mask(transit_time=bls.transit_time_at_max_power.value,
                                            period=bls.period_at_max_power.value,
                                            duration=bls.duration_at_max_power.value
                                        )

        lc_all = lc_all.flatten(mask=msk)

        self.time = lc_all.time.value
        self.flux = lc_all.flux.value
        self.flux_err = lc_all.flux_err.value

        if show_plot:
            self._plot_lc()





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
        self.get_stellar_params()

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

        












