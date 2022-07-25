#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tess trAnsiT fittER (TATER)

TATER downloads TESS photometry and implements an MCMC
ensemble sampler to fit a BATMAN transit model to the data.
"""

# Futures
# [...]

# Built-in/Generic Imports
import os, datetime
# [...]

# Libs
import batman, emcee, corner
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import get_cmap
from pandas import DataFrame
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import curve_fit
from scipy.stats import linregress, binned_statistic
from scipy.special import erfcinv
from lightkurve import search_lightcurve
from transitleastsquares import transitleastsquares, transit_mask, cleaned_array
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
from wotan import flatten
from tqdm import tqdm
import pandas as pd
from os.path import exists


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

# ignore plotting warning
plt.rcParams.update({'figure.max_open_warning': 0})


# </header>
# Code begins here.

class TransitFitter(object):
    """Main class for TATER"""

    def __init__(self, tic_id, auto_params=False, ask_user=False, assume_solar=False):
        """Initialization

        @param tic_id: TIC ID number
        @type tic_id: int

        @param auto_params: load stellar parameters automatically from MAST and interpolate LD coeffs
        @type auto_params: bool (optional; default=False)

        @param ask_user: ask user to input info missing from MAST (ignored if auto_params=False)
        @type ask_user: bool (default=False)

        @param assume_solar: fill in solar estimates for missing MAST values (ignored if auto_params=False or ask_user=True)
        @type assume_solar: bool (default=False)
        """

        # check user input
        if (not isinstance(tic_id, int)) | (tic_id <= 0):
            raise ValueError("TIC ID ({}) must be a positive integer.".format(tic_id))

        # initialize variables
        self.tic_id = "TIC {}".format(tic_id)
        self.labels = ["$P$", "$T_0$", "$r_p/R_*$", "$a/R_*$", "$b$"]

        # initialize
        self.TCEs = []
        self.planet_candidates = []
        self.lc_figure = None

        # initialize stellar params
        self.R_star = None
        self.R_star_uncert = None
        self.M_star = None
        self.M_star_uncert = None
        self.logg = None
        self.Teff = None
        self.Fe_H = None
        self.u1 = None
        self.u2 = None

        # option to automatically get stellar params from MAST
        self.missing = []
        if auto_params:
            print("   retrieving stellar parameters...")
            missing = self._get_stellar_params_(
                ask_user=ask_user, assume_solar=assume_solar)
            # check and store whether MAST is missing info
            self.missing = missing

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
        self.nwalkers = 100
        self.nsteps = 350
        self.nburn = 250

        self.injection_recovery_results = None

    def download_data(self, window_size=3.0, n_sectors=None, show_plot=False, save_lc=True):
        """Function to download data with Lightkurve and flatten raw light curve

        @param window_size: median smoothing filter window size in days
        @type window_size: float (optional; default=3.0)

        @param n_sectors: number of TESS sectors to load (default is all)
        @type n_sectors: int (optional; default=None)

        @param show_plot: show plot of raw and flattened LC
        @type show_plot: bool (optional; default=False)

        @return None

        """

        # download SPOC data from all sectors at 120s exp time
        print("   acquiring TESS data...")
        search_result = search_lightcurve(self.tic_id, mission="TESS", author="SPOC", exptime=120)
        #Catch case where no data are nsec_foundif nsec_found == 0:
        if len(search_result) == 0:
            return 0

        # in rare cases (e.g. TIC 96246348), the wrong targets are included in search output
        if sum(search_result.target_name.data != self.tic_id[len('TIC '):]) > 0:
                print("Additional targets found by lightkurve")
                print(search_result)
                print("Removing additional targets")
                search_result = search_result[np.where(search_result.target_name.data == self.tic_id[len('TIC '):])]
                if len(search_result) == 0:
                    return 0
        nsec_found = len(search_result)

        self.lc = search_result.download_all(quality_bitmask="default")
        self.time_raw, self.f_raw, self.ferr_raw = np.concatenate([(sector.remove_nans().time.value,
                                                                    sector.remove_nans().flux.value,
                                                                    sector.remove_nans().flux_err.value
                                                                    ) for sector in self.lc], axis=1)

        # initialize light curve figure
        fig, axes = plt.subplots(2, 1,
                                 figsize=(10, 8),
                                 sharex=True,
                                 gridspec_kw=dict(wspace=0.0, hspace=0.0, height_ratios=[1, 1])
                                 )
        axes[0].set_title(self.tic_id)
        axes[0].set_ylabel("Flux")
        axes[1].set_xlabel("Time (days)")
        axes[1].set_ylabel("Relativeflux ")

        # only use first n_sectors of data (optional)
        lightcurves = self.lc[:n_sectors] if type(n_sectors) == int else self.lc

        # flatten and concatenate light curve
        for lc in tqdm(lightcurves, desc="   flattening light curve"):
            # clean data arrays
            time = lc.remove_nans().time.value
            flux = lc.remove_nans().flux.value
            flux_err = lc.remove_nans().flux_err.value

            self.time = np.concatenate((self.time, time))
            self.f_err = np.concatenate((self.f_err, flux_err / flux))

            # flatten flux with running median filter (Wotan package)
            flatten_lc, trend_lc = flatten(
                time,                                   # Array of time values
                flux,                                   # Array of flux values
                method='median',                        # median filter
                window_length=window_size,              # The length of the filter window in units of ``time``
                edge_cutoff=0.5,                        # length (in units of time) to be cut off each edge.
                break_tolerance=0.5,                    # Split into segments at breaks longer than that
                return_trend=True                       # Return trend and flattened light curve
            )

            self.f = np.concatenate((self.f, flatten_lc))
            self.trend = np.concatenate((self.trend, trend_lc))

            # plot data, trend, and flattened lc
            axes[0].scatter(time, flux, c='k', s=1, alpha=0.2, rasterized=True)
            axes[0].plot(time, trend_lc, "b-", lw=2)
            axes[1].scatter(time, flatten_lc, c='k', s=1, alpha=0.2, rasterized=True)

        # show plot if option is true
        if show_plot:
            plt.ion(), plt.show(), plt.pause(0.001)

        # save figure
        self.lc_figure = fig

        plt.close()

        print("   done.")

        return nsec_found

    def use_local_data(self, ddir = '/Users/courtney/Documents/data/toi_paper_data/triceratops_tess_lightcurves/', show_plot=False):
            """Function to load local lightcurve data

            @param show_plot: show plot of raw and flattened LC
            @type show_plot: bool (optional; default=False)

            @return None

            """

            # load lightcurve
            ticnumber = self.tic_id[4:]
            lcfile = ddir + str(ticnumber)+'_lightcurve_detrended.csv'
            print('looking for file ', lcfile)
            if not exists(lcfile):
                print('no file found for '+str(self.tic_id))
                return 0 #0 = no sectors of data found
            lc = pd.read_csv(lcfile)

            #Load ld_values
            #Converting to np arrays is important because otherwise
            #they will be loaded as series and the indexing won't work
            #properly when masking transits during TLS search
            self.time_raw = np.array(lc.time)
            self.time = np.array(lc.time)
            self.f_raw = np.array(lc.initial_flux)
            self.ferr_raw = np.array(lc.err)
            self.f_err = np.array(lc.err)
            self.flux = np.array(lc.flux)
            self.f = np.array(lc.flux)
            self.trend = np.array(lc.trend)

            #Make lightcurve figures
            fig, axes = plt.subplots(2, 1,
                                    figsize=(10, 8),
                                    sharex=True,
                                    gridspec_kw=dict(wspace=0.0, hspace=0.0, height_ratios=[1, 1])
                                    )
            axes[0].set_title(self.tic_id)
            axes[0].set_ylabel("Flux")
            axes[1].set_xlabel("Time (days)")
            axes[1].set_ylabel("Relativeflux ")

            # plot data, trend, and flattened lc
            axes[0].scatter(self.time, self.f_raw, c='k', s=1, alpha=0.2, rasterized=True)
            axes[0].plot(self.time, self.trend, "b-", lw=2)
            axes[1].scatter(self.time, self.flux, c='k', s=1, alpha=0.2, rasterized=True)

            # show plot if option is true
            if show_plot:
                plt.ion(), plt.show(), plt.pause(0.001)

            # save & close figure
            self.lc_figure = fig
            plt.close()

            print("   done.")


            return 999 #just don't return 0


    def find_planets(self, time=None, flux=None, flux_err=None,
        max_iterations=7, tce_threshold=8.0,
        period_min=0.5, period_max=100, show_plots=False, save_results=True):
        """Function to identify transits using TLS, then perform model fit with MCMC

        @param time: time array
        @type time: numpy array (optional; default=None)

        @param flux: flux array
        @type flux: numpy array (optional; default=None)

        @param flux_err: flux uncertainty array
        @type flux_err: numpy array (optional; default=None)

        @param max_iterations: maximum number of search iterations if SDE threshold is never reached
        @type max_iterations: int (optional; default=7)

        @param tce_threshold: Minimum Signal Detection Efficiency (SDE) that counts as a Threshold Crossing Event (TCE)
        @type tce_threshold: float (optional; default=8.0)

        @param period_min: minimum orbital period for TLS to explore
        @type period_min: float (optional; default=0.8)

        @param period_max: maximum orbital period for TLS to explore
        @type period_max: float (optional; default=100)

        @param show_plots: show plots of periodogram and best TLS model
        @type show_plots: bool (optional; default=False)

        @param save_results: save results of fit
        @type save_results: bool (optional; default=True)

        @return self.planet_candidates: list of planet candidates

        """


        """
        ###### TEST INJECTION AND RECOVERY ##################
        print("    Running test injection/recovery...")
        test_time, test_flux, test_flux_err = cleaned_array(self.time, self.f, self.f_err)
        test_per = 28.7
        test_t0 = self.time[0] + test_per
        test_rp = 0.02
        test_a = 25.0
        test_inc = 90.
        test_injected_lc, ground_truth = self._inject_(test_time, test_flux, test_t0, test_per, test_rp, test_a, test_inc)
        test_recover = self._recover_(test_time, test_injected_lc, test_flux_err, test_t0, test_per, ground_truth,
                                      raw_flux=False)
        print("    Test injection/recovery done.\n")
        print("    Recovered injected planet? {} \n".format(test_recover))
        ###### END TEST INJECTION AND RECOVERY ##################
        """



        TCEs,_ = self._tls_search_(max_iterations, tce_threshold,
                                 time=time, flux=flux, flux_err=flux_err,
                                 period_min=period_min, period_max=period_max,
                                 show_plots=True) if show_plots \
            else self._tls_search_(max_iterations, tce_threshold,
                                   time=time, flux=flux, flux_err=flux_err,
                                   period_min=period_min, period_max=period_max)

        self.TCEs = TCEs

        # check whether any planets were found
        if not len(TCEs) >= 1:
            raise ValueError("No TCEs were found.")

        # do MCMC fit for each planet in candidate list
        for i, TCE in enumerate(TCEs):

            print("   Running MCMC for TCE with $P = {:.6f}$ days (SDE={:.6f})".format(TCE.period, TCE.SDE))

            # initialize parameters
            theta_0 = dict(per=TCE.period,
                           t0=TCE.T0,
                           rp_rs=TCE.rp_rs,
                           a_rs=max(1, self._P_to_a_(TCE.period)),
                           b=0.1
                           )

            # mask previous transits
            intransit = np.zeros(len(self.time_raw), dtype=bool)
            if i > 0:
                for previous_tce in TCEs[0:i]:
                    intransit += transit_mask(self.time_raw, previous_tce.period,
                                             2.5 * previous_tce.duration,
                                             previous_tce.T0)

            # Analyzing all the data is computationally inefficient. Therefore we will only fit a transit to
            # the data in and around the transits.
            time, flux, flux_err = self._get_intransit_flux_(TCE, msk=intransit)

            # Need at least one transit of data to do MCMC
            cadence = 2. / 1440  # 2-min cadence
            if not len(time) >= TCE.duration / cadence:
                TCE.FP = "Insufficient data"
                print("    Insufficient in-transit data to execute MCMC. Continuing... \n")
                continue

            # run the MCMC (option to show plots)
            if show_plots:
                planet_fit, walker_fig, corner_fig, best_fig, best_full_fig = self._execute_mcmc_(
                    theta_0, time, flux, flux_err, show_plots=True)
            else:
                planet_fit, walker_fig, corner_fig, best_fig, best_full_fig = self._execute_mcmc_(
                    theta_0, time, flux, flux_err)

            # save figures to candidate object dictionary
            TCE.fit_results = planet_fit
            TCE.mcmc_fig = walker_fig
            TCE.corner_fig = corner_fig
            TCE.result_fig = best_fig
            TCE.result_full_fig = best_full_fig

            plt.close()

            # save results of fit
            if save_results:
                print("   saving results...")
                self._save_tce_pdf_(TCE, i)
                print("   MCMC COMPLETE.")
                print(" ")

        return self.TCEs


    def vet_TCEs(self, save_results=True):
        """Function to perform vetting of TCEs (to promote to candidate)

        @param save_results: save results of fit
        @type save_results: bool (optional; default=True)

        @return None

        """

        print("   Vetting TCEs...")  # ref Zink+2020

        # get ephemerides
        self._get_ephems_(self.TCEs)

        # previous planet check
        self.TCEs = self._vet_previous_planets_(self.TCEs)

        # line test
        self.TCEs = self._vet_line_test_(self.TCEs)

        # odd even test
        self.TCEs = self._vet_odd_even_(self.TCEs)

        # other vetting functions here?
        # ............
        # ............


        for i, TCE in enumerate(self.TCEs):
            if TCE.FP == "No":
                self.planet_candidates.append(TCE)
            if save_results & (TCE.FP != "Insufficient data"):
                self._save_vetting_pdf_(TCE, i)

        print("   Vetting recovered {} planet candidate(s) from {} TCE(s).".format(len(self.planet_candidates),
                                                                                   len(self.TCEs)))
        print("   VETTING COMPLETE.")

        # save summary figure of vetted candidates
        """
        if save_results & (len(self.planet_candidates) > 0):
            # self._save_results_image_()
            self._save_system_overview_()
        """

        print(" ")
        print("   TATER DONE.")

        return self.planet_candidates


    def _save_tce_pdf_(self, planet_dict, planet_ind):
        """Helper function to save TCE figures and fit results to one PDF file

        @param planet_dict: TLS result/planet parameter dictionary
        @type planet_dict: dict

        @param planet_ind: index for current planet
        @type planet_ind: int

        @return None

        """

        # get TIC number
        tic_no = self.tic_id[4:]

        # make output directory
        outputs_directory = "{}/outputs".format(os.getcwd())
        if not os.path.isdir(outputs_directory):
            os.mkdir(outputs_directory)

        # directory to save results to
        save_to_path = "{}/outputs/{}".format(os.getcwd(), tic_no)
        if not os.path.isdir(save_to_path):
            os.mkdir(save_to_path)

        # use PdfPages to make a new PDF document
        with PdfPages("{}/tater_report_{}_0{}.pdf".format(save_to_path, tic_no, planet_ind + 1)) as pdf:

            # save all the figures to the PDF
            pdf.savefig(self.lc_figure)
            pdf.savefig(planet_dict.periodogram_fig)
            pdf.savefig(planet_dict.model_fig)
            pdf.savefig(planet_dict.mcmc_fig)
            pdf.savefig(planet_dict.corner_fig)
            if planet_dict.result_fig is not None:
                pdf.savefig(planet_dict.result_fig)
            if planet_dict.result_full_fig is not None:
                pdf.savefig(planet_dict.result_full_fig)

            # make a table with the fit results and save it to the PDF
            table_fig, _ = self._render_mpl_table_(planet_dict.fit_results)
            pdf.savefig(table_fig)

            # random pdf stuff
            d = pdf.infodict()
            d['Title'] = 'TATER Report {}'.format(self.tic_id)
            d['Author'] = 'TATER'
            d['Keywords'] = 'TESS, TLS, TATER'
            d['CreationDate'] = datetime.datetime.today()

        # save summary of fit results to a txt file (also in LaTeX format!)
        with open("{}/tater_report_{}_0{}.txt".format(save_to_path, tic_no, planet_ind + 1), "w") as text_file:

            text_file.write("{}\n".format(self.tic_id))
            text_file.write("TCE: {}\n \n".format(planet_ind + 1))

            # save star params
            text_file.write("------------------------------\n")
            for key in self.star_info:
                text_file.write("{} : {} \n".format(key, self.star_info[key]))
            text_file.write("------------------------------\n \n")

            # save planet params
            for key in planet_dict.keys():
                text_file.write("{} : {} \n".format(key, planet_dict[key]))

            text_file.write("\n LaTeX table : \n")
            text_file.write(planet_dict.fit_results.to_latex())

        return None


    def _save_vetting_pdf_(self, planet_dict, planet_ind):
        """Helper function to save vetting figures and fit results to a PDF file

        @param planet_dict: TLS result/planet parameter dictionary
        @type planet_dict: dict

        @param planet_ind: index for current planet
        @type planet_ind: int

        @return None

        """

        # get TIC number
        tic_no = self.tic_id[4:]

        # make output directory
        outputs_directory = "{}/outputs".format(os.getcwd())
        if not os.path.isdir(outputs_directory):
            os.mkdir(outputs_directory)

        # directory to save results to
        save_to_path = "{}/outputs/{}".format(os.getcwd(), tic_no)
        if not os.path.isdir(save_to_path):
            os.mkdir(save_to_path)

        # use PdfPages to make a new PDF document
        with PdfPages("{}/tater_vet_{}_0{}.pdf".format(save_to_path, tic_no, planet_ind + 1)) as pdf:

            # save figures to the PDF
            pdf.savefig(planet_dict.periodogram_fig)
            pdf.savefig(planet_dict.result_fig)
            pdf.savefig(planet_dict.corner_fig)
            pdf.savefig(planet_dict.line_test_fig)
            pdf.savefig(planet_dict.oddeven_fig)
            pdf.savefig(planet_dict.tdepths_fig)
            pdf.savefig(planet_dict.ttv_fig)

        # save summary of fit results to a txt file
        with open("{}/tater_vet_{}_0{}.txt".format(save_to_path, tic_no, planet_ind + 1), "w") as text_file:
            text_file.write("{}\n".format(self.tic_id))
            text_file.write("TCE: {}\n".format(planet_ind + 1))
            if planet_dict.FP == "No":
                text_file.write("vet: PASS.\n \n")
            else:
                text_file.write("vet: FAIL.\n \n")

            text_file.write("Period : {} \n \n".format(planet_dict.period))

            text_file.write("FP : {} \n".format(planet_dict.FP))
            text_file.write("Delta_chi2_straightline : {} (>30.863?) \n".format(planet_dict.delta_chi2))
            text_file.write("Z_oddeven : {} (<5?) \n".format(planet_dict.Z_oddeven))


        # save transit times to a separate txt file for TTV analysis
        ttv_text_file = "{}/tater_transittimes_{}_0{}.txt".format(save_to_path, tic_no, planet_ind + 1)
        np.savetxt(ttv_text_file, planet_dict.ttv_data, header="transit #, t0, uncert")

        return None


    def _tls_search_(self, max_iterations, tce_threshold,
        time=None, flux=None, flux_err=None, mask=None,
        period_min=0.5, period_max=100,
        make_plots=True, show_plots=False):
        """Helper function to run TLS search for transits in light curve

        @param max_iterations: maximum number of search iterations if SDE threshold is never reached
        @type max_iterations: int

        @param tce_threshold: Minimum Signal Detection Efficiency (SDE) that counts as a Threshold Crossing Event (TCE)
        @type tce_threshold: float

        @param time: time array
        @type time: numpy array (optional; default=None)

        @param flux: flux array
        @type flux: numpy array (optional; default=None)

        @param flux_err: flux uncertainty array
        @type flux_err: numpy array (optional; default=None)

        @param mask: in-transit boolean mask array
        @type mask: numpy array (optional; default=None)

        @param period_min: minimum orbital period for TLS to explore
        @type period_min: float (optional; default=0.8)

        @param period_max: maximum orbital period for TLS to explore
        @type period_max: float (optional; default=100)

        @param make_plots: make plots of periodogram and best TLS model
        @type make_plots: bool (optional; default=True)

        @param show_plots: show plots of periodogram and best TLS model (only used if make_plots = True)
        @type show_plots: bool (optional; default=False)

        @return self.TCEs, intransit: list of threshold crossing events (TCEs) and in-transit mask

        """

        # copy time, flux, uncertainty arrays
        if time is None:
            time = self.time
        if flux is None:
            flux = self.f
        if flux_err is None:
            flux_err = self.f_err

        # set maximum period to search (if not given)
        if period_max is None:
            period_max = np.ptp(time) / 2  # require at least 2 transits
        #if period_max > 100:
        #    period_max = 100 # no more than 100 days period

        # initialize in-transit mask
        if mask is None:
            intransit = np.zeros(len(time), dtype="bool")
        else:
            intransit = mask

        # TCE list
        TCEs = []

        # do the TLS search. stop searching after at most max_iterations
        for i in range(max_iterations):

            # clean arrays for TLS, masking out out-of-transit flux
            new_time, new_flux, new_flux_err = cleaned_array(time[~intransit], flux[~intransit], flux_err[~intransit])
            print('just applied mask')
            # leave loop if previous TLS iterations have completely masked all data
            if len(time) == 0:
                break

            # initializes TLS
            tls = transitleastsquares(new_time, new_flux, new_flux_err)

            # Get TLS power spectrum; use stellar params
            tls_results = tls.power(
                R_star=self.R_star.value,
                R_star_min=max(0.07, self.R_star.value - self.R_star_uncert.value),  # set min and max stellar values to 1-sigma uncertainties from MAST
                R_star_max=self.R_star.value + self.R_star_uncert.value,
                M_star=self.M_star.value,
                M_star_min=max(0.07, self.M_star.value - self.M_star_uncert.value),
                M_star_max=self.M_star.value + self.M_star_uncert.value,
                u=[self.u1, self.u2],
                period_max=period_max,
                period_min=period_min, use_threads=1
                #period_min=period_min#, use_threads=1
            )

            # check whether TCE threshold is reached
            if round(tls_results.SDE) <= tce_threshold:

                # stop searching if there are no more TCEs
                print(f"   No additional TCEs found above SDE={tce_threshold}.")
                print("   TRANSIT SEARCH COMPLETE.")
                print(" ")
                break

            else:
                print("   TCE at $P = {:.6f}$ days (SDE = {:.6f})".format(tls_results.period, tls_results.SDE))
                print(f"   Transit search iteration {i} done.")
                print(" ")

            # don't trust the TLS duration! (different cadences/data gaps can mess this up)
            # instead, estimate from period and stellar density
            tls_results.duration = self._estimate_duration_(tls_results.period)

            if make_plots:
                # initialize periodogram figure
                fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
                ax1.set_xlim([tls_results.periods.min(), tls_results.periods.max()])
                ax1.set_xlabel("period (days)")
                ax1.set_ylabel("power")
                ax1.set_title("Peak at {:.4f} days (SDE = {:.4f})".format(tls_results.period, tls_results.SDE))

                # label TCE threshold, best period
                ax1.axhline(tce_threshold, ls="--", c="r", alpha=0.6)
                ax1.axvline(tls_results.period, alpha=0.2, lw=6, c="b")

                # label alias periods
                for j in range(2, 15):
                    ax1.axvline(tls_results.period * j, alpha=0.2, lw=1, c="b", ls='--')
                    ax1.axvline(tls_results.period / j, alpha=0.2, lw=1, c="b", ls='--')

                # plot periodogram
                ax1.plot(tls_results.periods, tls_results.power, 'k-', lw=1)

                # initialize TLS transit model figure
                fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
                ax2.set_title("TLS transit model (preliminary)")
                ax2.set_xlim([-tls_results.duration*24, tls_results.duration*24])
                ax2.set_xlabel("phase (hrs)")
                ax2.set_ylabel("relative flux")

                # create a batman model using the initial TLS parameters
                tls_model = self._model_single_transit_(t=new_time,
                                                        t0=tls_results.T0,
                                                        per=tls_results.period,
                                                        rp=tls_results.rp_rs,
                                                        a=self._P_to_a_(tls_results.period)
                                                        )

                # fold model and data
                phase = (new_time - tls_results.T0 + 0.5 * tls_results.period) % tls_results.period - 0.5 * tls_results.period
                tls_model = tls_model[np.argsort(phase)]
                f_fold = new_flux[np.argsort(phase)]
                phase = np.sort(phase)

                # plot folded data and TLS transit model
                ax2.scatter(phase * 24, f_fold, s=1, c='k', rasterized=True)
                ax2.plot(phase * 24, tls_model, "b-", lw=3)

                # option to show plots
                if show_plots:
                    plt.ion(), plt.show(), plt.pause(0.001)

                # add "False Positive" keyword + plots to tls_results object
                tls_results.periodogram_fig = fig1
                tls_results.model_fig = fig2
                tls_results.FP = "No"

                plt.close()

            # mask the detected transit signal before next iteration of TLS
            # length of mask is 2.5 x the transit duration
            intransit += transit_mask(time, tls_results.period, 2.5 * tls_results.duration, tls_results.T0)
            print('set intransit')
            print('len(intransit)', len(intransit))
            print('intransit.dtype', intransit.dtype)
            # make planet vetting figures
            # self._generate_vet_figures_(tls_results)

            # append tls_results to TCE list
            TCEs.append(tls_results)

        return TCEs,intransit


    def _get_intransit_flux_(self, tce_dict, msk=[]):
        """Helper function to extract and re-normalize in-transit flux
        to save compute time for MCMC fit

        @param tce_dict: TCE results dictionary
        @type tce_dict: dict

        @param msk: previous transit mask
        @type msk: array

        @return new_t, new_f, new_ferr: normalized in-transit time, flux, and uncert

        """

        # get the individual transit times and transit duration to create mask
        transit_times = tce_dict.transit_times
        duration = tce_dict.duration

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


    def _vet_previous_planets_(self, TCE_list):
        """Helper function to do previous planet check defined by Zink+2020
        https://ui.adsabs.harvard.edu/abs/2020AJ....159..154Z/abstract

        @param TCE_list: list of TCE dictionaries
        @type TCE_list: list

        @return vetted TCE list

        """

        print("   -> previous planet check:")

        for i in range(len(TCE_list)):

            if TCE_list[i].FP == "Insufficient data":
                continue

            for j in range(0, i):

                if TCE_list[j].FP == "Insufficient data":
                    continue

                P_A, P_B = np.sort((TCE_list[i].fit_results["median"]["$P$"],
                                    TCE_list[j].fit_results["median"]["$P$"]))
                delta_P = (P_B - P_A) / P_A
                sigma_P = np.sqrt(2) * erfcinv(np.abs(delta_P - round(delta_P)))

                delta_t0 = np.abs(TCE_list[i].fit_results["median"]["$T_0$"] -
                                  TCE_list[j].fit_results["median"]["$T_0$"]) / TCE_list[i].duration

                delta_SE1 = np.abs(TCE_list[i].fit_results["median"]["$T_0$"] -
                                   TCE_list[j].fit_results["median"]["$T_0$"] +
                                   TCE_list[i].fit_results["median"]["$P$"] / 2) / TCE_list[i].duration

                delta_SE2 = np.abs(TCE_list[i].fit_results["median"]["$T_0$"] -
                                   TCE_list[j].fit_results["median"]["$T_0$"] -
                                   TCE_list[i].fit_results["median"]["$P$"] / 2) / TCE_list[i].duration

                if (sigma_P > 2.0) & (TCE_list[j].FP != "No"):
                    TCE_list[i].FP = "Previous"
                    break

                if (sigma_P > 2.0) & (delta_t0 < 1):
                    TCE_list[i].FP = "Previous"
                    break

                elif (sigma_P > 2.0) & ((delta_SE1 < 1) | (delta_SE2 < 1)):
                    TCE_list[i].FP = "Previous"
                    break

            if TCE_list[i].FP == "No":
                print("      {}...PASS.".format(i))
            else:
                print("      {}...FAIL.".format(i))

        return TCE_list


    def _vet_odd_even_(self, TCE_list):
        """Helper function to do odd vs even check defined by Zink+2020
        https://ui.adsabs.harvard.edu/abs/2020AJ....159..154Z/abstract

        @param TCE_list: list of TCE dictionaries
        @type TCE_list: list

        @return vetted TCE list

        """

        print("   -> odd vs even transit test:")

        for i in range(len(TCE_list)):

            if TCE_list[i].FP == "Insufficient data":
                continue

            # mask previous transits
            previous_intransit = np.zeros(len(self.time_raw), dtype=bool)
            if i > 0:
                for previous_tce in TCE_list[0:i]:
                    previous_intransit += transit_mask(self.time_raw, previous_tce.period,
                                              2.5 * previous_tce.duration,
                                              previous_tce.T0)

            # Only care about data near transits
            time, flux, flux_err = self._get_intransit_flux_(TCE_list[i], msk=previous_intransit)

            # get useful info from dict
            transit_times = TCE_list[i].transit_times
            duration = TCE_list[i].duration
            period = TCE_list[i].period
            depth = TCE_list[i].depth

            # empty arrays
            transits_odd = []
            t0_odd = []
            time_odd = []
            flux_odd = []
            depth_odd = []
            tfold_odd = []

            transits_even = []
            t0_even = []
            time_even = []
            flux_even = []
            depth_even = []
            tfold_even = []

            # fit each individual transit separately
            for j, t0 in enumerate(transit_times):

                transit_msk = (time > t0 - 3 * duration) & (time < t0 + 3 * duration)

                # skip if there are fewer than 70% of data points
                cadence = 2. / 1440  # 2-min cadence
                N_min = 0.7 * (6 * duration / cadence)
                if np.sum(transit_msk) < N_min:
                    continue

                # Fit transit model --> get t0, depths for single transit
                p0 = [t0, np.sqrt(depth)]
                popt, _ = curve_fit(
                    lambda t, t0_best, rp_best: self._model_single_transit_(
                        t, t0=t0_best,
                        per=period,
                        rp=rp_best,
                        a=self._P_to_a_(period)
                    ), time[transit_msk], flux[transit_msk], p0, sigma=flux_err[transit_msk]
                )

                # odd transits
                if j % 2:
                    transits_odd.append(j)
                    t0_odd.append(popt[0])
                    time_odd.append(time[transit_msk])
                    flux_odd.append(flux[transit_msk])
                    depth_odd.append(popt[1])

                # even transits
                else:
                    transits_even.append(j)
                    t0_even.append(popt[0])
                    time_even.append(time[transit_msk])
                    flux_even.append(flux[transit_msk])
                    depth_even.append(popt[1])

            # make odd vs. even figure
            gridspec = dict(wspace=0.0, hspace=0.0, width_ratios=[1, 1])
            oddeven_fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True, gridspec_kw=gridspec)
            cmap = get_cmap('viridis')

            # plot odd transits
            depth_odd = np.array(depth_odd)
            if len(transits_odd) > 0:
                ax = axes[1]
                mean_depth_odd = np.mean(depth_odd[depth_odd < 0.9])
                std_depth_odd = np.std(depth_odd[depth_odd < 0.9])
                ax.axhline(1 - mean_depth_odd**2, c='b', ls='-', lw=2, alpha=0.2)
                ax.axhline(1 - (mean_depth_odd**2 + std_depth_odd**2), c='b', ls='--', lw=1)
                ax.axhline(1 - (mean_depth_odd**2 - std_depth_odd**2), c='b', ls='--', lw=1)
                for k in range(len(time_odd)):
                    color_values = np.linspace(0, 1, len(time_odd))
                    t_fold = (time_odd[k] - t0_odd[k] + 0.5 * period) % period - 0.5 * period
                    tfold_odd.append(t_fold)
                    ax.scatter(t_fold * 24, flux_odd[k], s=4, alpha=0.5, color=cmap(color_values[k]),
                               rasterized=True,
                               label=transits_odd[k])
                ax.legend(title="transit #")
                # binned
                flux_fold_odd = np.array([item for sublist in flux_odd for item in sublist])
                time_fold_odd = np.array([item for sublist in tfold_odd for item in sublist])
                flux_fold_odd = flux_fold_odd[np.argsort(time_fold_odd)]
                time_fold_odd = np.sort(time_fold_odd)
                ax.errorbar(*self._resample_(time_fold_odd * 24, flux_fold_odd),
                            fmt='rx', fillstyle="none", elinewidth=1, zorder=200, alpha=0.7)

            # plot even transits
            depth_even = np.array(depth_even)
            if len(transits_even) > 0:
                ax = axes[0]
                mean_depth_even = np.mean(depth_even[depth_even < 0.9])
                std_depth_even = np.std(depth_even[depth_even < 0.9])
                ax.axhline(1 - mean_depth_even**2, c='b', ls='-', lw=2, alpha=0.2)
                ax.axhline(1 - (mean_depth_even**2 + std_depth_even**2), c='b', ls='--', lw=1)
                ax.axhline(1 - (mean_depth_even**2 - std_depth_even**2), c='b', ls='--', lw=1)
                ax.set_ylabel("relative flux")
                for k in range(len(time_even)):
                    color_values = np.linspace(0, 1, len(time_even))
                    t_fold = (time_even[k] - t0_even[k] + 0.5 * period) % period - 0.5 * period
                    tfold_even.append(t_fold)
                    ax.scatter(t_fold * 24, flux_even[k], s=4, alpha=0.5, color=cmap(color_values[k]),
                               rasterized=True,
                               label=transits_even[k])
                ax.legend(title="transit #")
                # binned
                flux_fold_even = np.array([item for sublist in flux_even for item in sublist])
                time_fold_even = np.array([item for sublist in tfold_even for item in sublist])
                flux_fold_even = flux_fold_even[np.argsort(time_fold_even)]
                time_fold_even = np.sort(time_fold_even)
                ax.errorbar(*self._resample_(time_fold_even * 24, flux_fold_even),
                            fmt='rx', fillstyle="none", elinewidth=1, zorder=200, alpha=0.7)

            # test discrepancy significance
            if (len(transits_odd) > 0) & (len(transits_even) > 0):

                Z = abs(mean_depth_even - mean_depth_odd) / np.sqrt(std_depth_even ** 2 + std_depth_odd ** 2)
                TCE_list[i].Z_oddeven = Z

                if Z > 5:
                    oddeven_fig.suptitle("$Z$ = {:.3f} (FAIL)".format(Z))
                    if TCE_list[i].FP != "No":
                        TCE_list[i].FP += ", odd_even"
                    else:
                        TCE_list[i].FP = "odd_even"
                    print("      {}...FAIL.".format(i))

                else:
                    oddeven_fig.suptitle("$Z$ = {:.3f} (PASS)".format(Z))
                    print("      {}...PASS.".format(i))

            else:
                TCE_list[i].Z_oddeven = np.nan
                print("      Insufficient transits for odd/even test.")

            for ax in axes:
                ax.axhline(1.0, c='k', ls='--', lw=1, zorder=0)
                ax.set_xlabel("phase (hrs)")

            # save and close figure
            TCE_list[i].oddeven_fig = oddeven_fig
            plt.close()

            # Now plot transit depth vs transit number
            tdepths_fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_title("transit depth check")
            ax.set_xlabel("transit number")
            ax.set_ylabel("transit depth")

            ax.axhline(1.0, c='k', ls='--')

            # plot each transit depth
            odd_depths = np.array([(1 - np.square(depth_odd))[i] for i in np.argsort(t0_odd)])
            even_depths = np.array([(1 - np.square(depth_even))[i] for i in np.argsort(t0_even)])
            ax.plot(np.sort(transits_odd)[odd_depths > 0.1], odd_depths[odd_depths > 0.1],
                    "k.", ms=12, zorder=10, label="odd")
            ax.plot(np.sort(transits_even)[even_depths > 0.1], even_depths[even_depths > 0.1],
                    "kx", ms=12, zorder=10, label="even")
            ax.legend()

            # plot average transit depth
            mean_depth = np.nanmean(np.concatenate((odd_depths[odd_depths > 0.1], even_depths[even_depths > 0.1])))
            std_depth = np.nanstd(np.concatenate((odd_depths[odd_depths > 0.1], even_depths[even_depths > 0.1])))
            ax.axhline(mean_depth, c='b', ls='-', lw=2, alpha=0.2)
            ax.axhline(mean_depth + std_depth, c='b', ls='--', lw=1)
            ax.axhline(mean_depth - std_depth, c='b', ls='--', lw=1)

            # save and close figure
            TCE_list[i].tdepths_fig = tdepths_fig
            plt.close()

        return TCE_list


    def _vet_line_test_(self, TCE_list):
        """Helper function to compare linear fit to transit model fit

        @param TCE_list: list of TCE dictionaries
        @type TCE_list: list

        @return vetted TCE list

        """

        print("   -> straight line test:")

        for i in range(len(TCE_list)):

            if TCE_list[i].FP == "Insufficient data":
                continue

            # mask previous transits
            previous_intransit = np.zeros(len(self.time_raw), dtype=bool)
            if i > 0:
                for previous_tce in TCE_list[0:i]:
                    previous_intransit += transit_mask(self.time_raw, previous_tce.period,
                                              2.5 * previous_tce.duration,
                                              previous_tce.T0)

            # Only care about data near transits
            time, flux, flux_err = self._get_intransit_flux_(TCE_list[i], msk=previous_intransit)

            # grab parameters from fit
            t0_best = TCE_list[i].fit_results["median"]["$T_0$"]
            p_best = TCE_list[i].fit_results["median"]["$P$"]
            a_best = TCE_list[i].fit_results["median"]["$a/R_*$"]
            b_best = TCE_list[i].fit_results["median"]["$b$"]
            rp_best = TCE_list[i].fit_results["median"]["$r_p/R_*$"]

            # phase to best period and t0
            t_fold = (time - t0_best + 0.5 * p_best) % p_best - 0.5 * p_best

            # sort arrays by phase
            flux = flux[np.argsort(t_fold)]
            flux_err = flux_err[np.argsort(t_fold)]
            t_fold = np.sort(t_fold)

            # set up figure for residuals
            gridspec = dict(wspace=0.0, hspace=0.0, height_ratios=[1, 1])
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw=gridspec)

            # select first axis for transit model residual plot
            ax = axes[0]
            ax.set_ylabel("transit model residuals")
            ax.set_xlim([t_fold.min() * 24 + 0.75, t_fold.max() * 24 - 0.75])

            # convert to inclination angle
            inc = np.arccos(b_best / a_best) * (180 / np.pi)

            # batman model
            params = batman.TransitParams()
            params.t0 = 0  # time of inferior conjunction
            params.per = p_best  # orbital period
            params.rp = rp_best  # planet radius (in units of stellar radii)
            params.a = a_best  # semi-major axis (in units of stellar radii)
            params.inc = inc  # orbital inclination (in degrees)
            params.ecc = 0.  # eccentricity
            params.w = 90.  # longitude of periastron (in degrees)
            params.u = [self.u1, self.u2]  # limb darkening coefficients []
            params.limb_dark = "quadratic"  # limb darkening model
            model = batman.TransitModel(params, t_fold)  # initializes model

            # flux from batman model
            flux_m = model.light_curve(params)  # calculates light curve

            # compute residuals and chi2
            resid_tmodel = flux - flux_m
            chi2_tmodel = np.sum(resid_tmodel ** 2 / flux_err ** 2)

            # plot residuals for "best-fit" model
            ax.errorbar(t_fold * 24, resid_tmodel, yerr=flux_err, fmt='k.', ms=1, alpha=0.1, rasterized=True)  # plot data
            ax.plot(t_fold * 24, resid_tmodel, 'k.', ms=1, alpha=0.5, rasterized=True)

            # plot binned residuals
            ax.errorbar(*self._resample_(t_fold * 24, resid_tmodel), fmt='r.', fillstyle="none", elinewidth=1,
                        zorder=200, label="binned transit model residual")

            # set y limits
            ax.set_ylim([-5 * np.std(resid_tmodel), 5 * np.std(resid_tmodel)])

            # line at y=0
            ax.axhline(0.0, color="b", alpha=0.75, label=r"$\chi^2 = {:.3f}$".format(chi2_tmodel))
            ax.legend()

            # Now fit a straight line to the data
            linear_func = lambda x, m, b: m * x + b
            popt, _ = curve_fit(linear_func, t_fold, flux, [0, 1], sigma=flux_err)
            flux_line = linear_func(t_fold, *popt)

            # compute residuals and chi2
            resid_line = flux - flux_line
            chi2_line = np.sum(resid_line**2 / flux_err**2)

            # select second axis for straight line residual plot
            ax = axes[1]
            ax.set_ylabel("straight line residuals")
            ax.set_xlabel("phase (hrs)")

            # plot residuals for "best-fit" model
            ax.errorbar(t_fold * 24, resid_line, yerr=flux_err, fmt='k.', ms=1, alpha=0.1,
                        rasterized=True)  # plot data
            ax.plot(t_fold * 24, resid_line, 'k.', ms=1, alpha=0.5, rasterized=True)

            # plot binned residuals
            ax.errorbar(*self._resample_(t_fold * 24, resid_line), fmt='r.', fillstyle="none", elinewidth=1,
                        zorder=200, label="binned straight line residual")

            # set y limits
            ax.set_ylim([-5 * np.std(resid_line), 5 * np.std(resid_line)])

            # line at y=0
            ax.axhline(0.0, color="b", alpha=0.75, label=r"$\chi^2 = {:.3f}$".format(chi2_line))
            ax.legend()

            # test discrepancy significance
            delta_chi2 = chi2_line - chi2_tmodel
            TCE_list[i].delta_chi2 = delta_chi2

            if delta_chi2 < 30.863:
                fig.suptitle(r"$\Delta \chi^2$ = {:.3f} (FAIL)".format(delta_chi2))
                if TCE_list[i].FP != "No":
                    TCE_list[i].FP += ", straight_line"
                else:
                    TCE_list[i].FP = "straight_line"
                print("      {}...FAIL.".format(i))

            else:
                fig.suptitle(r"$\Delta \chi^2$ = {:.3f} (PASS)".format(delta_chi2))
                print("      {}...PASS.".format(i))

            # save figure to TCE object
            TCE_list[i].line_test_fig = fig
            plt.close()

        return TCE_list


    def _get_ephems_(self, TCE_list):
        """Helper function to get transit times and fit linear ephemeris

        @param TCE_list: list of TCE dictionaries
        @type TCE_list: list

        @return None

        """

        print("   -> saving transit times")

        for i in range(len(TCE_list)):

            if TCE_list[i].FP == "Insufficient data":
                continue

            # mask previous transits
            previous_intransit = np.zeros(len(self.time_raw), dtype=bool)
            if i > 0:
                for previous_tce in TCE_list[0:i]:
                    previous_intransit += transit_mask(self.time_raw, previous_tce.period,
                                                       2.5 * previous_tce.duration,
                                                       previous_tce.T0)

            # Only care about data near transits
            time, flux, flux_err = self._get_intransit_flux_(TCE_list[i], msk=previous_intransit)

            # get useful info from dict
            transit_times = TCE_list[i].transit_times
            duration = TCE_list[i].duration
            period = TCE_list[i].period
            depth = TCE_list[i].depth

            # empty arrays
            transit_number_list = []
            t0_list = []
            t0_uncert_list = []

            # fit each individual transit separately
            for j, t0 in enumerate(transit_times):

                transit_msk = (time > t0 - 3 * duration) & (time < t0 + 3 * duration)

                # skip if there are fewer than 70% of data points
                cadence = 2. / 1440   # 2-min cadence
                N_min = 0.7 * (6 * duration / cadence)
                if np.sum(transit_msk) < N_min:
                    continue

                # Fit transit model --> get t0, depths for single transit
                p0 = [t0, np.sqrt(depth)]
                popt, pcov = curve_fit(
                    lambda t, t0_best, rp_best: self._model_single_transit_(
                        t, t0=t0_best,
                        per=period,
                        rp=rp_best,
                        a=self._P_to_a_(period)
                    ), time[transit_msk], flux[transit_msk], p0, sigma=flux_err[transit_msk]
                )

                # get 1-sigma uncert from cov matrix
                perr = np.sqrt(np.diag(pcov))

                # save results to arrays
                transit_number_list.append(j)
                t0_list.append(popt[0])
                t0_uncert_list.append(perr[0])


            # make transit TTV figure
            gridspec = dict(wspace=0.0, hspace=0.0, height_ratios=[1, 1])
            ttv_fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw=gridspec)

            linear_func = lambda x, m, b: m * x + b

            # plot transit center time vs. transit number
            ax = axes[0]
            ax.set_title("Ephemeris check")
            ax.set_ylabel("transit center (d)")
            ax.errorbar(np.array(transit_number_list), np.array(t0_list),
                        yerr=np.array(t0_uncert_list), fmt='k.', ms=12)
            # only fit and plot regression line if at least two transits
            if (len(transit_number_list) > 1):
                slope, intercept, _, _, _ = linregress(np.array(transit_number_list), np.array(t0_list))
                y_fit = linear_func(np.array(transit_number_list), slope, intercept)
                ax.plot(np.array(transit_number_list), y_fit, 'b-')
                ax.text(0, t0_list[-1],
                        " $P$ = {:.5f} d \n $T_0$ = {:.5f} d \n $y = Px + T_0$".format(slope, intercept))

            # plot residuals
            ax = axes[1]
            ax.set_xlabel("transit number")
            ax.set_ylabel("O - C (d)")
            ax.axhline(0.0, c='b', ls='--', lw=1, zorder=0)
            # only fit and plot regression line residuals if at least two transits
            if (len(transit_number_list) > 1):
                ax.errorbar(np.array(transit_number_list), np.array(t0_list) - y_fit,
                            yerr=np.array(t0_uncert_list), fmt='k.', ms=12)

            # save and close figure
            TCE_list[i].ttv_fig = ttv_fig
            TCE_list[i].ttv_data = np.array((transit_number_list, t0_list, t0_uncert_list)).T
            plt.close()

        return None


    def _model_single_transit_(self, t, t0, per, rp, a,
        inc=90., baseline=1., q1=None, q2=None):
        """Helper function to generate a single batman transit model

        @param t: time
        @type t: numpy array

        @param t0: time of first transit
        @type t0: float

        @param per: period
        @type per: float

        @param rp: planet radius (units of stellar radius)
        @type rp: float

        @param a: semi-major axis (units of stellar radius)
        @type a: float

        @param inc: planet inclination (units of degrees)
        @type inc: float (optional; default=90.)

        @param baseline: average out-of-transit flux
        @type baseline: float (optional; default=1.)

        @param q1: Kipping et al. (2013) LD parameter #1
        @type q1: float (optional; default=None)

        @param q2: Kipping et al. (2013) LD parameter #2
        @type q2: float (optional; default=None)

        @return: light curve flux

        """

        # use specific user input LD parameters if given
        if q1 is not None and q2 is not None:
            u1 = 2*np.sqrt(q1)*q2
            u2 = np.sqrt(q1)*(1-2*q2)
        else:
            u1,u2 = self.u1, self.u2

        # initialize batman model
        params = batman.TransitParams()
        params.t0 = t0  # time of inferior conjunction
        params.per = per  # orbital period
        params.rp = rp  # planet radius (in units of stellar radii)
        params.a = a  # semi-major axis (in units of stellar radii)
        params.inc = inc  # orbital inclination (in degrees)
        params.ecc = 0.  # eccentricity
        params.w = 90.  # longitude of periastron (in degrees)
        params.u = [u1, u2]  # limb darkening coefficients []
        params.limb_dark = "quadratic"  # limb darkening model
        model = batman.TransitModel(params, t)  # initializes model

        # get model flux
        lc = model.light_curve(params)  # calculates light curve

        # include effect of baseline != 1
        lc *= baseline

        return lc


    def _get_stellar_params_(self, ask_user=False, assume_solar=False):
        """Helper function that loads stellar parameters from MAST and interpolates LD coeffs from table
        If no parameter found, asks user for input (unless ask_user=False, then function skips)
        If no parameter found and ask_user=False, solar values can be assumed (only if assume_solar=True)
        LD source: https://ui.adsabs.harvard.edu/abs/2017A%26A...600A..30C/abstract

        @param ask_user: ask user to input info missing from MAST
        @type ask_user: bool (default=False)

        @param assume_solar: fill in solar estimates for missing MAST values (ignored if ask_user=True)
        @type assume_solar: bool (default=False)

        @return missing (bool for whether MAST is missing some input)

        """

        # initialize empty list for values missing from MAST
        missing = []

        # get stellar params from MAST
        tic_table = Catalogs.query_object(self.tic_id, catalog="TIC", radius=0.01)[0]

        # stellar radius
        self.R_star = tic_table["rad"] * u.R_sun
        if not np.isfinite(self.R_star.value):
            print("   Could not locate valid 'R_star'.")
            missing += ['R_star']
            if ask_user:
                self.R_star = self._ask_user_("R_star [R_sun]", limits=(0, 1000)) * u.R_sun
            elif assume_solar:
                print("   Solar value of 'R_star' used instead: 1 R_sun.")
                self.R_star = 1. * u.R_sun

        # stellar radius uncert
        self.R_star_uncert = tic_table["e_rad"] * u.R_sun
        if not np.isfinite(self.R_star_uncert.value):
            print("   Could not locate valid 'R_star_uncert'.")
            missing += ['R_star_uncert']
            if ask_user:
                self.R_star_uncert = self._ask_user_("R_star_uncert [R_sun]", limits=(0, 1000)) * u.R_sun
            elif assume_solar:
                print("   Solar value of 'R_star' used instead: assuming uncert of 0.1 R_sun.")
                self.R_star_uncert = 0.1 * u.R_sun

        # stellar mass
        self.M_star = tic_table["mass"] * u.M_sun
        if not np.isfinite(self.M_star.value):
            print("   Could not locate valid 'M_star'.")
            missing += ['M_star']
            if ask_user:
                self.M_star = self._ask_user_("M_star [M_sun]", limits=(0, 1000)) * u.M_sun
            elif assume_solar:
                print("   Solar value of 'M_star' used instead: 1 M_sun.")
                self.M_star = 1. * u.M_sun

        # stellar mass uncert
        self.M_star_uncert = tic_table["e_mass"] * u.M_sun
        if not np.isfinite(self.M_star_uncert.value):
            print("   Could not locate valid 'M_star_uncert'.")
            missing += ['M_star_uncert']
            if ask_user:
                self.M_star_uncert = self._ask_user_("M_star_uncert [M_sun]", limits=(0, 1000)) * u.M_sun
            elif assume_solar:
                print("   Solar value of 'M_star' used instead: assuming uncert of 0.1 M_sun.")
                self.M_star_uncert = 0.1 * u.M_sun

        # stellar surface gravity
        self.logg = tic_table["logg"]
        if not 0 < self.logg < 5:
            print("   Could not locate valid 'logg'.")
            missing += ['logg']
            if ask_user:
                self.logg = self._ask_user_("logg", limits=(0, 5))
            elif assume_solar:
                print("   Solar value of 'logg' used instead: 4.4374.")
                self.logg = 4.4374 # Smalley et al. (2005)

        # stellar effective temperature
        self.Teff = tic_table["Teff"]
        if not 3500 < self.Teff < 50000:
            print("   Could not locate valid 'Teff'.")
            missing += ['Teff']
            if ask_user:
                self.Teff = self._ask_user_("Teff [K]", limits=(3500, 50000))
            elif assume_solar:
                print("   Solar value of 'Teff' used instead: 5777 K.")
                self.Teff = 5777 # https://en.wikipedia.org/wiki/Sun

        # stellar metallicity
        self.Fe_H = tic_table["MH"]
        if not -5 < self.Fe_H < 1:
            print("   Could not locate valid 'Fe_H'.")
            missing += ['Fe_H']
            if ask_user:
                self.Fe_H = self._ask_user_("Fe_H", limits=(-5, 1))
            elif assume_solar:
                print("   Solar value of 'Fe_H' used instead: 0.")
                self.Fe_H = 0.

        # interpolate limb darkening coefficients
        # LD table from https://ui.adsabs.harvard.edu/abs/2017A%26A...600A..30C/abstract
        Vizier.ROW_LIMIT = -1
        ld_table = Vizier.get_catalogs("J/A+A/600/A30/table25")[0]
        ld_table = ld_table[ld_table["xi"] == 2.0]
        ld_points = np.array([ld_table["logg"], ld_table["Teff"], ld_table["Z"]]).T
        ld_values = np.array([ld_table["aLSM"], ld_table["bLSM"]]).T
        ld_interpolator = LinearNDInterpolator(ld_points, ld_values)
        self.u1, self.u2 = ld_interpolator(self.logg, self.Teff, self.Fe_H)

        # print stellar info
        self.star_info = dict(zip(
            ["TIC ID", "R_star", "M_star", "logg", "Teff", "[Fe/H]", "u1", "u2"],
            [self.tic_id, self.R_star, self.M_star, self.logg, self.Teff, self.Fe_H, self.u1, self.u2]
        )
        )

        print("------------------------------")
        for key in self.star_info: print("   {}:\t{}".format(key, self.star_info[key]))
        print("------------------------------")

        tic_no = self.tic_id[4:]
        save_to_path = "{}/outputs/{}".format(os.getcwd(), tic_no)
        if not os.path.isdir(save_to_path):
            os.mkdir(save_to_path)
        missing_text_file = "{}/missing_data_{}.txt".format(save_to_path, tic_no)
        with open(missing_text_file,'w') as f:
            for i in missing:
                f.write(i + '\n')

        return missing


    def _ask_user_(self, variable_name, limits=None):
        """Helper function to ask user for input

        @param variable_name: name of user input variable
        @type variable_name: string

        @param limits: upper and lower bounds on allowed user input
        @type limits: tuple (optional; default=None)

        @return response

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
        """Helper function to compute approx transit duration given period.
        (Eq 19 of Winn (2010) in Exoplanets)

        @param period: planet orbital period in days
        @type period: float

        @return duration: approximate transit duration in days

        """

        # stellar density
        rho_star = (self.M_star / self.R_star ** 3).value  # solar units

        # period in years
        P_yr = period / 365.

        # transit duration in days
        duration = (13. / 24) * P_yr**(1 / 3) * rho_star**(-1 / 3)  # days

        return duration


    def _P_to_a_(self, per):
        """Helper function to convert P to a using Kepler's third law

        @param per: orbital period
        @type per: float

        @return a/R_star: semimajor axis in units of stellar radii

        """

        # conversions
        M_star = self.M_star.to(u.kg).value
        R_star = self.R_star.to(u.m).value

        # Kepler's 3rd law
        a = ((6.67e-11 * M_star) / (4 * np.pi ** 2) * (per * 86400.) ** 2) ** (1 / 3)

        return a / R_star


    def _log_likelihood_(self, theta, x, y, yerr):
        """Helper function that returns log-likelihood for MCMC

        @param theta: parameter vector (period, t0, rp, a, b)
        @type theta: numpy array

        @param x: time array
        @type x: numpy array

        @param y: flux array
        @type y: numpy array

        @param yerr: flux uncertainty array
        @type yerr: numpy array

        @return log_like

        """

        # unpack parameters
        per, t0, rp, a, b = theta

        # get inclination angle
        inc = np.arccos(b / a) * (180 / np.pi)

        # initialize batman model
        params = batman.TransitParams()
        params.t0 = t0  # time of inferior conjunction
        params.per = per  # orbital period
        params.rp = rp  # planet radius (in units of stellar radii)
        params.a = a  # semi-major axis (in units of stellar radii)
        params.inc = inc  # orbital inclination (in degrees)
        params.ecc = 0.  # eccentricity
        params.w = 90.  # longitude of periastron (in degrees)
        params.u = [self.u1, self.u2]  # limb darkening coefficients []
        params.limb_dark = "quadratic"  # limb darkening model
        model = batman.TransitModel(params, x)  # initializes model

        # get batman model flux
        flux_m = model.light_curve(params)  # calculates light curve

        # log likelihood
        log_like = -0.5 * np.sum((y - flux_m) ** 2 / yerr ** 2 + np.log(2 * np.pi * yerr ** 2))

        return log_like


    def _log_prior_(self, theta, theta_0):
        """Helper function that returns log-prior for MCMC

        @param theta: parameter vector (period, t0, rp, a, b)
        @type theta: numpy array

        @param theta_0: initialization parameter vector (period, t0, rp, a, b)
        @type theta_0: numpy array

        @return prior

        """

        # unpack parameters
        per, t0, rp, a, b = theta

        # set uniform priors
        if (theta_0["per"] - 0.1 < per < theta_0["per"] + 0.1) \
                and (theta_0["t0"] - 0.083 < t0 < theta_0["t0"] + 0.083) \
                and (0.0 < rp < 0.3) \
                and (max(1, 0.5 * theta_0["a_rs"]) < a < 10.0 * theta_0["a_rs"]) \
                and (0.0 < b < 1.0):
            prior = 0.0

        else:
            prior = -np.inf

        return prior

    def _log_probability_(self, theta, theta_0, x, y, yerr):
        """Helper function that returns posterior log-probability for MCMC

        @param theta: parameter vector (period, t0, rp, a, b)
        @type theta: numpy array

        @param theta_0: initialization parameter vector (period, t0, rp, a, b)
        @type theta_0: numpy array

        @param x: time array
        @type x: numpy array

        @param y: flux array
        @type y: numpy array

        @param yerr: flux uncertainty array
        @type yerr: numpy array

        @return log_prob

        """

        # priors
        lp = self._log_prior_(theta, theta_0)

        # uniform priors
        if not np.isfinite(lp):
            return -np.inf

        # Bayes
        log_prob = lp + self._log_likelihood_(theta, x, y, yerr)

        return log_prob

    def _execute_mcmc_(self, theta_0, t, f, f_err, show_plots=False):
        """Helper function to run MCMC

        @param theta_0: parameter vector (period, t0, rp, a, b)
        @type theta_0: numpy array

        @param t: time array
        @type t: numpy array

        @param f: flux array
        @type f: numpy array

        @param f_err: flux uncertainty array
        @type f_err: numpy array

        @param show_plots: show MCMC plots
        @type show_plots: bool (optional; default=False)

        @return results_df, walker_fig, corner_fig, best_fig, best_full_fig

        """

        np.random.seed(42)

        # initial state
        pos = np.array([theta_0["per"], theta_0["t0"], theta_0["rp_rs"], theta_0["a_rs"], theta_0["b"]]) \
              + 5e-5 * np.random.randn(self.nwalkers, self.ndim)

        # initialize sampler
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self._log_probability_,
                                        args=(theta_0, t, f, f_err)
                                        )

        # run the MCMC
        sampler.run_mcmc(pos, self.nsteps, progress=True)

        # grab the chain
        flat_samples = sampler.get_chain(discard=self.nburn, flat=True)
        samples = sampler.get_chain()

        # generate plots (option to display plots)
        if show_plots:
            walker_fig, corner_fig = self._plot_mcmc_diagnostics_(samples, flat_samples, show_plot=True)
            if len(t) > 0:
                best_fig, best_full_fig = self._plot_best_fit_(t, f, f_err, flat_samples, show_plot=True)
            else:
                best_fig, best_full_fig = None, None
        else:
            walker_fig, corner_fig = self._plot_mcmc_diagnostics_(samples, flat_samples)
            if len(t) > 0:
                best_fig, best_full_fig = self._plot_best_fit_(t, f, f_err, flat_samples)
            else:
                best_fig, best_full_fig = None, None

        # pack up fit results in a DataFrame
        results_dict = dict(zip(["median", "lower", "upper"], np.quantile(flat_samples, [0.5, 0.16, 0.84], axis=0)))
        results_df = DataFrame(results_dict, index=self.labels)
        results_df["(+)"] = results_df["median"] - results_df["lower"]
        results_df["(-)"] = results_df["upper"] - results_df["median"]

        # add convenient units to dataframe
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

        # add units to dataframe
        results_df["units"] = ["d", "d", "-", "-", "-", "R_Earth", "R_Jup", "AU"]

        return results_df, walker_fig, corner_fig, best_fig, best_full_fig


    def _plot_mcmc_diagnostics_(self, samples, flat_samples, show_plot=False):
        """Helper function for plotting MCMC walkers and posterior distributions in a corner plot

        @param samples: MCMC chain samples
        @type samples: numpy array

        @param flat_samples: flattened MCMC chain samples
        @type flat_samples: numpy array

        @param show_plot: show walker and corner plots
        @type show_plot: bool (optional; default=False)

        @return: walker_fig, corner_fig

        """

        # initialize figure
        walker_fig, axes = plt.subplots(self.ndim, figsize=(12, 10), sharex=True)
        axes[-1].set_xlabel("step number")

        # local variable for corner plot labels
        labels = self.labels

        # plot walkers from MCMC chain
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[self.nburn:, :, i], alpha=0.3, rasterized=True)
            ax.set_xlim(0, len(samples) - self.nburn)
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        # make corner plot
        corner_fig = corner.corner(flat_samples,
                                   labels=labels,
                                   quantiles=[0.16, 0.5, 0.84],
                                   show_titles=True)

        # option to show plot
        if show_plot:
            plt.ion(), plt.show(), plt.pause(0.001)

        return walker_fig, corner_fig


    def _resample_(self, x, y, nbins=60):
        """Helper function to resample (bin) light curve for better display

        @param x: time
        @type x: numpy array

        @param y: flux
        @type y: numpy array

        @return binned_x, binned_y, binned_yerr, binned_xerr

        """

        # get bin means and edges
        bin_means, bin_edges, _ = binned_statistic(x, y, statistic='mean', bins=nbins)

        # get bin standard deviations for y-uncertainty
        bin_stds, _, _ = binned_statistic(x, y, statistic='std', bins=nbins)

        # count data points in each bin
        bin_counts, _, _ = binned_statistic(x, y, statistic='count', bins=nbins)

        # get bin widths for x-uncertainty
        bin_width = (bin_edges[1] - bin_edges[0])

        # get bin centers
        bin_centers = bin_edges[1:] - bin_width / 2

        return bin_centers, bin_means, bin_stds / np.sqrt(bin_counts), bin_width / 2


    def _plot_best_fit_(self, t, f, f_err, flat_samples, show_plot=False):
        """Helper function for plotting the folded light curve, best-fit model, and residuals
        (plots 100 random samples from MCMC)

        @param t: time array
        @type t: numpy array

        @param f: flux array
        @type f: numpy array

        @param f_err: flux uncertainty array
        @type f_err: numpy array

        @param flat_samples: flattened MCMC chain samples
        @type flat_samples: numpy array

        @param show_plot: show best fit figure
        @type show_plot: bool (optional; default=False)

        @return fig, fig2: phase-folded LC and full LC

        """

        # initialize phase-folded LC figure
        gridspec = dict(wspace=0.0, hspace=0.0, height_ratios=[2, 1])
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw=gridspec)  # set up figure

        # get median "best" values from MCMC results
        p_best, t0_best, rp_best, a_best, b_best = np.quantile(flat_samples, 0.5, axis=0)

        # phase to best period and t0
        t_fold = (t - t0_best + 0.5 * p_best) % p_best - 0.5 * p_best

        # sort arrays by phase
        f = f[np.argsort(t_fold)]
        f_err = f_err[np.argsort(t_fold)]
        t_fold = np.sort(t_fold)

        # select first axis
        ax = axes[0]
        ax.set_ylabel("relative flux")
        ax.set_title("P = {:.3f} days;    T0 = {:.3f};    rp_rs = {:.3f};    a_rs = {:.3f}".format(
            p_best, t0_best, rp_best, a_best)
        )

        # plot folded light curve and model
        ax.errorbar(t_fold * 24, f, yerr=f_err, fmt='k.', ms=1, alpha=0.1, rasterized=True)  # plot data
        ax.plot(t_fold * 24, f, 'k.', ms=1, alpha=0.5, rasterized=True)

        # plot binned data
        ax.errorbar(*self._resample_(t_fold * 24, f), fmt='r.', fillstyle="none", elinewidth=1, zorder=200)

        # choose 100 random draws from MCMC to plot
        inds = np.random.randint(len(flat_samples), size=100)

        # plot 100 random models
        for ind in inds:

            # get parameters
            per, t0, rp, a, b = flat_samples[ind]

            # convert to incination angle
            inc = np.arccos(b / a) * (180 / np.pi)

            # initialize batman model
            params = batman.TransitParams()
            params.t0 = 0  # time of inferior conjunction
            params.per = per  # orbital period
            params.rp = rp  # planet radius (in units of stellar radii)
            params.a = a  # semi-major axis (in units of stellar radii)
            params.inc = inc  # orbital inclination (in degrees)
            params.ecc = 0.  # eccentricity
            params.w = 90.  # longitude of periastron (in degrees)
            params.u = [self.u1, self.u2]  # limb darkening coefficients []
            params.limb_dark = "quadratic"  # limb darkening model
            model = batman.TransitModel(params, t_fold)  # initializes model

            # get batman model flux
            flux_model = model.light_curve(params)  # calculates light curve

            # plot model
            ax.plot(t_fold * 24, flux_model, 'b-', lw=1, alpha=0.2)

        # select second axis for residual plot
        ax = axes[1]
        ax.set_ylabel("median residuals")
        ax.set_xlabel("phase (hrs)")
        ax.set_xlim([t_fold.min() * 24 + 0.75, t_fold.max() * 24 - 0.75])

        # convert to inclination angle
        inc = np.arccos(b_best / a_best) * (180 / np.pi)

        # batman model
        params = batman.TransitParams()
        params.t0 = 0  # time of inferior conjunction
        params.per = p_best  # orbital period
        params.rp = rp_best  # planet radius (in units of stellar radii)
        params.a = a_best  # semi-major axis (in units of stellar radii)
        params.inc = inc  # orbital inclination (in degrees)
        params.ecc = 0.  # eccentricity
        params.w = 90.  # longitude of periastron (in degrees)
        params.u = [self.u1, self.u2]  # limb darkening coefficients []
        params.limb_dark = "quadratic"  # limb darkening model
        model = batman.TransitModel(params, t_fold)  # initializes model

        # flux from batman model
        flux_m = model.light_curve(params)  # calculates light curve

        # plot residuals for median "best-fit" model
        ax.errorbar(t_fold * 24, f - flux_m, yerr=f_err, fmt='k.', ms=1, alpha=0.1, rasterized=True)  # plot data
        ax.plot(t_fold * 24, f - flux_m, 'k.', ms=1, alpha=0.4, rasterized=True)

        # plot binned residuals
        ax.errorbar(*self._resample_(t_fold * 24, f - flux_m), fmt='r.', fillstyle="none", elinewidth=1, zorder=200)

        # line at y=0
        ax.axhline(0.0, color="b", alpha=0.75)

        # set up figure for full light curve with best model
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        ax2.set_ylabel("relative flux")
        ax2.set_xlabel("time (days)")

        # plot data
        ax2.scatter(self.time, self.f, c='k', s=1, alpha=0.2, rasterized=True)

        # plot best model
        params.t0 = t0_best
        model = batman.TransitModel(params, self.time)  # initializes model
        flux_m = model.light_curve(params)  # calculates light curve
        ax2.plot(self.time, flux_m, "b-", alpha=0.75)

        # option to show plots
        if show_plot:
            plt.ion(), plt.show(), plt.pause(0.001)

        return fig, fig2


    def _render_mpl_table_(self, data, col_width=3.0, row_height=0.625,
                           header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                           bbox=[0, 0, 1, 1], header_columns=0,
                           ax=None, **kwargs):
        """Helper function to convert pandas df table to a figure
        source: https://stackoverflow.com/questions/19726663/how-to-save-the-pandas-dataframe-series-data-as-a-figure

        @param data: data to put into table
        @type data: numpy array

        *use link for description of other parameters*

        @return ax.get_figure(), ax

        """

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
                cell.set_facecolor(row_colors[k[0] % len(row_colors)])

        return ax.get_figure(), ax

    """
    def _save_system_overview_(self):
        # Helper function to save system overview plots


        # get number of candidates
        N_candidates = len(self.planet_candidates)

        # save directory
        tic_no = self.tic_id[4:]
        save_to_path = "{}/outputs/{}".format(os.getcwd(), tic_no)
        if not os.path.isdir(save_to_path):
            os.mkdir(save_to_path)

        # set up figure
        fig, axes = plt.subplots(1 + N_candidates, 2)

        # 1) T0 vs period plot
        ax = axes[0, 0]
        ax.set_xlabel("Period (days)")
        ax.set_ylabel("T0")

        for i in range(N_candidates):

            period = self.planet_candidates[i].fit_results["median"]["$P$"]
            t0 = self.planet_candidates[i].fit_results["median"]["$T_0$"]

            ax.plot(period, t0, "o")




        # save figure
        fig.savefig("{}/{}_system_overview.pdf".format(save_to_path, tic_no))
        plt.close()
    """


    def _save_results_image_(self):
        """Helper function to save a visualization of the system

        @param None
        @type None

        @return None

        """

        # save directory
        tic_no = self.tic_id[4:]
        save_to_path = "{}/outputs/{}".format(os.getcwd(), tic_no)
        if not os.path.isdir(save_to_path):
            os.mkdir(save_to_path)

        n_planets = len(self.planet_candidates)

        fig, axes = plt.subplots(1 + n_planets, 1, figsize=(15, 5 + 5 * n_planets), sharex=True)

        # define color of star
        if self.Teff <= 4000:
            star_color = "tomato"
        elif 4000 < self.Teff <= 5100:
            star_color = "wheat"
        elif 5100 < self.Teff <= 6100:
            star_color = "khaki"
        elif 6100 < self.Teff <= 7500:
            star_color = "lightyellow"
        else:
            star_color = "lightsteelblue"

        axes[0].plot([-5, 100], [0, 0], 'k--', zorder=0)

        # plot star, label parameters
        star = plt.Circle((0, 0), 1.0, color=star_color, ec="k", zorder=10)
        axes[0].add_patch(star)
        axes[0].text(-4, 2, s=("Star: " + format(self.tic_id) + "\n" + \
                               "$T_{eff}$ = " + format(self.Teff) + " K\n" + \
                               "$R_{*}$ = " + format(self.R_star.value) + " $R_{sun}$\n" + \
                               "$M_{*}$ = " + format(self.M_star.value) + " $M_{sun}$"
                               ), size=15)

        # plot planets
        max_x = 20
        for i, planet in enumerate(self.planet_candidates):

            inc = np.arccos(planet.fit_results["median"]["$b$"] / planet.fit_results["median"]["$a/R_*$"])
            x = planet.fit_results["median"]["$a/R_*$"] * np.sin(inc)
            y = planet.fit_results["median"]["$a/R_*$"] * np.cos(inc)
            if x > max_x:
                max_x = x

            axes[0].plot(x, y, "x", c="darkslategray", ms=15)
            axes[0].text(x - 3, y + 1, "SDE = {:.4f}".format(planet.SDE), size=13)

            star = plt.Circle((x, 0), 5.0, color=star_color, ec="k", zorder=10)
            axes[i + 1].add_patch(star)

            p = plt.Circle((x, planet.fit_results["median"]["$b$"] * 5),
                           planet.fit_results["median"]["$r_p/R_*$"] * 5,
                           color='darkslategray', zorder=20)
            axes[i + 1].add_patch(p)

            axes[i + 1].plot([x - 8, x + 8],
                             [planet.fit_results["median"]["$b$"] * 5, planet.fit_results["median"]["$b$"] * 5],
                             'k:', zorder=15, alpha=0.7)
            axes[i + 1].plot([x, x],
                             [0, planet.fit_results["median"]["$b$"] * 5],
                             'k:', zorder=15, alpha=0.7)
            axes[i + 1].plot(x, 0, 'k.', zorder=20)

            axes[i + 1].text(x - 6.5, 5, s=("$P$ = " + format(planet.fit_results["median"]["$P$"], ".6f") + \
                                            " [{:.6f}, {:.6f}]".format(planet.fit_results["(-)"]["$P$"],
                                                                       planet.fit_results["(+)"]["$P$"]) + " d\n" + \
                                            "$r_p/R_*$ = " + format(planet.fit_results["median"]["$r_p/R_*$"], ".5f") + \
                                            " [{:.5f}, {:.5f}]".format(planet.fit_results["(-)"]["$r_p/R_*$"],
                                                                       planet.fit_results["(+)"]["$r_p/R_*$"]) + "\n" + \
                                            "       (" + format(planet.fit_results["median"]["$r_p$"][0], ".5f") + \
                                            " [{:.5f}, {:.5f}]".format(planet.fit_results["(-)"]["$r_p$"][0],
                                                                       planet.fit_results["(+)"]["$r_p$"][
                                                                           0]) + " EarthRad)\n" + \
                                            "$a/R_*$ = " + format(planet.fit_results["median"]["$a/R_*$"], ".4f") + \
                                            " [{:.4f}, {:.4f}]".format(planet.fit_results["(-)"]["$a/R_*$"],
                                                                       planet.fit_results["(+)"]["$a/R_*$"]) + "\n" + \
                                            "       (" + format(planet.fit_results["median"]["$a$"], ".5f") + \
                                            " [{:.5f}, {:.5f}]".format(planet.fit_results["(-)"]["$a$"],
                                                                       planet.fit_results["(+)"]["$a$"]) + " AU)\n" + \
                                            "$b$ = " + format(planet.fit_results["median"]["$b$"], ".4f") + \
                                            " [{:.4f}, {:.4f}]".format(planet.fit_results["(-)"]["$b$"],
                                                                       planet.fit_results["(+)"]["$b$"]) + "\n"
                                            ), size=13)

        # adjust axes
        for ax in axes:
            ax.set_aspect('equal')
            ax.set_aspect('equal')
            ax.set_ylim([-5.5, 5.5])
            ax.set_xlim([-5, max_x + 20])
            ax.axis("off")

        # save figure
        fig.savefig("{}/system_overview_{}.pdf".format(save_to_path, tic_no))
        plt.close()

        return None


    def _inject_(self, time, flux, t0, per, rp, a, inc, baseline=1., q1=None, q2=None):
        """Helper function to generate a single batman transit model

        @param time: time array
        @type time: numpy array

        @param flux: flux array
        @type flux: numpy array

        @param t0: time of first transit
        @type t0: float

        @param per: period
        @type per: float

        @param rp: planet radius (units of stellar radius)
        @type rp: float

        @param a: semi-major axis (units of stellar radius)
        @type a: float

        @param inc: planet inclination (units of degrees)
        @type inc: float (optional; default=90.)

        @param baseline: average out-of-transit flux
        @type baseline: float (optional; default=1.)

        @param q1: Kipping et al. (2013) LD parameter #1
        @type q1: float (optional; default=None)

        @param q2: Kipping et al. (2013) LD parameter #2
        @type q2: float (optional; default=None)

        @return: initialized light curve with injected planet signal, model light curve
        """

        lc = self._model_single_transit_(time, t0, per, rp, a,
            inc=inc, baseline=baseline, q1=q1, q2=q2)

        return np.array(flux * lc), lc


    def _recover_(self, time, flux, flux_err, t0, period, ground_truth_model, t0_tolerance = 0.01,
        max_iterations=7, tce_threshold=8.0, make_plots=False, show_plots=False,
        raw_flux=True, window_size=3.0):

        """Function to recover signals from light curve

        @param time: time array
        @type time: numpy array

        @param flux: flux array
        @type flux: numpy array

        @param flux_err: flux uncertainty array
        @type flux_err: numpy array

        @param period: orbital period of signal to recover (units of days)
        @type period: float

        @param t0: transit time of signal to recover
        @type t0: float

        @param ground_truth_model: the ground truth injected transit model used for initial SNR heuristic
        @type ground_truth_model: array

        @param t0_tolerance: required t0 agreement of signal (as fraction of period)
        @type t0_tolerance: float

        @param max_iterations: maximum number of search iterations if SDE threshold is never reached
        @type max_iterations: int (optional; default=7)

        @param tce_threshold: Minimum Signal Detection Efficiency (SDE) that counts as a Threshold Crossing Event (TCE)
        @type tce_threshold: float (optional; default=8.0)

        @param make_plots: make plots of periodogram and best TLS model
        @type make_plots: bool (optional; default=True)

        @param show_plots: show plots of periodogram and best TLS model
        @type show_plots: bool (optional; default=False)

        @param raw_flux: bool for whether flux is raw (and therefore should be flattened)
        @type raw_flux: bool (optional; default=True)

        @param window_size: median smoothing filter window size in days (only used if raw_flux == True)
        @type window_size: float (optional; default=3.0)

        @return: recovery success (bool)
        """

        # flatten light curve if input is not already flattened
        ##########################################
        ### Needs debugging @ Andy ##############
        if raw_flux:
            flux_err = flux_err / flux
            # flatten flux with running median filter (Wotan package)
            flux = flatten(
                time,                                   # Array of time values
                flux,                                   # Array of flux values
                method='median',                        # median filter
                window_length=window_size,              # The length of the filter window in units of ``time``
                edge_cutoff=0.0,                        # length (in units of time) to be cut off each edge.
                break_tolerance=0.5,                    # Split into segments at breaks longer than that
                return_trend=False                      # Return trend and flattened light curve
                )
        ##########################################


        # check if ground truth model does better than a straight line ~~~~~~~~
        # chi2 of ground truth model
        ground_truth_chi2 = np.sum((flux - ground_truth_model) ** 2 / flux_err ** 2)

        # fit a line
        linear_func = lambda x, m, b: m * x + b
        popt, _ = curve_fit(linear_func, time, flux, [0, 1], sigma=flux_err)
        flux_line = linear_func(time, *popt)

        # chi2 of line fit
        line_chi2 = np.sum((flux - flux_line) ** 2 / flux_err ** 2)

        delta_chi2 = line_chi2 - ground_truth_chi2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        recovery = False

        # run TLS search only if ground truth model is preferred at 5 sigma (Dressing+2015)
        if delta_chi2 > 30.863:
            # mask previous transits
            intransit = np.zeros(len(self.time_raw), dtype=bool)

            # see if injected signal is recovered
            for i in range(max_iterations):
                TCEs,intransit = self._tls_search_(max_iterations=1, tce_threshold=tce_threshold, time=time, flux=flux,
                                     flux_err=flux_err, mask=intransit, period_min=period*0.95, period_max=period*1.05, make_plots=make_plots, show_plots=show_plots)
                                     #flux_err=flux_err, mask=intransit, make_plots=make_plots, show_plots=show_plots)

                # stop searching if no TCEs found
                if len(TCEs) == 0:
                    break

                TCE = TCEs[0] # only run _tls_search_ one iteration at a time

                if abs(TCE.period - period) / TCE.period_uncertainty > 5:
                    continue

		# fractional phase difference between injected and recovered T0
                phase_diff = (abs(TCE.T0 - t0) % period) / period
		# check both sides of phase difference
                if min(phase_diff, abs(phase_diff - 1)) > t0_tolerance:
                    continue

                if TCE.SDE < tce_threshold:
                    continue

                recovery = True

                break

        return recovery


    def _explore_(self, time, flux, flux_err, mstar, rstar,
        smart_search=False, smart_search_Nbins=10, smart_search_Nperbin=10,
        baseline_min=1, baseline_max=1, baselines=None,
        q1_min=0.25, q1_max=0.25, q1s=None,
        q2_min=0.25, q2_max=0.25, q2s=None,
        t0_min=None, t0_max=None, t0s=None,
        period_min=0.5, period_max=200, periods=None,
        radius_min=0.5, radius_max=4, radii=None,
        b_min=0, b_max=1, bs=None,
        N=25, seed=None, t0_tolerance=0.01,
        max_iterations=7, tce_threshold=8.0, show_plots=False,
        raw_flux=True, window_size=3.0):
        """Function to explore multiple injections & recoveries

        @param time: time array
        @type time: numpy array

        @param flux: flux array
        @type flux: numpy array

        @param flux_err: flux uncertainty array
        @type flux_err: numpy array

        @param mstar: stellar mass (Solar mass)
        @type mstar: float

        @param rstar: stellar radius (Solar radii)
        @type mstar: float

        @smart_search:

        baseline_min: flux baseline minimum
        baseline_max: flux baseline maximum
        baselines: flux array (overrides baseline_min and baseline_max)
        q1_min: q1 limb darkening minimum
        q1_max: q1 limb darkening maximum
        q1s: q1 limb darkening array (overrides q1_min and q1_max)
        q2_min: q2 limb darkening minimum
        q2_max: q2 limb darkening maximum
        q2s: q2 limb darkening array (overrides q2_min and q2_max)
        t0_min: t0 transit time minimum
        t0_max: t0 transit time maximum
        t0s: t0 transit time array (overrides t0_min and t0_max)
        period_min: orbital period minimum (days)
        period_max: orbital period maximum (days)
        periods: orbital period array (overrides period_min and period_max)
        radius_min: planetary radius minimum (Earth radii)
        radius_max: planetary radius maximum (Earth radii)
        radii: planetary radius array (overrides radius_min and radius_max)
        b_min: impact parameter minimum
        b_max: impact parameter maximum
        bs: impact parameter array (overrides b_min and b_max)
        N: number of sample draws to run and return
        seed: random seed input to gaurantee repeatability

        @param t0_tolerance: required t0 agreement of signal (as fraction of period)
        @type t0_tolerance: float

        @param max_iterations: maximum number of search iterations if SDE threshold is never reached
        @type max_iterations: int (optional; default=7)

        @param tce_threshold: Minimum Signal Detection Efficiency (SDE) that counts as a Threshold Crossing Event (TCE)
        @type tce_threshold: float (optional; default=8.0)

        @param show_plots: show plots of periodogram and best TLS model
        @type show_plots: bool (optional; default=False)

        @param raw_flux: bool for whether flux is raw (and therefore should be flattened)
        @type raw_flux: bool (optional; default=True)

        @param window_size: median smoothing filter window size in days (only used if raw_flux == True)
        @type window_size: float (optional; default=3.0)
        """

        # random seed initialization
        if isinstance(seed,int) or isinstance(seed,float):
            np.random.seed(int(seed))

        # function to calculate transit duration
        def calc_t_tot(ars, inc, rprs, b, per):
            '''
            ars: semi-major axis / stellar radius
            inc: inclination (degrees)
            rprs: planet radius / stellar radius
            b: impact parameter
            per: orbital period

            returns t_tot: transit duration (same units at per)
            '''
            asini = ars*np.sin(np.pi*inc/180)
            t_tot = per*np.arcsin(np.sqrt((1+rprs)**2-b**2)/asini)/np.pi
            return t_tot

        # function to calculate inclination
        def calc_inc(acosi,asini):
            return 90. if acosi == 0. \
                else (180./np.pi)*np.arctan(asini/acosi)

        # calculates semimajor axis in solar radii
        # given input period (days) and stellar mass (solar masses)
        def calc_a(mstar,per):
            G = 6.67408e-8 # cm^3 g^-1 s^-2
            # convert from solar masses to grams
            mstar_in_g = 1.988e33*mstar
            # convert from days to seconds
            per_in_s = 24.*3600*per
            cm_in_solar_radius = 6.957e10
            return ((G*mstar_in_g*per_in_s**2)/(4*np.pi**2))**(1./3)\
                /cm_in_solar_radius

        # calculates a/R* given input period (days),
        # stellar mass (solar masses), and stellar radius (solar radii)
        def calc_ars(mstar,per,rstar):
            semaxis = calc_a(mstar,per)
            return semaxis/rstar

        # Solar radius / Earth radius = 109.1
        rsun_to_rearth = 109.1

        # overwrite and recalculate N if using smart_search
        if smart_search:
            N = smart_search_Nbins * smart_search_Nperbin

        # create input parameter samples (as needed)
        if baselines is not None:
            assert len(baselines) == N, 'baselines not length N'
        else:
            baselines = np.random.uniform(baseline_min,baseline_max,N)
        if q1s is not None:
            assert len(q1s) == N, 'q1s not length N'
        else:
            q1s = np.random.uniform(q1_min,q1_max,N)
        if q2s is not None:
            assert len(q2s) == N, 'q2s not length N'
        else:
            q2s = np.random.uniform(q2_min,q2_max,N)
        if bs is not None:
            assert len(bs) == N, 'bs not length N'
        else:
            bs = np.random.uniform(b_min,b_max,N)
        if t0s is not None:
            assert len(t0s) == N, 't0s not length N'
        else:
            if t0_min is None:
                t0_min = min(time)
            if t0_max is None:
                t0_max = max(time)
            t0s = np.random.uniform(t0_min,t0_max,N)

        # pick and explore all periods and radii at once
        if not smart_search:
            if periods is not None:
                assert len(periods) == N, 'periods not length N'
            else:
                periods = 10**np.random.uniform(
                    np.log10(period_min),np.log10(period_max),N)
            if radii is not None:
                assert len(radii) == N, 'radii not length N'
            else:
                radii = np.random.uniform(radius_min,radius_max,N)/\
                    (rsun_to_rearth*rstar)

            # calculate remaining needed parameters
            ars = calc_ars(mstar,periods,rstar)
            asinis = np.sqrt(ars**2 - bs**2)
            incs = list(map(calc_inc,bs,asinis))

            # assemble parameters into single array
            thetas = np.array((baselines,q1s,q2s,
                t0s,periods,radii,ars,incs)).T

            # helper function to map onto in order to perform injection/recovery
            def helper(theta,time=time,flux=flux,flux_err=flux_err):
                baseline,q1,q2,t0,per,rp,ars,inc = theta
                print(theta)
                flux,flux_model = self._inject_(time, flux, t0, per, rp, ars, inc, baseline, q1, q2)
                return self._recover_(time, flux, flux_err, t0, per, flux_model,
                    t0_tolerance, max_iterations, tce_threshold, show_plots,
                    raw_flux, window_size)
            results = list(map(helper,thetas))

            # save output in object and return
            self.injection_recovery_results = [thetas,results]

        # if smart_search, bin periods and explore radii sequentially
        else:
            thetas = []
            results = []

            # create period bins in log space
            period_bins = np.logspace(np.log10(period_min),np.log10(period_max),smart_search_Nbins+1)

            for i in range(len(period_bins)-1):
                period_bin_min = period_bins[i]
                period_bin_max = period_bins[i+1]
                per = np.random.uniform(period_bin_min,period_bin_max)

                # start at 1% transit depth radius (if within range)
                rprs = np.sqrt(0.01)
                radius = rprs * rsun_to_rearth * rstar
                radius = max(radius,radius_min)
                radius = min(radius,radius_max)
                rprs = radius / (rsun_to_rearth * rstar)

                for j in range(smart_search_Nperbin):
                    print('Testing Rp = ' + str(radius) + ' Rearth, P = ' + str(per) + ' d')
                    baseline = baselines[i*smart_search_Nbins + j]
                    q1 = q1s[i*smart_search_Nbins + j]
                    q2 = q2s[i*smart_search_Nbins + j]
                    b = bs[i*smart_search_Nbins + j]
                    t0 = t0s[i*smart_search_Nbins + j]

                    # calculate remaining needed parameters
                    ar = calc_ars(mstar,per,rstar)
                    asini = np.sqrt(ar**2 - b**2)
                    inc = calc_inc(b,asini)
                    rprs = radius / (rsun_to_rearth * rstar)

                    # run injection/recovery once and save input/output
                    theta = np.array((baseline,q1,q2,t0,per,rprs,ar,inc))
                    new_flux,new_flux_model = self._inject_(time, flux, t0, per, rprs, ar, inc, baseline, q1, q2)
                    result = self._recover_(time, new_flux, flux_err, t0, per, new_flux_model,
                        t0_tolerance, max_iterations, tce_threshold, show_plots,
                        raw_flux, window_size)
                    thetas += [theta]
                    results += [result]
                    print('PASS') if result else print('FAIL')

                    ###
                    #look at 85% lower limit of passes and 85% upper limit of failures
                    #average of two values should be estimate of boundary median
                    #half the difference should be estimate of boundary std dev
                    #draw from norm(boundary median, boundary std dev) truncated at radius boundaries
                    #repeat ad nauseum
                    ###

                    # don't find new period and radius when you just did the last injection in your current period bin
                    if j+1 < smart_search_Nperbin:
                        # select radii inside period bin range
                        bin_rprs = np.array(thetas)[np.logical_and(np.array(thetas)[:,4] > period_bin_min, np.array(thetas)[:,4] < period_bin_max)][:,5]
                        bin_radii = bin_rprs * rsun_to_rearth * rstar
                        bin_results = np.array(results)[np.logical_and(np.array(thetas)[:,4] > period_bin_min, np.array(thetas)[:,4] < period_bin_max)]
                        # separate radii that passed injection from those that failed
                        pass_radii = np.append(bin_radii[bin_results],radius_max) # include radius_max as defacto pass as initial distribution broadener
                        fail_radii = np.append(bin_radii[~bin_results],radius_min) # include radius_min as defacto fail as initial distribution broadener
                        # estimate pass/fail boundary as normal CDF
                        lower_1sigma_est = np.percentile(pass_radii,15.865,interpolation='linear') # CDF percentile to get lower end of 1 sigma normal PDF
                        upper_1sigma_est = np.percentile(fail_radii,84.135,interpolation='linear') # CDF percentile to get upper end of 1 sigma normal PDF
                        norm_med_est = (upper_1sigma_est+lower_1sigma_est)/2 # average provides estimate of boundary center
                        norm_std_est = abs(upper_1sigma_est-lower_1sigma_est)/2 # half the difference provides estimats of boundary std dev width
                        print('1')

                        # draw next radius to explore from estimated normal dist (truncated at radius_min and radius_max
                        radius = np.random.normal(loc=norm_med_est,scale=norm_std_est)
                        count = 0
                        while radius > radius_max or radius < radius_min:
                            count += 1
                            if count == 10:
                                import pdb; pdb.set_trace()
                            print('2')
                            print(norm_med_est,norm_std_est)
                            radius = np.random.normal(loc=norm_med_est,scale=norm_std_est)

                        # draw a new random period as well
                        per = np.random.uniform(period_bin_min,period_bin_max)
                        print('3')

                print('Period bin ' + str(i+1) + ' of ' + str(smart_search_Nbins) + ' (' + str(period_bin_min) + ' d - ' + str(period_bin_max) + 'd) complete')

            # save output in object and return
            self.injection_recovery_results = [thetas,results]

        return thetas, results




if __name__ == "__main__":
	# TESTS
	print("    Running test injection/recovery...")
	test_time = np.linspace(1,76,3600)
	test_flux_err = np.array([1e-3]*len(test_time))
	test_flux = 1 + np.random.randn(len(test_time))*test_flux_err
	#test_time, test_flux, test_flux_err = cleaned_array(self.time, self.f, self.f_err)
	test_per = 28.7
	#test_t0 = self.time[0] + test_per
	test_t0 = test_time[0] + test_per
	#test_rp = 0.02
	test_rp = 0.1
	test_a = 25.0
	test_inc = 90.
	test_baseline = 1.
	test_q1 = 0.25
	test_q2 = 0.25
	mstar = 1 * u.M_sun # solar mass
	rstar = 1 * u.R_sun # solar radius
	TransitFitterObject = TransitFitter(111111111)
	TransitFitterObject.R_star = rstar
	TransitFitterObject.M_star = mstar
	TransitFitterObject.u1 = test_q1
	TransitFitterObject.u2 = test_q2

	N = 50
	test_periods = np.random.shuffle(np.linspace(0.5,100,N))
	test_radii = np.random.shuffle(np.linspace(0.5,8,N))
	test_injected_lc, ground_truth = TransitFitter._inject_(TransitFitterObject, test_time, test_flux, test_t0, test_per, test_rp, test_a, test_inc, test_baseline, test_q1, test_q2)

	TransitFitterObject.lc = test_injected_lc
	TransitFitterObject.time_raw = test_time
	TransitFitterObject.f_raw = test_flux
	TransitFitterObject.ferr_raw = test_flux_err

	trial_planets,results = TransitFitter._explore_(TransitFitterObject, test_time, test_injected_lc, test_flux_err, mstar.value, rstar.value,
		smart_search=True,smart_search_Nbins=20, smart_search_Nperbin=20, periods=test_periods,radii=test_radii,N=N,show_plots=False,raw_flux=False, radius_min=1, radius_max=10,period_max=100)
	#test_recover = TransitFitter._recover_(TransitFitterObject, test_time, test_injected_lc, test_flux_err, test_t0, test_per, ground_truth,
	#			      raw_flux=False)
	#print("    Test injection/recovery done.\n")
	#print("    Recovered injected planet? {} \n".format(test_recover))


	#t = np.linspace(0,75,3600)
	#yerr = 1e-3
	#y = 1 + np.random.randn(len(t))*yerr
	##periods = np.linspace(5,50,46)
	##radii = np.linspace(0.5,5,10)/109.1 # convert from earth to sun radii
	##durations = np.linspace(1,5,5)/24
	#
	##results = explore(t, y, yerr, periods, radii, durations)
	##import pdb; pdb.set_trace()
	##results = np.random.randint(0,2,len(periods)*len(radii)).reshape((len(periods),len(radii)))
	#
	#mstar = 1 # solar mass
	#rstar = 1 # solar radius
	#TransitFitterObject = TransitFitter(111111111)
	#trial_planets,results = TransitFitter._explore_(TransitFitterObject,t, y, yerr, mstar, rstar)
	print(trial_planets,results)


	fig, ax = plt.subplots(1,figsize=(10,10))
	for i in range(len(trial_planets)):
		baseline,q1,q2,t0,per,rp,a,inc = trial_planets[i]
		recovered = results[i]
		fmt = 'bo' if recovered else 'ro'
		ax.plot(per,rp*109.1,fmt)

	#for i,period in enumerate(periods):
	#	for j,rp in enumerate(radii):
	#		if results[i][j]:
	#			ax.plot(period,rp*109.1,'ko')

	ax.set_xlabel('Period (d)')
	ax.set_ylabel('Radius (R_earth)')
	fig.savefig('test_recovery.png')
	plt.show()
