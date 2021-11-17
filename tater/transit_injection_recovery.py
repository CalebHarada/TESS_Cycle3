import numpy as np
import matplotlib.pyplot as plt
import batman
from astropy.timeseries import BoxLeastSquares as bls
import astropy.units as u
import sys
sys.path.insert(1,'/Users/mayo/Dropbox/packages')
from semaxis import calc_a, calc_ars

nparams_per_planet = 5

# model light curve using Kreidberg's BATMAN model
# input t is the time series
# input theta takes following form:
#     [baseline level, log(Jitter), q1, q2,
#      conjunction time 1, per 1, log10(Rp/R*) 1, t_tot 1, b 1,
#      ...
#      conjunction time n, per n, log10(Rp/R*) n, t_tot n, b n]
# where n=nplanets. Total # of parameters = 4+5*n
def model(theta, t):
	bline,q1,q2 = theta[:3]

	# reparameterize limb darkening parameters (see Kipping (2013))
	u1 = 2*np.sqrt(q1)*q2
	u2 = np.sqrt(q1)*(1-2*q2)

	# reshape list of remaining parameters into planet-per-row array
	planets = theta[3:].reshape(-1,nparams_per_planet)

	m_flux = np.zeros(len(t))
	for planet in planets:
		'''
		t0,per,rp,t_tot,b = planet
		# assumes a is in units of R*
		acosi = b
		# derived from eqn for total transit duration
		# assumes a and rp are in units of R*
		asini = np.sqrt((1+rp)**2-b**2)/\
			np.sin(np.pi*t_tot/per)
		a = np.sqrt(acosi**2+asini**2)
		inc = 90. if acosi == 0. else (180./np.pi)*np.arctan(asini/acosi)
		'''

		t0,per,rp,a,inc = planet

		params = batman.TransitParams()
		params.t0        = t0           # time of inferior conjunction
		params.per       = per          # orbital period
		params.rp        = rp           # planet radius (stellar radii)
		params.a         = a            # semi-major axis (stellar radii)
		params.inc       = inc          # orbital inclination (degrees)
		params.ecc       = 0.           # eccentricity
		params.w         = 0.           # longitude of periastron (degrees)
		params.u         = [u1,u2]      # limb darkening coefficients
		params.limb_dark = "quadratic"  # limb darkening model

		t = np.copy(t) # bugfix from underlying BATMAN code
		exp_time = 1764.944/60/60/24 # exact exposure time for Kepler/K2 in days
		m = batman.TransitModel(params, t, nthreads=1,
			supersample_factor=30, exp_time=exp_time)  # initializes model
		m_flux += m.light_curve(params)                # add on new light curve

	m_flux -= len(planets)-1 # turn sum of planet light curves into average
	m_flux *= bline # allow for possible systematic offset of baseline
	return m_flux

# for usage on normalized data only
def inject(theta, t, y):
	transit = model(theta, t)
	return y*transit

def recover(t, y, yerr, duration, period, snr = 5, tolerance = 0.01, 
	minimum_period=None, maximum_period=None):
	'''
	Input
	t: time series
	y: photometry
	yerr: photometric uncertainties
	duration: transit duration value or array of values to explore
	period: injected period
	snr: SNR over which signal must be detected for successful recovery
	tolerance: allowed deviation between injected and recovered period

	Output: (period, snr, success)
	period: period of recovered signal
	snr: SNR of recovered signal
	success: bool representing whether injected signal was recovered or not
	'''
	bls_obj = bls(t, y, dy=yerr)
	periodogram = bls_obj.autopower(duration, objective="snr",
		minimum_period=minimum_period,maximum_period=maximum_period)
	ixs = np.logical_and(periodogram.period > period*(1-tolerance),
		periodogram.period < period*(1+tolerance))
	success = False if max(periodogram.power[ixs]) < snr else True
	best_ix = np.argmax(periodogram.power[ixs])
	period = periodogram.period[ixs][best_ix]
	snr = periodogram.power[ixs][best_ix]
	return period,snr,success

def explore(t, y, yerr, mstar, rstar,
	baseline_min=1, baseline_max=1, baselines=None,
	q1_min=0.25, q1_max=0.25, q1s=None,
	q2_min=0.25, q2_max=0.25, q2s=None,
	t0_min=None, t0_max=None, t0s=None,
	period_min=0.5, period_max=200, periods=None,
	radius_min=0.5, radius_max=4, radii=None,
	b_min=0, b_max=1, bs=None,
	N=25,seed=None):
	'''
	t: input time array
	y: input flux array
	yerr: input flux uncertainties array
	mstar: stellar mass (Solar masses)
	rstar: stellar radius (Solar radii)
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
	'''

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

	# Solar radius / Earth radius = 109.1
	rsun_to_rearth = 109.1

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
	if t0s is not None:
		assert len(t0s) == N, 't0s not length N'
	else:
		if t0_min is None:
			t0_min = min(t)
		if t0_max is None:
			t0_max = max(t)	
		t0s = np.random.uniform(t0_min,t0_max,N)
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
	if bs is not None:
		assert len(bs) == N, 'bs not length N'
	else:
		bs = np.random.uniform(b_min,b_max,N)

	# calculate remaining needed parameters
	ars = calc_ars(mstar,periods,rstar)
	asinis = np.sqrt(ars**2 - bs**2)
	incs = list(map(calc_inc,bs,asinis))

	# assemble parameters into single array
	thetas = np.array((baselines,q1s,q2s,
		t0s,periods,radii,ars,incs)).T

	# helper function to map onto in order to perform injection/recovery
	def helper(theta):
		baseline,q1,q2,t0,per,rp,ars,inc = theta
		b = ars*np.cos(np.pi*inc/180)
		t_tot = calc_t_tot(ars, inc, rp, b, per)
		return recover(t, inject(theta, t, y), yerr, t_tot, per,
			minimum_period=period_min, maximum_period=period_max)
	'''
	for i,period in enumerate(periods):
		for j,rp in enumerate(radii):
			# integrate across variety of t0, t_tot, and b LATER
			t0, t_tot, b = 0, durations[int(len(durations)/2)], 0
			theta = np.array([baseline, q1, q2, t0, period, rp, t_tot, b])
			injected_y = inject(theta, t, y)
			recovered = recover(t, injected_y, yerr, durations, period,
				minimum_period=min(periods),maximum_period=max(periods))
			recovered_array[i][j] = int(recovered[2])
			print(period,rp*109.1,recovered)
	return recovered_array
	'''
	results = list(map(helper,thetas))
	return thetas,results


t = np.linspace(0,75,3600)
yerr = 1e-3
y = 1 + np.random.randn(len(t))*yerr
#periods = np.linspace(5,50,46)
#radii = np.linspace(0.5,5,10)/109.1 # convert from earth to sun radii
#durations = np.linspace(1,5,5)/24

#results = explore(t, y, yerr, periods, radii, durations)
#import pdb; pdb.set_trace()
#results = np.random.randint(0,2,len(periods)*len(radii)).reshape((len(periods),len(radii)))

mstar = 1 # solar mass
rstar = 1 # solar radius
trial_planets,results = explore(t, y, yerr, mstar, rstar)
print(trial_planets,results)


fig, ax = plt.subplots(1,figsize=(10,10))
for i in range(len(trial_planets)):
	baseline,q1,q2,t0,per,rp,a,inc = trial_planets[i]
	recovered = results[i][2]
	fmt = 'bo' if recovered else 'ro'
	ax.plot(per,rp*109.1,fmt)

'''
for i,period in enumerate(periods):
	for j,rp in enumerate(radii):
		if results[i][j]:
			ax.plot(period,rp*109.1,'ko')
'''

ax.set_xlabel('Period (d)')
ax.set_ylabel('Radius (R_earth)')
fig.savefig('test_recovery.png')
plt.show()
