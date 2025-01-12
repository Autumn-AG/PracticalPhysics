import pickle
import random
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2


font = {'family' : 'DejaVu Sans', 'weight' : 'normal', 'size'   : 16}
rc('font', **font)
fontsize = 14

def myGauss(x, A, mean, width, base):
    return A*np.exp(-(x-mean)**2/(2*width**2)) + base
    # Given fitting function, a Gaussian with a uniform background.

def myLognormal(x, A, mean, width, offset):
    return A * np.exp(- ( (np.log(x + offset) - mean)**2  /  (2* (width**2)) ))
    # Given fitting function, a Lognormal with a uniform background.

def pulse_shape(t_rise, t_fall):
    xx=np.linspace(0, 4095, 4096)
    yy = -(np.exp(-(xx-1000)/t_rise)-np.exp(-(xx-1000)/t_fall))
    yy[:1000]=0
    yy /= np.max(yy)
    return yy

def fit_pulse(x, A):
    _pulse_template = pulse_shape(20,80)
    xx=np.linspace(0, 4095, 4096)
    return A*np.interp(x, xx, _pulse_template)

def myExp(x, A, o):
    output = A * (x - o) * np.exp(- A * (x - o))
    output = np.where(output < 0, 0, output)
    return output

# Stupid pulses
pulse_plotting = False
if pulse_plotting:
    with open("data/calibration_p3.pkl","rb") as file:
        calibration_data=pickle.load(file)
    # pulse_template = pulse_shape(20,80)
    # plt.plot(pulse_template/2000, label='Pulse Template', color='#dd00ed', alpha=1)
    for i in range(1):
        index = random.randint(0, 999)
        plt.plot(calibration_data[f'evt_{index}'], color='#023ADA', alpha=0.5, label="Pulse")

    plt.xlabel(r'Time ($\mu$s)', fontsize=16)
    plt.ylabel('Readout Voltage (V)', fontsize=16)
    plt.legend(loc=1)
    plt.grid(which='major', color='#CCCCCC', linewidth=0.8)
    plt.grid(which='minor', color='#DDDDDD', linewidth=0.5)
    plt.minorticks_on()

    plt.show()

def plot_with_estimator(subplot, en_data, num_bins: int, bin_range: tuple, units: str, fit, clr='b', p0=None) -> float:
    n, bin_edges, _ = subplot.hist(en_data, bins=num_bins, range=bin_range, color='k', histtype='step', label='data', alpha=0.7)
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    sig = np.sqrt(n)
    sig = np.where(sig==0, 1, sig)

    if not(p0):
        popt, pcov = curve_fit(fit, bin_centers, n, sigma = sig, absolute_sigma=True)
    else:
        popt, pcov = curve_fit(fit, bin_centers, n, sigma = sig, absolute_sigma=True, p0=p0)
    
    n_fit = fit(bin_centers, *popt)
    chisquared = np.sum(((n - n_fit)/sig)**2)
    dof = num_bins - len(popt)

    x_bestfit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    y_bestfit = fit(x_bestfit, *popt)
  
    subplot.set_xlabel(f'Calibrated Detector Response ({units})')
    subplot.set_ylabel(f'Events / %4.3f {units}'%((bin_range[-1]-bin_range[0])/num_bins))
    subplot.set_xlim(bin_range)
    subplot.grid(which='major', color='#CCCCCC', linewidth=0.8)

    subplot.errorbar(bin_centers, n, yerr=sig, fmt='none', c='k', alpha = 0.5)
    subplot.plot(x_bestfit, y_bestfit, label='fit', ls="--", c=clr, alpha=0.7)
    
    print(r'$\mu$ = %3.3f V ms'%(popt[1]),
          r'$\sigma$ = %3.3f V ms'%(popt[2]),
          r'$\chi^2$/DOF = %3.2f/%i'%(chisquared,dof),
          r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisquared,dof)))
    print("")
    return float(popt[1])
    # return (float(chisquared/dof), float(1-chi2.cdf(chisquared,dof)))

def plot_residuals(subplot, en_data, num_bins: int, bin_range: tuple, units: str, fit, clr, p0=None) -> tuple:
    n, bin_edges = np.histogram(en_data, bins=num_bins, range=bin_range)
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    sig = np.sqrt(n)
    sig = np.where(sig==0, 1, sig)

    if not(p0):
        popt, pcov = curve_fit(fit, bin_centers, n, sigma = sig, absolute_sigma=True)
    else:
        popt, pcov = curve_fit(fit, bin_centers, n, sigma = sig, absolute_sigma=True, p0=p0)
    
    n_fit = fit(bin_centers, *popt)

    residuals = (n - n_fit)
    subplot.errorbar(bin_centers, residuals, yerr=np.ones_like(sig), fmt='o', c='k', alpha=0.7)
    subplot.axhline(0, ls='--', c=clr)  # Reference line at zero for residuals
    subplot.set_xlabel(f'Calibrated Detector Response ({units})')
    subplot.set_ylabel('Residuals (number of events)')
    subplot.set_xlim(bin_range)
    subplot.grid(which='major', color='#CCCCCC', linewidth=0.8)

    return None

def gen_full_plot(subplot, resplot, mean, en_data, num_bins: int, bin_range: tuple, units: str, fit, fitlabel, clr, p0=None, nodl=False):
    
    scale_factor = (10 / mean)
    data = en_data * scale_factor

    if not nodl:
        n, bin_edges, _ = subplot.hist(data, bins=num_bins, range=bin_range, color='k', histtype='step', label='Data', alpha=0.7)
    else:
        n, bin_edges= np.histogram(data, bins=num_bins, range=bin_range)
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])

    sig = np.sqrt(n)
    sig = np.where(sig==0, 1, sig)

    if not(p0):
        popt, pcov = curve_fit(fit, bin_centers, n, sigma = sig, absolute_sigma=True)
    else:
        popt, pcov = curve_fit(fit, bin_centers, n, sigma = sig, absolute_sigma=True, p0=p0)
    
    n_fit = fit(bin_centers, *popt)
    chisquared = np.sum(((n - n_fit)/sig)**2)   
    dof = num_bins - len(popt)

    x_bestfit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    y_bestfit = fit(x_bestfit, *popt)
  
    subplot.set_xlabel(f'Calibrated Energy Reading ({units})')
    subplot.set_ylabel(f'Events / %4.3f {units}'%((bin_range[-1]-bin_range[0])/num_bins))
    subplot.set_xlim(bin_range)
    subplot.grid(which='major', color='#CCCCCC', linewidth=0.8)

    subplot.errorbar(bin_centers, n, yerr=sig, fmt='none', c='k', alpha = 0.5)
    subplot.plot(x_bestfit, y_bestfit, label=fitlabel, ls="--", c=clr, alpha=0.7)
    subplot.legend(loc=1)

    residuals = (n - n_fit)
    resplot.errorbar(bin_centers, residuals, yerr=np.ones_like(sig), fmt='o', c='k', alpha=0.7, label="Residuals")
    resplot.axhline(0, ls='--', c=clr, label="Best Fit")  # Reference line at zero for residuals
    resplot.set_xlabel(f'Calibrated Detector Response ({units})')
    resplot.set_ylabel('Offset from fit (residuals)')
    resplot.set_xlim(bin_range)
    resplot.grid(which='major', color='#CCCCCC', linewidth=0.8)
    resplot.legend(loc=1)
    
    print("params =", *popt)
    print(r'$\chi^2$/DOF = %3.2f'%(chisquared / dof), 
          r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisquared,dof)))
    print("")

    return None
    return (float(chisquared/dof), float(1-chi2.cdf(chisquared,dof)))



"""Energy Estimators"""
# amp1  = np.zeros(1000)      # max - min estimate 
# amp2  = np.zeros(1000)      # max - baseline estimate 
# area1 = np.zeros(1000)      # Sum across whole trace integral 
# area2 = np.zeros(1000)      # Sum minus baseline integral 58
# area3 = np.zeros(1000)      # Limited integral 32
# pulse_fit = np.zeros(1000)  # Chi-Squared Fit


# with open("data/calibration_p3.pkl","rb") as file:
#     calibration_data=pickle.load(file)
# for ievt in range(1000): 
#     current_data = calibration_data['evt_%i'%ievt]
#     mxm = np.max(current_data)
#     min = np.min(current_data)
#     baseline = np.average(current_data[:1000])
#     # amp calc
#     # amp1[ievt] = mxm - min
#     amplit = float(mxm - baseline)
#     amp2[ievt] = max(amplit, 0)

#     # x = np.linspace(0, 4095, 4096)
#     # cur_popt, cur_pcov = curve_fit(fit_pulse, x, current_data)
#     # pulse_fit[ievt] = np.max(fit_pulse(x, *cur_popt))


#     # amp2[ievt] = max(amplit, 0)
#     # # area calc
#     # area1[ievt] = (np.sum(current_data))
#     # area2[ievt] = (area1[ievt] - (baseline * 4096))
#     # area3[ievt] = (np.sum(current_data[995:1994])) 
# # convert from V to mV
# amp1 *= 1000                 
# amp2 *= 1000
# pulse_fit *= 1000


fig, ((s1)) = plt.subplots(nrows=1, ncols=1, layout='constrained')
fig, ((a1, s2), (s3, s4)) = plt.subplots(nrows=2, ncols=2, layout='constrained')

mean = 0.240319760114011

# # noise = np.zeros(1000) 
# # with open("data/noise_p3.pkl","rb") as file:
# #     noise_data=pickle.load(file)
# # for ievt in range(1000): 
# #     current_data = noise_data['evt_%i'%ievt]
# #     mxm = np.max(current_data)
# #     min = np.min(current_data)
# #     baseline = np.average(current_data[:1000])
# #     amplit = float(mxm - baseline)
# #     noise[ievt] = max(amplit, 0)
# # noise *= 1000

signal = np.zeros(1000) 
with open("data/signal_p3.pkl","rb") as file:
    signal_data=pickle.load(file)
for ievt in range(1000): 
    current_data = signal_data['evt_%i'%ievt]
    mxm = np.max(current_data)
    min = np.min(current_data)
    baseline = np.average(current_data[:1000])
    amplit = float(mxm - baseline)
    signal[ievt] = max(amplit, 0)
signal *= 1000

signal_first = np.delete(signal, np.where(signal*10/mean >= 5))
signal_second = np.delete(signal, np.where(signal*10/mean < 5))

#gen_full_plot(n1, n2, mean, noise, 60, (1.9, 3.8), "keV", fit=myLognormal, fitlabel="Fit", clr='#8000d0')

gen_full_plot(s1, s2, mean, signal, 70, (0, 20), "keV", fit=myExp, fitlabel="Total Fit", clr='r')
gen_full_plot(s1, s3, mean, signal_first, 70, (0, 20), "keV", fit=myExp, fitlabel="Fit >= 5", clr='g', nodl = True)
gen_full_plot(s1, s4, mean, signal_second, 70, (0, 20), "keV", fit=myExp, fitlabel="Fit < 5", clr='b', p0=(80, 1.2, 5), nodl = True)

plt.show()

# # gen_full_plot(a2, mean, pulse_fit, 40, (0.15, 0.38), "keV", fit=myGauss, clr='#8000d0', p0=(70, 10, 2, 4))
# # fig, ((a3)) = plt.subplots(nrows =1, ncols=1) 
# # plot_residuals(a3, pulse_fit, 37, (0.15, 0.38), "mV", fit=myGauss, clr='#8000d0')
# # plt.show()

# # fig, ((a1, a2, a3), (a4, a5, a6)) = plt.subplots(nrows=2, ncols=3, layout='constrained')
# # mean = plot_with_estimator(a1, pulse_fit, 37, (0.15, 0.28), "mV", fit=myGauss, clr='#8000d0')

# # for i in range (210, 600):
# #     try:
# #         if i == 211: i == 119
# #         a = gen_full_plot(a2, mean, pulse_fit, i, (0.15, 0.28), "keV", fit=myGauss, clr='#8000d0')
# #         meow = a[0]         
# #         if 0.5 > abs(meow - 1):
# #             print(i, a)
# #     except RuntimeError or Val:
# #         pass

# # 33, 46
# # mean = plot_with_estimator(a2, amp2, 33, (0.17, 0.31), "mV", fit=myGauss, clr = 'b')
# # gen_full_plot(a3, mean, amp2, 46, (0.17, 0.31), "keV", fit=myGauss, clr='#8000d0', p0=(40, 10, 1, 4))
# # plt.show()

