import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

##########  FUNCTIONS  #############

def myLin(x, m, b) -> float:
    """Estimator function for scipy.optimize.curve_fit, that estimates a linear function y = mx + b"""
    return (m * x) + b

def v_corr(v_m, d) -> float:
    return v_m / (1 - 2.104*(d/D)  + 2.089*((d/D)**3))

def mySqrt(x, a, b) -> float:
    return a * (x)**0.5 + b

def theoreticalX(x, length, angle, gam) -> float:
    # x = time
    ins = angle * np.exp(-gam * np.abs(x))
    return length * (ins)

##############  CONSTANTS  ################

PI = np.pi
D = 0.05
D_UNC = 0.001   

filenames = ['d1','d2','d3','d4','d5','d6',     # Diff angles for 50g
             'v1','v2','v3','v4',               # Diff angles for 200g
             'v5','v6','v7','v8',               # Diff angles for 100g
             't50','t100','t200','t500','v1','v6','l1','v10',        # Diff weights for 45deg, 28
             'l2']         # Diff lengths for 45deg, 100g

masses =    [50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
             200.0, 200.0, 200.0, 200.0,
             100.0, 100.0, 100.0, 100.0,
             50.0,  100.0, 200.0, 500.0,
             100.0, 100.0, 100.0, 100.0, 100.0]

masses_unc = np.full_like(masses, 0.1)

def DegToRad(angle: int) -> float:
    return PI * (angle / 180)

angles =    [-30, -45, -75, +30, +45, +75,
             +45, -60, +60, -45,
             +30, +45, +60, -75,
             +45, +45, +45, +45,
             +45, +45, +45, +45 , +45]

for i in range(len(angles)):
    angles[i] = DegToRad(angles[i])

angles_unc = np.full_like(angles, DegToRad(2))

lengths =   [0.275, 0.275, 0.275, 0.275, 0.275, 0.275,
             0.275, 0.275, 0.275, 0.275,
             0.275, 0.275, 0.275, 0.275, 
             0.5, 0.5, 0.5, 0.5,
             0.224, 0.275, 0.31, 0.325, 0.34]

lengths_unc = np.full_like(masses, 0.001)

tp = []
tp_unc = []
edc = []
edc_unc = []

for i in range(len(filenames)):
    fn = filenames[i]
    print(fn, "Start")
    curr_filename = f"data/{fn}.txt"
    curr_data = np.transpose(np.loadtxt(curr_filename, delimiter=",", skiprows=1))
    # Get daya
    time = curr_data[0]
    x = curr_data[1]
    # Rebase time
    t0 = time[0]
    time = np.where(time > 0, time - t0, time)
    # Find Peaks
    x_tops = []
    unc_tops = []
    t_tops = []
    x_bots = []
    unc_bots =[]
    t_bots = []

    for i in range(1, len(x) - 1):
        if (x[i] >= x[i - 1]) and (x[i] >= x[i+1]):
            x_tops.append(x[i])
            unc_tops.append(0.01)
            t_tops.append(time[i])
        if (x[i] <= x[i - 1]) and (x[i] <= x[i+1]):
            x_bots.append(x[i])
            unc_bots.append(0.01)
            t_bots.append(time[i])
    
    # # Rebase Time Again
    # t0 = min(t_bots, t_tops)
    # bb = np.full(len(t_bots), t0)
    # tb = np.full(len(t_tops), t0)
    # t_tops = t_tops - tb    
    # t_bots = t_bots - bb

    # Time Period Calculation
    top_times = np.diff(t_tops)
    bot_times = np.diff(t_bots)
    tims = np.concatenate((top_times, bot_times))
    tim_mean = np.mean(tims)
    tim_stderr = np.sqrt(np.std(tims))
    tp.append(round(float(tim_mean), 3))
    tp_unc.append(round(float(tim_stderr), 3))

    # Exp Boi Calculation
    top_popt, top_pcov = curve_fit(theoreticalX, t_tops, x_tops, sigma=unc_tops, absolute_sigma=True)
    bot_popt, bot_pcov = curve_fit(theoreticalX, t_bots, np.abs(x_bots), sigma=unc_bots, absolute_sigma=True)


    edc.append(float(top_popt[2] + bot_popt[2]) / 2)
    
    terr = np.sqrt(np.diag(top_pcov))
    berr = np.sqrt(np.diag(bot_pcov))
    edc_unc.append(float(terr[2] + berr[2]) /2)

    fig, ((p1, p2)) = plt.subplots(nrows=1, ncols=2, layout='constrained')

    tops_x_bestfit = np.linspace(t_tops[0], t_tops[-1], 200)
    tops_y_bestfit = theoreticalX(tops_x_bestfit, *top_popt)
    p1.plot(tops_x_bestfit, tops_y_bestfit, label='Crests Fit', c='g', ls="--", alpha=0.5)
    
    bots_x_bestfit = np.linspace(t_bots[0], t_bots[-1], 200)
    bots_y_bestfit = theoreticalX(bots_x_bestfit, *bot_popt)
    p1.plot(bots_x_bestfit, -bots_y_bestfit, label='Troughs Fit', c='r', ls="--", alpha=0.5)
       
    p1.plot(time, x, label='X-Data', c='k', alpha = 0.25)
    #p1.errorbar(t_tops, x_tops, label="Extrema", yerr=unc_tops, fmt='o', c='blue', alpha=0.4) #, capsize=0.2)
    
    #p1.errorbar(t_bots, x_bots, yerr=unc_bots, fmt='o', c='blue', alpha=0.4) #, capsize=0.2)
    p1.errorbar(t_tops, x_tops, yerr=unc_tops, solid_capstyle='projecting', capsize=1.5, fmt='o', c='g', ms=2, alpha=0.45, label="Crest data")
    p1.errorbar(t_bots, x_bots, yerr=unc_bots, solid_capstyle='projecting', capsize=1.5, fmt='o', c='r', ms=2, alpha=0.45, label="Trough data")


    p1.set_xlabel("Time (s)", fontsize = 13)
    p1.set_ylabel("Horizontal displacement (m)", fontsize = 13)
    p1.grid(which='major', color='#CCCCCC', linewidth=0.8)
    p1.grid(which='minor', color='#DDDDDD', linewidth=0.5)
    #p1.title(f"X plot for data {fn}")
    p1.minorticks_on()
    p1.legend(loc=1, fontsize=10)

    top_fit = theoreticalX(t_tops, *top_popt)
    top_res = (x_tops - top_fit)
    
    top_dof = len(x_tops) - 1
    top_chisq = np.sum((top_res)**2 / top_fit)

     
    bot_fit = theoreticalX(t_bots, *bot_popt)
    bot_res = -(x_bots + bot_fit)
    
    bot_dof = len(x_bots) - 1
    bot_chisq = np.sum((bot_res)**2 / bot_fit)

    p2.errorbar(t_tops, top_res, yerr=unc_tops, solid_capstyle='projecting', capsize=1.5, fmt='o', c='g', ms=2, alpha=0.45, label="Crest Residuals")
    p2.errorbar(t_bots, bot_res, yerr=unc_bots, solid_capstyle='projecting', capsize=1.5, fmt='o', c='r', ms=2, alpha=0.45, label="Trough Residuals")

    p2.axhline(0, ls='--', c="k", label="Best Fits")  # Reference line at zero for residuals
    p2.set_xlabel("Time (s)", fontsize = 13)
    p2.set_ylabel("Residuals of displacement (m)", fontsize = 13)
    p2.grid(which='major', color='#CCCCCC', linewidth=0.8)
    p2.legend(loc=4, fontsize=10)

    print(fn, "End", "\n")

    print("TOP", "\n params:", *top_popt, "unc params", np.sqrt(np.diag(top_pcov)), "\n Chisq: ", "%f"%top_chisq, "Dof: ", "%.2f"%top_dof, " CHSQP: ", "%.2f"%(1-chi2.cdf(top_chisq,top_dof))    )
    print("BOT", "\n params:", *bot_popt, "unc params", np.sqrt(np.diag(bot_pcov)), "\n Chisq: ", "%f"%bot_chisq, "Dof: ", "%.2f"%bot_dof, " CHSQP: ", "%.2f"%(1-chi2.cdf(bot_chisq,bot_dof))    )

    plt.show()

print(edc)
print(edc_unc)
