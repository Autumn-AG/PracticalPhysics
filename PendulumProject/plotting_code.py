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

def GetXFromTime(x, length, tp, angle, ) -> float:
    # x = time
    pass       

##############  CONSTANTS  ################

PI = np.pi
D = 0.048
D_UNC = 0.003   

filenames = ['d1','d2','d3','d4','d5','d6',     # Diff angles for 50g
             'v1','v2','v3','v4',               # Diff angles for 200g
             'v5','v6','v7','v8',               # Diff angles for 100g
             't50','t100','t200','t500',        # Diff weights for 45deg, 28
             'v9','v6','l1','v10','l2']         # Diff lengths for 45deg, 100g

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
             0.225, 0.275, 0.31, 0.325, 0.34]

lengths_unc = np.full_like(masses, 0.001)

for i in range(len(lengths)):
    lengths[i] += D
    lengths_unc[i] += D_UNC


tp =        [1.140, 1.177, 1.234, 1.149, 1.181, 1.22,
             1.198, 1.229, 1.224, 1.191,
             1.151, 1.157, 1.172, 1.173, 
             1.358, 1.358, 1.371, 1.383,
             1.058, 1.157, 1.214, 1.25, 1.276]

tp_unc =    [0.141, 0.127, 0.141, 0.129, 0.128, 0.126,
             0.148, 0.129, 0.138, 0.119, 
             0.129, 0.132, 0.156, 0.169, 
             0.149, 0.150, 0.152, 0.156, 
             0.139, 0.132, 0.136, 0.147, 0.144]

gam = [0.010458335335121503, 0.013247652068825343, 0.009904609889511043, 0.013105510635313535, 0.01214853489943396, 0.011500096515356677,
       0.012580352654617944, 0.013220946651987895, 0.013630543610345951, 0.013174622521843864,
       0.01003164797349607, 0.009912694780531143, 0.010882733653438108, 0.012354875602465157, 
       0.01335731603169881, 0.010827006279744087, 0.011532402943123805, 0.012513085227963766, 
       0.009986577850603516, 0.009912694780531143, 0.014737040682874934, 0.008372580296589459, 0.013520019521597206]

gam_unc = [0.004865736792663557, 0.0020676988224864745, 0.0014905125064174837, 0.00663624493863826, 0.002588673322584778, 0.001500849023341298,
           0.001260086143818321, 0.0013759861573432554, 0.0013694168912830693, 0.0012356047946558157,
           0.0001753934018954738, 0.00018740817287225478, 0.00015349148328803604, 0.00012290509381595422, 
           5.0833882492973425e-05, 2.6490593918087442e-05, 5.0516963019294445e-05, 5.398714495638439e-05, 
           0.00013301097017932794, 0.00018740817287225478, 5.5835561181287796e-05, 0.00012265924145652018, 5.2346302171867704e-05]

# tau = 1/np.array(gam)
# tau_unc = []
# for i in range(len(tau)):
#     ratio = gam_unc[i] / gam[i]
#     tau_unc.append(tau[i] * ratio)
    
#     tau[i] = round(tau[i], 3)
#     tau_unc[i] = round(tau_unc[i], 3)

tau = [77.618,  75.485, 78.963,  82.304,  80.314,  83.956, 
       79.489,  75.638,  73.365,  75.904,  
       99.685, 100.881,  91.889,  80.940,
       74.865,  92.362,  86.712,  79.916,
       100.134, 100.881,  109.856, 119.437, 123.964] 

tau_unc = [7.486, 7.782, 7.194, 7.638, 7.540, 7.348,
           7.962,  7.872,  7.371,  7.119,
           1.743,  1.907,  1.296,  0.805, 
           0.285,  0.226,  0.380,  0.345,  
           1.334,  1.907,  0.257,  1.750,  0.286]


tp = np.array(tp)
tp_unc = np.array(tp_unc)
tau_unc = np.array(tau_unc)

mass_ind = [14, 15, 16, 17]
angle_ind = [0, 1, 2, 3, 4, 5]
length_ind = [18, 19, 20, 21, 22]

# filenames = ['d1','d2','d3','d4','d5','d6',     # Diff angles for 50g
#              'v1','v2','v3','v4',               # Diff angles for 200g
#              'v5','v6','v7','v8',               # Diff angles for 100g
#              # Ind 14 onwards
#              't50','t100','t200','t500',        # Diff weights for 45deg, 28
#              # Ind 18 onwards
#              'v9','v6','l1','v10','l2']         # Diff lengths for 45deg, 100g

def plotr(a1, a2, xdata, x_unc, ydata, y_unc, fit, name, xlab, ylab):
    clr="#8000d0"
    print(name)
    popt, pcov = curve_fit(fit, xdata, ydata, sigma = y_unc, absolute_sigma=True)

    x_bestfit = np.linspace(min(xdata)*0.9, max(xdata)*1.1, 1000)
    y_bestfit = fit(x_bestfit, *popt)

    a1.scatter(xdata, ydata, marker='x', s=30, alpha=1, label="Data", c='#ff9b00')
    a1.errorbar(xdata, ydata, xerr=x_unc, yerr=y_unc, solid_capstyle='projecting', capsize=1.8, fmt='none', c='#2050C0', alpha = 0.4)
    a1.plot(x_bestfit, y_bestfit, alpha=0.85, ls='--', label=" Best Fit", c=clr)

    a1.set_xlabel(xlab, fontsize=14)
    a1.set_ylabel(ylab, fontsize=14)
    a1.grid(which='major', color='#CCCCCC', linewidth=0.8)
    a1.grid(which='minor', color='#DDDDDD', linewidth=0.3)
    a1.minorticks_on()  
    a1.legend(loc=1, fontsize=11)
    
    # print(xdata, type(xdata))
    # print(*popt)
    # print("\nmiau\n")

    y_fit = fit(xdata, *popt)
    residuals = (ydata - y_fit)

    dof = len(xdata) - 1
    chisq = np.sum((residuals)**2 / y_fit)

    resylab = "Residuals of " + ylab
    
    a2.scatter(xdata, residuals, marker='x', s=30, alpha=1, label="Data", c='#ff9b00')
    a2.errorbar(xdata, residuals, xerr=x_unc, yerr=y_unc, solid_capstyle='projecting', capsize=1.5, fmt='none', c='#2050C0', ms=5, alpha=0.45, label="Residuals")
    a2.axhline(0, ls='--', c=clr, label="Best Fit")  # Reference line at zero for residuals
    a2.set_xlabel(xlab, fontsize=14)
    a2.set_ylabel(resylab, fontsize=14)
    a2.grid(which='major', color='#CCCCCC', linewidth=0.8)
    a2.legend(loc=1, fontsize=11)

    print(name, "\n params:", *popt, "unc params", np.sqrt(np.diag(pcov)), "\n Chisq: ", "%f"%chisq, "Dof: ", "%.2f"%dof, " CHSQP: ", "%.2f"%(1-chi2.cdf(chisq,dof))    )

if True:
    a1 = []
    a1_unc = []
    tp1 = []
    tp1_unc = []
    tau1 = []
    tau1_unc = []

    for ind in angle_ind:
        a1.append(angles[ind])   
        a1_unc.append(angles_unc[ind])
        tp1.append(tp[ind])
        tp1_unc.append(tp_unc[ind])
        tau1.append(tau[ind])
        tau1_unc.append(tau_unc[ind])

    a1 = np.abs(np.array(a1))
    a1_unc = np.abs(np.array(a1_unc))
    tp1 = np.array(tp1)
    tp1_unc = np.array(tp1_unc)
    tau1 = np.array(tau1)
    tau1_unc = np.array(tau1_unc)

    l2 = []
    l2_unc = []
    tp2 = []
    tp2_unc = []
    tau2 = []
    tau2_unc = []

    for ind in length_ind:
        l2.append(lengths[ind])   
        l2_unc.append(lengths_unc[ind])
        tp2.append(tp[ind])
        tp2_unc.append(tp_unc[ind])
        tau2.append(tau[ind])
        tau2_unc.append(tau_unc[ind])

    l2 = np.array(l2)
    l2_unc = np.array(l2_unc)
    tp2 = np.array(tp2)
    tp2_unc = np.array(tp2_unc)
    tau2 = np.array(tau2)
    tau2_unc = np.array(tau2_unc)

    m3 = []
    m3_unc = []
    tp3 = []
    tp3_unc = []
    tau3 = []
    tau3_unc = []

    for ind in mass_ind:
        m3.append(masses[ind])   
        m3_unc.append(masses_unc[ind])
        tp3.append(tp[ind])
        tp3_unc.append(tp_unc[ind])
        tau3.append(tau[ind])
        tau3_unc.append(tau_unc[ind])

    m3 = np.array(m3)
    m3_unc = np.array(m3_unc)
    tp3 = np.array(tp3)
    tp3_unc = np.array(tp3_unc)
    tau3 = np.array(tau3)
    tau3_unc = np.array(tau3_unc)


angler = True
lengther = False
masser = False
plotting = True

# Angle
if angler:
    # fig, ((p1, p2)) = plt.subplots(nrows=1, ncols=2, layout='constrained')
    # plotr(p1, p2, a1, a1_unc, tp1, tp1_unc, myLin, "Angles vs Time Period", "Angle of release (rad)", "Period of Oscillation (s)")
    fig, ((p3, p4)) = plt.subplots(nrows=1, ncols=2, layout='constrained')
    plotr(p3, p4, a1, a1_unc, tau1, tau1_unc, myLin, "Angles vs Decay Constant", "Angle of release (rad)", r"Constant of Decay $\tau$ (s)")

# Length
if lengther:
    # fig, ((p5, p6)) = plt.subplots(nrows=1, ncols=2, layout='constrained')
    # plotr(p5, p6, l2, l2_unc, tp2, tp2_unc, mySqrt, "Length vs Time Period", "Length of Pendulum (m)", "Period of Oscillation (s)")
    fig, ((p7, p8)) = plt.subplots(nrows=1, ncols=2, layout='constrained')

    plotr(p7, p8, l2, l2_unc, tau2, tau2_unc, mySqrt, "Length vs Decay Constant", "Length of Pendulum (m)", r"Constant of Decay $\tau$ (s)")

# Mass
if masser:
    # fig, ((p9, p10)) = plt.subplots(nrows=1, ncols=2, layout='constrained')
    # plotr(p9, p10, m3, m3_unc, tp3, tp3_unc, myLin, "Mass vs Time Period", "Mass of Pendulum (g)", "Period of Oscillation (s)")
    fig, ((p11, p12)) = plt.subplots(nrows=1, ncols=2, layout='constrained')
    plotr(p11, p12, m3, m3_unc, tau3, tau3_unc, myLin, "Mass vs Decay Constant", "Mass of Pendulum (g)", r"Constant of Decay $\tau$ (s)")

if plotting:
    plt.show()
