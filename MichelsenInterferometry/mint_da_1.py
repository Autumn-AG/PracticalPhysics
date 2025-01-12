import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

MU_0 = 4 * np.pi * 10**(-7) # Tm/A permiability of free space
N = 130                     # Number of turns in each coil
R = 16.0 * (0.01)           # Radius/Separation of the coils (cm to m)

Voltages = [350.38,
            300.43, 300.43, 300.43,
            275.45, 275.45, 275.45,
            250.48, 250.48, 250.48, 250.48,
            225.51, 225.51, 225.51, 225.51,
            200.53, 200.53, 200.53, 200.53,
            175.52, 175.52, 175.52
            #, 175.52 , 150.52
            ]

Currents = [1.437,
            1.440, 1.648, 1.842,
            1.445, 1.659, 1.834,
            1.350, 1.459, 1.545, 1.645,
            1.259, 1.369, 1.447, 1.641,
            1.201, 1.326, 1.451, 1.556,
            1.151, 1.288, 1.439 
            #, 1.697, 1.615
            ]

Diameters = [11.5,
             10.6, 9.4, 8.4,
             10.2, 8.9, 8.0,
             10.5, 9.7, 9.1, 8.4,
             10.5, 9.6, 8.9, 7.9,
             9.9, 9.0, 8.2, 7.8,
             9.8, 9.0, 7.9
             #, 6.5, 6.1
            ]

v_dict = {#300.4 : [[1.44, 1.65, 1.84],       [10.6, 9.4, 8.4]     ],
          #275.5 : [[1.45, 1.66, 1.83],       [10.2, 8.9, 8.0]     ],
          #250.5 : [[1.35, 1.46, 1.55, 1.65], [10.5, 9.7, 9.1, 8.4]],
          225.5 : [[1.26, 1.37, 1.45, 1.63], [10.5, 9.6, 8.9, 7.8]],
          #200.5 : [[1.20, 1.33, 1.45, 1.56], [9.9 , 9.0, 8.2, 7.7]],
          #175.5 : [[1.15, 1.29, 1.44, 1.70], [9.8 , 9.0, 7.9, 6.5]]
          }

volt_unc = 0.1
curr_unc = 0.01
diam_unc = 0.2

def myLin(x, m, b):
    return (m * x) + b

def calcB (curr, rad):
    const = (4/5) ** (3/2)
    prod = MU_0 * N * curr / R
    Base = const * prod
    return corrB(Base, rad)

def corrB(B, rad):
    const = (rad/R) ** 2
    factor = 1 - ((const **2) / (0.6583 + (0.29 * const)) ** 2)
    return factor * B


plot_currvsdiam = False
if plot_currvsdiam:
    for i in v_dict:
        fig, (a1, a2) = plt.subplots(nrows=1, ncols=2, layout='constrained')
        curr, diam = np.array(v_dict[i][0]), np.array(v_dict[i][1])

        L = len(curr)
        cerr = np.full(L, curr_unc)
        derr = np.full(L, diam_unc)

        popt, pcov = curve_fit(myLin, curr, diam, sigma = derr, absolute_sigma=True)
        punc = np.sqrt(np.diag(pcov))

        d_fit = myLin(curr, *popt)
        residuals = diam - d_fit
        chisq = np.sum((residuals/derr)**2)
        dof = L - 1
        
        x_bestfit = np.linspace(curr[0], curr[-1], 150)
        y_bestfit = myLin(x_bestfit, *popt)
 
        a1.errorbar(curr, diam, xerr=cerr, yerr=derr, fmt='o', c="#023AD0", alpha=0.6, label="I vs. 2r data")
        a1.plot(x_bestfit, y_bestfit, c="#8000E0", ls="--", alpha=0.75, label="Linear Fit")
        a1.set_xlabel(r'Current (A) for %3.1f $\pm$ %2.1f V'%(i, volt_unc), fontsize=14)
        a1.set_ylabel("Diameter of electron Ring (cm)", fontsize=14)
        a1.legend(loc=1)
        a1.grid(which='major', color='#CCCCCC', linewidth=0.6)

        a2.errorbar(curr, residuals, xerr=cerr, yerr=derr, fmt='o', c='#023ADA', alpha=0.6, label="Residuals")
        a2.axhline(0, ls='--', c="#8000E0", label="Best Fit", alpha=0.75)  # Reference line at zero for residuals
        a2.set_xlabel(r'Current (A) for %3.1f $\pm$ %2.1f V'%(i, volt_unc), fontsize=14)
        a2.set_ylim(-0.7, +0.7)
        a2.set_ylabel('Diameter of electron Ring residual (cm)', fontsize=14)
        a2.grid(which='major', color='#CCCCCC', linewidth=0.6)
        a2.legend(loc=1)

        print("VOLTAGE:", i, "\n\t",
            "CHI:", r"%3.2f"%(chisq/dof), r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisq,dof)), "\n\t",
            "Params:", *popt, "\n\t",
            "Unc Params:", punc)

    plt.show()

c_dict = {1.44: [[300.4, 275.5, 225.5, 175.5], 
                 [10.6 , 10.2 , 8.9  , 8.0]]}

vvsdiam = False
if vvsdiam:
    for i in c_dict:
        fig, (v1, v2) = plt.subplots(nrows=1, ncols=2, layout='constrained')
        volts, diam = np.array(c_dict[i][0]), np.array(c_dict[i][1])

        L = len(volts)
        verr = np.full(L, volt_unc)
        derr = np.full(L, diam_unc)

        popt, pcov = curve_fit(myLin, volts, diam, sigma = derr, absolute_sigma=True)

        d_fit = myLin(volts, *popt)
        residuals = diam - d_fit
        chisq = np.sum((residuals/derr)**2)
        dof = 3

        x_bestfit = np.linspace(volts[-1], volts[0], 200)
        y_bestfit = myLin(x_bestfit, *popt)

        unc = np.sqrt(np.diag(pcov))
        
        v1.errorbar(volts, diam, xerr=derr, yerr=verr, fmt='o', c='#023ADA', alpha=0.6, label="V vs d Data")
        v1.plot(x_bestfit, y_bestfit, color="#8000E0", ls=":", alpha=0.75, label = "Linear fit" )
        v1.set_xlabel(r'Volts (V) for %3.2f $\pm$ %3.2f A'%(i, curr_unc), fontsize=14)
        v1.set_ylabel('Diameter of Ring (cm)', fontsize=14)
        v1.legend(loc=2)
        v1.grid(which='major', color='#CCCCCC', linewidth=0.6)
                
        v2.errorbar(volts, residuals, xerr=derr, yerr=verr, fmt='o', c='#023ADA', alpha=0.6, label="Residuals")
        v2.axhline(0, ls='--', c="#8000E0", label="Best Fit", alpha=0.75)  # Reference line at zero for residuals
        v2.set_xlabel(r'Voltage Residual (V)', fontsize=14)
        v2.set_ylabel('Diameter Residual (cm)', fontsize=14)
        v2.set_ylim(-0.35, 0.35)
        v2.grid(which='major', color='#CCCCCC', linewidth=0.6)
        v2.legend(loc=1)


        print("CURR:", i, "\n\t",
            "CHI:", r"%3.2f"%(chisq/dof), r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisq,dof)), "\n\t",
            "Params:", *popt, "\n\t",
            "Unc Params:", unc)

    plt.show()


plot_final = False

if plot_final:
    fig, (b1, b2) = plt.subplots(nrows=1, ncols=2, layout='constrained')

    r_list =[]
    r_rat = []
    prod_list = []
    prod_rat = []

    for d_val in Diameters:
        ind = Diameters.index(d_val)
        c_val = Currents[ind]
        v_val = Voltages[ind]

        r_adj = d_val * 0.01 / 2
        rat_adj = 0.001 / r_adj

        r_list.append(1 / r_adj)
        r_rat.append(rat_adj)
        prod_list.append(calcB(c_val, r_adj) / np.sqrt(v_val))
        prod_rat.append((1/130 + (0.5 * volt_unc/v_val) + curr_unc/c_val + 2/160))
    
    r_rat = np.array(r_rat)
    prod_rat = np.array(prod_rat)

    bv = np.array(prod_list)
    rad = np.array(r_list)

    rad_err = np.multiply(rad, r_rat)
    bv_err = np.multiply(bv, prod_rat)

    popt, pcov = curve_fit(myLin, bv, rad, sigma = rad_err, absolute_sigma=True)
    x_bestfit = np.linspace(min(bv), max(bv), 100)
    y_bestfit = myLin(x_bestfit, *popt)

    rad_fit = myLin(bv, *popt)
    residuals = (rad - rad_fit)
    chisq = np.sum(((residuals)/rad_err)**2)   
    dof = len(Diameters) - 1


    b1.errorbar(bv, rad, xerr=bv_err, yerr = rad_err, color='#8000E0', ls="none", marker='o', alpha=0.55, label="Data")
    b1.plot(x_bestfit, y_bestfit, color="#023AD0", ls="--", alpha=0.7, label = "Linear fit")
    b1.legend(loc=2)

    b1.set_ylabel(r"1/radius ($m^{-1}$)", fontsize=15)
    x_string = r"$\frac{Magnetic \;\; Field}{\sqrt{Accelerating \;\; Potential}}$ with units ($kg^{1/2}$ $m^{-1}$ $s^{-1/2}$ $A^{-1/2}$)"
    b1.set_xlabel(x_string, fontsize=15)

    b1.grid(which='major', color='#CCCCCC', linewidth=0.6)
    b1.grid(which='minor', color='#DDDDDD', linewidth=0.3)
    b1.minorticks_on()

    print("Params for mx + B:", *popt)
    print("Uncertinty in params:", np.sqrt(np.diag(pcov)))
    print("Chi_Sq:", r"%3.2f"%(chisq/dof))
    print("Chi_Sq Probability:", r'$\chi^2$ prob.= %2.2f'%(1-chi2.cdf(chisq,dof)))

    b2.errorbar(bv, residuals, xerr=bv_err, yerr = rad_err, color='#8000E0', ls="none", marker='o', alpha=0.55, label="Residuals")
    b2.axhline(0, ls='--', c="#023AD0", label="Best Fit")  # Reference line at zero for residuals
    b2.legend(loc=2)

    b2.set_ylabel(r'1/radius Residual ($m^{-1}$)', fontsize=15)
    b2.set_xlabel(r'$B/\sqrt{\Delta V}$ (with units as before)', fontsize=15)

    b2.grid(which='major', color='#CCCCCC', linewidth=0.6)
    b2.grid(which='minor', color='#DDDDDD', linewidth=0.3)
    b2.minorticks_on()

    plt.show()

plot_final = True

if plot_final:
    fig, (b1, b2) = plt.subplots(nrows=1, ncols=2, layout='constrained')

    r_list =[]
    r_rat = []
    B_list = []
    B_rat = []

    for d_val in Diameters:
        ind = Diameters.index(d_val)
        c_val = Currents[ind]
        v_val = Voltages[ind]

        r_adj = d_val * 0.01 / 2
        rat_adj = 0.001 / r_adj

        r_list.append(1 / r_adj)
        r_rat.append(rat_adj)
        B_list.append(calcB(c_val, r_adj))
        B_rat.append((1/130 + curr_unc/c_val + 2/160))
    
    r_rat = np.array(r_rat)
    B_rat = np.array(B_rat)

    bv = np.array(B_list)
    rad = np.array(r_list)

    rad_err = np.multiply(rad, r_rat)
    bv_err = np.multiply(bv, B_rat)

    popt, pcov = curve_fit(myLin, bv, rad, sigma = rad_err, absolute_sigma=True)
    x_bestfit = np.linspace(min(bv), max(bv), 100)
    y_bestfit = myLin(x_bestfit, *popt)

    rad_fit = myLin(bv, *popt)
    residuals = (rad - rad_fit)
    chisq = np.sum(((residuals)/rad_err)**2)   
    dof = len(Diameters) - 1

    b1.plot(x_bestfit, y_bestfit, color="#023AD0", ls="--", alpha=0.7, label = "Linear fit")
    b1.errorbar(bv, rad, xerr=bv_err, yerr = rad_err, color='#8000E0', ls="none", marker='o', alpha=0.7, label="Data")
    b1.legend(loc=2)

    b1.set_ylabel(r"1/radius ($m^{-1}$)", fontsize=13)
    x_string = r"Magnetic Field (Tesla)"
    b1.set_xlabel(x_string, fontsize=13)

    b1.grid(which='major', color='#CCCCCC', linewidth=0.6)
    b1.grid(which='minor', color='#DDDDDD', linewidth=0.3)
    b1.minorticks_on()

    print("Params for mx + B:", *popt)
    print("Uncertinty in params:", np.sqrt(np.diag(pcov)))
    print("Chi_Sq:", r"%3.2f"%(chisq/dof))
    print("Chi_Sq Probability:", r'$\chi^2$ prob.= %2.2f'%(1-chi2.cdf(chisq,dof)))

    b2.errorbar(bv, residuals, xerr=bv_err, yerr = rad_err, color='#8000E0', ls="none", marker='o', alpha=0.6, label="Residuals")
    b2.axhline(0, ls='--', c="#023AD0", label="Best Fit")  # Reference line at zero for residuals
    b2.legend(loc=2)

    b2.set_ylabel(r'1/radius Residual ($m^{-1}$)', fontsize=13)
    b2.set_xlabel(r'Magnetic Field (T)', fontsize=13)

    b2.grid(which='major', color='#CCCCCC', linewidth=0.6)
    b2.grid(which='minor', color='#DDDDDD', linewidth=0.3)
    b2.minorticks_on()

    plt.show()
