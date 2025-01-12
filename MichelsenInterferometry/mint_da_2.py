import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

# LRT
def getAngle(x: tuple[int]) -> float:
    """ Returns Angle value in radians as a float """
    base = x[0] + (x[1]/60)
    return (np.pi * (base) / 180)

def getRefractionRHS(t, wavelength, theta_val) -> float:
    """ Required RHS for the x-axis """
    return (t * (theta_val ** 2) / wavelength)

def getRefractionUncFrac(t, unc_t, wL, unc_wL, th, unc_th) -> float:
    a = unc_t / t
    b = unc_wL / wL
    c = unc_th / th
    return a + b + (2 * c)

def getThermalRHS(L, dT, lambd) -> float:
    """ Required RHS for the y-axis """
    return (2 * L * dT / lambd)

def getThermalUncFrac(L, unc_L, dT, unc_dT, lambd, unc_lambd) -> float:
    a = unc_L / L
    b = unc_dT / dT
    c = unc_lambd / lambd
    return (a + b + c)

# Plotting Functions
def myLin (x, m, b):
    return (m * x) + b
    

""" Wavelength Calc Data """
# Plot N (x-axis) vs 2*dx (y-axis) -> get wavelength as slope
dx_list = [5.0, 5.0, 5.0, 4.0,
           8.0, 8.0, 8.0, 8.0,
           11.0, 10.0, 11.0, 11.0,
           13.0, 13.0, 13.0, 14.0,
           16.0, 15.0]
unc_dx = 1.0

wavel_N = [10, 10, 10, 10,
           20, 20, 20, 20,
           30, 30, 30, 30,
           40, 40, 40, 40,
           50, 50]
unc_N = 1

""" Index of Refraction Data """
# Plot N (x-axis) vs [t * theta^2 / lambda (from part 1)] gives (n/n-1) as slope with n as refr. index
refr_N = [-30, -30, -20, -20, -10, -10,
           10,  10,  20,  20,  30,  30]

theta_zero = (82, 35) 
unc_theta =  (0, 5)

theta_vals = [(78, 10), (78, 10), (79, 0) , (79, 0) , (80, 0), (80, 5),
              (85, 15), (85, 20), (86, 25), (86, 20), (87, 5), (87, 10)]

UNIT_CONSTANT = 10 ** 6         # Convert mm to nm

t = 7.71 * UNIT_CONSTANT        # in nm
unc_t = 0.01 * UNIT_CONSTANT    # in nm

""" Thermal Data """ 
# Plot [2 L0 dT / lambda] (x-axis) vs N (y-axis) get coefficient of therm. exp. (alpha) as slope.
start_temp = [27.2, 27.9, 27.8, 23.5,
              26.7, 24.8, 27.2, 26.4,
              27.5, 26.5, 27.9]

end_temp =   [28.1, 28.9, 29.1, 24.9,
              28.3, 26.6, 29.1, 28.2,
              29.4, 28.7, 30.4]

unc_temp = 0.1  # Kelvin
L_zero = 88.6 * UNIT_CONSTANT  # mm to nm
unc_Lz = 0.1  * UNIT_CONSTANT  # mm to nm

thermal_N = [10, 12, 13, 14,
             15, 16, 17, 18,
             20, 22, 24]

""" Data Analysis """

# Plotting
plotting = True

plot1=False
plot2=False
plot3=True

wavelength_calc = True
index_calc = True
therm_calc = True

# Wavelength Calculation: Part 1

global greenWavelength
global unc_gW
greenWavelength = -1
unc_gW = -1

if wavelength_calc:
    # numpy-ify
    y_data = 2 * np.array(dx_list)
    yerr = np.full(len(dx_list), 2 * unc_dx)
    x_data = np.array(wavel_N)
    xerr = np.full(len(dx_list), unc_N)

    # scipy moment
    popt, pcov = curve_fit(myLin, xdata=x_data, ydata=y_data, sigma = yerr, absolute_sigma=True)
    
    y_fit = myLin(x_data, *popt)
    residuals = y_data - y_fit
    chisq = np.sum((residuals/yerr)**2)
    dof = len(x_data) - 1

    x_bestfit = np.linspace(5, 55, 200)
    y_bestfit = myLin(x_bestfit, *popt)

    punc = np.sqrt(np.diag(pcov))

    # plotting
    if plot1:
        fig, (a1, a2) = plt.subplots(nrows=1, ncols=2, layout='constrained')
        
        a1.errorbar(x_data, y_data, xerr=xerr, yerr=yerr, fmt='o', c="#023AD0", alpha=0.5, label="Data")
        a1.plot(x_bestfit, y_bestfit, c="#8000E0", ls="--", alpha=0.75, label="Linear Fit")
        a1.set_ylabel(r"Path length diff. $2\Delta x$ $(\mu m)$", fontsize=14)
        a1.set_xlabel("Number of fringe variations, N", fontsize=14)
        a1.legend(loc=2)
        a1.grid(which='major', color='#CCCCCC', linewidth=0.6)

        a2.errorbar(x_data, residuals, xerr=xerr, yerr=yerr, fmt='o', c="#023AD0", alpha=0.5, label="Residuals")
        a2.axhline(0, ls='--', c="#8000E0", label="Best Fit", alpha=0.75)  # Reference line at zero for residuals
        a2.set_ylabel(r"Residuals $(\mu m)$", fontsize=14)
        a2.set_ylim(-7, 7)
        a2.set_xlabel("Number of fringe variations, N", fontsize=14)
        a2.grid(which='major', color='#CCCCCC', linewidth=0.6)
        a2.legend(loc=1)

    # Assign global values
    greenWavelength = round(1000 * popt[0], 3)  # in nm
    unc_gW = round(1000 * punc[0], 3)           # in nm

    print("\nWavelength:", r"%3.3f nm"%(greenWavelength), r"with unc %3.3f nm"%(unc_gW),  "\n",
        "CHI:", r" Reduced $\chi^2$ %3.2f"%(chisq/dof), r'Prob.= %1.1f'%(1-chi2.cdf(chisq,dof)), "\n",
        "Params:", r"m = %2.4f $\mu m$"%(popt[0]), r"b = %2.3f $\mu m$"%(popt[1]), "\n",
        "Param Uncertainty:", r"m: %2.3f $\mu m$"%(punc[0]), r"b: %2.3f $\mu m$\n"%(punc[1]))
    
# Index of Refraction Calculation: Part 2
if index_calc:
    y_data = []
    yerr = []

    unc_angle = getAngle(unc_theta)
    for i in range(len(theta_vals)):
        dAngle = abs(getAngle(theta_vals[i]) - getAngle(theta_zero))
        yi = getRefractionRHS(t, greenWavelength, dAngle)
        rat_i = getRefractionUncFrac(t, unc_t, greenWavelength, unc_gW, dAngle, unc_angle)
        unc_yi = yi * rat_i
        y_data.append(yi)
        yerr.append(unc_yi)

    # numpy-ify
    y_data = np.array(y_data)
    yerr = np.array(yerr)
    x_data = np.array(refr_N)
    x_data = np.abs(x_data)
    xerr = np.full(len(x_data), unc_N)

    # scipy moment
    popt, pcov = curve_fit(myLin, xdata=x_data, ydata=y_data, sigma = yerr, absolute_sigma=True)
    punc = np.sqrt(np.diag(pcov))
    
    y_fit = myLin(x_data, *popt)    
    residuals = y_data - y_fit
    chisq = np.sum((residuals/yerr)**2)
    dof = len(x_data) - 1

    x_bestfit = np.linspace(7.5, 32.5, 200)
    y_bestfit = myLin(x_bestfit, *popt)

    # plotting
    if plot2:
        fig, (b1, b2) = plt.subplots(nrows=1, ncols=2, layout='constrained')
        
        b1.errorbar(x_data, y_data, xerr=xerr, yerr=yerr, fmt='o', c="#023AD0", alpha=0.45, label="Data")
        b1.plot(x_bestfit, y_bestfit, c="#8000E0", ls="--", alpha=0.75, label="Linear Fit")
        b1.set_ylabel(r"$(t\theta^2) / \lambda$ ($rad^2$)", fontsize=14)
        b1.set_xlabel("Number of fringe variations, N", fontsize=14)
        b1.legend(loc=2)
        b1.grid(which='major', color='#CCCCCC', linewidth=0.6)

        b2.errorbar(x_data, residuals, xerr=xerr, yerr=yerr, fmt='o', c="#023AD0", alpha=0.45, label="Residuals")
        b2.axhline(0, ls='--', c="#8000E0", label="Best Fit", alpha=0.75)  # Reference line at zero for residuals
        b2.set_ylabel(r"Residuals of $(t\theta^2) / \lambda$ ($rad^2$)", fontsize=14)
        b2.set_ylim(-17, 17)
        b2.set_xlabel("Number of fringe variations, N", fontsize=14)
        b2.grid(which='major', color='#CCCCCC', linewidth=0.6)
        b2.legend(loc=2)

    m = popt[0] 
    unc_m = punc[0]
    n = m / (m - 1)
    unc_n = n * unc_m / m

    print("\n Index of Refraction:", n, "with unc.", unc_n, "\n",
        "CHI:", r"Reduced $\chi^2$ %3.2f"%(chisq/dof), r'Prob.= %1.1f'%(1-chi2.cdf(chisq,dof)), "\n",
        "Params:", r"m = %2.3f "%(popt[0]), r"b = %2.3f"%(popt[1]), "\n",
        "Param Uncertainty:", r"m: %2.3f"%(punc[0]), r"b: %2.3f"%(punc[1]))


# Thermal Expansion Coefficient Calculation: Part 3
if therm_calc:
    x_data = []
    xerr = []

    for i in range(len(start_temp)):
        dTemp = end_temp[i] - start_temp[i]
        xi = getThermalRHS(L_zero, dTemp, greenWavelength) / 1000
        rat_xi = getThermalUncFrac(L_zero, unc_Lz, dTemp, unc_temp, greenWavelength, unc_gW)
        unc_xi = abs(xi * rat_xi)
        x_data.append(xi)
        xerr.append(unc_xi)

    # numpy-ify
    y_data = np.array(x_data)
    yerr = np.array(xerr)

    x_data = np.array(thermal_N)
    xerr = np.full(len(x_data), unc_N)

    # scipy moment
    popt, pcov = curve_fit(myLin, xdata=x_data, ydata=y_data, sigma = yerr, absolute_sigma=True)
    punc = np.sqrt(np.diag(pcov))
    
    y_fit = myLin(x_data, *popt)    
    residuals = y_data - y_fit
    chisq = np.sum((residuals/yerr)**2)
    dof = len(x_data) - 1

    x_bestfit = np.linspace(min(x_data) - 1, max(x_data) + 1, 150)
    y_bestfit = myLin(x_bestfit, *popt)

    # plotting
    if plot3:
        fig, (c1, c2) = plt.subplots(nrows=1, ncols=2, layout='constrained')
        
        c1.errorbar(x_data, y_data, xerr=xerr, yerr=yerr, fmt='o', c="#023AD0", alpha=0.45, label="Data")
        c1.plot(x_bestfit, y_bestfit, c="#8000E0", ls="--", alpha=0.75, label="Linear Fit")
        c1.set_ylabel(r"$2\;L_0\;\Delta T / \lambda$ with units (kilo K)", fontsize=14)
        c1.set_xlabel("Number of fringe variations, N", fontsize=14)
        c1.legend(loc=2)
        c1.grid(which='major', color='#CCCCCC', linewidth=0.6)

        c2.errorbar(x_data, residuals, xerr=xerr, yerr=yerr, fmt='o', c="#023AD0", alpha=0.45, label="Residuals")
        c2.axhline(0, ls='--', c="#8000E0", label="Best Fit", alpha=0.75)  # Reference line at zero for residuals
        c2.set_ylabel(r"Residuals of$2\;L_0\;\Delta T / \lambda$ (kilo K)", fontsize=14)
        #c2.set_xlim(-17, 17)
        c2.set_xlabel("Number of fringe variations, N", fontsize=14)
        c2.grid(which='major', color='#CCCCCC', linewidth=0.6)
        c2.legend(loc=2)

    alpha = 1 / 1000 / popt[0]
    unc_alpha = alpha * punc[0] / popt[0]

    print("\nCoefficient of Expansion:", alpha, "with unc.", unc_alpha, "\n",
        "CHI:", r"Reduced $\chi^2$ %3.2f"%(chisq/dof), r'Prob.= %1.1f'%(1-chi2.cdf(chisq,dof)), "\n",
        "Params:", r"m = %2.3f "%(popt[0]), r"b = %2.3f"%(popt[1]), "\n",
        "Param Uncertainty:", r"m: %2.3f"%(punc[0]), r"b: %2.3f"%(punc[1]))

if plotting:
    plt.show()

