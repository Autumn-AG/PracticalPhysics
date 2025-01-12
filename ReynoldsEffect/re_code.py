import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

D = 93.0            # pm 1.0
WAT_RHO = 0.99      # pm 0.01
NY_RHO = 1.12
WAT_NU = 9.46e-3    # pm 0.1e-3 -> https://www.omnicalculator.com/physics/water-viscosity#:~:text=The%20viscosity%20of%20water%20is,C%20is%200.354%20mPa%C2%B7s%20.

GLY_RHO = 1.26      # pm 0.01
TEF_RHO = 2.2   
GLY_NU = 11.37      # pm 0.01  -> https://www.met.reading.ac.uk/~sws04cdw/viscosity_calc.html

SIZE_SIG = 0.01

def myLin(x, m, b) -> float:
    """Estimator function for scipy.optimize.curve_fit, that estimates a linear function y = mx + b"""
    return (m * x) + b

def v_corr(v_m, d) -> float:
    return v_m / (1 - 2.104*(d/D)  + 2.089*((d/D)**3))

def mySq(x, a, b) -> float:
    return a * (x)**2 + b

def mySqrt(x, a, b) -> float:
    return a * (x)**0.5 + b

def watReyn(dia, vel, vel_unc) -> float:
    '''Adjusted for units mm/s'''
    Reyn = WAT_RHO * dia * vel / WAT_NU / 100
    frac = (0.01/WAT_RHO) + (0.1e-3/WAT_NU) + (0.01/dia) + (vel_unc/vel)
    return (Reyn, Reyn*frac)

def glyReyn(dia, vel, vel_unc) -> float:
    '''Adjusted for units mm/s'''
    Reyn = GLY_RHO * dia * vel / GLY_NU / 100
    frac = (0.01/GLY_RHO) + (0.1e-3/GLY_NU) + (0.01/dia) + (vel_unc/vel)
    return (Reyn, Reyn*frac)


def theo_Gly_Vel(dia) -> float:
    r = dia/2
    V = (4 / 3) * np.pi * (r**3)
    rho = TEF_RHO - GLY_RHO
    m = V * rho
    g = 9.81
    theo_Vel = (m * g) / (6 * np.pi * GLY_NU * r)
    return theo_Vel * 10
    

def theo_Wat_Vel(dia) -> float:
    r = dia/2
    A = np.pi * (r**2)
    C_d = 0.4
    V = (4 / 3) * np.pi * (r**3)
    rho = TEF_RHO - GLY_RHO
    m = V * rho
    g = 9.81
    top = 2 * m * g
    bot = WAT_RHO * C_d * A
    theo_Vel = np.sqrt(top / bot)
    return theo_Vel * 10 


sizes = ['a', 'b', 'c', 'd', 'e']
test_number = [1, 2, 3]
colour_dict = {1: 'r', 2: 'g', 3: 'b'}
len_data = len(sizes) * len(test_number)
            
wat_sizes = [2.30, 2.29, 2.32,
             3.14, 3.10, 3.10,
             3.94, 3.92, 3.96,
             4.75, 4.74, 4.75,
             6.34, 6.34, 6.35]

gly_sizes = [1.52, 1.57, 1.57,
             2.38, 2.35, 2.35,
             3.16, 3.16, 3.17,
             4.74, 4.75, 4.74, 
             6.35, 6.35] 

wat_vel = []
gly_vel = []


for size in sizes:
    ind = sizes.index(size)
    grat = 8.5 / 10
    wrat = 5 / 10

    if True:
        for tn in test_number:
            if not (size == 'e') or not (tn == 3):
                sub_ind = test_number.index(tn)
                ''' Glycerine '''
                gly_filename = f"Glycerine/G-{size}{tn}d.txt"
                gly_data = np.transpose(np.loadtxt(gly_filename, delimiter="\t", skiprows=2))
                gly_len = len(gly_data[0])
                time = gly_data[0]
                pos = gly_data[1]
                time -= time[0]
                pos -= pos[0]

                peak = max(pos)
                divider = grat * peak

                time = np.delete(time, np.where(pos < divider))
                pos = np.delete(pos, np.where(pos < divider))

                xerr = np.full(len(time), 0.01)
                yerr = np.full(len(pos), 0.2)

                popt, pcov = curve_fit(myLin, time, pos, sigma=yerr, absolute_sigma=True)

                # print(pos)

                pos_fit = myLin(time, *popt)
                chisq = np.sum((pos - pos_fit)**2/pos_fit)
                dof = 1

                # Smoothed line of best fit (1000 points)
                plt.scatter(time, pos, s=5, label=f"{size}{tn} data", c=colour_dict[tn], alpha=0.4)
                plt.errorbar(time, pos, xerr=xerr, yerr=yerr, fmt='none', c='k', alpha = 0.5)
                x_bestfit1 = np.linspace(time[0], time[-1], 1000)
                y_bestfit1 = myLin(x_bestfit1, *popt)
                plt.plot(x_bestfit1, y_bestfit1, label='Fit', c='k', ls="--", alpha=0.4)
                print(f"Glycerine {size}{tn} velocity: ", "%.2f"%popt[0], " Chisq: ", "%.2f"%chisq, " CHSQP: ", "%.2f"%(1-chi2.cdf(chisq,dof)))

                fv = v_corr(popt[0], gly_sizes[ind + sub_ind])
                gly_vel.append(fv)

            plt.title(f"Position vs Time for size {size} in Glycerine")
            plt.xlabel("Time from first frame captured")
            plt.ylabel("Position from first frame captured")
            plt.grid(which='major', color='#CCCCCC', linewidth=0.8)
            plt.grid(which='minor', color='#DDDDDD', linewidth=0.5)
            plt.minorticks_on()
            plt.legend(loc=1)
            plt.show()

    if True:
        for tn in test_number:
            ''' Water '''
            wat_filename = f"Water/W-{size}{tn}d.txt"
            wat_data = np.transpose(np.loadtxt(wat_filename, delimiter="\t", skiprows=2))
            wat_len = len(wat_data[0])
            time = wat_data[0]
            pos = wat_data[1]
            time -= time[0]
            pos -= pos[0]

            peak = max(pos)
            divider = wrat * peak

            time = np.delete(time, np.where(pos < divider))
            pos = np.delete(pos, np.where(pos < divider))

            xerr = np.full(len(time), 0.01)
            yerr = np.full(len(pos), 1.5)

            popt, pcov = curve_fit(myLin, time, pos, sigma=yerr, absolute_sigma=True)

            # print(pos)

            pos_fit = myLin(time, *popt)
            chisq = np.sum((pos - pos_fit)**2/pos_fit)
            dof = 1

            # # Smoothed line of best fit (1000 points)
            # plt.scatter(time, pos, s=5, label=f"{size}{tn} data", c=colour_dict[tn], alpha=0.3)
            # plt.errorbar(time, pos, xerr=xerr, yerr=yerr, fmt='none', c='k', alpha = 0.4)
            # x_bestfit1 = np.linspace(time[0], time[-1], 1000)
            # y_bestfit1 = myLin(x_bestfit1, *popt)
            # plt.plot(x_bestfit1, y_bestfit1, label='Fit', c='k', ls="--", alpha=0.7)
            # print(f"Water {size}{tn} velocity: ", "%.2f"%popt[0], " Chisq: ", "%.2f"%chisq, " CHSQP: ", "%.2f"%(1-chi2.cdf(chisq,dof)))

            fv = v_corr(popt[0], wat_sizes[ind + sub_ind])
            wat_vel.append(fv)

        # plt.title(f"Position vs Time for size {size} in Water")
        # plt.xlabel("Time from first frame captured")
        # plt.ylabel("Position from first frame captured")
        # plt.grid(which='major', color='#CCCCCC', linewidth=0.8)
        # plt.grid(which='minor', color='#DDDDDD', linewidth=0.5)
        # plt.minorticks_on()
        # plt.legend(loc=1)
        # plt.show()


unc_ratio = 0.2

gs = np.array(gly_sizes)
gv = np.array(gly_vel)
ws = np.array(wat_sizes)
wv = np.array(wat_vel)

def plotr(a1, a2, xdata, ydata, fit, unc_ratio, med, clr, ylim):
    
    yerr = ydata*unc_ratio
    popt, pcov = curve_fit(fit, xdata, ydata, sigma = yerr, absolute_sigma=True)

    x_bestfit = np.linspace(min(xdata)*0.9, max(xdata)*1.1, 1000)
    y_bestfit = fit(x_bestfit, *popt)


    a1.scatter(xdata, ydata, marker='x', s=30, alpha=0.8, label="Data", c='#ff9b00')
    a1.errorbar(xdata, ydata, yerr=yerr, solid_capstyle='projecting', capsize=1.8, fmt='none', c='#2050C0', alpha = 0.4)
    a1.plot(x_bestfit, y_bestfit, alpha=0.85, ls='--', label="Fit", c=clr)

    a1.set_xlabel(f'Size of ball dropped in {med} (mm)', fontsize=14)
    a1.set_ylabel(f'Terminal velocity of ball in medium (mm/s)', fontsize=14)
    a1.grid(which='major', color='#CCCCCC', linewidth=0.8)
    a1.grid(which='minor', color='#DDDDDD', linewidth=0.3)
    a1.minorticks_on()  
    a1.legend(loc=2, fontsize=11)
    
    y_fit = fit(xdata, *popt)
    residuals = (ydata - y_fit)
    chisq = np.sum((residuals)**2 / y_fit)
    dof = 2
    
    a2.errorbar(xdata, residuals, yerr=yerr, solid_capstyle='projecting', capsize=1.5, fmt='o', c='#2050C0', ms=5, alpha=0.45, label="Residuals")
    a2.axhline(0, ls='--', c=clr, label="Best Fit")  # Reference line at zero for residuals
    a2.set_xlabel(f'Size of ball dropped in {med} (mm)', fontsize=14)
    a2.set_ylabel('Offset from fit (mm/s)', fontsize=14)
    a2.set_ylim(ylim)
    a2.grid(which='major', color='#CCCCCC', linewidth=0.8)
    a2.legend(loc=2, fontsize=11)

    print(f"{med} velocity: ", *popt, "\n Chisq: ", "%.2f"%chisq, " CHSQP: ", "%.2f"%(1-chi2.cdf(chisq,dof)), "unc vel: ", np.sqrt(np.diag(pcov)))


# fig, ((a1, a2)) = plt.subplots(nrows=1, ncols=2, layout='constrained')
# plotr(a1, a2, gs, gv, unc_ratio=0.04, fit=mySq, med="Glycerine", clr="#8000d0", ylim=(-2, 2))

# print("\n")
# fig, ((b1, b2)) = plt.subplots(nrows=1, ncols=2, layout='constrained')
# plotr(b1, b2, ws, wv, unc_ratio=0.04, fit=mySqrt, med="Water", clr="g", ylim=(-20, 20))

# plt.show()  

# print("\n")


wRE = []
for i in range(len(wat_vel)):
    wRE.append([wat_sizes[i]  , watReyn(wat_sizes[i], wat_vel[i], wat_vel[i]*0.04)])


gRE = []
for i in range(len(gly_vel)):
    gRE.append([gly_sizes[i]  , glyReyn(gly_sizes[i], gly_vel[i], gly_vel[i]*0.04)])

# print("Water")
# for i in range(len(wat_vel)):
#     print(wRE[i][0],
#           "mm : velocity",
#           r"$%3.2f "%(wat_vel[i]),
#           r"\pm %3.2f "%(wat_vel[i]*0.04),
#           r"$ & $%3.2f"%(theo_Wat_Vel(wRE[i][0])),
#           r"$ & $%3.2f"%(float(wRE[i][1][0])),
#           r" \pm %3.2f"%(float(wRE[i][1][1])), "$" 
#           )

# print("\n Glycerine")

# for i in range(len(gly_vel)):
#     print(gRE[i][0],
#           "mm : ", 
#           (gly_vel[i]),
#           (gly_vel[i]*0.04),
#           (theo_Gly_Vel(gRE[i][0])),
#           (float(gRE[i][1][0])),
#           (float(gRE[i][1][1]))
#           )

# (dia, vel, vel_unc)

dia = 4.74

r = dia/2
V = (4 / 3) * np.pi * (r**3)
rho = TEF_RHO - GLY_RHO
print(rho, "=", TEF_RHO, "-", GLY_RHO)
m = V * rho
print(m, "=", V, "*", rho)
g = 9.81
theo_Vel = (m * g) / (6 * np.pi * GLY_NU * r) * 10
print(theo_Vel, (m * g), "/", "(", 6, "*", np.pi, "*", GLY_NU, "*", r, ")")

