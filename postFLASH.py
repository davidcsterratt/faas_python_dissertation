import time

from preFLASH import *
# from numba import njit, jit
from scipy.integrate import odeint
import scipy.integrate as integrate

# import time

## Equations
# @njit(parallel=True)
def f_postflash(y, t, params):
    CaDMn_s, CaDMn_f, DMn_s, DMn_f, CaPP, PP, Ca, OGB5N, CaOGB5N, NtNt, CtCt, CaNtNr, CaCtCr, CaNrCaNr, CaCrCaCr = y

    # K_off_CaDMn, K_on_CaDMn, K_off_D, K_on_D, K_off_CaPP, K_f, K_s, K_on_TN, K_on_TC, K_on_RN, K_on_RC, K_off_TN, K_off_TC, K_off_RN, K_off_RC = params
    K_off_CaDMn = params[0]
    K_on_CaDMn = params[1]
    K_off_D = params[2]
    K_on_D = params[3]
    K_off_CaPP = params[4]
    K_f = params[5]
    K_s = params[6]
    K_on_TN = params[7]
    K_on_TC = params[8]
    K_on_RN = params[9]
    K_on_RC = params[10]
    K_off_TN = params[11]
    K_off_TC = params[12]
    K_off_RN = params[13]
    K_off_RC = params[14]

    f = np.asarray([
        -K_off_CaDMn * CaDMn_s + K_on_CaDMn * DMn_s * Ca - CaDMn_s * K_s,  # CaDMn_s
        -K_off_CaDMn * CaDMn_f + K_on_CaDMn * DMn_f * Ca - CaDMn_f * K_f,  # CaDMn_f
        K_off_CaDMn * CaDMn_s - K_on_CaDMn * DMn_s * Ca - DMn_s * K_s,  # DMn_s
        K_off_CaDMn * CaDMn_f - K_on_CaDMn * DMn_f * Ca - DMn_f * K_f,  # DMn_f
        -K_off_CaPP * CaPP + K_on_CaDMn * PP * Ca + CaDMn_s * K_s + CaDMn_f * K_f,  # CaPP
        K_off_CaPP * CaPP - K_on_CaDMn * PP * Ca + 2 * DMn_s * K_s + 2 * DMn_f * K_f + CaDMn_s * K_s + CaDMn_f * K_f,
        # PP
        K_off_CaDMn * CaDMn_s + K_off_CaDMn * CaDMn_f - K_on_CaDMn * DMn_s * Ca - \
        K_on_CaDMn * DMn_f * Ca - K_on_D * OGB5N * Ca + K_off_D * CaOGB5N + \
        K_off_CaPP * CaPP - K_on_CaDMn * PP * Ca - 2 * K_on_TN * NtNt * Ca + \
        K_off_TN * CaNtNr - K_on_RN * CaNtNr * Ca + 2 * K_off_RN * CaNrCaNr \
        - 2 * K_on_TC * CtCt * Ca + K_off_TC * CaCtCr - K_on_RC * CaCtCr * Ca + \
        2 * K_off_RC * CaCrCaCr,  # -K_on_CB*CB*Ca + K_off_CB*CaCB,                                        Ca
        - K_on_D * OGB5N * Ca + K_off_D * CaOGB5N,  # OGB5N
        K_on_D * OGB5N * Ca - K_off_D * CaOGB5N,  # CaOGB5N
        -2 * K_on_TN * NtNt * Ca + K_off_TN * CaNtNr,  # NtNt
        -2 * K_on_TC * CtCt * Ca + K_off_TC * CaCtCr,  # CtCt
        2 * K_on_TN * NtNt * Ca - K_off_TN * CaNtNr - K_on_RN * CaNtNr * Ca + 2 * K_off_RN * CaNrCaNr,  # CaNtNr
        2 * K_on_TC * CtCt * Ca - K_off_TC * CaCtCr - K_on_RC * CaCtCr * Ca + 2 * K_off_RC * CaCrCaCr,  # CaCtCr
        K_on_RN * CaNtNr * Ca - 2 * K_off_RN * CaNrCaNr,  # CaNrCaNr
        K_on_RC * CaCtCr * Ca - 2 * K_off_RC * CaCrCaCr  # CaCrCaCr
    ])

    return f


# ## Equations
# @jit(parallel=True)
def f_postflash_jac(y, t, params):
    CaDMn_s, CaDMn_f, DMn_s, DMn_f, CaPP, PP, Ca, OGB5N, CaOGB5N, NtNt, CtCt, CaNtNr, CaCtCr, CaNrCaNr, CaCrCaCr = y

    # K_off_CaDMn, K_on_CaDMn, K_off_D, K_on_D, K_off_CaPP, K_f, K_s, K_on_TN, K_on_TC, K_on_RN, K_on_RC, K_off_TN, K_off_TC, K_off_RN, K_off_RC = params
    K_off_CaDMn = params[0]
    K_on_CaDMn = params[1]
    K_off_D = params[2]
    K_on_D = params[3]
    K_off_CaPP = params[4]
    K_f = params[5]
    K_s = params[6]
    K_on_TN = params[7]
    K_on_TC = params[8]
    K_on_RN = params[9]
    K_on_RC = params[10]
    K_off_TN = params[11]
    K_off_TC = params[12]
    K_off_RN = params[13]
    K_off_RC = params[14]

    f = np.array([[-K_off_CaDMn - K_s, 0, Ca * K_on_CaDMn, 0, 0, 0, DMn_s * K_on_CaDMn, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, -K_f - K_off_CaDMn, 0, Ca * K_on_CaDMn, 0, 0, DMn_f * K_on_CaDMn, 0, 0, 0, 0, 0, 0, 0, 0],
                  [K_off_CaDMn, 0, -Ca * K_on_CaDMn - K_s, 0, 0, 0, -DMn_s * K_on_CaDMn, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, K_off_CaDMn, 0, -Ca * K_on_CaDMn - K_f, 0, 0, -DMn_f * K_on_CaDMn, 0, 0, 0, 0, 0, 0, 0, 0],
                  [K_s, K_f, 0, 0, -K_off_CaPP, Ca * K_on_CaDMn, K_on_CaDMn * PP, 0, 0, 0, 0, 0, 0, 0, 0],
                  [K_s, K_f, 2 * K_s, 2 * K_f, K_off_CaPP, -Ca * K_on_CaDMn, -K_on_CaDMn * PP, 0, 0, 0, 0, 0, 0, 0, 0],
                  [K_off_CaDMn, K_off_CaDMn, -Ca * K_on_CaDMn, -Ca * K_on_CaDMn, K_off_CaPP, -Ca * K_on_CaDMn,
                   -CaCtCr * K_on_RC - CaNtNr * K_on_RN - 2 * CtCt * K_on_TC - DMn_f * K_on_CaDMn - DMn_s * K_on_CaDMn - K_on_CaDMn * PP - K_on_D * OGB5N - 2 * K_on_TN * NtNt,
                   -Ca * K_on_D, K_off_D, -2 * Ca * K_on_TN, -2 * Ca * K_on_TC, -Ca * K_on_RN + K_off_TN,
                   -Ca * K_on_RC + K_off_TC, 2 * K_off_RN, 2 * K_off_RC],
                  [0, 0, 0, 0, 0, 0, -K_on_D * OGB5N, -Ca * K_on_D, K_off_D, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, K_on_D * OGB5N, Ca * K_on_D, -K_off_D, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, -2 * K_on_TN * NtNt, 0, 0, -2 * Ca * K_on_TN, 0, K_off_TN, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, -2 * CtCt * K_on_TC, 0, 0, 0, -2 * Ca * K_on_TC, 0, K_off_TC, 0, 0],
                  [0, 0, 0, 0, 0, 0, -CaNtNr * K_on_RN + 2 * K_on_TN * NtNt, 0, 0, 2 * Ca * K_on_TN, 0,
                   -Ca * K_on_RN - K_off_TN, 0, 2 * K_off_RN, 0],
                  [0, 0, 0, 0, 0, 0, -CaCtCr * K_on_RC + 2 * CtCt * K_on_TC, 0, 0, 0, 2 * Ca * K_on_TC, 0,
                   -Ca * K_on_RC - K_off_TC, 0, 2 * K_off_RC],
                  [0, 0, 0, 0, 0, 0, CaNtNr * K_on_RN, 0, 0, 0, 0, Ca * K_on_RN, 0, -2 * K_off_RN, 0],
                  [0, 0, 0, 0, 0, 0, CaCtCr * K_on_RC, 0, 0, 0, 0, 0, Ca * K_on_RC, 0, -2 * K_off_RC]])

    return f


## Compute sensitivity equations
## Uncomment for hessian
# f_s <- sensitivitiesSymb(f)
## Generate ODE function
# func <- funC(f, nGridpoints=0)
# func_s <- funC(c(f, f_s), nGridpoints=0)

def postflash(theta=0, phi=get_exp(0)['par'], epsilon=0, time_points=0, hessian=False, alpha_change=0):
    ## Experimental - try to kill runaway computations after 1s - whether
    ## this works depends on if the cOde code checks for interrupts
    # setTimeLimit(cpu=5)

    ## all rate parameters

    parms = pd.concat([phi["K_off_CaDMn"] * 1000,
                       phi["K_on_CaDMn"] * 1000,
                       phi["K_off_D"] * 1000,
                       phi["K_on_D"] * 1000,
                       phi["K_off_CaPP"] * 1000,
                       1 / (phi["tau_f"] / 1000),  # Kf
                       1 / (phi["tau_s"] / 1000),  # Ks
                       theta])

    parms.index = ['K_off_CaDMn', 'K_on_CaDMn', 'K_off_D', 'K_on_D', 'K_off_CaPP', 'K_f', 'K_s'] + list(theta.index)
    ## uncaging and fast fraction

    alpha = max(float(1 + epsilon) * float(phi['delay'] * theta['m_alpha'] + theta['alpha0']), 0)
    alpha = alpha + alpha*alpha_change

    x = phi['x']

    ## same time points as experiment
    times = np.squeeze(np.asarray(time_points.T)) / 1000
    # times = np.arange(0,37, 0.01)

    ## read in pre-flash values and set concentrations
    pre_out = pd.Series(get_preflash_ss(theta, phi))

    pre_out.index = ['CaDMn', 'DMn', 'Ca', 'OGB5N', 'CaOGB5N', 'NtNt', 'CtCt', 'CaNtNr', 'CaCtCr', 'CaNrCaNr',
                     'CaCrCaCr']

    y = [
        float((1 - x) * alpha * pre_out["CaDMn"]),  # CaDMn_s
        float(x * alpha * pre_out["CaDMn"]),  # CaDMn_f
        float((1 - x) * alpha * pre_out["DMn"]),  # DMn_s
        float(x * alpha * pre_out["DMn"]),  # DMn_f
        0,  # CaPP
        0]  # PP
    y = y + list(
        pre_out[["Ca", "OGB5N", "CaOGB5N", "NtNt", "CtCt", "CaNtNr", "CaCtCr", "CaNrCaNr", "CaCrCaCr"]].to_list())
    ## solve the ODEs
    if (hessian):
        ## Not yet
        pass
    else:
        # jac = lambda y, t: f_postflash_jac(y,t,parms.to_numpy())
        post_out, full_output = odeint(f_postflash, y, times, args=(parms.to_numpy(),), Dfun=f_postflash_jac,
                                       full_output=1)
        if full_output['message'] != 'Integration successful.':
            raise ValueError


    ## Value of F_max/F_min
    F_ratio = phi['Ratio_D']
    ## time points to compare simulation and experiment at\
    relevantID = np.arange(post_out.shape[0])
    # print(post_out[relevantID, 7])
    ## evaluate fluorescence time course
    ## create list of F-ratios
    F_ratio_course = (post_out[relevantID, 7] + float(F_ratio) * post_out[relevantID, 8]) / (
                pre_out["OGB5N"] + float(F_ratio) * pre_out["CaOGB5N"])

    ## Missing some bits to do with the hessian here

    return F_ratio_course
