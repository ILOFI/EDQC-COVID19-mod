import numpy as np
import pandas as pd
import pickle
import json
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np
import pandas as pd
import datetime as dt
from pymc import *
from time import time
import pymc
from sklearn.metrics import r2_score, mean_squared_error


def str_to_dt(datestr):
    return dt.datetime.strptime(datestr, '%Y/%m/%d').date()
def dt_to_str(datedt):
    return datedt.strftime('%Y/%m/%d')


# In[3]:

def save_obj(obj, name ):
    with open('objs/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name ):
    with open('objs/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# In[4]:

def get_3dayavg(x):
    res = [x[0],]
    for t in range(1, len(x)-1):
        res.append(np.mean([x[t-1], x[t], x[t+1]]))
    res.append(x[-1])
    return np.array(res)


# In[5]:

# Data of cumulative E
Edata = pd.read_csv('modeldata/Edata.csv')


# In[6]:

# Data of cumulative I
Idata = pd.read_csv('modeldata/Idata.csv')


# In[7]:

# Data of cumulative R
Rdata = pd.read_csv('modeldata/Rdata.csv')


# In[8]:

# Average duration from symptom onset to report
confirmrate_data = pd.read_csv('modeldata/confirm_duration.csv')


# In[9]:

# Intensity of human mobility
bmi_data = pd.read_csv('modeldata/mobility.csv')


# In[10]:

# Intensity of human mobility without intervention
bmi_data_nocontrol = pd.read_csv('modeldata/mobility_noctrl.csv')


# In[11]:

# eta1: close-contact tracing, eta2: community-based NAT
eta_data = pd.read_csv('modeldata/eta12.csv')


# In[12]:

# eta220: community-based NAT stopped after June 20
eta220_data = pd.read_csv('modeldata/eta220.csv')


# In[13]:

# starting from June 4 (first case onset)
start_t = 2
population = np.array([22000000,])
Etrue = Edata.drop(['ID', 'city'], axis= 1)
Etrue = np.array(Etrue)[:, start_t:]
Itrue = Idata.drop(['ID', 'city',], axis= 1)
Itrue = np.array(Itrue)[:, start_t:]
Rtrue = Rdata.drop(['ID', 'city',], axis= 1)
Rtrue = np.array(Rtrue)[:, start_t:]

Etrue = Etrue - Itrue 
Itrue = Itrue - Rtrue
confirmrate = np.array(confirmrate_data['confirm_duration']).flatten()[start_t:]
confirmrate = get_3dayavg(confirmrate)
bmi = np.array(bmi_data['bmi']).flatten()[start_t:]
bmi_noctrl = np.array(bmi_data_nocontrol['bmi']).flatten()[start_t:]

# In[15]:

alldates = list(Edata.columns)[2+start_t:]
alldates_dt = [dt.datetime.strptime(d, '%Y/%m/%d').date() for d in alldates]
allday_t = len(alldates)


# In[16]:

eta1 = (np.array(eta_data['eta1']))[start_t:]
eta2 = (np.array(eta_data['eta2']))[start_t:]
eta220 = (np.array(eta220_data['eta220']))[start_t:]


# In[17]:

eta1_noctrl = np.zeros(eta1.shape)
eta2_noctrl = np.zeros(eta2.shape)


# In[18]:

'''
N: total population of each region, order by valid_names
initI: first day's infection num, order by valid_names
'''
pres_d = 2
N = population
initE = [x for x in Etrue[:, 0]]
initP = [Itrue[0, int(pres_d)] - Itrue[0, 0]]
initI = [x for x in Itrue[:, 0]]
initR = [x for x in Rtrue[:, 0]]


# In[19]:

def get_sigma(t, sigma):
    if t < len(sigma):
        return sigma[t]
    return sigma[-1]


# In[20]:

def get_bmi(t, bmi):
    if t < len(bmi):
        return bmi[t]
    while t >= len(bmi):
        t -= 7
    return bmi[t]


# In[21]:

def get_arr(t, arr):
    if t < len(arr):
        return arr[t]
    return arr[-1]


# In[22]:

def sim_seiisr(INPUT, alpha1, bmi, beta1, sigma, q, t_range):
    T = len(t_range)
    S = np.zeros(T)
    E = np.zeros(T)
    P = np.zeros(T)
    I = np.zeros(T)
    IS = np.zeros(T)
    R = np.zeros(T)
    S[0] = INPUT[0]
    E[0] = INPUT[1]
    P[0] = INPUT[2]
    I[0] = INPUT[3]
    IS[0] = INPUT[4]
    R[0] = INPUT[5]
    print(INPUT)
    beta1 = 1 / (beta1 - pres_d)
    for j in range(1, T):
        eta1j = get_arr(j+1, eta1)
        eta2j = get_arr(j+1, eta2)
        bmij = get_bmi(j+1, bmi)
        sigmaj = get_sigma(j+1, sigma)
        S[j] = S[j - 1] - alpha1 * S[j - 1] * bmij * I[j - 1] / population[0] - q * alpha1 * S[j - 1] * bmij * P[j - 1] / population[0]
        E[j] = E[j - 1] + alpha1 * S[j - 1] * bmij * I[j - 1] / population[0] + q * alpha1 * S[j - 1] * bmij * P[j - 1] / population[0]  - beta1 * E[j - 1]
        P[j] = P[j - 1] + beta1 * E[j - 1] - P[j - 1] / pres_d
        I[j] = I[j - 1] + (1-eta1j-eta2j) * P[j - 1] / pres_d - sigmaj * I[j - 1]
        IS[j] = IS[j - 1] + (eta1j+eta2j) * P[j - 1] / pres_d - sigmaj * IS[j - 1]
        R[j] = R[j - 1] + sigmaj * (I[j - 1] + IS[j - 1])

    return np.array([S, E, P, I, IS, R]).T


# In[23]:

def simulate_single(alpha1, e2, beta1, sigma, q):

    if alpha1 < 0 or beta1 < 0:
        print('Warning ', alpha1, beta1)
    
    INPUT1 = np.zeros((6))
    INPUT1[0] = population[0]
    INPUT1[1] = initE[streetid]
    INPUT1[2] = initP[streetid]
    INPUT1[3] = initI[streetid] * (1 - eta1[0] - eta2[0])
    INPUT1[4] = initI[streetid] * (eta1[0] + eta2[0])
    INPUT1[5] = initR[streetid]
    
    t_range = np.arange(0.0, t, 1.0)
    RES1 = sim_seiisr(INPUT1, alpha1, e2, beta1, sigma, q, t_range)
    
    return RES1


# In[24]:

import numpy as np

def SEIR_MODEL1(Ecumm, Icumm, Rcumm, init_S, init_E, init_P, init_I, init_R, bmi, sigma):
    T = len(Ecumm)
    print(T)
#     case_data = Icumm
    case_data = np.concatenate([[Icumm[0],], np.diff(Icumm)])
    
    alpha1 = Uniform('alpha1', 1e-3, 5, value = 0.9)
    beta1 = Weibull('beta1', alpha = 1.681228, beta = 6.687700 )
    q0 = Uniform('q0', 0.01, 1.5, value = 0.8)
    
    @deterministic
    def sim(alpha1 = alpha1, beta1 = beta1, q0 = q0):
#         beta1 = 5.37
        S = np.zeros(T)
        E = np.zeros(T)
        P = np.zeros(T)
        I = np.zeros(T)
        IS = np.zeros(T)
        S[0] = init_S
        E[0] = init_E
        P[0] = init_P
        I[0] = init_I * (1 - eta1[0] - eta2[0])
        IS[0] = init_I * (eta1[0] + eta2[0])
        cumulative_cases = np.zeros(T)
#         cumulative_cases[0] = Ecumm[0]
        cumulative_cases[0] = Icumm[0]
#         cumulative_cases[2*T] = Rcumm[0]

        for j in range(1, T):
            eta1j = get_arr(j+1, eta1)
            eta2j = get_arr(j+1, eta2)
            bmij = get_bmi(j+1, bmi)
            sigmaj = get_sigma(j+1, sigma)
            
            S[j] = S[j - 1] - alpha1 * S[j - 1] * bmij * I[j - 1] / population[0] - q0 * alpha1 * S[j - 1] * bmij * P[j - 1] / population[0]
            E[j] = E[j - 1] + alpha1 * S[j - 1] * bmij * I[j - 1] / population[0] + q0 * alpha1 * S[j - 1] * bmij * P[j - 1] / population[0] - E[j - 1] / (beta1 - pres_d)
            P[j] = P[j - 1] + E[j - 1] / (beta1 - pres_d) - P[j - 1] / pres_d
            I[j] = I[j - 1] + (1-eta1j-eta2j) * P[j - 1] / pres_d - sigmaj * I[j - 1]
            IS[j] = IS[j - 1] + (eta1j+eta2j) * P[j - 1] / pres_d - sigmaj * IS[j - 1]
#             cumulative_cases[j] = E[j] 
#             cumulative_cases[j] = cumulative_cases[j - 1] + E[j - 1] / beta1
            cumulative_cases[j] = P[j - 1] / pres_d
#             cumulative_cases[j + 2*T] = cumulative_cases[j + 2*T - 1] + sigmaj * (I[j - 1] + IS[j - 1])
            
        return cumulative_cases[:]
    cases = Lambda('cases', lambda sim = sim : sim)
    A = Poisson('A', mu = cases, value = case_data, observed = True)
    return locals()

def get_result_with_CI(params, bmi, eta1, eta2):
    allres = []
    alphas = params['alpha_trace']
#     betas = np.array([5.37 for i in range(len(alphas))])
    betas = params['beta_trace']
    qs = params['q_trace']
    
    for alpha, beta, q in zip(alphas, betas, qs):
        allres.append(simulate_exp( 
                          alpha, bmi, 
                          beta, sigma, q, eta1, eta2
                            )
                     )
    print(len(allres))
    result = np.zeros(allres[0].shape)
    resultup = np.zeros(allres[0].shape)
    resultdown = np.zeros(allres[0].shape)
    for i in range(allres[0].shape[0]):
        for j in range(allres[0].shape[1]):
            tmp = [x[i, j] for x in allres]
            result[i, j] = np.percentile(tmp, 50)
            resultup[i, j] = np.percentile(tmp, 97.5)
            resultdown[i, j] = np.percentile(tmp, 2.5)
    return result, resultup, resultdown

def get_result_with_CI2(params, bmi, eta1, eta2, sigma):
    allres = []
    alphas = params['alpha_trace']
#     betas = np.array([5.37 for i in range(len(alphas))])
    betas = params['beta_trace']
    qs = params['q_trace']
    print(sigma)
    for alpha, beta, q in zip(alphas, betas, qs):
        allres.append(simulate_exp( 
                          alpha, bmi, 
                          beta, sigma, q, eta1, eta2
                            )
                     )
    print(len(allres))
    result = np.zeros(allres[0].shape)
    resultup = np.zeros(allres[0].shape)
    resultdown = np.zeros(allres[0].shape)
    for i in range(allres[0].shape[0]):
        for j in range(allres[0].shape[1]):
            tmp = [x[i, j] for x in allres]
            result[i, j] = np.percentile(tmp, 50)
            resultup[i, j] = np.percentile(tmp, 97.5)
            resultdown[i, j] = np.percentile(tmp, 2.5)
    return result, resultup, resultdown

def save_resultforsimu(result, resultup, resultdown, orgres, orgresup, orgresdown, filename, dt_range):
    df = pd.DataFrame(data = {
        'Date': dt_range,
        'reported_I': Itrue[streetid, :] + Rtrue[streetid, :],
        'incidence_conbined_intervention_I': orgres[:, 6],
        'incidence_conbined_intervention_I_up': orgresup[:, 6],
        'incidence_conbined_intervention_I_down': orgresdown[:, 6],
        'incidence_simulation_I': result[:, 6],
        'incidence_simulation_I_up': resultup[:, 6],
        'incidence_simulation_I_down': resultdown[:, 6],
        'simulation_I': result[:, 3] + result[:, 4] + result[:, 5],
        'simulation_I_up': resultup[:, 3] + resultup[:, 4] + resultup[:, 5],
        'simulation_I_down': resultdown[:, 3] + resultdown[:, 4] + resultdown[:, 5],

    })
    df.to_csv(filename, index=False)

def sim_seiisr_exp(INPUT, alpha1, bmi, beta1, sigma, q, t_range, eta1, eta2):
    T = len(t_range)
    S = np.zeros(T)
    E = np.zeros(T)
    P = np.zeros(T)
    I = np.zeros(T)
    IS = np.zeros(T)
    R = np.zeros(T)
    newI = np.zeros(T)
    S[0] = INPUT[0]
    E[0] = INPUT[1]
    P[0] = INPUT[2]
    I[0] = INPUT[3]
    IS[0] = INPUT[4]
    R[0] = INPUT[5]
    newI[0] = INPUT[3]+INPUT[4]+INPUT[5]
    beta1 = 1 / (beta1 - pres_d)
    for j in range(1, T):
        eta1j = get_arr(j+1, eta1)
        eta2j = get_arr(j+1, eta2)
        bmij = get_bmi(j+1, bmi)
        sigmaj = get_sigma(j+1, sigma)
        S[j] = S[j - 1] - alpha1 * S[j - 1] * bmij * I[j - 1] / population[0] - q * alpha1 * S[j - 1] * bmij * P[j - 1] / population[0]
        E[j] = E[j - 1] + alpha1 * S[j - 1] * bmij * I[j - 1] / population[0] + q * alpha1 * S[j - 1] * bmij * P[j - 1] / population[0] - beta1 * E[j - 1]
        P[j] = P[j - 1] + beta1 * E[j - 1] - P[j - 1] / pres_d
        I[j] = I[j - 1] + (1-eta1j-eta2j) * P[j - 1] / pres_d - sigmaj * I[j - 1]
        IS[j] = IS[j - 1] + (eta1j+eta2j) * P[j - 1] / pres_d - sigmaj * IS[j - 1]
        R[j] = R[j - 1] + sigmaj * (I[j - 1] + IS[j - 1])
        newI[j] = P[j - 1] / pres_d

    return np.array([S, E, P, I, IS, R, newI]).T

def simulate_exp(alpha1, bmi, beta1, sigma, q, eta1, eta2):

#     k = 30
    
    if alpha1 < 0 or beta1 < 0:
        print('Warning ', alpha1, beta1)
    
    INPUT1 = np.zeros((6))
    INPUT1[0] = population[0]
    INPUT1[1] = initE[streetid]
    INPUT1[2] = initP[streetid]
    INPUT1[3] = initI[streetid] * (1 - eta1[0] - eta2[0])
    INPUT1[4] = initI[streetid] * (eta1[0] + eta2[0])
    INPUT1[5] = initR[streetid]
    
    t_range = np.arange(0.0, t, 1.0)
    RES1 = sim_seiisr_exp(INPUT1, alpha1, bmi, beta1, sigma, q, t_range, eta1, eta2)
    
    return RES1

matplotlib.rcParams['font.size'] = 12

for sce_name in ['Main', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']:
    dir_name = 'simuresult' if sce_name == 'Main' else 'sensitivity_result'
    gen_plot = True if sce_name == 'Main' else False
    street_params = load_obj('Beijing_tracetest_sensitivity_10m_{}'.format(sce_name))
    streetname = 'Beijing'
    streetid = 0
    e2 = bmi
    sigma = np.divide(1, confirmrate)
    sigma2 = sigma[:8]

    params = street_params['Beijing']

    if sce_name == 'S1':
        initE[0] = 35
    elif sce_name == 'S2':
        initE[0] = 43
    else:
        initE[0] = 37

    if sce_name == 'S5':
        pres_d = 1.1
    elif sce_name == 'S6':
        pres_d = 3.0
    else:
        pres_d = 2.
    initP = [Itrue[0, int(pres_d)] - Itrue[0, 0]]

    t = len(alldates) + 0
    based_time = str_to_dt(alldates[0])
    t_range_subdt = [based_time + dt.timedelta(days = x) for x in range(t)]

    orgres, orgresup, orgresdown = get_result_with_CI(params, bmi, eta1, eta2)
    save_resultforsimu(orgres, orgresup, orgresdown, orgres, orgresup, orgresdown, './{}/{}_resultsimu111.csv'.format(dir_name, sce_name), t_range_subdt)

    from matplotlib.dates import DateFormatter
    formatter = DateFormatter("%d %B")

    plt.figure(figsize=(10, 5))
    plt.plot(t_range_subdt[1:], orgres[1:, 6], 'b', label = 'Simulation')
    plt.fill_between(t_range_subdt[1:], orgresdown[1:, 6], orgresup[1:, 6], alpha=0.5, color='lightblue')
    plt.plot(t_range_subdt[1:len(Itrue[streetid, :])], np.diff(Itrue[streetid, :] + Rtrue[streetid, :]), 'ko', label = 'Observation')
    plt.xlabel('Date')
    plt.ylabel('Number of cases')
    span_xticks = 3
    x = t_range_subdt
    plt.xticks(np.array(x)[np.arange(0, len(x), span_xticks)])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim(0, 50)
    plt.xlim(t_range_subdt[0])
    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()
    plt.legend(loc='upper left', frameon=False)
    if gen_plot:
        plt.savefig('./{}/{}_FigS8A_incidence.pdf'.format(dir_name, sce_name), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t_range_subdt, orgres[:, 3] + orgres[:, 4] + orgres[:, 5], 'b', label = 'Estimated')
    plt.fill_between(t_range_subdt, orgresdown[:, 3] + orgresdown[:, 4] + orgresdown[:, 5], orgresup[:, 3] + orgresup[:, 4] + orgresup[:, 5], alpha=0.5, color='lightblue')
    plt.plot(t_range_subdt[:len(Itrue[streetid, :])], Itrue[streetid, :] + Rtrue[streetid, :], 'ko', label = 'Reported')
    plt.xlabel('Date')
    plt.ylabel('Number of cumulative cases')
    span_xticks = 3
    x = t_range_subdt
    plt.xticks(np.array(x)[np.arange(0, len(x), span_xticks)])
    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim(0,400)
    plt.xlim(t_range_subdt[0])
    plt.legend(loc='upper left', frameon=False)
    if gen_plot:
        plt.savefig('./{}/{}_FigS8B_cumulative.pdf'.format(dir_name, sce_name), bbox_inches='tight')
    plt.close()

    reported_I = Itrue[streetid, :] + Rtrue[streetid, :]
    print("RMSE for scenario {} is {:.3f}".format(sce_name, np.sqrt(mean_squared_error(np.diff(reported_I), orgres[1:, 6]))))

    result, resultup, resultdown = get_result_with_CI2(params, bmi_noctrl, eta1_noctrl, eta2_noctrl, sigma2)
    save_resultforsimu(result, resultup, resultdown, orgres, orgresup, orgresdown, './{}/{}_resultsimu000.csv'.format(dir_name, sce_name), t_range_subdt)
    
    plt.figure(figsize=(10, 5))
    plt.plot(t_range_subdt, orgres[:, 6], 'b', label = 'Combined interventions', clip_on=False, zorder=3)
    plt.fill_between(t_range_subdt, orgresdown[:, 6], orgresup[:, 6], alpha=0.5, color='lightblue', clip_on=False, zorder=3)
    plt.plot(t_range_subdt, result[:, 6], 'orchid', label = 'No NPIs')
    plt.fill_between(t_range_subdt, resultdown[:, 6], resultup[:, 6], alpha=0.7, color='lavender')
    plt.xlabel('Date')
    plt.ylabel('Number of cases')
    span_xticks = 3
    x = t_range_subdt
    plt.xticks(np.array(x)[np.arange(0, len(x), span_xticks)])
    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()
    # plt.grid()
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim(0, 10000)
    plt.xlim(t_range_subdt[0])
    plt.legend(loc='upper left', frameon=False)
    if gen_plot:
        plt.savefig('./{}/{}_FigS7A_incidence.pdf'.format(dir_name, sce_name), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(t_range_subdt, orgres[:, 6], 'b', label = 'A: Combined interventions')
    plt.fill_between(t_range_subdt, orgresdown[:, 6], orgresup[:, 6], alpha=0.5, color='lightblue')

    result, resultup, resultdown = get_result_with_CI(params, bmi_noctrl, eta1, eta2)
    save_resultforsimu(result, resultup, resultdown, orgres, orgresup, orgresdown, './{}/{}_resultsimu011.csv'.format(dir_name, sce_name), t_range_subdt)
    plt.plot(t_range_subdt, result[:, 6], 'r', label = 'C: No localized lockdown')
    plt.fill_between(t_range_subdt, resultdown[:, 6], resultup[:, 6], alpha=0.5, color='pink')

    result, resultup, resultdown = get_result_with_CI(params, bmi, eta1_noctrl, eta2)
    save_resultforsimu(result, resultup, resultdown, orgres, orgresup, orgresdown, './{}/{}_resultsimu101.csv'.format(dir_name, sce_name), t_range_subdt)
    plt.plot(t_range_subdt, result[:, 6], 'orange', label = 'D: No close-contact tracing')
    plt.fill_between(t_range_subdt, resultdown[:, 6], resultup[:, 6], alpha=0.5, color='wheat')

    result, resultup, resultdown = get_result_with_CI(params, bmi, eta1, eta2_noctrl)
    save_resultforsimu(result, resultup, resultdown, orgres, orgresup, orgresdown, './{}/{}_resultsimu110.csv'.format(dir_name, sce_name), t_range_subdt)
    plt.plot(t_range_subdt, result[:, 6], 'orchid', label = 'E: No community-based NAT')
    plt.fill_between(t_range_subdt, resultdown[:, 6], resultup[:, 6], alpha=0.7, color='lavender')

    plt.xlabel('Date')
    plt.ylabel('Number of cases')
    span_xticks = 3
    x = t_range_subdt
    plt.xticks(np.array(x)[np.arange(0, len(x), span_xticks)])
    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()
    # plt.grid()
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim(0, 80)
    plt.xlim(t_range_subdt[0], t_range_subdt[-1])
    plt.legend(loc='upper left', frameon=False)
    if gen_plot:
        plt.savefig('./{}/{}_FigS7B_incidence.pdf'.format(dir_name, sce_name), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(t_range_subdt, orgres[:, 6], 'b', label = 'A: Combined interventions')
    plt.fill_between(t_range_subdt, orgresdown[:, 6], orgresup[:, 6], alpha=0.5, color='lightblue')

    result, resultup, resultdown = get_result_with_CI(params, bmi, eta1_noctrl, eta2_noctrl)
    save_resultforsimu(result, resultup, resultdown, orgres, orgresup, orgresdown, './{}/{}_resultsimu100.csv'.format(dir_name, sce_name), t_range_subdt)
    plt.plot(t_range_subdt, result[:, 6], 'r', label = 'F: Localized lockdown only')
    plt.fill_between(t_range_subdt, resultdown[:, 6], resultup[:, 6], alpha=0.5, color='pink')

    result, resultup, resultdown = get_result_with_CI(params, bmi_noctrl, eta1, eta2_noctrl)
    save_resultforsimu(result, resultup, resultdown, orgres, orgresup, orgresdown, './{}/{}_resultsimu010.csv'.format(dir_name, sce_name), t_range_subdt)
    plt.plot(t_range_subdt, result[:, 6], 'orange', label = 'G: Close-contact tracing only')
    plt.fill_between(t_range_subdt, resultdown[:, 6], resultup[:, 6], alpha=0.5, color='wheat')

    result, resultup, resultdown = get_result_with_CI(params, bmi_noctrl, eta1_noctrl, eta2)
    save_resultforsimu(result, resultup, resultdown, orgres, orgresup, orgresdown, './{}/{}_resultsimu001.csv'.format(dir_name, sce_name), t_range_subdt)
    plt.plot(t_range_subdt, result[:, 6], 'orchid', label = 'H: Community-based NAT only')
    plt.fill_between(t_range_subdt, resultdown[:, 6], resultup[:, 6], alpha=0.7, color='lavender')

    plt.xlabel('Date')
    plt.ylabel('Number of cases')
    span_xticks = 3
    x = t_range_subdt
    plt.xticks(np.array(x)[np.arange(0, len(x), span_xticks)])
    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()
    # plt.grid()
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim(0, 250)
    plt.xlim(t_range_subdt[0], t_range_subdt[-1])
    plt.legend(loc='upper left', frameon=False)
    if gen_plot:
        plt.savefig('./{}/{}_Fig4B_incidence.pdf'.format(dir_name, sce_name), bbox_inches='tight')
    plt.close()

    result, resultup, resultdown = get_result_with_CI(params, bmi, eta1, eta220)
    save_resultforsimu(result, resultup, resultdown, orgres, orgresup, orgresdown, './{}/{}_resultsimu110Jun20.csv'.format(dir_name, sce_name), t_range_subdt)

    result, resultup, resultdown = get_result_with_CI(params, bmi_noctrl, eta1_noctrl, eta220)
    save_resultforsimu(result, resultup, resultdown, orgres, orgresup, orgresdown, './{}/{}_resultsim001beforejun21.csv'.format(dir_name, sce_name), t_range_subdt)

    print("{} Done".format(sce_name))
