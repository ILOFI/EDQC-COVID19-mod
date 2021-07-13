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

simu_name = 'Main'

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


# In[27]:

streetname = 'Beijing'
streetid = 0
e2 = bmi
sigma = np.divide(1, confirmrate)
Ecumm = Etrue[streetid, :]
Icumm = Itrue[streetid, :]
Rcumm = Rtrue[streetid, :]

print("Sampling parameters for {}".format(simu_name))
print(N[streetid], initE[streetid], initI[streetid], initR[streetid], streetid, streetname)
np.random.seed(202018)
mod1 = SEIR_MODEL1(Ecumm, Icumm + Rcumm, Rcumm, N[streetid], initE[streetid], initP[streetid], initI[streetid], initR[streetid], 
                  e2, sigma)
mc = MCMC(mod1)
mc.use_step_method(AdaptiveMetropolis, [mod1['alpha1'], mod1['beta1'], mod1['q0']])#, mod1['s0'], mod1['e0']
mc.sample(iter = 10001000, burn = 1000, thin = 1000, verbose = 0) #23000 35000


# In[29]:

street_params = {}
street_params[streetname] = {'alpha_trace': mod1['alpha1'].trace(), 'beta_trace': mod1['beta1'].trace(), 'q_trace': mod1['q0'].trace()}


# In[31]:

plt.plot(street_params['Beijing']['alpha_trace'])
plt.ylim([0.2, 1.6])
plt.title(r'$\beta_0$', size=15)
plt.xlabel(r'Iteration ($\times 1000$)', size=12)
plt.ylabel('Value', size=12)
plt.xlim([0, 10000])
plt.savefig('./simuresult/{}_trace_plot_beta.png'.format(simu_name), dpi=500)
plt.close()

plt.plot(street_params['Beijing']['q_trace'])
plt.ylim([0., 1.0])
plt.title(r'$q_0$', size=15)
plt.xlabel(r'Iteration ($\times 1000$)', size=12)
plt.ylabel('Value', size=12)
plt.xlim([0, 10000])
plt.savefig('./simuresult/{}_trace_plot_q.png'.format(simu_name), dpi=500)
plt.close()


# In[68]:

street_params['Beijing']['alpha_trace']

save_obj(street_params, 'Beijing_tracetest_sensitivity_10m_{}'.format(simu_name))

print("{} Done".format(simu_name))