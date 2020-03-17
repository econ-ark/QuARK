# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: collapsed,code_folding
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Making Structural Estimates From Empirical Results
#
# This notebook conducts a quick and dirty structural estimation based on Table 9 of "MPC Heterogeneity and Household Balance Sheets" by Fagereng, Holm, and Natvik <cite data-cite="6202365/SUE56C4B"></cite>, who use Norweigian administrative data on income, household assets, and lottery winnings to examine the MPC from transitory income shocks (lottery prizes).  Their Table 9 reports an estimated MPC broken down by quartiles of bank deposits and
# prize size; this table is reproduced here as $\texttt{MPC_target_base}$.  In this demo, we use the Table 9 estimates as targets in a simple structural estimation, seeking to minimize the sum of squared differences between simulated and estimated MPCs by changing the (uniform) distribution of discount factors.  The essential question is how well their results be rationalized by a simple one-asset consumption-saving model.  
#
#
# The function that estimates discount factors includes several options for estimating different specifications:
#
# 1. TypeCount : Integer number of discount factors in discrete distribution; can be set to 1 to turn off _ex ante_ heterogeneity (and to discover that the model has no chance to fit the data well without such heterogeneity).
# 2. AdjFactor : Scaling factor for the target MPCs; user can try to fit estimated MPCs scaled down by (e.g.) 50%.
# 3. T_kill    : Maximum number of years the (perpetually young) agents are allowed to live.  Because this is quick and dirty, it's also the number of periods to simulate.
# 4. Splurge   : Amount of lottery prize that an individual will automatically spend in a moment of excitement (perhaps ancient tradition in Norway requires a big party when you win the lottery), before beginning to behave according to the optimal consumption function.  The patterns in Table 9 can be fit much better when this is set around \$700 --> 0.7.  That doesn't seem like an unreasonable amount of money to spend on a memorable party.
# 5. do_secant : Boolean indicator for whether to use "secant MPC", which is average MPC over the range of the prize.  MNW believes authors' regressions are estimating this rather than point MPC.  When False, structural estimation uses point MPC after receiving prize.  NB: This is incompatible with Splurge > 0.
# 6. drop_corner : Boolean for whether to include target MPC in the top left corner, which is greater than 1.  Authors discuss reasons why the MPC from a transitory shock *could* exceed 1.  Option is included here because this target tends to push the estimate around a bit.

# %% {"code_folding": [0]}
# Import python tools

import sys
import os

import numpy as np
from copy import deepcopy

# %% {"code_folding": [0]}
# Import needed tools from HARK

from HARK.utilities import approxUniform, getPercentiles
from HARK.parallel import multiThreadCommands
from HARK.estimation import minimizeNelderMead
from HARK.ConsumptionSaving.ConsIndShockModel import *
from HARK.cstwMPC.SetupParamsCSTW import init_infinite

# %% {"code_folding": [0]}
# Set key problem-specific parameters

TypeCount = 8    # Number of consumer types with heterogeneous discount factors
AdjFactor = 1.0  # Factor by which to scale all of MPCs in Table 9
T_kill = 100     # Don't let agents live past this age
Splurge = 0.0    # Consumers automatically spend this amount of any lottery prize
do_secant = True # If True, calculate MPC by secant, else point MPC
drop_corner = False # If True, ignore upper left corner when calculating distance

# %% {"code_folding": [0]}
# Set standard HARK parameter values

base_params = deepcopy(init_infinite)
base_params['LivPrb'] = [0.975]
base_params['Rfree'] = 1.04/base_params['LivPrb'][0]
base_params['PermShkStd'] = [0.1]
base_params['TranShkStd'] = [0.1]
base_params['T_age'] = T_kill # Kill off agents if they manage to achieve T_kill working years
base_params['AgentCount'] = 10000
base_params['pLvlInitMean'] = np.log(23.72) # From Table 1, in thousands of USD
base_params['T_sim'] = T_kill  # No point simulating past when agents would be killed off

# %% {"code_folding": [0]}
# Define the MPC targets from Fagereng et al Table 9; element i,j is lottery quartile i, deposit quartile j

MPC_target_base = np.array([[1.047, 0.745, 0.720, 0.490],
                            [0.762, 0.640, 0.559, 0.437],
                            [0.663, 0.546, 0.390, 0.386],
                            [0.354, 0.325, 0.242, 0.216]])
MPC_target = AdjFactor*MPC_target_base

# %% {"code_folding": [0]}
# Define the four lottery sizes, in thousands of USD; these are eyeballed centers/averages

lottery_size = np.array([1.625, 3.3741, 7.129, 40.0])

# %% {"code_folding": [0]}
# Make several consumer types to be used during estimation

BaseType = IndShockConsumerType(**base_params)
EstTypeList = []
for j in range(TypeCount):
    EstTypeList.append(deepcopy(BaseType))
    EstTypeList[-1](seed = j)

# %% {"code_folding": [0]}
# Define the objective function

def FagerengObjFunc(center,spread,verbose=False):
    '''
    Objective function for the quick and dirty structural estimation to fit
    Fagereng, Holm, and Natvik's Table 9 results with a basic infinite horizon
    consumption-saving model (with permanent and transitory income shocks).

    Parameters
    ----------
    center : float
        Center of the uniform distribution of discount factors.
    spread : float
        Width of the uniform distribution of discount factors.
    verbose : bool
        When True, print to screen MPC table for these parameters.  When False,
        print (center, spread, distance).

    Returns
    -------
    distance : float
        Euclidean distance between simulated MPCs and (adjusted) Table 9 MPCs.
    '''
    # Give our consumer types the requested discount factor distribution
    beta_set = approxUniform(N=TypeCount,bot=center-spread,top=center+spread)[1]
    for j in range(TypeCount):
        EstTypeList[j](DiscFac = beta_set[j])

    # Solve and simulate all consumer types, then gather their wealth levels
    multiThreadCommands(EstTypeList,['solve()','initializeSim()','simulate()','unpackcFunc()'])
    WealthNow = np.concatenate([ThisType.aLvlNow for ThisType in EstTypeList])

    # Get wealth quartile cutoffs and distribute them to each consumer type
    quartile_cuts = getPercentiles(WealthNow,percentiles=[0.25,0.50,0.75])
    for ThisType in EstTypeList:
        WealthQ = np.zeros(ThisType.AgentCount,dtype=int)
        for n in range(3):
            WealthQ[ThisType.aLvlNow > quartile_cuts[n]] += 1
        ThisType(WealthQ = WealthQ)

    # Keep track of MPC sets in lists of lists of arrays
    MPC_set_list = [ [[],[],[],[]],
                     [[],[],[],[]],
                     [[],[],[],[]],
                     [[],[],[],[]] ]

    # Calculate the MPC for each of the four lottery sizes for all agents
    for ThisType in EstTypeList:
        ThisType.simulate(1)
        c_base = ThisType.cNrmNow
        MPC_this_type = np.zeros((ThisType.AgentCount,4))
        for k in range(4): # Get MPC for all agents of this type
            Llvl = lottery_size[k]
            Lnrm = Llvl/ThisType.pLvlNow
            if do_secant:
                SplurgeNrm = Splurge/ThisType.pLvlNow
                mAdj = ThisType.mNrmNow + Lnrm - SplurgeNrm
                cAdj = ThisType.cFunc[0](mAdj) + SplurgeNrm
                MPC_this_type[:,k] = (cAdj - c_base)/Lnrm
            else:
                mAdj = ThisType.mNrmNow + Lnrm
                MPC_this_type[:,k] = cAdj = ThisType.cFunc[0].derivative(mAdj)

        # Sort the MPCs into the proper MPC sets
        for q in range(4):
            these = ThisType.WealthQ == q
            for k in range(4):
                MPC_set_list[k][q].append(MPC_this_type[these,k])

    # Calculate average within each MPC set
    simulated_MPC_means = np.zeros((4,4))
    for k in range(4):
        for q in range(4):
            MPC_array = np.concatenate(MPC_set_list[k][q])
            simulated_MPC_means[k,q] = np.mean(MPC_array)

    # Calculate Euclidean distance between simulated MPC averages and Table 9 targets
    diff = simulated_MPC_means - MPC_target
    if drop_corner:
        diff[0,0] = 0.0
    distance = np.sqrt(np.sum((diff)**2))
    if verbose:
        print(simulated_MPC_means)
    else:
        print (center, spread, distance)
    return distance


# %% {"code_folding": [0]}
# Conduct the estimation

guess = [0.92,0.03]
f_temp = lambda x : FagerengObjFunc(x[0],x[1])
opt_params = minimizeNelderMead(f_temp, guess, verbose=True)
print('Finished estimating for scaling factor of ' + str(AdjFactor) + ' and "splurge amount" of $' + str(1000*Splurge))
print('Optimal (beta,nabla) is ' + str(opt_params) + ', simulated MPCs are:')
dist = FagerengObjFunc(opt_params[0],opt_params[1],True)
print('Distance from Fagereng et al Table 9 is ' + str(dist))

# %% [markdown]
# ### PROBLEM
#
# See what happens if you do not allow a splurge amount at all.  Hint: Think about how this question relates to the `drop_corner` option.
#
# Explain why you get the results you do, and comment on possible interpretations of the "splurge" that might be consistent with economic theory.    
# Hint: What the authors are able to measure is actually the marginal propensity to EXPEND, not the marginal propensity to CONSUME as it is defined in our benchmark model.

# %%
# Put your solution here
drop_corner = True # If True, ignore upper left corner when calculating distance
Splurge = 0.0    # Consumers automatically spend this amount of any lottery prize (default = 0.0)

f_temp = lambda x : FagerengObjFunc(x[0],x[1])
opt_params = minimizeNelderMead(f_temp, guess, verbose=True)
print('Finished estimating for scaling factor of ' + str(AdjFactor) + ' and "splurge amount" of $' + str(1000*Splurge))
print('Optimal (beta,nabla) is ' + str(opt_params) + ', simulated MPCs are:')
dist = FagerengObjFunc(opt_params[0],opt_params[1],True)
print('Distance from Fagereng et al Table 9 is ' + str(dist))

# %%
print('\n\
When not matching the lowest MPC-wealth-cell, the distrance to the \
empirical distribution is lower than when matching the whole \
distribution. \n\
Furthermore, the estimated discount factor is higher, implying that \
households are more patient on average. \n\
The spread in discount rates is also lower. \n\
The combined effect is that both most impatient, and the most patient household \
is more patient than the corresponding household when matching the full \
MPC-wealth distribution. \n\
In order to generate a higher average MPC and a larger spread in MPCs, the \
first estimation requires both lower patience, on average, and more \
heterogeneity in patience. \n\
\n\
Not matching the upper-left corner of the MPC-distribution is equivalent to \
saying that the current model is not a good model for those households. These \
are households which have low wealth and win small prizes. \
One way to think about the high MPC is that these households either have \
    \n 1) future financial committments which they settle today, or \
    \n 2) the lottery win allows them to purchase durable goods today \n\
As long as these "uses" are of constant dollar value (i.e. do to scale with \
the size of the prize) we would expect the MPC to be declining in the value \
of the prize. This is also the case in the data. \
')

# %% Estimating with splurge > 0
drop_corner = False # If True, ignore upper left corner when calculating distance
Splurge = 0.43*lottery_size[0]    # Consumers automatically spend this amount of any lottery prize (default = 0.0)

f_temp = lambda x : FagerengObjFunc(x[0],x[1])
opt_params = minimizeNelderMead(f_temp, guess, verbose=True)
print('Finished estimating for scaling factor of ' + str(AdjFactor) + ' and "splurge amount" of $' + str(1000*Splurge))
print('Optimal (beta,nabla) is ' + str(opt_params) + ', simulated MPCs are:')
dist = FagerengObjFunc(opt_params[0],opt_params[1],True)
print('Distance from Fagereng et al Table 9 is ' + str(dist))

# %% [markdown]
# ### PROBLEM
#
# Call the _Marginal Propensity to Continue Consuming_ (MPCC) in year `t+n` the proportion of lottery winnings that get spent in year `t+n`.  That is, if consumption is higher in year `t+2` by an amount corresponding to 14 percent of lottery winnings, we would say  _the MPCC in t+2 is 14 percent.
#
# For the baseline version of the model with the "splurge" component, calculate the MPCC's for years `t+1` through `t+3` and plot them together with the MPC in the first year (including the splurge component)
#

center = 0.7898188094496416
spread = 0.16098056707531155

ahead = 3

# %%
# # def MPCC(ahead):
# #     '''
# #     Calculates the margina propensity to continue consuming (MPCC) in

# #     Parameters
# #     ----------
# #     ahead : integer
# #         Number of periods ahead to calculate the MPCC.
    
# #     Returns
# #     -------
# #     MPCC : float
# #         The MPCC n-periods ahead (n=ahead).
# #     '''
#     # Give our consumer types the requested discount factor distribution
# beta_set = approxUniform(N=TypeCount,bot=center-spread,top=center+spread)[1]
# for j in range(TypeCount):
#     EstTypeList[j](DiscFac = beta_set[j])

# # Solve and simulate all consumer types, then gather their wealth levels
# multiThreadCommands(EstTypeList,['solve()','initializeSim()','simulate()','unpackcFunc()'])
# WealthNow = np.concatenate([ThisType.aLvlNow for ThisType in EstTypeList])

# # Get wealth quartile cutoffs and distribute them to each consumer type
# quartile_cuts = getPercentiles(WealthNow,percentiles=[0.25,0.50,0.75])
#     for ThisType in EstTypeList:
#         WealthQ = np.zeros(ThisType.AgentCount,dtype=int)
#         for n in range(3):
#             WealthQ[ThisType.aLvlNow > quartile_cuts[n]] += 1
#         ThisType(WealthQ = WealthQ)

#     # Keep track of MPC sets in lists of lists of arrays
#     MPC_set_list = [ [[],[],[],[]],
#                       [[],[],[],[]],
#                       [[],[],[],[]],
#                       [[],[],[],[]] ]

#     # Calculate the MPC for each of the four lottery sizes for all agents
#     for ThisType in EstTypeList:
#         ThisType.simulate(1)
#         c_base = ThisType.cNrmNow
#         MPC_this_type = np.zeros((ThisType.AgentCount,4))
#         for k in range(4): # Get MPC for all agents of this type
#             Llvl = lottery_size[k]
#             Lnrm = Llvl/ThisType.pLvlNow
#             if do_secant:
#                 SplurgeNrm = Splurge/ThisType.pLvlNow
#                 mAdj = ThisType.mNrmNow + Lnrm - SplurgeNrm
#                 cAdj = ThisType.cFunc[0](mAdj) + SplurgeNrm
#                 MPC_this_type[:,k] = (cAdj - c_base)/Lnrm
#             else:
#                 mAdj = ThisType.mNrmNow + Lnrm
#                 MPC_this_type[:,k] = cAdj = ThisType.cFunc[0].derivative(mAdj)

#         # Sort the MPCs into the proper MPC sets
#         for q in range(4):
#             these = ThisType.WealthQ == q
#             for k in range(4):
#                 MPC_set_list[k][q].append(MPC_this_type[these,k])

#     # Calculate average within each MPC set
#     simulated_MPC_means = np.zeros((4,4))
#     for k in range(4):
#         for q in range(4):
#             MPC_array = np.concatenate(MPC_set_list[k][q])
#             simulated_MPC_means[k,q] = np.mean(MPC_array)
            
# # %%
# MPCC(4)

# %%
BaseType = IndShockConsumerType(**base_params)
EstTypeList = []
for j in range(TypeCount):
    EstTypeList.append(deepcopy(BaseType))
    EstTypeList[-1](seed = j)

center = 0.78981881 
spread = 0.16098057
nInit = 95
nExtra = 3

# Give our consumer types the requested discount factor distribution
beta_set = approxUniform(N=TypeCount,bot=center-spread,top=center+spread)[1]
for j in range(TypeCount):
    EstTypeList[j](DiscFac = beta_set[j])
    EstTypeList[j].track_vars = ['aNrmNow','mNrmNow','cNrmNow','pLvlNow','PermShkNow','TranShkNow']

# Solve and simulate all consumer types, then gather their wealth levels
multiThreadCommands(EstTypeList,['solve()','initializeSim()','simulate(' +str(nInit) +')','unpackcFunc()'])
WealthNow = np.concatenate([ThisType.aLvlNow for ThisType in EstTypeList])

# Get wealth quartile cutoffs and distribute them to each consumer type
quartile_cuts = getPercentiles(WealthNow,percentiles=[0.25,0.50,0.75])
for ThisType in EstTypeList:
    WealthQ = np.zeros(ThisType.AgentCount,dtype=int)
    for n in range(3):
        WealthQ[ThisType.aLvlNow > quartile_cuts[n]] += 1
    ThisType(WealthQ = WealthQ)

# Keep track of MPC sets in lists of lists of arrays
MPC_set_list = [ [[],[],[],[]],
                 [[],[],[],[]],
                 [[],[],[],[]],
                 [[],[],[],[]] ]


# Calculate the MPC for each of the four lottery sizes for all agents
Rfree=base_params['Rfree']
for ThisType in EstTypeList:
    MPC_this_type = np.zeros((ThisType.AgentCount,4))
    ThisType.simulate(nExtra)
    c_base = ThisType.cNrmNow_hist[nInit]
    print(ThisType.cNrmNow)
    print(ThisType.cNrmNow_hist[nInit + nExtra - 1])
    
    mAdj_hist = ThisType.mNrmNow_hist
    # I AM HERE
    mAdj_hist[nInit:nInit+nExtra-1] = np.zeros((nExtra, ))
    
    for k in range(4): # Get MPC for all agents of this type
        Llvl = lottery_size[k]
        Lnrm = Llvl/ThisType.pLvlNow_hist[95]
        SplurgeNrm = Splurge/ThisType.pLvlNow_hist[95]
        mAdj = ThisType.mNrmNow + Lnrm - SplurgeNrm
        cAdj = ThisType.cFunc[0](mAdj) + SplurgeNrm
        MPC_this_type[:,k] = (cAdj - c_base)/Lnrm
        
        
        # Sort the MPCs into the proper MPC sets
        for q in range(4):
            these = ThisType.WealthQ == q
            for k in range(4):
                MPC_set_list[k][q].append(MPC_this_type[these,k])
                
                
# Calculate average within each MPC set
simulated_MPC_means = np.zeros((4,4))
for k in range(4):
    for q in range(4):
        MPC_array = np.concatenate(MPC_set_list[k][q])
        simulated_MPC_means[k,q] = np.mean(MPC_array)
        
        
print(simulated_MPC_means)



























































