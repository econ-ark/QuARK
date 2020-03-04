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
#
# # A Life Cycle Model: Data and Theory
#
# National registry data on income and wealth from Scandinavian countries (esp. Norway) have recently become available (with a lot of security) to some (lucky!) researchers.   These data offer a uniquely powerful tool for testing (and improving) our models of consumption and saving behavior over the life cycle.
#
# This notebook is an example of how to construct a life cycle model with the HARK toolkit that makes predictions that can be compared to the raw data statistics that now are becoming available.
#
# For example, existing papers have tabulated information about the **growth rate** of assets at different ages over the life cycle. 
#
# The default parameters of the HARK life cycle model have not been optmized to match features of the Norwegian data; a first step in a real "structural" estimation would be to use Norwegian calibrate the inputs to the model (like the profile of income, and the magnitude of income shocks, over the life cycle), and then to find the values of parameters like the time preference rate that allow the model to fit the data best.  (See [SolvingMicroDSOPs](https://econ.jhu.edu/people/ccarroll/SolvingMicroDSOPs) for how this can be done, and search for the corresponding HARK content using [our documentation](https://hark.readthedocs.io)).

# %% {"code_folding": []}
# Initial imports and notebook setup, click arrow to show

import HARK.ConsumptionSaving.ConsIndShockModel as cShksModl        # The consumption-saving micro model
import HARK.SolvingMicroDSOPs.Calibration.EstimationParameters as Params    # Parameters for the consumer type and the estimation
from HARK.utilities import plotFuncsDer, plotFuncs              # Some tools
import pandas as pd 

import numpy as np


# %% {"code_folding": [0]}
# Set up default values for CRRA, DiscFac, and simulation variables in the dictionary 
Params.init_consumer_objects["CRRA"]= 2.00            # Default coefficient of relative risk aversion (rho)
Params.init_consumer_objects["DiscFac"]= 0.97         # Default intertemporal discount factor (beta)
Params.init_consumer_objects["PermGroFacAgg"]= 1.0    # Aggregate permanent income growth factor 
Params.init_consumer_objects["aNrmInitMean"]= -10.0   # Mean of log initial assets 
Params.init_consumer_objects["aNrmInitStd"]= 1.0      # Standard deviation of log initial assets
Params.init_consumer_objects["pLvlInitMean"]= 0.0     # Mean of log initial permanent income 
Params.init_consumer_objects["pLvlInitStd"]= 0.0      # Standard deviation of log initial permanent income


# %%
# Make an instance of a lifecycle consumer to be used for estimation
LifeCyclePop = cShksModl.IndShockConsumerType(**Params.init_consumer_objects)


# %% {"code_folding": [0]}
# Solve and simulate the model (ignore the "warning" message)
LifeCyclePop.solve()                            # Obtain consumption rules by age 
LifeCyclePop.unpackcFunc()                      # Expose the consumption rules

# Which variables do we want to track
LifeCyclePop.track_vars = ['aNrmNow','pLvlNow','mNrmNow','cNrmNow','TranShkNow']

LifeCyclePop.T_sim = 120                        # Nobody lives to be older than 145 years (=25+120)
LifeCyclePop.initializeSim()                    # Construct the age-25 distribution of income and assets
LifeCyclePop.simulate()                         # Simulate a population behaving according to this model


# %% {"code_folding": [0]}
# Plot the consumption functions during working life

print('Consumption as a function of market resources while working:')
mMin = min([LifeCyclePop.solution[t].mNrmMin for t in range(LifeCyclePop.T_cycle)])
plotFuncs(LifeCyclePop.cFunc[:LifeCyclePop.T_retire],mMin,5)


# %% {"code_folding": [0]}
# Define the saving rate function
def savRteFunc(SomeType, m, t):
    """
    Parameters:
    ----------
        SomeType: 
             Agent type that has been solved and simulated.
        m:
            normalized market resources of agent
        t:
            age of agent (from starting in the workforce)
        
        
    Returns:
    --------
        savRte: float
    
    """
    inc = (SomeType.Rfree -1.)*(m-1.)+1. # Normalized by permanent labor income
    cns = SomeType.solution[t].cFunc(m)  # Consumption (normalized)
    sav = inc - cns                      # Flow of saving this period
    savRte = sav / inc                   # Saving Rate
    return savRte  


# %% {"code_folding": []}
# Create a giant matrix gathering useful data:
# 't_now', 'aNrmNow_hist', 'cNrmNow_hist', employment-status in date t and date t-1,
# aLvlGro_hist, Saving rate

w, h = 1, LifeCyclePop.T_cycle
giant_list = [[0 for x in range(w)] for y in range(h)]
savRte_list = []

import warnings
warnings.filterwarnings("ignore") # Suppress some disturbing but harmless warnings

for t in range(1,LifeCyclePop.T_cycle+1):
    #aLvlGro_hist[0] = 0 # set the first growth rate to 0, since there is no data for period 0
    aLvlGroNow = np.log((LifeCyclePop.aNrmNow_hist[t]   *LifeCyclePop.pLvlNow_hist[t])/ \
                         LifeCyclePop.aNrmNow_hist[t-1] *LifeCyclePop.pLvlNow_hist[t-1]) # (10000,)

    # Call the saving rate function defined above 
    savRte = savRteFunc(LifeCyclePop, LifeCyclePop.mNrmNow_hist[t] , t)
      
    savRte_list.append(savRte) # Add this period's saving rate to the list 

    # Create elements of matrix list
    matrix_list = [0 for number in range(7)]
    matrix_list[0] = t
    matrix_list[1] = LifeCyclePop.aNrmNow_hist[t]
    matrix_list[2] = LifeCyclePop.cNrmNow_hist[t]
    matrix_list[3] = LifeCyclePop.TranShkNow_hist[t]
    matrix_list[4] = LifeCyclePop.TranShkNow_hist[t-1]
    matrix_list[5] = aLvlGroNow
    matrix_list[6] = savRte
    
    giant_list[t-1] = matrix_list

# %% {"code_folding": [0]}
# Construct the level of assets A from a*p where a is the ratio to permanent income p
# Remember 41 is "years after entering workforce" (=age 25); 66 is the year right after retirement
LifeCyclePop.aLvlNow_hist = LifeCyclePop.aNrmNow_hist*LifeCyclePop.pLvlNow_hist
aGro41=LifeCyclePop.aLvlNow_hist[41]/LifeCyclePop.aLvlNow_hist[40]
aGro41NoU=aGro41[aGro41[:]>0.2] # Throw out extreme outliers; don't want growth rates relative to 0 income!


# %% {"code_folding": [0]}
# Plot the (truncated) distribution of growth rates of wealth between age 65 and 66 (=25 + 41)
from matplotlib import pyplot as plt
n, bins, patches = plt.hist(aGro41NoU,50,density=True)


# %% [markdown]
# # PROBLEM : Distribution of aNrm and aLev
#
# Perhaps more interesting than the distribution of asset growth rates over the life cycle is the distribution of the level of assets, or the ratio of assets to permanent income.
#
# Construct a plot similar to the one above for the disributions of $\texttt{aNrm}$ and $\texttt{aLev}$ in the period just BEFORE retirement (44 periods from the start).
#

# %%
# put your solution here

# %% [markdown]
# # PROBLEM : "Luck" and Saving
#
# In this model, each consumer experiences a set of draws of permanent income shocks over their lifetime.  Some will be lucky and draw a mostly positive series of shocks (and unlucky people experience negative shocks).
#
# This problem asks you to examine the consequences of these shocks for the lifetime pattern of saving.
#
# The first step is to recalibrate the model so that there is no difference in initial assets, then reconstruct the initial conditions and simulate the model:

# %%
# put your answer here

# %% [markdown]
# # PROBLEM : "Luck" and Saving (cont)
#
# Now we are interested in comparing the people who were "lucky" vs those who were "unlucky"
#
# The easiest way to measure this is by the cumulated level of noncapital (labor) income they have experienced over their working life.
#
# For consumer in period 41 (age 66), calculate this object, then plot it against the $\texttt{aNrm}$ ratio at age 66.

# %%
# put your answer here

# %% [markdown]
# # PROBLEM : "Luck" and Saving (cont)
#
# You can have luck in transitory income shocks or in permanent income shocks.  Their consequences are quite different.  With a permanent shock, you expect your (noncapital) income to change forever, and (according to Friedman (1957)) you should adjust your consumption nearly one-for-one.  With a transitory shock, you expect your income to return to its "permanent" level so you do not consume. So if you get a positive transitory shock, you will mostly save it.
#
# The existence of transitory shocks therefore means that people who have on average experienced positive transitory shocks over their lifetimes should have higher saving rates.  That would bias the relationship between lifetime income and the $\texttt{aNrm}$ ratio upward.
#
# To see how important this might be, redo the same exercise as before, but using the level of (noncapital) permanent income (rather than overall income including transitory and permanent) over the lifetime.  Comment on the result

# %%
# put your solution here

# %% [markdown]
# # PROBLEM : Saving Rates and Wealth Ratios
#
# The [Haig-Simons definition of "saving"](https://en.wikipedia.org/wiki/Haig%E2%80%93Simons_income) is basically the amount by which your wealth changes from one period to the next. This definition includes the consequences of any capital gains (or losses) for your wealth.  
#
# In recent work, [Faegering, Holm, Natvik, and Moll](http://www.nber.org/papers/w26588) have proposed that instead households largely ignore the consequences of capital gains and losses.  That is, their consumption is largely unchanged by asset price movements.
#
# Specifically, they define "active saving" as the difference between income and consumption _neglecting_ any contriubutions from "buy and hold" assets like houses or stocks.  The "active saving rate" is the quantity of active saving divided by the level of income. They find that the "active saving rate" is remarkably stable over the range from roughly the 20th percentile to the 95th percentile of the wealth distribution (see the figures below from their paper).
#
# The basic model considered above does not allow for capital gains or losses, so it can be used to calculate directly the saving behavior of people who do not anticipate capital gains and losses.  So, the saving rate computed by the $\texttt{savRte}$ function above should correspond to their "active saving rate."
#
# Your problem: For the entire population simulated above, calculate what the model predicts about the saving rate they measure.  You will do this by grouping the population into vigntile bins, and calculating the average active saving rate for all the households in each vigntile, and then plotting the wealth vigntiles against their saving rates.
#
# ![fbbn-sav-by-age-and-ed.jpg](https://github.com/llorracc/Figures/blob/master/fbbn-sav-by-age-and-ed.jpg?raw=true)
#

# %%
# put your solution here

# %% [markdown]
# # Saving Rates and Lifetime Income Growth
#
# We are interested in how income growth over the lifetime of the agent affects their saving rate and asset ratio $a=A/P$.
#

# %%
cumulative_income_first_half = np.sum(LifeCyclePop.pLvlNow_hist[0:20,:]*LifeCyclePop.TranShkNow_hist[0:20,:],0)
cumulative_income_second_half = np.sum(LifeCyclePop.pLvlNow_hist[20:40,:]*LifeCyclePop.TranShkNow_hist[20:40,:],0)
lifetime_growth = cumulative_income_second_half/cumulative_income_first_half

t=39
vigntiles = qcut(lifetime_growth,20,labels=False)
savRte = savRteFunc(LifeCyclePop, LifeCyclePop.mNrmNow_hist[t] , t)
savRtgueseByVigtile = np.zeros(20)
assetsByVigtile = np.zeros(20)
assetsNrmByVigtile = np.zeros(20)
for i in range(20):
    savRteByVigtile[i] = np.mean(savRte[vigntiles==i])
    assetsByVigtile[i] = np.mean(LifeCyclePop.aLvlNow_hist[t][vigntiles==i])
    assetsNrmByVigtile[i] = np.mean(LifeCyclePop.aNrmNow_hist[t][vigntiles==i])
plt.plot(np.array(range(20)), savRteByVigtile)
plt.title("Saving Rate at age 65, by Vigntile of Lifetime Income Growth")
plt.xlabel("Vigntile of Lifetime Income Growth")
plt.ylabel("Savings Rate")

plt.figure()
plt.plot(np.array(range(20)), assetsByVigtile)
plt.title("Assets at age 65, by Vigntile of Lifetime Income Growth")
plt.xlabel("Vigntile of Lifetime Income Growth")
plt.ylabel("Assets")

plt.figure()
plt.plot(np.array(range(20)), assetsNrmByVigtile)
plt.title("Normalized Assets at age 65, by Vigntile of Lifetime Income Growth")
plt.xlabel("Vigntile of Lifetime Income Growth")
plt.ylabel("Normalized Assets")
