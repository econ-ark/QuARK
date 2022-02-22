# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: ExecuteTime,-autoscroll,collapsed
#     notebook_metadata_filter: all,-widgets,-varInspector
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.9
#   latex_envs:
#     LaTeX_envs_menu_present: true
#     autoclose: false
#     autocomplete: false
#     bibliofile: biblio.bib
#     cite_by: apalike
#     current_citInitial: 1
#     eqLabelWithNumbers: true
#     eqNumInitial: 1
#     hotkeys:
#       equation: Ctrl-E
#       itemize: Ctrl-I
#     labels_anchors: false
#     latex_user_defs: false
#     report_style_numbering: false
#     user_envs_cfg: false
# ---

# %% [markdown]
# # Perfect Foresight Model Impatience Conditions

# %% code_folding=[0]
# Initial notebook set up

# %matplotlib inline
import matplotlib.pyplot as plt

# The first step is to be able to bring things in from different directories
import sys 
import os

sys.path.insert(0, os.path.abspath('../lib'))

import numpy as np
import HARK 
import time
from copy import deepcopy
mystr = lambda number : "{:.4f}".format(number)
from HARK.utilities import plot_funcs

# These last two will make our charts look nice
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Dark2')

# %% [markdown]
# After using the Jupyter notebook [Gentle-Intro-To-HARK-PerfForesightCRRA](https://github.com/econ-ark/DemARK/blob/Course-Choice/notebooks/Gentle-Intro-To-HARK-PerfForesightCRRA.ipynb) to learn the basics of HARK, answer the following questions:

# %% [markdown]
#
#
# [PerfectForesightCRRA](http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/Consumption/PerfForesightCRRA) defines several 'impatience' conditions that are useful in understanding the model.  We will use here the HARK toolkit's solution to the permanent-income-normalized version of the model, which constructs a consumption function for the ratio of consumption to permanent income.
#
# The handout claims that in order for the perfect foresight consumption model to be useful, it is necessary to impose
# the 'return impatience condition' (RIC):
#
# \begin{eqnarray}
#   \frac{(R \beta)^{1/\rho}}{R} & < & 1
# \end{eqnarray}
#
# and defines some other similar inequalities that help characterize what happens in the model (or whether it has a solution at all).
#
# This question asks you to explore numerically what happens to the consumption function as these conditions get close to failing.
#
# Specifically, given the default set of parameter values used in the notebook below, you should:
#
# 1. Plot the consumption function for a perfect foresight consumer with those defaultparameter values, along with the "sustainable" level of consumption that would preserve wealth
# 1. Calculate the numerical values of the three impatience conditions
# 0. Calculate the values of $\beta$ and $G$ such that the impatience factors on the LHS of the two equations would be exactly equal to 1
#
# Next, along with the sustainable consumption function, you should plot a sequence of consumption functions of a HARK `PerfForesightConsumerType` consumer, for a set of parameter values that go from the default value toward some interesting point:
#
# 1. For some sequence of values of $\beta$ that go from the default value to some value very close to the point where the RIC fails
#    * Actually, we do this one for you to show how to do it generically
# 0. For some sequence of values of $G$ that go from the default value to some value just below the maximum possible value of $G$.  (Why is it the maximum possible value?)
# 0. For some sequence of values of $\rho$ that go from the default value to some value that is very large
#
# and in each case you should explain, using analytical mathematical reasoning, the numerical result you get.  (You can just type your answers in the notebook).

# %% [markdown]
#
#
#
#
#

# %%
# Import the machinery for solving the perfect foresight model and the default parameters

from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType # Import the consumer type
import HARK.ConsumptionSaving.ConsumerParameters as Params # Import default parameters

# Now extract the default values of the parameters of interest

CRRA       = Params.CRRA 
Rfree      = Params.Rfree 
DiscFac    = Params.DiscFac
PermGroFac = Params.PermGroFac
rfree      = Rfree-1

# %%
# Now create a perfect foresight consumer example, 
PFagent = PerfForesightConsumerType(**Params.init_perfect_foresight)
PFagent.cycles = 0 # We need the consumer to be infinitely lived
PFagent.LivPrb = [1.0] # Suppress the possibility of dying

# Solve the agent's problem
PFagent.solve()

# %%
# Plot the consumption function 

# Remember, after invoking .solve(), the consumption function is stored as PFagent.solution[0].cFunc

# Set out some range of market resources that we want to plot consumption for

mMin = 0
mMax = 20
numPoints = 100
m_range  = np.linspace(mMin, mMax, numPoints) # This creates an array of points in the given range

# Feed our range of market resources into our consumption function in order to get consumption at each point

cHARK = PFagent.solution[0].cFunc(m_range) # Because the input m_range is an array, the output cHARK is too

# Construct the 45 degree line where value on vertical axis matches horizontal
degree45 = m_range # This will be the array of y points identical to the x points

# Find the value of consumption at the largest value of m
c_max    = PFagent.solution[0].cFunc([mMax])

# Use matplotlib package (imported in first cell) to plot the consumption function
plt.figure(figsize=(9,6)) # set the figure size
plt.ylim(0.,c_max[0]*1.1)     # set the range for the vertical axis with a 10 percent margin at top
plt.plot(m_range, cHARK, 'b', label='Consumption Function from HARK') # Plot m's on the x axis, versus our c on the y axis, and make the line blue, with a label
plt.xlabel('Market resources m') # x axis label
plt.ylabel('Consumption c')      # y axis label

# The plot is named plt and it hangs around like a variable 
# but is not displayed until you do a plt.show()

plt.plot(m_range, degree45  , 'g', label='c = m') # Add 45 degree line
plt.legend() # construct the legend

plt.show() # show the plot

# %%
dir(PFagent)

# %%
((Rfree*DiscFac)**(1/CRRA))/Rfree

# %%
hNrm = -1/((PermGroFac[0]/Rfree)-1)
hNrm

# %%
# QUESTION: Now calculate and plot the "sustainable" level of consumption that leaves wealth untouched
# and plot it against the perfect foresight solution

mMax = 200
numPoints = 100
m_range  = np.linspace(mMin, mMax, numPoints) # This creates an array of points in the given range

cSustainable = 1. + (rfree/Rfree)*(m_range-1+hNrm) # For any given level of m, the level of c that would leave wealth unchanged
# Obviously, 0 is the wrong formula here -- you should fill in the right one

plt.figure(figsize=(9,6)) # set the figure size
plt.xlabel('Market resources m') # x axis label
plt.ylabel('Consumption c') # y axis label

plt.plot(m_range, cSustainable  , 'k', label='Sustainable c') # Add sustainable c line
plt.plot(m_range, cHARK, 'b', label='c Function')
plt.legend()

plt.show() # show the plot

# %%
# Compute the values of the impatience conditions under default parameter values

Pat_df  = (Rfree*DiscFac)**(1/CRRA) # Plug in the formula for the absolute patience factor
PatR_df = Pat_df/Rfree # Plug in the formula for the return patience factor
PatG_df = Pat_df/PermGroFac[0] # Plug in the formula for the growth patience factor

DiscFac_lim  = (Rfree)**(CRRA-1)

PermGroFac_lim = PermGroFac[0]



# %%
# The code below is an example to show you how to plot a set of consumption functions
# for a sequence of values of the discount factor.  You should be
# to adapt this code to solve the rest of the sproblem posed above

howClose=0.01 # How close to come to the limit where the impatience condition fails
DiscFac_min = 0.8
DiscFac_max = DiscFac_lim-howClose # 
numPoints = 10
DiscFac_list = np.linspace(DiscFac_min, DiscFac_max, numPoints) # Create a list of beta values

plt.figure(figsize=((9,6))) # set the plot size

plt.plot(m_range, cSustainable  , 'k', label='Sustainable c') # Add sustainable c line
for i in range(len(DiscFac_list)):
    PFagent.DiscFac = DiscFac_list[i]
    PFagent.solve()
    cHARK = PFagent.solution[0].cFunc(m_range)
    plt.plot(m_range, cHARK, label='Consumption Function, $\\beta$= '+str(PFagent.DiscFac))

PFagent.DiscFac = Params.DiscFac # return discount factor to default value
PFagent.solve() # It's polite to leave the PFagent back with its default solution
plt.xlabel('Market resources m') # x axis label
plt.ylabel('Consumption c')      # y axis label
plt.legend()                     # show legend
plt.show()                       # plot chart


# %%
# Now plot the consumption functions for alternate values of G as described above
# Note the tricky fact that PermGroFac is a list of values because it could 
# be representing some arbitrary sequence of growth rates

# %%
# PROBLEM: What is the upper bound for possible values of G?  Why?

PermGro_min = PermGroFac[0]
PermGro_max = PermGro_min # Replace with correct answer
PermGroArray = np.linspace(PermGro_min, PermGro_max, numPoints, endpoint=True)
PermGroList = PermGroArray.tolist() # Make growth factors a list

# Copy and modify the code above for plotting the consumption functions (starting with plt.figure(figsize=...))

# %%
# PROBLEM: 
# Now plot the consumption functions for values of rho above the default value
