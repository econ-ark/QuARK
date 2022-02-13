# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: ExecuteTime,-autoscroll,collapsed
#     cell_metadata_json: true
#     formats: ipynb,py:percent
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
#     version: 3.8.8
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
# # Theoretical Foundations of Buffer Stock Saving
#
# <cite data-cite="6202365/8AH9AXN2"></cite>
#
# <p style="text-align: center;"><small><small><small>Generator: BufferStockTheory-make/notebooks_byname</small></small></small></p>
#
# [![econ-ark.org](https://img.shields.io/badge/Powered%20by-Econ--ARK-3e8acc.svg)](https://econ-ark.org/materials/bufferstocktheory)
#

# %% [markdown]
# <a id='interactive-dashboard'></a>
#
# [This notebook](https://econ-ark.org/materials/bufferstocktheory?launch) uses the [Econ-ARK/HARK](https://github.com/econ-ark/HARK) toolkit to reproduce and illustrate key results of the paper [Theoretical Foundations of Buffer Stock Saving](https://econ-ark.github.io/BufferStockTheory/).
#
# An [interactive dashboard](https://econ-ark.org/materials/bufferstocktheory?dashboard) allows you to modify parameters to see how (some of) the figures change.
#
#
# - In JupyterLab, click on the $\bullet$$\bullet$$\bullet$ patterns to expose the runnable code
# - in either a Jupyter notebook or JupyterLab:
#
#     * Click the double triangle <span class=reload>&#x23e9;</span> above to execute the code and generate the figures

# %% [markdown]
# `# Setup Python Below`

# %% {"jupyter": {"source_hidden": true}, "tags": []}
# Import required python packages
import os.path
import sys
import subprocess
import logging
import numpy as np
from copy import deepcopy
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore") # Ignore some harmless but alarming warning messages

# Make sure other required tools (like HARK) are installed
if os.path.isdir('binder'):  # Folder defining requirements exists
    # File requirements.out should be created first time notebook is run
    if not os.path.isfile('./binder/requirements.out'):
        try:
            output = subprocess.check_output(
                [sys.executable, '-m', 'pip', 'install','--user','-r','./binder/requirements.txt'],stderr=subprocess.STDOUT) 
            requirements_out = open("./binder/requirements.out","w")
            requirements_out.write(output.decode("utf8"))
        except subprocess.CalledProcessError as e:
            print(output.decode("utf8"))
            print(e.output.decode("utf8"),e.returncode)

# %% [markdown]
# `# Setup HARK Below`

# %% {"jupyter": {"source_hidden": true}, "tags": []}
from HARK import __version__ as HARKversion
from HARK.utilities import (
    plot_funcs, find_gui, make_figs, determine_platform,
    test_latex_installation, setup_latex_env_notebook)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    PerfForesightConsumerType, IndShockConsumerType, init_perfect_foresight, init_idiosyncratic_shocks)

# Code to allow a master "Generator" and derived "Generated" versions
# Generator marking  - allows "$nb-Problems-And-Solutions → $nb-Problems → $nb" 
Generator = True  # Is this notebook the master or is it generated?

# Whether to save the figures to Figures_dir
saveFigs = True

# Whether to draw the figures
drawFigs = True

pf = determine_platform() # latex checking depends on platform
try: # test whether latex is installed on command line 
    latexExists = test_latex_installation(pf)
except ImportError:  # windows and MacOS requires manual latex install
    latexExists = False

setup_latex_env_notebook(pf, latexExists)

# check if GUI is present; if not then switch drawFigs to False and force saveFigs to be True
if not find_gui():
    drawFigs, saveFigs = False, True

# Font sizes for figures
fssml, fsmid, fsbig = 18, 22, 26

def makeFig(figure_name, target_dir="../../Figures"):
    print('')
    make_figs(figure_name, saveFigs, drawFigs, target_dir)
    print('')
    
base_params = deepcopy(init_idiosyncratic_shocks)
# Uninteresting housekeeping and details
# Make global variables for the things that were lists above 
PermGroFac, PermShkStd, TranShkStd = base_params['PermGroFac'][0], base_params['PermShkStd'][0], base_params['TranShkStd'][0]

# Some technical settings that are not interesting for our purposes
base_params['LivPrb'] = [1.0]   # 100 percent chance of living to next period
base_params['BoroCnstArt'] = None    # No artificial borrowing constraint



# %% [markdown] {"jp-MarkdownHeadingCollapsed": true, "tags": []}
# ## [The Problem](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory/BufferStockTheory3.html#The-Problem)
#
# The paper [calibrates](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#Calibration) a small set of parameters:
#
# \begin{align}
#  &
# \newcommand\maththorn{\mathord{\pmb{\text{\TH}}}}
# \newcommand{\aLvl}{\mathbf{a}}
# \newcommand{\aNrm}{{a}}
# \newcommand{\BalGroRte}{\tilde}
# \newcommand{\Bal}{\check}
# \newcommand{\bLvl}{{\mathbf{b}}}
# \newcommand{\bNrm}{{b}}
# \newcommand{\cFunc}{\mathrm{c}}
# \newcommand{\cLvl}{{\mathbf{c}}}
# \newcommand{\cNrm}{{c}}
# \newcommand{\CRRA}{\rho}
# \newcommand{\DiscFac}{\beta}
# \newcommand{\dLvl}{{\mathbf{d}}}
# \newcommand{\dNrm}{{d}}
# \newcommand{\Ex}{\mathbb{E}}
# \newcommand{\hLvl}{{\mathbf{h}}}
# \newcommand{\hNrm}{{h}}
# \newcommand{\IncUnemp}{\mu}
# \newcommand{\mLvl}{{\mathbf{m}}}
# \newcommand{\mNrm}{{m}}
# \newcommand{\MPC}{\kappa}
# \newcommand{\PatFac}{\pmb{\unicode[0.55,0.05]{0x00DE}}}
# \newcommand{\PatRte}{\pmb{\unicode[0.55,0.05]{0x00FE}}}
# \newcommand{\PermGroFacAdj}{\tilde{\Phi}}
# \newcommand{\PermGroFac}{\pmb{\Phi}}
# \newcommand{\PermShkStd}{\sigma_{\PermShk}}
# \newcommand{\PermShk}{\pmb{\Psi}} % New
# \newcommand{\pLvl}{{\mathbf{p}}}
# \newcommand{\Rfree}{\mathsf{R}}
# \newcommand{\RNrm}{\mathcal{R}}
# \newcommand{\Thorn}{\pmb{\TH}}
# \newcommand{\TranShkAll}{\pmb{\xi}}
# \newcommand{\TranShkStd}{\sigma_{\TranShk}}
# \newcommand{\TranShk}{\pmb{\theta}}
# \newcommand{\Trg}{\hat}
# \newcommand{\uFunc}{\mathrm{u}}
# \newcommand{\UnempPrb}{\wp}
# \newcommand{\vLvl}{{\mathbf{v}}}
# \newcommand{\vNrm}{{v}}
# \renewcommand{\APFac}{\pmb{\unicode[0.55,0.05]{0x00DE}}}
# \end{align}
#
# | Parameter | Description | Python Variable | Value |
# |:---:      | :---:       | :---:  | :---: |
# | $\PermGroFac$ | Permanent Income Growth Factor | $\texttt{PermGroFac}$ | 1.03 |
# | $\Rfree$ | Interest Factor | $\texttt{Rfree}$ | 1.04 |
# | $\DiscFac$ | Time Preference Factor | $\texttt{DiscFac}$ | 0.96 |
# | $\CRRA$ | Coeﬃcient of Relative Risk Aversion| $\texttt{CRRA}$ | 2 |
# | $\UnempPrb$ | Probability of Unemployment | $\texttt{UnempPrb}$ | 0.005 |
# | $\TranShk^{\large u}$ | Income when Unemployed | $\texttt{IncUnemp}$ | 0. |
# | $\PermShkStd$ | Std Dev of Log Permanent Shock| $\texttt{PermShkStd}$ | 0.1 |
# | $\TranShkStd$ | Std Dev of Log Transitory Shock| $\texttt{TranShkStd}$ | 0.1 |
#
# that define the preferences and environment of microeconomic consumers as detailed below.  (For notational conventions used here and in the paper, see the [NARK](https://github.com/econ-ark/HARK/blob/BST-HARK-pre-release-v4/Documentation/NARK/NARK.pdf).)
#
# The objective of such a consumer with a horizon of $n$ periods is to maximize the value obtained from the sequence of consumption choices __**c**__ from period $t=T-n$ to a terminal period $T$:
#
# \begin{equation}
# \mathbf{v}_{t} = \sum_{i=0}^{n} \DiscFac^{n}\mathrm{u}(\mathbf{c}_{t+n})
# \end{equation}
#
# The infinite-horizon solution is defined as the limit of the first period solution $\mathrm{c}_{T-n}$ as the horizon $n$ goes to infinity.

# %% [markdown] {"tags": []}
# ### Details
# For a microeconomic consumer who begins period $t$ with __**m**__arket resources boldface $\mLvl_{t}$ (=net worth plus current income), the amount that remains after __**c**__onsumption of $\cLvl_{t}$ will be end-of-period __**a**__ssets $\aLvl_{t}$,
#
# <!-- Next period's 'Balances' $B_{t+1}$ reflect this period's $\aLvl_{t}$ augmented by return factor $R$:-->

# %% [markdown]
# \begin{eqnarray}
# \aLvl_{t}   &=&\mLvl_{t}-\cLvl_{t}. \notag
# \end{eqnarray}
#
# The consumer's __**p**__ermanent noncapital income $\pLvl$ grows by a predictable factor $\PermGroFac$ and is subject to an unpredictable multiplicative shock $\Ex_{t}[\PermShk_{t+1}]=1$,
#
# \begin{eqnarray}
# \pLvl_{t+1} & = & \pLvl_{t} \PermGroFac \PermShk_{t+1}, \notag
# \end{eqnarray}
#
# and, if the consumer is employed, actual income is permanent income multiplied by a transitory shock $\TranShk^{\large e}$.  There is also a probability $\UnempPrb$ that the consumer will be temporarily unemployed and experience income of $\TranShk^{\large u}  = 0$.  We construct $\TranShk^{\large e}$ so that its mean value is $1/(1-\UnempPrb)$ because in that case the mean level of the transitory shock (accounting for both unemployed and employed states) is exactly
#
# \begin{eqnarray}
# \Ex_{t}[\TranShk_{t+1}] & = & \TranShk^{\large{u}}  \times \UnempPrb + (1-\UnempPrb) \times \Ex_{t}[\TranShk^{\large{e}}_{t+1}] \notag
# \\ & = & 0 \times \UnempPrb + (1-\UnempPrb) \times 1/(1-\UnempPrb)  \notag
# \\ & = & 1. \notag
# \end{eqnarray}
#
#   We can combine the unemployment shock $\TranShk^{\large u}$ and the transitory shock to employment income $\TranShk^{\large e}$ into $\TranShkAll_{t+1}$, so that next period's market resources are
# \begin{eqnarray}
#     \mLvl_{t+1} &=& \aLvl_{t}\Rfree +\pLvl_{t+1}\TranShkAll_{t+1}.  \notag
# \end{eqnarray}

# %% [markdown]
# When the consumer has a CRRA utility function $u(\cLvl)=\frac{\cLvl^{1-\CRRA}}{1-\CRRA}$, the paper shows that the problem can be written in terms of ratios (nonbold font) of level (bold font) variables to permanent income, e.g. $m_{t} \equiv \mLvl_{t}/\pLvl_{t}$, and the Bellman form of [the problem reduces to](https://econ-ark.github.io/BufferStockTheory/#The-Related-Problem):
#
# \begin{eqnarray*}
# v_t(m_t) &=& \max_{c_t}~~ u(c_t) + \DiscFac~\Ex_{t} [(\PermGroFac\PermShk_{t+1})^{1-\CRRA} v_{t+1}(m_{t+1}) ] \\
# & s.t. & \\
# a_t &=& m_t - c_t \\
# m_{t+1} &=& a_t \Rfree/(\PermGroFac \PermShk_{t+1}) + \TranShkAll_{t+1} \\
# \end{eqnarray*}

# %% {"tags": []}
# Set the parameters for the baseline results in the paper
base_params['PermGroFac'] = [1.03]         # Permanent income growth factor
base_params['Rfree'] = Rfree = 1.04        # Interest factor on assets
base_params['DiscFac'] = DiscFac = 0.96    # Time Preference Factor
base_params['CRRA'] = CRRA = 2.00          # Coefficient of relative risk aversion
# Probability of unemployment (e.g. Probability of Zero Income in the paper)
base_params['UnempPrb'] = UnempPrb = 0.005
base_params['IncUnemp'] = IncUnemp = 0.0   # Induces natural borrowing constraint
base_params['PermShkStd'] = [0.1]          # Standard deviation of log permanent income shocks
base_params['TranShkStd'] = [0.1]          # Standard deviation of log transitory income shocks
# %% [markdown] {"tags": []}
# ## Convergence of the Consumption Rules
#
# Under the given parameter values, [the paper's first figure](https://econ-ark.github.io/BufferStockTheory/#Convergence-of-the-Consumption-Rules) depicts the successive consumption rules that apply in the last period of life $(c_{T}(m))$, the second-to-last period, and earlier periods $(c_{T-n})$.  The consumption function to which these converge is $c(m)$:
#
# \begin{equation}
# c(m) = \lim_{n \uparrow \infty} c_{T-n}(m) \notag
# \end{equation}

# %% [markdown]
# `# Create a buffer stock consumer instance:`

# %% {"jupyter": {"source_hidden": true}, "tags": []}
# Create a buffer stock consumer instance by invoking the IndShockConsumerType class
# with the parameter dictionary "base_params"

base_params['cycles'] = 100  # periods to solve from end
# Construct finite horizon agent with baseline parameters
baseAgent_Fin = \
    IndShockConsumerType(**base_params,
                         quietly=True)  # Don't babble during setup

baseAgent_Fin.solve(quietly=True)  # Solve the model quietly

baseAgent_Fin.unpack('cFunc')  # Retrieve consumption functions
cFunc = baseAgent_Fin.cFunc    # Shortcut


# %% [markdown]
# `# Plot the consumption rules:`

# %% {"jupyter": {"source_hidden": true}, "tags": []}
# Plot the different consumption rules for the different periods

mPlotMin = 0
mLocCLabels = 9.6  # Defines horizontal limit of figure
mPlotTop = 6.5     # Defines maximum m value where functions are plotted
mPts = 1000        # Number of points at which functions are evaluated

mBelwLabels = np.linspace(mPlotMin, mLocCLabels-0.1, mPts) # Range of m below loc of labels
m_FullRange = np.linspace(mPlotMin, mPlotTop, mPts)        # Full plot range

# c_Tm0  defines the last period consumption rule (c=m)
T = -1  # Solution in the last period
c_Tm0 = m_FullRange
# c_Tm1 defines the second-to-last period consumption rule
c_Tm1  = cFunc[T-1 ](mBelwLabels)
c_Tm5  = cFunc[T-5 ](mBelwLabels) # c_Tm5  defines the T-5  period consumption rule
c_Tm10 = cFunc[T-10](mBelwLabels) # c_Tm10 defines the T-10 period consumption rule
c_Limt = cFunc[0   ](mBelwLabels) # limiting inﬁnite-horizon consumption rule

plt.figure(figsize=(12, 9))
plt.rcParams['font.size'], plt.rcParams['font.weight'] = fsmid, 'bold'

xMin, xMax = 0, 11
yMin, yMax = 0, 7
plt.xlim(xMin, xMax)
plt.ylim(yMin, yMax)

plt.plot(mBelwLabels, c_Limt, color='black')
plt.plot(mBelwLabels, c_Tm1, color='black')
plt.plot(mBelwLabels, c_Tm5, color='black')
plt.plot(mBelwLabels, c_Tm10, color='black')
plt.plot(m_FullRange, c_Tm0, color='black')
plt.text(yMax, yMax-1    , r'$c_{T   }(m) = 45$ degree line')
plt.text(mLocCLabels, 5.3, r'$c_{T-1 }(m)$')
plt.text(mLocCLabels, 2.6, r'$c_{T-5 }(m)$')
plt.text(mLocCLabels, 2.1, r'$c_{T-10}(m)$')
plt.text(mLocCLabels, 1.7, r'$c(m)       $')
plt.arrow(6.9, 6.05, -0.6, 0, head_width=0.1, width=0.001,
          facecolor='black', length_includes_head='True')
plt.tick_params(labelbottom=False, labelleft=False, left='off',
                right='off', bottom='off', top='off')
plt.text(0, 7.05, "$c$", fontsize=fsbig)
plt.text(xMax+0.1, 0, "$m$", fontsize=fsbig)

# Save the figure
makeFig('cFuncsConverge')  # Comment out if you want to run uninterrupted

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# Use the [interactive dashboard](#interactive-dashboard) to explore the effects of changes in patience, risk aversion, or risk

# %% [markdown] {"tags": []}
# ### PROBLEM: Natural Borrowing Constraint Approaches Artificial Constraint
#
# Show numerically the result that is proven analytically in [The-Liquidity-Constrained-Solution-as-a-Limit](https://econ-ark.github.io/BufferStockTheory/#The-Liquidity-Constrained-Solution-as-a-Limit), by solving the model for successively smaller values of $\UnempPrb$.
#    * You need only to solve for the second-to-last period of life to do this
#       * `TwoPeriodModel = IndShockConsumerType(**base_params)`
#       * `TwoPeriodModel.cycles = 2   # Make this type have a two period horizon (Set T = 2)`
#
#    * You should show the consumption rules for different values of $\UnempPrb$ on the same graph
#       * To make this easier, you will want to use the plot_funcs command:
#          * `from HARK.utilities import plot_funcs_der, plot_funcs`
#
# Create a cell or cells in the notebook below this cell and put your solution there; comment on the size of $\UnempPrb$ needed to make the two models visually indistinguishable

# %% [markdown] {"tags": []}
# ## Factors and Conditions
#
# ### [The Finite Human Wealth Condition](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#Human-Wealth)
#
# Human wealth for a perfect foresight consumer is the present discounted value of future noncapital income:
#
# \begin{eqnarray}\notag
# \hLvl_{t} & = & \Ex_{t}[\pLvl_{t} + \Rfree^{-1} \pLvl_{t+1} + \Rfree^{2} \pLvl_{t+2} ... ] \\ \notag
#       & = & \pLvl_{t} \left(1 + (\PermGroFac/\Rfree) + (\PermGroFac/\Rfree)^{2} ... \right)
# \end{eqnarray}
#
# which approaches infinity as the horizon extends if $\PermGroFac/\Rfree \geq 1$.  We say that the 'Finite Human Wealth Condition' [(FHWC)](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#FHWC) holds if
# $0 \leq (\PermGroFac/\Rfree) < 1$.

# %% [markdown] {"tags": []}
# ### [Absolute Patience and the AIC](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#AIC)
#
# The paper defines the Absolute Patience Factor [(APF)](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#APFac) as being equal to the ratio $\cLvl_{t+1}/\cLvl_{t}$ for a perfect foresight consumer.  (The Old English character [Thorn](https://en.wikipedia.org/wiki/Thorn_(letter)) used for this object in the paper cannot reliably be rendered in Jupyter notebooks; it may appear as capital Phi):
#
# \begin{equation}
# \PatFac = (\Rfree \DiscFac)^{1/\CRRA}
# \end{equation}
#
# If $\APFac = 1$, a perfect foresight consumer will spend at exactly the level of $\cLvl$ that can be sustained perpetually (given their current and future resources).  If $\APFac < 1$ (the consumer is 'absolutely impatient'; or, 'the absolute impatience condition holds'), the consumer is consuming more than the sustainable amount, so consumption will fall, and if the consumer is 'absolutely patient' with $\APFac > 1$ consumption will grow over time.
#
#

# %% [markdown] {"tags": []}
# ### [Growth Patience and the GICRaw](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#GIC)
#
# For a [perfect foresight consumer](https://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/consumption/PerfForesightCRRA), whether the ratio $c$=__**c**__/__**p**__ is rising, constant, or falling depends on the relative growth rates of consumption and permanent income; that ratio is measured by the [Perfect Foresight Growth Patience Factor](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#PFGPF):
#
# \begin{eqnarray}
# \APFac_{\PermGroFac} & = & \APFac/\PermGroFac
# \end{eqnarray}
# and whether the $c$ is falling or rising over time depends on whether $\APFac_{\PermGroFac}$ is below or above 1.
#
# An analogous condition can be defined when there is uncertainty about permanent income.  Defining $\tilde{\PermGroFac} = (\Ex[\PermShk^{-1}])^{-1}\PermGroFac$, the
# ['Growth Impatience Condition'](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#GIC) determines whether, _in expectation_, the stochastic value of $c$ is rising, constant, or falling over time:
#
# \begin{eqnarray}
#   \APFac/\tilde{\PermGroFac} & < & 1.
# \end{eqnarray}
#
# ### [The Finite Value of Autarky Condition (FVAC)](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#Autarky-Value)

# %% [markdown]
# The paper [shows](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#Autarky-Value) that a consumer who planned to spend his permanent noncapital income $\{ \pLvl_{t}, \pLvl_{t+1}, ...\} $ in every period would have value defined by
#
# \begin{equation*}
# \vLvl_{t}^{\text{autarky}} = \uFunc(\pLvl_{t})\left(\frac{1}{1-\DiscFac \PermGroFac^{1-\CRRA} \Ex[\PermShk^{1-\CRRA}]}\right)
# \end{equation*}
#
# and defines the ['Finite Value of Autarky Condition'](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#Autarky-Value) as the requirement that the denominator be a positive finite number:
#
# \begin{equation*}
# \DiscFac \PermGroFac^{1-\CRRA} \Ex[\PermShk^{1-\CRRA}] < 1
# \end{equation*}

# %% [markdown]
# ### [The Weak Return Impatience Condition (WRIC)](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#WRIC)
#
# The [Return Impatience Condition](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#RIC) $\APFac/\Rfree < 1$ has long been understood to be required for the perfect foresight model to have a nondegenerate solution (a common special case is when $\CRRA=1$; in this case $\APFac = \Rfree \DiscFac$ so $\APFac<1$ reduces to the familiar condition $\DiscFac < \Rfree$).
#
# If the RIC does not hold, the consumer is so patient that the optimal consumption function approaches zero as the horizon extends indefinitely.
#
# When the probability of unemployment is $\UnempPrb$, the paper articulates an analogous (but weaker) return impatience condition:
#
# \begin{eqnarray}
#  \UnempPrb^{1/\CRRA} \APFac/\Rfree & < & 1
# \end{eqnarray}

# %% [markdown]
# # Key Results
#
# ## [Nondegenerate Solution Requires FVAC and WRIC](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#Sufficient-Conditions-For-Nondegenerate-Solution)
#
# A main result of the paper is that the conditions required for the model to have a nondegenerate limiting solution ($0 < c(m) < \infty$ for feasible $m$) are that the Finite Value of Autarky (FVAC) and Weak Return Impatience Condition (WRIC) hold.

# %% [markdown]
# ## [Natural Borrowing Constraint limits to Artificial Borrowing Constraint](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#The-Liquidity-Constrained-Solution-as-a-Limit)

# %% [markdown]
# Defining $\chi(\UnempPrb)$ as the consumption function associated with any particular probability of a zero-income shock $\UnempPrb,$ and defining $\hat{\chi}$ as the consumption function that would apply in the absence of the transitory zero-income shocks but in the presence of an 'artificial' borrowing constraint requiring $a \geq 0$ (_a la_ Deaton (1991)), the paper shows that
#
# \begin{eqnarray}
# \lim_{\UnempPrb \downarrow 0}~\chi(\UnempPrb) & = & \hat{\chi}
# \end{eqnarray}
#
# That is, as $\UnempPrb$ approaches zero the problem with uncertainty becomes identical to the problem that instead has constraints.  (See [Precautionary Saving and Liquidity Constraints](https://econ-ark.github.io/LiqConstr) for a full treatment of the relationship between precautionary saving and liquidity constraints).

# %% [markdown]
# ## [$\cFunc(m)$ can be Finite Even When Human Wealth Is Infinite](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#When-The-GICRaw-Fails)
#
# In the perfect foresight model, if $\Rfree < \PermGroFac$ the PDV of future labor income approaches infinity as the horizon extends and so the limiting consumption function is $c(m) = \infty$ for all $m$.  Many models have no well-defined limiting solution when human wealth is infinite.
#
# The presence of uncertainty changes this: Even when limiting human wealth is infinite, the limiting consumption function is finite for all values of $m$.
#
# This is because uncertainty imposes a "natural borrowing constraint" that deters the consumer from borrowing against their unbounded (but uncertain) future labor income.

# %% [markdown]
# A [table](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#Sufficient-Conditions-For-Nondegenerate-Solution) puts this result in the context of implications of other conditions and restrictions.
#
#

# %% [markdown]
# ## [Unique and Stable Values of $\mNrm$](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#Unique-Stable-Points)
#
# Assuming that the **FVAC** and **WRIC** hold so that the problem has a nondegenerate solution, under more stringent conditions its dynamics can also be shown to exhibit certain kinds of stability.  Two particularly useful kinds of stability are existence of a 'target' value of market resources $\Trg{\mNrm}$ (`mNrmFacTrg` in the toolkit) and a 'pseudo-steady-state' value $\Bal{\mNrm}$ (`mBalLvl` in the toolkit).
#
# ### [If the GIC-Nrm Holds, $\exists$ a finite 'target' $\mNrm$](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#onetarget)
#
# Section [Individual Target Wealth](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#onetarget) shows that, under parameter values for which the limiting consumption function exists, if the [GICMod](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#GICMod) holds then there will be a value $\Trg{m}$ such that:
#
# \begin{eqnarray*}
# \Ex[m_{t+1}] & > & m_{t}~\text{if $m_{t} < \Trg{m}$} \\
# \Ex[m_{t+1}] & < & m_{t}~\text{if $m_{t} > \Trg{m}$} \\
# \Ex[m_{t+1}] & = & m_{t}~\text{if $m_{t} = \Trg{m}$}
# \end{eqnarray*}
#
# [An equation](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#mTargImplicit) in the paper tells us that, for the expected normalized interest factor $\bar{\RNrm}=\mathbb{E}[\Rfree/(\PermGroFac \PermShk)]$, if $\mNrm_{t}=\Trg{m}$ then:
#
# \begin{align}
# (\Trg{\mNrm}-\cFunc(\Trg{\mNrm}))\bar{\RNrm}+1 & = \Trg{\mNrm}
# %\\ \Trg{\mNrm}(1-\bar{\RNrm}^{-1})+\bar{\RNrm}^{-1} & = \Trg{\cNrm}
# %\\ \Trg{\cNrm} & = \Trg{\mNrm} - (\Trg{\mNrm} - 1)\bar{\RNrm}^{-1}
# \end{align}
#
# which can be solved numerically for the unique $\Trg{\mNrm}$ that satisfies it.
#
# ### [If the GIC-Raw Holds, $\exists$ a balanced growth 'pseudo-steady-state' $\mNrm$](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#pseudo-steady-state)
#
# Section [Individual Balanced-Growth 'pseudo steady state'](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#pseudo-steady-state) shows that, under parameter values for which the limiting consumption function exists, if the **GIC** holds then there will be a value $\Bal{m}$ such that:
#
# \begin{eqnarray*}
# \Ex_{t}[\mLvl_{t+1}/\mLvl_{t}] & > & \PermGroFac~\text{if $m_{t} < \Bal{m}$} \\
# \Ex_{t}[\mLvl_{t+1}/\mLvl_{t}] & < & \PermGroFac~\text{if $m_{t} > \Bal{m}$} \\
# \Ex_{t}[\mLvl_{t+1}/\mLvl_{t}] & = & \PermGroFac~\text{if $m_{t} = \Bal{m}$}
# \end{eqnarray*}
#
# [An equation](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#balgrostableSolve) in the paper tells us that if $\mNrm_{t}=\Bal{m}$ then:
#
# \begin{align}
# (\Bal{\mNrm}-\cFunc(\Bal{\mNrm}))\RNrm+1 & = \Bal{\mNrm}
# \end{align}
#
# which can be solved numerically for the unique $\Bal{\mNrm}$ that satisfies it.
#
#
# ### [Example With Finite Pseudo-Steady-State But Infinite Target Wealth](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#GICModFailsButGICRawHolds)
#
# [A figure](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#GICModFailsButGICRawHolds) depicts a solution when the **FVAC** [(Finite Value of Autarky Condition)](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#FVAC) and [**WRIC**](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#FVAC) hold (so that the model has a solution), the [**GIC**](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#GICRaw) holds, so the model has a pseudo-steady-state $\Bal{\mNrm}$, but the [**GIC-Nrm**](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#GICMod) fails, so the model does not have an individual target wealth ratio $\Trg{\mNrm}$ (or, rather, the target wealth ratio is infinity, as can be seen by the fact that the level of $\cNrm$ is always below the level that would keep $\Ex_{t}[\Delta \mNrm_{t+1}] = 0$).
#
# This example was constructed by quadrupling the variance of the permanent shocks from the baseline parameterization.  The extra precautionary saving induced by increased uncertainty is what pushes the agent into the region without a target wealth ratio.

# %% [markdown]
# `# Create an example consumer instance where the GICMod fails but the GIC Holds:`

# %% {"tags": []}
# GICModFailsButGICRawHolds Example

base_params['cycles'] = 0  # revert to default of infinite horizon
GICModFailsButGICRawHolds_params = dict(base_params)

# Increase patience by increasing risk
GICModFailsButGICRawHolds_params['PermShkStd'] = [0.2]

# Create an agent with these parameters
GICModFailsButGICRawHolds = \
    IndShockConsumerType(**GICModFailsButGICRawHolds_params,
                         quietly=True,messaging_level=logging.CRITICAL  # If True, output suppressed
                         )
# %% [markdown]
# `# Solve that consumer's problem:`

# %% {"tags": []}
# Solve the model for these parameter values
GICModFailsButGICRawHolds.tolerance = 0.0001  # Declare victory at ...
# Suppress output during solution
GICModFailsButGICRawHolds.solve(quietly=False,messaging_level=logging.CRITICAL) 

# Because we are trying to solve a problem very close to the poised patience
# values, we want to do it with extra precision to be sure we've gotten the
# answer right.  We can retrieve the distance between the last two solutions:

distance_original = GICModFailsButGICRawHolds.solution[0].distance_last

# But high precision would have slowed things down if we used it from the start

# Instead, we can take the solution obtained above, and continue it but with
# parameters that will yield a more precise answer:

# Solve with larger than normal range
GICModFailsButGICRawHolds.aXtraMax = GICModFailsButGICRawHolds.aXtraMax * 10

# Solve using four times as many gridpoints
GICModFailsButGICRawHolds.aXtraCount = GICModFailsButGICRawHolds.aXtraCount * 4

GICModFailsButGICRawHolds.update_assets_grid()

# Solve to a 10 times tighter degree of error tolerance
GICModFailsButGICRawHolds.tolerance = GICModFailsButGICRawHolds.tolerance/10

# When the solver reaches its tolerance threshold, it changes the solver
# attribute stge_kind to have 'iter_status' of 'finished'
# If we want to continue the solution (having changed something, as above)
# To continue the solution from where we left off, we just change the
# 'iter_status' to 'iterator' and tell it to ".solve()" again

GICModFailsButGICRawHolds.solution[0].stge_kind['iter_status'] = 'iterator'
# continue solving

# Setting messaging_level to NOTSET prints all info including progress
GICModFailsButGICRawHolds.solve(messaging_level=logging.NOTSET, quietly=False)

# Test whether the new solution meets a tighter tolerance than before:
distance_now = GICModFailsButGICRawHolds.solution[0].distance_last
print('\ndistance_now < distance_original: ' +
      str(distance_now < distance_original))

# Again increase the range
GICModFailsButGICRawHolds.aXtraMax = GICModFailsButGICRawHolds.aXtraMax * 10

# and gridpoints
GICModFailsButGICRawHolds.aXtraCount = GICModFailsButGICRawHolds.aXtraCount * 2

# construct grid with the extra gridpoints and expanded range
GICModFailsButGICRawHolds.update_assets_grid()

# and decrease error tolerance
GICModFailsButGICRawHolds.tolerance = GICModFailsButGICRawHolds.tolerance/100

# mark as not finished but ready to continue iterating
GICModFailsButGICRawHolds.solution[0].stge_kind['iter_status'] = 'iterator'

# continue solving
GICModFailsButGICRawHolds.solve(messaging_level=logging.DEBUG, quietly=False)

# Test whether the new solution meets a tighter tolerance than before:
distance_now = GICModFailsButGICRawHolds.solution[0].distance_last
print('\ndistance_now < distance_original: ' +
      str(distance_now < distance_original))

# %% [markdown]
# `# Plot the results:`

# %% {"pycharm": {"is_executing": true}, "tags": []}
# Plot https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#GICModFailsButGICRawHolds

soln = GICModFailsButGICRawHolds.solution[0]  # Short alias for solution

# Get objects that have been Built, parameters configured, and expectations 
Bilt, Pars, E_tp1_ = soln.Bilt, soln.Pars, soln.E_Next_

# consumption function
cFunc = Bilt.cFunc

# Shortcuts to useful items
RPFac = Bilt.RPFac
G = Pars.PermGroFac

# https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#MPCminDefn
mpc_Min = 1.0-RPFac 
# https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#hNrmDefn
h_inf = (1.0/(1.0-G/Rfree))

# Perfect foresight consumption function (unused but convenient for explorations)
def cFunc_Uncnst(m): return mpc_Min * m + (h_inf - 1) * mpc_Min

# Initialize figure setup
fig, ax = plt.subplots(figsize=(12, 8))

[xMin, xMax] = [0.0, 8.0]
yMin = 0.0
yMax = E_tp1_.c_where_E_Next_m_tp1_minus_m_t_eq_0(xMax)*1.3

mPltVals = np.linspace(xMin, xMax, mPts)

if latexExists:
    c_Stable_TrgNrm_txt = "$\Ex_{t}[\Delta m_{t+1}] = 0$"
    c_Stable_BalLvl_txt = "$\Ex_{t}[{\mathbf{m}}_{t+1}/{\mathbf{m}}_{t}] = \PermGroFac$"
    c_Stable_BalLog_txt = "$\Ex_{t}[\log {\mathbf{m}}_{t+1} - \log {\mathbf{m}}_{t}] = \log \PermGroFac$"
    c_Unconstrained_txt = r'$\bar{\cFunc}(\mNrm)$'
else:
    c_Stable_TrgNrm_txt = "$\mathsf{E}_{t}[\Delta m_{t+1}] = 0$"
    c_Stable_BalLvl_txt = "$\mathsf{E}_{t}[\mathbf{m}_{t+1}/\mathbf{m}_{t}] = \Phi$"
    c_Stable_BalLog_txt = "$\mathsf{E}_{t}[\log \mathbf{m}_{t+1} - \log \mathbf{m}_{t}] = \log \Phi$"
    c_Unconstrained_txt = r'$\bar{\cFunc}(\mNrm)$'

cVals_Lmting_color = "black"
c_Stable_BalLvl_color = "black"  # or "blue"
c_Stable_BalLog_color = "blue"  # or "blue"
c_Stable_TrgNrm_color = "black"  # or "red"
c_Unconstrained_color = "black"  # or "red"

cVals_Lmting = cFunc(mPltVals)
c_Stable_TrgNrm = E_tp1_.c_where_E_Next_m_tp1_minus_m_t_eq_0(mPltVals)
c_Stable_BalLvl = E_tp1_.c_where_E_Next_PermShk_tp1_times_m_tp1_minus_m_t_eq_0(mPltVals)
c_Stable_BalLog = list(map(lambda mPltVal: soln.c_where_E_Next_mLog_tp1_minus_mLog_t_eq_0(mPltVal), mPltVals))
c_Unconstrained = list(map(lambda mPltVal: cFunc_Uncnst(mPltVal), mPltVals))

# To reduce clutter, results for PF soln and balanced-log-growth are omitted
cVals_Lmting_lbl, = ax.plot(mPltVals, cVals_Lmting, color=cVals_Lmting_color)
c_Stable_TrgNrm_lbl, = ax.plot(mPltVals, c_Stable_TrgNrm,
                            color=c_Stable_TrgNrm_color, linestyle="dashed", label=c_Stable_TrgNrm_txt)
c_Stable_BalLvl_lbl, = ax.plot(mPltVals, c_Stable_BalLvl,
                            color=c_Stable_BalLvl_color, linestyle="dotted", label=c_Stable_BalLvl_txt)
#c_Stable_BalLog_lbl, = ax.plot(mPltVals, c_Stable_BalLog,
#                            color=c_Stable_BalLog_color, linestyle="dotted", label=c_Stable_BalLog_txt)
# c_Unconstrained_lbl, = ax.plot(mPltVals, c_Unconstrained,color=c_Unconstrained_color, linestyle="dashdot", label=c_Unconstrained_txt)
ax.set_xlim(xMin, xMax)
ax.set_ylim(yMin, yMax)
ax.set_xlabel("$\mathit{m}$", fontweight='bold', fontsize=fsmid, loc="right")
ax.set_ylabel("$\mathit{c}$", fontweight='bold', fontsize=fsmid, loc="top", rotation=0)

ax.tick_params(labelbottom=False, labelleft=False, left='off',
               right='off', bottom='off', top='off')

ax.legend(handles=[c_Stable_TrgNrm_lbl, c_Stable_BalLvl_lbl])
#ax.legend(handles=[c_Stable_TrgNrm_lbl, c_Stable_BalLvl_lbl, c_Stable_BalLog_lbl])
ax.legend(prop=dict(size=fsmid))

mBalLvl = Bilt.mBalLvl
cNrmFacBalLvl = c_Stable_BalNrm = cFunc(mBalLvl)

ax.plot(mBalLvl, cNrmFacBalLvl, marker=".", markersize=15, color="black")  # Dot at Bal point
ax.text(1, 0.6, "$\mathrm{c}(m_{t})$", fontsize=fsmid)  # label cFunc

if latexExists:
    ax.text(mBalLvl+0.02, cNrmFacBalLvl-0.10, r"$\nwarrow$", fontsize=fsmid)
    ax.text(mBalLvl+0.25, cNrmFacBalLvl-0.18, r"$\hat{m}~$", fontsize=fsmid)
else:
    ax.text(mBalLvl+0.02, cNrmFacBalLvl-0.10, r"$\nwarrow$", fontsize=fsmid)
    ax.text(mBalLvl+0.25, cNrmFacBalLvl-0.18, r"$\check{m}~$", fontsize=fsmid)

makeFig('GICModFailsButGICRawHolds')
print('Finite mBalLvl but infinite mNrmFacTrgNrm')

# %% [markdown]
# In the [interactive dashboard](#interactive-dashboard), see what happens as changes in the time preference rate (or changes in risk $\PermShkStd$) change the consumer from _normalized-growth-patient_ $(\APFac > \PermGroFac)$ to _normalized-growth-impatient_ ($\APFac < \PermGroFac$)

# %% [markdown]
# As a foundation for the remaining figures, we define another instance of the class $\texttt{IndShockConsumerType}$, which has the same parameter values as the instance $\texttt{baseAgent}$ defined previously but is solved to convergence (our definition of an infinite horizon agent type) instead of only 100 periods

# %% [markdown]
# `# Construct infinite horizon solution for consumer with baseline parameters:`

# %% {"pycharm": {"is_executing": true}, "tags": []}
# Find the infinite horizon solution

base_params['aXtraCount'] = base_params['aXtraCount'] * 20
base_params['CubicBool'] = False
base_params['cycles'] = 0  # Default for infinite horizon model

baseAgent_Inf = IndShockConsumerType(
    **base_params,
    horizon='infinite',  # Infinite horizon
    quietly=True, messaging_level=logging.CRITICAL)  # construct it silently


# %% [markdown] {"tags": []}
# ### [Expected Consumption Growth, and Permanent Income Growth](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#AnalysisoftheConvergedConsumptionFunction)
#
# $\renewcommand{\PermShk}{\pmb{\Psi}}$
# The next figure, [Analysis of the Converged Consumption Function](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#cNrmTargetFig), shows expected growth factors for the levels of consumption $\cLvl$ and market resources $\mLvl$ as a function of the market resources ratio $\mNrm$ for a consumer behaving according to the converged consumption rule, along with the growth factor for $\mNrm$ itself, and the (constant) growth factors for consumption and expected permanent income, $\APFac$ and $\PermGroFac$.
#
# The growth factor for consumption can be computed without knowing the _level_ of the consumer's permanent income:
#
# \begin{eqnarray*}
# \Ex_{t}[\cLvl_{t+1}/\cLvl_{t}] & = & \Ex_{t}\left[\frac{\pLvl_{t+1}\cFunc(m_{t+1})}{\pLvl_{t}\cFunc(m_{t})}\right] \\
# % & = & \Ex_{t}\left[\frac{\PermGroFac \PermShk_{t+1} \pLvl_{t}}{\pLvl_{t}}\frac{\cFunc(m_{t+1})}{\cFunc(m_{t})}\right] \\
# & = & \left[\frac{\PermGroFac \PermShk_{t+1} \cFunc(m_{t+1})}{\cFunc(m_{t})}\right]
# \end{eqnarray*}

# %% [markdown] {"tags": []}
# and similarly the growth factor for the level of market resources is:
#
# \begin{eqnarray*}\renewcommand{\PermShk}{\pmb{\Psi}}
# \Ex_{t}[\mLvl_{t+1}/\mLvl_{t}]
# & = & \Ex_{t}\left[\frac{\PermGroFac \PermShk_{t+1} \mNrm_{t+1}} {\mNrm_{t}} \right]
# \\ & = & \Ex_{t}\left[\frac{\PermGroFac \PermShk_{t+1} (\aNrm_{t}\Rfree/(\PermGroFac \PermShk_{t+1}))+\PermGroFac \PermShk_{t+1}\TranShk_{t+1}}
# {\mNrm_{t}}\right]
# \\ & = & \Ex_{t}\left[\frac{\PermGroFac (\aNrm_{t}\RNrm+\PermShk_{t+1}\TranShk_{t+1})}
# {\mNrm_{t}}\right]
# \\ & = & \PermGroFac \left[\frac{\aNrm_{t}\RNrm+1}{\mNrm_{t}}\right]
# \end{eqnarray*}

# %% [markdown]
# For the ratio $\mNrm$ things are only slightly more complicated:
# \begin{eqnarray*}\renewcommand{\PermShk}{\pmb{\Psi}}
# \Ex_{t}[m_{t+1}]
# & = & \Ex_{t}\left[(m_{t}-c_{t})(\Rfree/(\PermShk_{t+1}\PermGroFac)) +\TranShk_{t+1}\right]\\
# & = & a_{t}\Rfree\Ex_{t}\left[(\PermShk_{t+1}\PermGroFac)^{-1}\right] +1 \\
# \Ex_{t}\left[m_{t+1}/m_{t}\right] & = & \left(\frac{a_{t}\bar{\RNrm}+1}{\mNrm_{t}}\right)
# \end{eqnarray*}

# %% [markdown] {"tags": []}
# <!-- The expectation of the growth in the log of $\mLvl$ is a downward-adjusted value of the log of the growth factor:
# \begin{eqnarray*}
# \Ex_{t}[\log(\mLvl_{t+1}/\mLvl_{t})]
# & = & \Ex_{t}\left[\log \PermGroFac \PermShk_{t+1} \mNrm_{t+1}\right] - \log \mNrm_{t}
# \\ & = & \Ex_{t}\left[\log \PermGroFac \left(\PermShk_{t+1} (\aNrm_{t}\Rfree/(\PermGroFac \PermShk_{t+1}))+\PermShk_{t+1}\TranShk_{t+1}\right)\right]-\log \mNrm_{t}
# \\ & = & \Ex_{t}\left[\log \PermGroFac (\aNrm_{t}\RNrm+\PermShk_{t+1}\TranShk_{t+1}+1-1)\right] - \log 
# {\mNrm_{t}}
# \\ & = & 
# \log \left(\PermGroFac 
# (\aNrm_{t}\RNrm+1)\Ex_{t}\left[\left(
# 1+\frac{\PermShk_{t+1}\TranShk_{t+1}-1}{(\aNrm_{t}\RNrm+1)}
# \right)
# \right]\right) - \log {\mNrm_{t}}
# \\ & = & \log \underbrace{\PermGroFac \left[\frac{\aNrm_{t}\RNrm+1}{\mNrm_{t}}\right]}_{\Ex_{t}[\mLvl_{t+1}/\mLvl_{t}]}+
# \log \Ex_{t}\left[\left(1+
# \frac{\PermShk_{t+1}\TranShk_{t+1}-1}{(\aNrm_{t}\RNrm+1)}
# \right)
# \right]
# \end{eqnarray*}
# -->

# %% [markdown]
# `# Solve problem of consumer with baseline parameters:`

# %% {"pycharm": {"is_executing": true, "name": "#%%\n"}, "tags": []}
# Solve baseline parameters agent
tweaked_params = deepcopy(base_params)
tweaked_params['DiscFac'] = 0.970  # Tweak to make figure clearer
baseAgent_Inf = IndShockConsumerType(
    **tweaked_params, quietly=True, messaging_level=logging.CRITICAL)  # construct it silently

baseAgent_Inf.solve(
    quietly=False, messaging_level=logging.INFO)  # Solve it with info

# %% [markdown] {"tags": []}
# `# Plot growth factors for various model elements at steady state:`

# %% {"pycharm": {"is_executing": true}, "tags": []}
# Plot growth rates

soln = baseAgent_Inf.solution[0]

# Built, parameters, expectations
Bilt, Pars, E_Next_ = soln.Bilt, soln.Pars, soln.E_Next_

# Retrieve parameters (makes code more readable)
Rfree, DiscFac, CRRA, PermGroFac = \
    Pars.Rfree, Pars.DiscFac, Pars.CRRA, Pars.PermGroFac

color_cons, color_mrktLev, color_mrktNrm, color_perm, color_mLogGroExp = \
    "blue", "red", "green", "black", "orange"

mPlotMin, mCalcMax, mPlotMax = 1.0, 50, 1.8

# Get steady state equilibrium and target values for m
mBalLvl, mNrmFacTrg = Bilt.mBalLvl, Bilt.mTrgNrm

pts_num = 200  # Plot this many points

m_pts = np.linspace(mPlotMin, mPlotMax, pts_num)   # values of m for plot
c_pts = soln.cFunc(m_pts)                   # values of c for plot
a_pts = m_pts - c_pts                       # values of a

# Get ingredients for calculating growth factors, then calculate them
Ex_cLvl_tp1_Over_pLvl_t = [ 
    soln.E_Next_.cLvl_tp1_Over_pLvl_t_from_a_t(a) for a in a_pts]
Ex_mLvl_tp1_Over_pLvl_t = [
    soln.E_Next_.mLvl_tp1_Over_pLvl_t_from_a_t(a) for a in a_pts]
Ex_mLog_tp1_minus_mLog_t_from_m_t = [
    soln.E_Next_.mLog_tp1_minus_mLog_t_from_m_t(m) for m in m_pts]
Ex_m_tp1_from_a_t = [
    soln.E_Next_.m_tp1_from_a_t(a) for a in a_pts]

Ex_cLvlGroFac = np.array(Ex_cLvl_tp1_Over_pLvl_t)/c_pts
Ex_mLvlGroFac = np.array(Ex_mLvl_tp1_Over_pLvl_t)/m_pts
Ex_mNrmGroFac = np.array(Ex_m_tp1_from_a_t)/m_pts
# Exponentiated growth rate; not used but available for exploration
Ex_mLogGroFac = np.exp(Ex_mLog_tp1_minus_mLog_t_from_m_t) 

# Absolute Patience Factor = lower bound of consumption growth factor
# https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#APF
APFac = Bilt.APFac

# Init figure and axes
fig, ax = plt.subplots(figsize=(12, 8))
plt.rcParams['font.size'], plt.rcParams['font.weight'] = fsmid, 'bold'

# Plot the Absolute Patience Factor line
ax.plot([0, mPlotMax], [APFac, APFac], color=color_cons)

# Plot the Permanent Income Growth Factor line
ax.plot([0, mPlotMax], [PermGroFac, PermGroFac]    , color=color_perm)

# Plot the expected consumption growth factor
ax.plot(m_pts, Ex_cLvlGroFac        , color=color_cons)

# Plot expected growth for the level of market resources
ax.plot(m_pts, Ex_mLvlGroFac        , color=color_mrktLev)

# Plot expected growth for the market resources ratio
ax.plot(m_pts, Ex_mNrmGroFac        , color=color_mrktNrm)

# To reduce clutter, the exponentiated log growth is left out
#ax.plot(m_pts, Ex_mLogGroExp        , color=color_mLogGroExp)

# Axes limits
GroFacMin, GroFacMax, xMin = 0.976, 1.06, 1.1

ax.set_xlim(xMin, mPlotMax * 1.1)
ax.set_ylim(GroFacMin, GroFacMax)

Thorn = u"\u00DE"

# If latex installed on system, plotting can look better
if latexExists:
    mNrmFacTrg_lbl = r'$1.00 = \Ex_{t}[\mNrm_{t+1}/\mNrm_{t}]:~ \Trg{m} \rightarrow~~$'
    PermGro_lbl = r"$\PermGroFac$"
    cLvlGroFac_lbl = r"$\Ex_{t}[\cLvl_{t+1}/\cLvl_{t}]$"
    mNrmGroFac_lbl = r"$\Ex_{t}[\mNrm_{t+1}/\mNrm_{t}] ^{\nearrow}$"
    mLvlGroFac_lbl = r"$\Ex_{t}[\mLvl_{t+1}/\mLvl_{t}]$"
    mBalLvl_lbl = r"$\check{\mNrm}_{\searrow}~$"    
    cLvlAPFac_lbl = r'$\pmb{\text{\TH}} = (\Rfree\DiscFac)^{1/\CRRA}$'
else:
    mNrmFacTrg_lbl = r'$\mathsf{E}_{t}[m_{t+1}/m_{t}]:~ \hat{m} \rightarrow~~$'
    PermGro_lbl = r"$\Phi$"
    cLvlGroFac_lbl = r"$\mathsf{E}_{t}[\mathbf{c}_{t+1}/\mathbf{c}_{t}]$"
    mNrmGroFac_lbl = r"$\mathsf{E}_{t}[m_{t+1}/m_{t}]^{\nearrow}$"
    mLvlGroFac_lbl = r"$\mathsf{E}_{t}[\mathbf{m}_{t+1}/\mathbf{m}_{t}]$"
    mBalLvl_lbl = r"$\check{m}_{\searrow}$"    
    cLvlAPFac_lbl = Thorn + r'$= (\mathsf{R}\beta)^{1/\rho}$'


if mNrmFacTrg:  # Do not try to plot it if it does not exist!
    ax.text(mNrmFacTrg-0.01, 1.0-0.001, 
            mNrmFacTrg_lbl, ha='right')

ax.plot(mBalLvl, G  , marker=".", markersize=12, color="black")  # Dot at mBalLvl 
ax.plot(mNrmFacTrg, 1.0, marker=".", markersize=12, color="black")  # Dot at mNrmFacTrg 

mLvlGroFac_lbl_xVal = mPlotMax
mLvlGroFac_lbl_yVal = soln.E_Next_.mLvl_tp1_Over_mLvl_t(mLvlGroFac_lbl_xVal)

mNrmGroFac_lbl_xVal = 0.92*mNrmFacTrg
mNrmGroFac_lbl_yVal = soln.E_Next_.m_tp1_Over_m_t(mNrmGroFac_lbl_xVal)

ax.text(mPlotMax+0.01, G-0.001,PermGro_lbl)
ax.text(mPlotMax+0.01, Ex_cLvlGroFac[-1]  ,cLvlGroFac_lbl)
ax.text(mPlotMax+0.01, APFac-0.001       ,cLvlAPFac_lbl)
ax.text(mBalLvl-0.06, G+0.001,mBalLvl_lbl              ,va='bottom',ha='left')
ax.text(mNrmGroFac_lbl_xVal-0.01, mNrmGroFac_lbl_yVal-0.003,mNrmGroFac_lbl,va='bottom',ha='right')
ax.text(mLvlGroFac_lbl_xVal+0.01, mLvlGroFac_lbl_yVal+0.001,mLvlGroFac_lbl,va='top')

# Ticks
ax.tick_params(labelbottom=True, labelleft=True, left='off', right='on', bottom='on', top='off')
plt.setp(ax.get_yticklabels(), fontsize=fssml)

# Label the mNrmFacTrg with vertical lines
plt.axvline(x=mNrmFacTrg,label='Individual Target', linestyle='dotted')
plt.legend()
ax.set_ylabel('Growth Factors')
makeFig('cNrmTargetFig')

# %% [markdown] {"tags": []}
# ### [Consumption Function Bounds](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#AnalysisOfTheConvergedConsumptionFunction)
# [The next figure](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#cFuncBounds)
# illustrates theoretical bounds for the consumption function.
#
# We define two useful variables: the lower bound of $\tilde{\MPC}$ (marginal propensity to consume) and the limit of $h$ (Human wealth), along with some functions such as the limiting perfect foresight consumption function $\bar{c}(m)$, the upper bound function $\bar{\bar c}(m)$, and the lower bound function $\tilde{c}$(m).

# %% [markdown] {"tags": []}
# `# Define bounds for figure:`

# %% {"pycharm": {"is_executing": true}, "tags": []}
# Define mpc_Min, h_inf and PF consumption function, upper and lower bound of c function

# construct and solve it silently
baseAgent_Inf = IndShockConsumerType(**base_params, quietly=True, messaging_level=logging.CRITICAL)
baseAgent_Inf.solve(quietly=True, messaging_level=logging.CRITICAL)  # Solve it with info
soln = baseAgent_Inf.solution[0]

UnempPrb = Pars.IncShkDstn.parameters['UnempPrb']

# Return Patience Factor
RPFacRaw = ((Rfree * DiscFac)**(1.0/CRRA)/Rfree)
RPFac = baseAgent_Inf.solution[0].Bilt.RPFac

# https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#MPCminDefn
mpc_Min = 1.0-RPFac 
# https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#MPCmaxDefn
mpc_Max = 1.0 - (UnempPrb**(1/CRRA)) * RPFac
# https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#hNrmDefn
h_inf = (1.0/(1.0-PermGroFac/Rfree))

def cFunc_Uncnst(m): return mpc_Min * m + (h_inf - 1) * mpc_Min
def cFunc_TopBnd(m): return mpc_Max * m
def cFunc_BotBnd(m): return mpc_Min * m


# %% [markdown]
# `# Plot figure showing bounds`

# %% {"pycharm": {"is_executing": true}, "tags": []}
# Plot the consumption function and its bounds
cMaxLabel = r'$\overline{c}(m)= (m-1+h)\tilde{\kappa}$'
cMinLabel = r'Lower Bound: $\tilde{c}(m)= (1-\pmb{\text{\TH}}_{\mathsf{R}})\tilde{\kappa}m$'
if not latexExists:
    cMaxLabel = r'$\bar{c}(m) = (m-1+h)\kappa$'  # Use unicode kludge
    cMinLabel = r'Lower Bound: c̲$(m)= (1-$'+Thorn+r'$_{\mathsf{R}})m = \kappa m$'
#    cMinLabel = r'Lower Bound: c̲$(m)= (1-$'+   ''+r'$_{\mathsf{R}})m = \kappa m$'
    cLvlAPFac_lbl = Thorn + r'$= (\mathsf{R}\beta)^{1/\rho}$'
#    cLvlAPFac_lbl = ' ' + r'$= (\mathsf{R}\beta)^{1/\rho}$'

    
mPlotMin = 0.0
mPlotMax = 25
# mKnk is point where the two upper bounds meet
mKnk = ((h_inf-1) * mpc_Min)/((1 - UnempPrb**(1.0/CRRA)*(Rfree*DiscFac)**(1.0/CRRA)/Rfree)-mpc_Min)
mBelwKnkPts = 300
mAbveKnkPts = 700
mBelwKnk = np.linspace(mPlotMin, mKnk, mBelwKnkPts)
mAbveKnk = np.linspace(mKnk, mPlotMax, mAbveKnkPts)
mFullPts = np.linspace(mPlotMin, mPlotMax, mBelwKnkPts+mAbveKnkPts)

plt.figure(figsize=(12, 8))
cTopMult = 1.12
plt.plot(mFullPts, soln.cFunc(mFullPts), color="black")
plt.plot(mBelwKnk, cFunc_Uncnst(mBelwKnk), color="black", linestyle="--")
plt.plot(mAbveKnk, cFunc_Uncnst(mAbveKnk), color="black", linewidth=2.5)
plt.plot(mBelwKnk, cFunc_TopBnd(mBelwKnk), color="black", linewidth=2.5)
plt.plot(mAbveKnk, cFunc_TopBnd(mAbveKnk), color="black", linestyle="--")
plt.plot(mBelwKnk, cFunc_BotBnd(mBelwKnk), color="black", linewidth=2.5)
plt.plot(mAbveKnk, cFunc_BotBnd(mAbveKnk), color="black", linewidth=2.5)
plt.tick_params(labelbottom=False, labelleft=False, left='off',
                right='off', bottom='off', top='off')
plt.xlim(mPlotMin, mPlotMax)
plt.ylim(mPlotMin, cTopMult*cFunc_Uncnst(mPlotMax))
plt.text(mPlotMin, cTopMult*cFunc_Uncnst(mPlotMax)+0.05, "$c$", fontsize=22)
plt.text(mPlotMax+0.1, mPlotMin, "$m$", fontsize=22)
plt.text(2.5, 1, r'$c(m)$', fontsize=22, fontweight='bold')
upper_upper_bound_m = 4.6

if latexExists:
    upper_upper_bound_m_lbl = r'$\leftarrow \overline{\overline{c}}(m)= \overline{\MPC}m = (1-\UnempPrb^{1/\CRRA}\pmb{\text{\TH}}_{\mathsf{R}})m$'
else:
    upper_upper_bound_m_lbl = r'$\overline{\overline{c}}(m)= \overline{\kappa}m = (1-\wp^{1/\rho}$'+Thorn+r'$_{\mathsf{R}})m$'
#    upper_upper_bound_m_lbl = r'$\overline{\overline{c}}(m)= \overline{\kappa}m = (1-\wp^{1/\rho}$'+   ''+r'$_{\mathsf{R}})m$'
    
plt.text(upper_upper_bound_m+0.6, cFunc_TopBnd(upper_upper_bound_m+0.5), 
             upper_upper_bound_m_lbl,
             fontsize=22, fontweight='bold')

upper_bound_m = 12
upper_bound_m_lbl=r'Upper Bound $ = $ Min $[\overline{\overline{c}}(m),\overline{c}(m)]$'
plt.text(upper_bound_m, cFunc_Uncnst(upper_bound_m)-0.3, upper_bound_m_lbl, fontsize=22, fontweight='bold')
plt.text(8, 0.9, cMinLabel, fontsize=22, fontweight='bold')
lower_unc_bound_m = 1.7
lower_unc_bound_c = cFunc_Uncnst(lower_unc_bound_m)-0.2
plt.text(lower_unc_bound_m, lower_unc_bound_c-0.2, cMaxLabel, fontsize=22, fontweight='bold')
plt.arrow(2.45, 1.05, -0.5, 0.02, head_width=0.05, width=0.001,
          facecolor='black', length_includes_head='True')
plt.arrow(lower_unc_bound_m, lower_unc_bound_c, -0.5, 0.1, head_width=0.05, width=0.001,
          facecolor='black', length_includes_head='True')
plt.arrow(upper_bound_m, cFunc_Uncnst(upper_bound_m)-0.2, -0.8, 0.05, head_width=0.1, width=0.015,
          facecolor='black', length_includes_head='True')
unconst_m = 4.5
plt.arrow(14, 0.70, 0.5, -0.1, head_width=0.05, width=0.001,
          facecolor='black', length_includes_head='True')

makeFig('cFuncBounds')


# %% [markdown]
# ### [Upper and Lower Limits of the Marginal Propensity to Consume](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#MPCLimits)
#
# The paper shows that as $m_{t}~\uparrow~\infty$ the consumption function in the presence of risk gets arbitrarily close to the perfect foresight consumption function.  Defining $\tilde{κ}$
# as the perfect foresight model's MPC, this implies that $\lim_{m_{t}~\uparrow~\infty} c^{\prime}(m) = \tilde{\kappa}$.
#
# The paper also derives an analytical limit $\bar{\MPC}$ for the MPC as $m$ approaches 0., its bounding value.  Strict concavity of the consumption function implies that the consumption function will be everywhere below a function $\bar{\MPC}m$, and strictly declining everywhere.  The last figure plots the MPC between these two limits.

# %% [markdown]
# `# Make and plot figure showing the upper and lower limits of the MPC:`

# %% {"pycharm": {"is_executing": true}, "tags": []}
# The last figure shows the upper and lower limits of the MPC

mPlotMin = 0
mPlotMax = 8

plt.figure(figsize=(12, 8))
# Set the plot range of m
m = np.linspace(0.001, mPlotMax, mPts)

# Use the HARK method derivative to get the derivative of cFunc, and which constitutes the MPC
MPC = soln.cFunc.derivative(m)

kappaDef = r'$\tilde{\kappa}\equiv(1-\pmb{\text{\TH}}_{\mathsf{R}})$'
if not latexExists:
    kappaDef = r'κ̲$\equiv(1-$'+Thorn+'$_{\mathsf{R}})$'

plt.plot(m, MPC, color='black')
plt.plot([mPlotMin, mPlotMax], [mpc_Max, mpc_Max], color='black')
plt.plot([mPlotMin, mPlotMax], [mpc_Min, mpc_Min], color='black')
plt.xlim(mPlotMin, mPlotMax)
plt.ylim(0, 1)  # MPC bounds are between 0 and 1

if latexExists:
    plt.text(1.5, 0.6, r'$\MPC(m) \equiv c^{\prime}(m)$', fontsize=26, fontweight='bold')
    plt.text(5, 0.87, r'$(1-\UnempPrb^{1/\CRRA}\pmb{\text{\TH}}_{\mathsf{R}})\equiv \overline{\MPC}$',
             fontsize=26, fontweight='bold')  # Use Thorn character
else:
    plt.text(1.5, 0.6, r'$\kappa(m) \equiv c^{\prime}(m)$', fontsize=26, fontweight='bold')
    plt.text(5, 0.87, r'$(1-\wp^{1/\rho}$'+Thorn+'$_{R})\equiv \bar{\kappa}$',
             fontsize=26, fontweight='bold')  # Use Phi instead of Thorn (alas)

plt.text(0.5, 0.07, kappaDef, fontsize=26, fontweight='bold')
plt.text(mPlotMax+0.05, mPlotMin, "$m$", fontsize=26)
plt.arrow(1.45, 0.61, -0.4, mPlotMin, head_width=0.02, width=0.001,
          facecolor='black', length_includes_head='True')
plt.arrow(2.2, 0.07, 0.2, -0.01, head_width=0.02, width=0.001,
          facecolor='black', length_includes_head='True')
plt.arrow(4.95, 0.895, -0.2, 0.03, head_width=0.02, width=0.001,
          facecolor='black', length_includes_head='True')

makeFig('MPCLimits')


# %% [markdown]
# # Summary
#
# [Two tables in the paper](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#Factors-Defined-And-Compared) summarize the various definitions, and then articulate conditions required for the problem to have a nondegenerate solution.  Among the nondegenerate cases, the most interesting result is that if the Growth Impatience Condition holds there will be a target level of wealth.

# %% [markdown] {"heading_collapsed": "true", "tags": []}
# ### Appendix: Options for Interacting With This Notebook <a id='optionsForInstalling'></a>
#
# 1. [View (static version)](https://nbviewer.org/github/econ-ark/BufferStockTheory/blob/master/BufferStockTheory.ipynb)
# 1. [Launch Online Interactive Version](https://econ-ark.org/materials/bufferstocktheory?launch)
# 1. For fast (local) execution, install [econ-ark](http://github.com/econ-ark) on your computer ([QUICK START GUIDE](https://github.com/econ-ark/HARK/blob/master/README.md)) then follow these instructions to retrieve the full contents of the `BufferStockTheory` [REMARK](https://github.com/econ-ark/BufferStockTheory):
#    1. At a command line, change the working directory to the one where you want to install
#        * On unix, if you install in the `/tmp` directory, the installation will disappear after a reboot:
#        * `cd /tmp`
#    1. `git clone https://github.com/econ-ark/BufferStockTheory`
#    1. `cd BufferStockTheory`
#    1. `jupyter lab BufferStockTheory.ipynb`

# %% [markdown] {"tags": []}
# ### Appendix: Perfect foresight agent failing both the FHWC and RIC
#
# An appendix shows the solution for the problem of a perfect foresight consumer whose parameters fail to satisfy both the FHWC and the RIC
#
# [Perfect Foresight Liquidity Constrained Solution](https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#ApndxLiqConstr)

# %% {"pycharm": {"is_executing": true}, "tags": []}
PFGICRawHoldsFHWCFailsRICFails_par = deepcopy(init_perfect_foresight)

# Replace parameters.
PFGICRawHoldsFHWCFailsRICFails_par['Rfree'] = 0.98
PFGICRawHoldsFHWCFailsRICFails_par['DiscFac'] = 1.0
PFGICRawHoldsFHWCFailsRICFails_par['PermGroFac'] = [0.99]
PFGICRawHoldsFHWCFailsRICFails_par['CRRA'] = 2
PFGICRawHoldsFHWCFailsRICFails_par['BoroCnstArt'] = 0.0
PFGICRawHoldsFHWCFailsRICFails_par['T_cycle'] = 1  # No seasonal cycles
PFGICRawHoldsFHWCFailsRICFails_par['T_retire'] = 0
PFGICRawHoldsFHWCFailsRICFails_par['cycles'] = 400  # This many periods
PFGICRawHoldsFHWCFailsRICFails_par['MaxKinks'] = 400
PFGICRawHoldsFHWCFailsRICFails_par['quiet'] = False
PFGICRawHoldsFHWCFailsRICFails_par['BoroCnstArt'] = 0.0  # Borrowing constraint
PFGICRawHoldsFHWCFailsRICFails_par['LivPrb'] = [1.0]

# Create the agent
HWRichButReturnPatientPFConstrainedAgent = \
    PerfForesightConsumerType(**PFGICRawHoldsFHWCFailsRICFails_par,
                              quietly=True
                              )
# Solve and report on conditions
this_agent = HWRichButReturnPatientPFConstrainedAgent
this_agent.solve(quietly=False, messaging_level=logging.DEBUG)

# Plot
mPlotMin, mPlotMax = 1, 9.5
plt.figure(figsize=(8, 4))
m_grid = np.linspace(mPlotMin, mPlotMax, 500)
plt.plot(m_grid-1, this_agent.solution[0].cFunc(m_grid), color="black")
plt.text(mPlotMax-1+0.05, 1, r"$b$", fontsize=26)
plt.text(mPlotMin-1, 1.017, r"$c$", fontsize=26)
plt.xlim(mPlotMin-1, mPlotMax-1)
plt.ylim(mPlotMin, 1.016)

makeFig('PFGICRawHoldsFHWCFailsRICFails')
