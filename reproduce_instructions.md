# `reproduce.py` Tests Notebooks In A Standardized Environment

The `reproduce` script allows you to test whether a notebook you have written will work on a machine that has a standard distribution of Econ-ARK and its dependencies installed.

It is particularly useful to verify that something you have written will work on someone else's computer. Particularly for a class assignment -- you don't want to turn in something and then have the recipient be unable to run it (for example, because you installed some special packages on your computer that you've forgotten about and that are not on the recipient's computer).

How to use reproduce script:

- Docker should be running in the background

- Go into QuARK directory

- Execute `python reproduce.py --local notebooks/NAME_OF_NOTEBOOK`, for example `python reproduce.py --local notebooks/LifeCycleModelExample-Problems-And-Solutions.ipynb`  from the command line to test the local solution inside the docker container.
This will create a new notebook `{Notebook}-reproduce.ipynb` (e.g., `LifeCycleModelExample-Problems-And-Solutions-reproduce.ipynb`), which should be the same as your original notebook but with the extra wiring to ensure that it is executed not on your computer but inside the docker container. This tool will be used to test against the standard econ-ark-notebook docker image so the environment is standardised.

- Once you create a pull request on GitHub you can test it again using `python reproduce.py --pr X notebooks/NAME_OF_NOTEBOOK` where `X` is the pull request number. For example to test https://github.com/econ-ark/QuARK/pull/1 and the LifeCycleModel notebook `python reproduce.py --pr 1 notebooks/LifeCycleModelExample-Problems-And-Solutions.ipynb`. [Make sure you are still in the QuARK directory]


Troubleshoot:

- If you get an error like 
```
 File "reproduce.py", line 22
    ORIGIN = f"https://github.com/econ-ark/QuARK"
                                                ^
SyntaxError: invalid syntax
```
use python3 (atleast python3.6) instead of python2. `python3 reproduce.py --pr X`

- If you are using linux and getting permission errors make sure docker is properly installed, https://docs.docker.com/install/linux/linux-postinstall/

