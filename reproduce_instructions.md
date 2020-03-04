How to use reproduce script.

- Docker should be running in the background

- Go into QuARK directory

- Execute `python reproduce.py --local notebooks/NAME_OF_NOTEBOOK`, for example `python reproduce.py --local notebooks/LifeCycleModelExample-Problems-And-Solutions.ipynb`  from the command line to test the local solution inside the docker container.
This will create a new notebook `{Notebook}-reproduce.ipynb` (in this case `LifeCycleModelExample-Problems-And-Solutions-reproduce.ipynb`), open this notebook to make sure everything is working as expected. This tool will be used to test against the standard econ-ark-notebook docker image so the environment is standardised.

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

