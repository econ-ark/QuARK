How to use reproduce script.

- Docker should be running in the background

- Go into QuARK directory

- Execute `python reproduce.py --local` from the command line to test the local solution inside the docker container.
This will create a new notebook `{Notebook}-reproduce.ipynb` (in this case `	LifeCycleModelExample-Problems-And-Solutions-reproduce.ipynb`), open this notebook to make sure everything is working as expected. This tool will be used to test against the standard econ-ark-notebook docker image so the environment is standardised.

- Once you create a pull request on GitHub you can test it again using `python reproduce.py --pr X` where `X` is the pull request number. For example to test https://github.com/econ-ark/QuARK/pull/1 `python reproduce.py --pr 1`. [Make sure you are still in the QuARK directory]

