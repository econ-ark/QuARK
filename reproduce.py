import argparse
import subprocess


def reproduce_quark():
    # Take the file as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local", help="reproduce local jupyter notebook", action='store_true'
        )
    parser.add_argument(
        "--pr", help="add the PR number to test"
        )
    args = parser.parse_args()

    RUN_LOCAL = args.local
    PR = args.pr
    if args.pr is None and RUN_LOCAL is False:
        print("Please provide a PR number if you want to test your PR to QuARK or use the command `$ python reproduce.py --local` to test the local notebook.")
        return

    ORIGIN = f"https://github.com/econ-ark/QuARK"
    DOCKER_IMAGE = f"econark/econ-ark-notebook"

    NOTEBOOK_NAME = f"LifeCycleModelExample-Problems-And-Solutions"



    pwd = subprocess.run(["pwd"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    mount = str(pwd.stdout)[2:-3] + ":/home/jovyan/work"
    # mount the present directory and start up a container
    container_id = subprocess.run(
        ["docker", "run", "-v", mount, "-d", DOCKER_IMAGE], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    container_id = container_id.stdout.decode("utf-8")[:-1]

    if not RUN_LOCAL:
        PATH_TO_NOTEBOOK = f"/home/jovyan/QuARK/notebooks/"
        # fetch the PR
        subprocess.run(
            [
                f'docker exec -it {container_id} bash -c "git clone {ORIGIN}; cd QuARK; git fetch {ORIGIN} +refs/pull/{PR}/merge; git checkout FETCH_HEAD"'
            ],
            shell=True,
        )
    else:
        PATH_TO_NOTEBOOK = f"/home/jovyan/work/notebooks/"

    # copy the notebook file to reproduce notebook
    subprocess.run(
        [
            f"docker exec -it  {container_id} bash -c 'cp {PATH_TO_NOTEBOOK}{NOTEBOOK_NAME}.ipynb {PATH_TO_NOTEBOOK}{NOTEBOOK_NAME}-reproduce.ipynb'"
        ],
        shell=True,
    )

    # execute the reproduce notebook
    subprocess.run(
        [
            f"docker exec -it  {container_id} bash -c 'jupyter nbconvert --to notebook --inplace --execute {PATH_TO_NOTEBOOK}{NOTEBOOK_NAME}-reproduce.ipynb'"
        ],
        shell=True,
    )

    if not RUN_LOCAL:
        # copy the reproduce notebook back to local machine
        subprocess.run(
            [f"docker exec -it  {container_id} bash -c 'cp {PATH_TO_NOTEBOOK}{NOTEBOOK_NAME}-reproduce.ipynb /home/jovyan/work/notebooks/'"],
            shell=True,
        )
    else:
        # the notebook is already running locally 
        pass

    subprocess.run([f"docker stop {container_id}"], shell=True)


reproduce_quark()
