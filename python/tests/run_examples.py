"""
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)

    This software is governed by the CeCILL-C license under French law and
    abiding by the rules of distribution of free software.  You can  use,
    modify and/ or redistribute the software under the terms of the CeCILL-C
    license as circulated by CEA, CNRS and INRIA at the following URL
    "http://www.cecill.info".

    As a counterpart to the access to the source code and  rights to copy,
    modify and redistribute granted by the license, users are provided only
    with a limited warranty  and the software's author,  the holder of the
    economic rights,  and the successive licensors  have only  limited
    liability.

    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-C license and that you accept its terms.
"""

from subprocess import run
from time import time
from os import getenv
from os.path import dirname, abspath
from os.path import join as join_path

DATA_PATH = getenv("N2D2_DATA")
if DATA_PATH is None:
    DATA_PATH="/local/DATABASE/"

DEVICE=0
EPOCHS=1
EXAMPLE_PATH = join_path(dirname(abspath(__file__)), "..","examples")

def path_to(script_name:str)->str:
    return join_path(EXAMPLE_PATH, script_name)

# Define the python binary to use
# We use the environment variable set by git lab (see: https://docs.gitlab.com/ee/ci/variables/predefined_variables.html)
if getenv("GITLAB_CI") is not None:
    print("Script run by gitlab")
    python_path = join_path(dirname(abspath(__file__)), "..", "venv", "bin", "python3.7")
else:
    print("Script run by a user")
    python_path = "python"

commands_to_call = [
    [python_path, path_to("data_augmentation.py"), f"--data_path={DATA_PATH}/mnist"],
    [python_path, path_to("graph_example.py")],
    [python_path, path_to("keras_example.py"),
            f"--data_path={DATA_PATH}/mnist",
            f"--dev={DEVICE}"],
    [python_path, path_to("torch_example.py"),
           f"--dev={DEVICE}",
            f"--epochs={EPOCHS}",],
    [python_path, path_to("lenet_onnx.py"),
            f"--device={DEVICE}",
            f"--data_path={DATA_PATH}/mnist",
            f"--onnx={path_to('LeNet.onnx')}"],
    [python_path, path_to("mnist_minimal.py"),
            f"--data_path={DATA_PATH}/mnist"],
    [python_path, path_to("performance_analysis.py"),
            f"--data_path={DATA_PATH}/GTSRB",
            f"--epochs={EPOCHS}"],
    [python_path, path_to("train_mobilenetv1.py"),
            f"--data_path={DATA_PATH}/ILSVRC2012",
            f"--label_path={DATA_PATH}/ILSVRC2012/synsets.txt",
            f"--dev={DEVICE}",
            f"--epochs={EPOCHS}"],
    [python_path, path_to("transfer_learning.py"),
            f"--data_path={DATA_PATH}/cifar-100-binary",
            f"--dev={DEVICE}",
            f"--epochs={EPOCHS}"],
    [python_path, path_to("load_numpy_data.py"),
            f"--dev={DEVICE}",
            f"--epochs={EPOCHS}"],
]

nb_faillure = 0
fail_outputs = {}
if __name__ == '__main__':
    for command in commands_to_call:
        print(f"Running : {command}")
        start_time = time()
        res = run(command, capture_output=True)
        duration = round(time() - start_time, 2)
        if res.returncode == 0:
            print(f" -> OK in {duration}s")
        else:
            nb_faillure += 1
            print(f" -> FAIL in {duration}s")
            fail_outputs[command[1].rstrip(".py").split("/")[-1]] = res.stderr.decode('UTF-8')
            # NOTE : We could save stdout in an external file to get a log.
    if nb_faillure != 0:
        print(f" => Total number of error : {nb_faillure}")
        for script_name, error  in fail_outputs.items():
            print(f"{script_name}\n{'*'*len(script_name)}\n{error}")
        exit(-1)
    else:
        print(" => All tests passed successfully !")
        exit(0)
