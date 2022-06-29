import subprocess
from time import time
from os.path import dirname, abspath
from os.path import join as join_path


example_path = join_path(dirname(abspath(__file__)), "..","examples")

commands_to_call = [
    ["python", join_path(example_path, "data_augmentation.py")],
    ["python", join_path(example_path, "graph_example.py")],
    ["python", join_path(example_path, "keras_example.py"),
            "--data_path=/local/DATABASE/mnist",
            "--dev=4"],
    ["python", join_path(example_path, "torch_example.py"),
            "--dev=6",
            "--epochs=2",],
    ["python", join_path(example_path, "lenet_onnx.py"),
            "-d=4",
            "--data_path=/local/DATABASE/mnist",
            f"--onnx={join_path(example_path, 'LeNet.onnx')}"],
    ["python", join_path(example_path, "mnist_minimal.py"),
            "--data_path=/local/DATABASE/mnist"],
    ["python", join_path(example_path, "performance_analysis.py"),
            "--data_path=/nvme0/DATABASE/GTSRB",
            "--epochs=1"],
    ["python", join_path(example_path, "performance_analysis.py"),
            "--data_path=/nvme0/DATABASE/GTSRB"],
    ["python", join_path(example_path, "train_mobilenetv1.py"),
            "--data_path=/nvme0/DATABASE/ILSVRC2012",
            "--label_path=/nvme0/DATABASE/ILSVRC2012/synsets.txt",
            "--dev=3",
            "--epochs=2"],
    ["python", join_path(example_path, "transfer_learning.py"),
            "--data_path=/nvme0/DATABASE/cifar-100-binary",
            "--dev=3",
            "--epochs=2"],
]

nb_faillure = 0
fail_outputs = {}
if __name__ == '__main__':
    for command in commands_to_call:
        print(f"Running : {command}")
        start_time = time()
        res = subprocess.run(command, capture_output=True)
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
