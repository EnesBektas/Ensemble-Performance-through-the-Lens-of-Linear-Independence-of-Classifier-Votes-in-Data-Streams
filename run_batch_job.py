import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True, choices=['ozabag', 'experiment'])
parser.add_argument('--real_data', type=int, default=0)
parser.add_argument('--dataset', required=True)
parser.add_argument('--classifiers', type=int, required=True)
parser.add_argument('--num_classes', type=int, required=True)
parser.add_argument('--num_of_iterations', type=int, required=True)
parser.add_argument('--max_samples', type=int, default=1000000 )
parser.add_argument('--start_index_of_iteration', type=int, default=0)

args = parser.parse_args()

script = "OzaBag.py" if args.mode == "ozabag" else "experiment.py"

command = [
    "python", script,
    "--real_data", str(args.real_data),
    "--dataset", "data/" + (args.dataset),
    "--classifiers", str(args.classifiers),
    "--num_classes", str(args.num_classes),
    "--num_of_iterations", str(args.num_of_iterations),
    "--max_samples", str(args.max_samples),
    "--start_index_of_iteration", str(args.start_index_of_iteration)
]

# print(f"Running: {' '.join(command)}")
subprocess.run(command, check=True)
