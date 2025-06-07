

## Usage

```bash
python run.py --mode <mode> --dataset <dataset> --classifiers <num_classifiers> \
  --num_classes <num_classes> --num_of_iterations <iterations> --real_data <real_data>
  --max_samples <max_samples> --start_index_of_iteration <start_index>
```
Argument	Description
--mode	Specifies which script to run. Must be one of: ozabag or goowe.
--dataset	The name of the dataset file (located in the data/ directory).
--classifiers	The number of classifiers to use in the ensemble.
--num_classes	The number of unique class labels in the dataset.
--real_data	Set to 1 if using a real dataset, otherwise 0 for synthetic data.
--max_samples	The maximum number of samples to process.
--num_of_iterations	The total number of iterations if the same experiment will be done multiple times.
--start_index_of_iteration	0	Useful for resuming from a specific iteration index.


Example
```bash
python run.py --mode ozabag --dataset rialto.csv --classifiers 16 \
  --num_classes 10 --num_of_iterations 1 --real_data 1
```

This command will run the OzaBag.py script using a real dataset rialto.csv with:

16 classifiers
7 classes
1 iterations
