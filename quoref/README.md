# Quoref minimal pairs and utility scripts

The json file contains the contrast sets built on top of the official [Quoref](https://allennlp.org/quoref) test set.

## Scripts

### Minimal command line interface for manual perturbation
This repository contains a simple script written in Python3 that takes a Quoref data file with answers in the official json format, and provides an interactive tool to enter new question-answer pairs that are expected to be perturbations of the questions in the dataset.

```python interface.py /path/to/data.json```
The output will be written in a new file in the current directory, and will contain a union of the instances in the original file, and the new QA pairs added in the current session. So you can provide as input the output from a previous session if you want to work in multiple sessions. The output file names will have timestamps appended to them.

The interface shows a random paragraph from the dataset and iterate over each QA pair that the dataset has for that paragraph. For each QA pair, you have the option of entering a new perturbed question, and enter the corresponding answer.

### Other utilities

```compute_metrics.py``` is a modification of the evaluation script to handle contrast sets.

```merge_perturbed_files.py``` merges outputs of multiple perturbation sessions.
