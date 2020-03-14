[MATRES](https://github.com/qiangning/MATRES) has a Platinum section which is the standard test set. This folder contains the minimal pairs we created for 239 instances in Platinum.

File description:
- Platinum_subset_minimal_pairs.csv: The raw file that hosts minimal pair annotations from two experts
- AnnotationCSV2XML.py: Use Platinum_subset_minimal_pairs.csv to generate Platinum_subset_original.xml and Platinum_subset_perturbed.xml
- Platinum_subset_original.xml and Platinum_subset_perturbed.xml: Raw input files used in CogCompTime2.0 separately
- proposed_elmo_lr0.001.original.output: The model output we got from running CogCompTime2.0 on Platinum_subset_original.xml. The three columns are id, gold label, predicted label index ({"BEFORE": 0, "AFTER": 1, "EQUAL": 2, "VAGUE": 3})
- proposed_elmo_lr0.001.perturbed.output: The model output we got from running CogCompTime2.0 on Platinum_subset_perturbed.xml
- proposed_elmo_lr0.001.merged.output: A merged version of the above two; used by consistency_analysis.py
- consistency_analysis.py: Gives the consistency metric reported in the paper

Numbers reported in the paper:
- 73.2% and 63.3%: Just look at proposed_elmo_lr0.001.original.output and proposed_elmo_lr0.001.perturbed.output and count how many times they're correct.
- 40.6%: `python consistency_analysis.py`
