# ROPES minimal pairs and scripts

The contrast sets and original instances that they were perturbed from the official [ROPES](https://allennlp.org/ropes) test set are in `data`. The predictions from the RoBERTa baseline are in `predictions`.


## Scripts

```evaluate_contrast_set.py``` is a modification of the ROPES evaluation script to handle contrast sets.

Example usage:
```python evaluate_contrast_set.py --original_dataset_path data/ropes_contrast_set_original_032820.json  --original_prediction_path predictions/predictions_original_032820.json  --contrast_dataset_path data/ropes_contrast_set_032820.json  --contrast_prediction_path predictions/predictions_contrast_set_032820.json --output_path metrics.json```
