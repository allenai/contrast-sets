# Evaluating consistency on contrast-sets test data

### Resouces
1. Clone `https://github.com/nitishgupta/MTMSN` to run evaluations

2. Download the pre-trained model from `https://github.com/huminghao16/MTMSN`

### Generate predictions

To generate predictions on the DROP-test data and DROP-constrast-sets test set:

1. Use the following script -- 
`https://github.com/nitishgupta/MTMSN/blob/master/evaluate_full.sh`

2. Supply the `DEV_DATA_JSON` with either DROP-test-set or contrast-sets test data.

3. Store the prediction files (`PREDICTIONS_JSON`) in a convinient location. 
We'll call call them `drop_full_test_predictions.json` and `drop_contrast_test_predictions.json`

### Measuring consistency

Run 

```
python consistency.py \
    --full_data_preds drop_full_test_predictions.json \
    --minimal_pairs_preds drop_contrast_test_predictions.json
```

