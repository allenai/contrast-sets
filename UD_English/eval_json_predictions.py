import sys
import json
from collections import defaultdict

import IPython as ipy


def eval_target_predictions(predict_originals_file, predict_altered_file, gold_file):
    with open(gold_file,'r') as f:
        true_attachments = json.loads(f.read())
    predictions_orig = []
    with open(predict_originals_file,'r') as f:
        for line in f:
            predictions_orig.append(json.loads(line))
    predictions_alt = []
    with open(predict_altered_file,'r') as f:
        for line in f:
            predictions_alt.append(json.loads(line))
    comparison = defaultdict(list)
    for variant in [predictions_orig, predictions_alt]:
        for i,prediction in enumerate(variant):
            gold = true_attachments[str(i)]
            target_adp = prediction['words'][gold[0]]
            pred_head_idx = prediction['predicted_heads'][gold[0]]
            pred_head = prediction['words'][pred_head_idx-1]
            pred_grandhead_idx = prediction['predicted_heads'][pred_head_idx-1]
            pred_grandhead = prediction['words'][pred_grandhead_idx-1]
            if variant == predictions_orig:
                # parser predictions are for the original version of the sentences
                gold_grandhead = prediction['words'][gold[1]]
            else:
                # parser predictions are on the altered version of the sentences 
                gold_grandhead = prediction['words'][gold[2]]
            comparison[i].append((pred_grandhead,
                                  gold_grandhead,
                                  pred_grandhead==gold_grandhead))
    return comparison


def eval_all_predictions(predict_file, gold_file):
    with open(gold_file,'r') as f:
        true_attachments = json.loads(f.read())
    predictions = []
    with open(predict_file,'r') as f:
        for line in f:
            predictions.append(json.loads(line))

    comparisons = defaultdict(list)
    for i,prediction in enumerate(predictions):
        gold_list = true_attachments[str(i)]
        for gold_annotation in gold_list:
            target_word = prediction['words'][gold_annotation[0]]
            pred_head_idx = prediction['predicted_heads'][gold_annotation[0]]
            pred_head = prediction['words'][pred_head_idx-1]
            pred_grandhead_idx = prediction['predicted_heads'][pred_head_idx-1]

            pred_grandhead = prediction['words'][pred_grandhead_idx-1]
            gold_grandhead = prediction['words'][gold_annotation[1]]
            comparisons[i].append((pred_grandhead,
                                   gold_grandhead,
                                   pred_grandhead==gold_grandhead))
    return comparisons


if __name__=="__main__":
    if len(sys.argv) > 3:
        comparison = eval_target_predictions(sys.argv[1],sys.argv[2],sys.argv[3])
        osum = sum([comparison[i][0][2] for i in comparison])
        print("Correct attachments when predicting original sentences: {}/150".format(osum))
        esum = sum([comparison[i][1][2] for i in comparison])
        print("Correct attachments when predicting edited sentences: {}/150".format(esum))
        consistency = sum([1 if (comparison[i][0][2] and comparison[i][1][2]) else 0 for i in comparison])
        print("Correct attachments when predicting both versions of a sentence: {}/150".format(consistency))

    else:
        full_results = eval_all_predictions(sys.argv[1],sys.argv[2])
        correct_count = 0
        total_count = 0
        for i in full_results:
            for j in range(len(full_results[i])):
                correct_count += full_results[i][j][2]
                total_count += 1
        print("Correct attachments for all words: {}/{} ({:.2f}%)".format(correct_count, total_count, 100*correct_count/total_count ))
        ipy.embed()
