from typing import List, Dict, Tuple
from collections import defaultdict
import json
import argparse

"""Script to measure consistency among MTMSN predictions."""


def read_json(input_json):
    with open(input_json, "r") as f:
        json_data = json.load(f)
    return json_data


def make_qid2f1_map(predictions_json):
    """Make a map from {query_id: F1} from MTMSN prediction json.

    The json is query_id: List[prediction_dict]; The complete list forms the predicted answer where each element is a
    span that is predicted as answer. *DROP allows multiple span answers*

    Each prediction_dict contains the key "f1" and "em" which is the score for the complete example; for multi-span
    answers, the f1 and em in each dict is the same and is computed considering the full prediction. It is copied
    to each dict individually for convenience.
    """
    qid2f1, qid2em = defaultdict(float), defaultdict(float)
    for qid, pred_list in predictions_json.items():
        f1 = pred_list[0]["f1"]
        em = pred_list[0]["em"]
        qid2f1[qid] = f1
        qid2em[qid] = em
    return qid2f1, qid2em


def original_qids(minimal_pairs_preds, full_data_preds):
    """Generate dict from original query_id to its perturbed ids for which example we perturbed.

    The perturbed query_id is original_query_id appended with _NUM where NUM=1,2,...
    """
    origqid2perturbedids = defaultdict(list)
    for qid in minimal_pairs_preds:
        orig_id = qid.split("_")[0]
        assert orig_id in full_data_preds, f"Orig id: {orig_id} does not exist in full data predictions"
        origqid2perturbedids[orig_id].append(qid)
    return origqid2perturbedids


def compute_consistency_score(origqid2perturbedids,
                              fulldata_qid2f1, fulldata_qid2em, minimal_pairs_qid2f1, minimal_pairs_qid2em,
                              f1_consistency: bool = False, f1thresh: float = None):
    """Compute consistency score for the given model predictions.

    Consistency for an example is 1.0 iff the prediction on the original question and all its perturbations is correct.

    Args:
        f1_consistency: `bool`
            If True, a prediction is judged correct if the F1 is above a predefined threshold, else EM=1.0 is used
        f1thresh: `float`
            If f1_consistency is True, then this threshold is used to judge if a prediction is correct.
    """
    if f1_consistency:
        print("--- Computing consistency based on F1 threshold of {} ---".format(f1thresh))
        assert f1thresh is not None, "F1-threshold is not provided"
    else:
        print("--- Computing consistency based on EM of 1.0 ---")

    def is_prediction_correct(f1, em):
        if f1_consistency:
            return f1 >= f1thresh
        else:
            return em == 1.0

    origqid2consistency = defaultdict(float)
    num_orig_ques = 0
    num_partially_correct = 0
    num_all_incorrect = 0
    num_all_correct = 0
    for orig_id, perturbed_ids in origqid2perturbedids.items():
        orig_correct = is_prediction_correct(fulldata_qid2f1[orig_id], fulldata_qid2em[orig_id])
        perturbations_correct = [is_prediction_correct(minimal_pairs_qid2f1[qid], minimal_pairs_qid2em[qid])
                                 for qid in perturbed_ids]
        perturbations_correct.append(orig_correct)
        consistency = float(all(perturbations_correct))
        origqid2consistency[orig_id] = consistency
        num_orig_ques += 1
        num_all_correct += float(all(perturbations_correct))
        num_all_incorrect += float(not any(perturbations_correct))
        num_partially_correct += float(any(perturbations_correct) and not all(perturbations_correct))

    avg_consistency = (num_all_correct/num_orig_ques) * 100.0
    avg_partial_correct = (num_partially_correct/num_orig_ques) * 100.0
    avg_all_incorrect= (num_all_incorrect/num_orig_ques) * 100.0
    print("Perc examples w/ all in-correct: {}".format(avg_all_incorrect))
    print("Perc examples w/ partially correct: {}".format(avg_partial_correct))
    print("Avg Consistency : {}".format(avg_consistency))

    return avg_consistency


def avg_f1(qid2f1):
    total_f1 = sum(qid2f1.values())
    avg_f1 = total_f1/len(qid2f1)
    return avg_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_data_preds", type=str, default="predictions/drop_dataset_test_preds.json")
    parser.add_argument("--minimal_pairs_preds", type=str, default="predictions/minimal_pairs_test_preds.json")
    args = parser.parse_args()

    full_data_preds = read_json(args.full_data_preds)
    minimal_pairs_preds = read_json(args.minimal_pairs_preds)

    fulldata_qid2f1, fulldata_qid2em = make_qid2f1_map(full_data_preds)
    minimal_pairs_qid2f1, minimal_pairs_qid2em = make_qid2f1_map(minimal_pairs_preds)

    origqid2perturbedids = original_qids(minimal_pairs_preds, full_data_preds)
    origques_qid2f1 = {qid: fulldata_qid2f1[qid] for qid in origqid2perturbedids}
    origques_qid2em = {qid: fulldata_qid2em[qid] for qid in origqid2perturbedids}

    print()
    print(f"Size of full data: {len(fulldata_qid2f1)}. Avg F1: {avg_f1(fulldata_qid2f1)}")
    print(f"Size of questions that were perturbed: {len(origques_qid2f1)}. Avg F1: {avg_f1(origques_qid2f1)}")
    print(f"Size of perturbed questions: {len(minimal_pairs_qid2f1)}. Avg F1: {avg_f1(minimal_pairs_qid2f1)}")
    print()

    avg_consistency_f1 = compute_consistency_score(origqid2perturbedids, fulldata_qid2f1, fulldata_qid2em,
                                                   minimal_pairs_qid2f1, minimal_pairs_qid2em, f1_consistency=True,
                                                   f1thresh=0.8)

    avg_consistency_em = compute_consistency_score(origqid2perturbedids, fulldata_qid2f1, fulldata_qid2em,
                                                minimal_pairs_qid2f1, minimal_pairs_qid2em, f1_consistency=False)