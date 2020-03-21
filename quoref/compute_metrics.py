"""
This evaluation script modifies code for the official Quoref evaluator (``allennlp/tools/quoref_eval.py``) to deal
with evaluating on contrast sets.
"""

import json
from typing import Dict, Tuple, List, Any, Set
import argparse
from collections import defaultdict
import numpy as np
from allennlp.tools import drop_eval


CONSISTENCY_F1_THRESHOLD = 0.8

def _get_contrast_sets(perturbed_gold_annotations: Dict[str, Any]) -> List[Set[str]]:
    grouped_instance_ids = defaultdict(set)
    for article_info in perturbed_gold_annotations["data"]:
        for paragraph_info in article_info["paragraphs"]:
            for qa_pair in paragraph_info["qas"]:
                query_id = qa_pair["id"]
                original_query_id = qa_pair["original_id"]
                grouped_instance_ids[original_query_id].add(original_query_id)
                grouped_instance_ids[original_query_id].add(query_id)

    return list(grouped_instance_ids.values())


def _get_questions_and_answers_from_data(annotations: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    If the annotations file is in the same format as the original data files, this method can be used to extract a
    dict of query ids and answers.
    """
    answers_dict: Dict[str, List[str]] = {}
    questions_dict: Dict[str, str] = {}

    for article_info in annotations["data"]:
        for paragraph_info in article_info["paragraphs"]:
            for qa_pair in paragraph_info["qas"]:
                query_id = qa_pair["id"]
                candidate_answers = [answer["text"] for answer in qa_pair["answers"]]
                answers_dict[query_id] = candidate_answers
                questions_dict[query_id] = qa_pair["question"]
    return answers_dict, questions_dict


def get_instance_metrics(annotations: Dict[str, Any],
                         predicted_answers: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """
    Takes gold annotations and predicted answers and  evaluates the predictions for each question
    in the gold annotations.  Both JSON dictionaries must have query_id keys, which are used to
    match predictions to gold annotations.

    The ``predicted_answers`` JSON must be a dictionary keyed by query id, where the value is a
    list of strings (or just one string) that is the answer.
    The ``annotations`` are assumed to have either the format of the dev set in the Quoref data release, or the
    same format as the predicted answers file.
    """
    instance_metrics: Dict[str, Tuple[float, float]] = {}
    if "data" in annotations:
        # We're looking at annotations in the original data format. Let's extract the answers.
        annotated_answers, questions_dict = _get_questions_and_answers_from_data(annotations)
    else:
        questions_dict = None
        annotated_answers = annotations
    for query_id, candidate_answers in annotated_answers.items():
        max_em_score = 0.0
        max_f1_score = 0.0
        if query_id in predicted_answers:
            predicted = predicted_answers[query_id]
            gold_answer = tuple(candidate_answers)
            em_score, f1_score = drop_eval.get_metrics(predicted, gold_answer)
            if gold_answer[0].strip() != "":
                max_em_score = max(max_em_score, em_score)
                max_f1_score = max(max_f1_score, f1_score)
        else:
            print("Missing prediction for question: {}".format(query_id))
            max_em_score = 0.0
            max_f1_score = 0.0
        instance_metrics[query_id] = max_em_score, max_f1_score

    return instance_metrics, questions_dict


def evaluate_contrast_sets(original_prediction_path: str,
                           original_gold_path: str,
                           perturbed_prediction_path: str,
                           perturbed_gold_path: str,
                           verbose: bool = False) -> None:
    """
    Takes a prediction files and gold files of original and perturbed sets, evaluates the predictions in both
    files, and computes individual metrics and consistency over contrast sets. All
    files must be json formatted and must have query_id keys, which are used to match predictions to gold
    annotations. Writes metrics to standard output.
    """
    # pylint: disable=too-many-locals,too-many-statements
    original_predicted_answers = json.load(open(original_prediction_path, encoding="utf-8"))
    original_annotations = json.load(open(original_gold_path, encoding="utf-8"))
    perturbed_predicted_answers = json.load(open(perturbed_prediction_path, encoding="utf-8"))
    perturbed_annotations = json.load(open(perturbed_gold_path, encoding="utf-8"))
    original_instance_metrics, original_questions = get_instance_metrics(original_annotations,
                                                                         original_predicted_answers)
    perturbed_instance_metrics, perturbed_questions = get_instance_metrics(perturbed_annotations,
                                                                           perturbed_predicted_answers)

    original_em_scores = [x[0] for x in original_instance_metrics.values()]
    original_f1_scores = [x[1] for x in original_instance_metrics.values()]
    global_original_em = np.mean(original_em_scores)
    global_original_f1 = np.mean(original_f1_scores)
    perturbed_em_scores = [x[0] for x in perturbed_instance_metrics.values()]
    perturbed_f1_scores = [x[1] for x in perturbed_instance_metrics.values()]
    global_perturbed_em = np.mean(perturbed_em_scores)
    global_perturbed_f1 = np.mean(perturbed_f1_scores)
    global_combined_em = np.mean(original_em_scores + perturbed_em_scores)
    global_combined_f1 = np.mean(original_f1_scores + perturbed_f1_scores)
    print("\nMetrics on original dataset")
    print("Exact-match accuracy {0:.2f}".format(global_original_em * 100))
    print("F1 score {0:.2f}".format(global_original_f1 * 100))
    print("\nMetrics on perturbed dataset")
    print("Exact-match accuracy {0:.2f}".format(global_perturbed_em * 100))
    print("F1 score {0:.2f}".format(global_perturbed_f1 * 100))
    print("\nMetrics on combined dataset")
    print("Exact-match accuracy {0:.2f}".format(global_combined_em * 100))
    print("F1 score {0:.2f}".format(global_combined_f1 * 100))

    contrast_sets = _get_contrast_sets(perturbed_annotations)
    set_sizes = [len(set_) for set_ in contrast_sets]
    mean_size = np.mean(set_sizes)
    std_sizes = np.std(set_sizes)
    all_instance_metrics = {key: value for key, value in list(original_instance_metrics.items()) +
                            list(perturbed_instance_metrics.items())}
    consistency_scores = []
    if original_questions is not None and perturbed_questions is not None:
        all_questions = {key: (value, "original") for key, value in original_questions.items()}
        all_questions.update({key: (value, "perturbed") for key, value in perturbed_questions.items()})
    elif verbose:
        print("Warning: verbose flag is set, but original data does not contain questions! Ignoring the flag.")
        verbose = False
    num_changed_questions = 0
    for set_ in contrast_sets:
        consistency = 1.0 if all([all_instance_metrics[query_id][1] > CONSISTENCY_F1_THRESHOLD
                                  for query_id in set_]) else 0.0
        consistency_scores.append(consistency)
        perturbed_set_questions = []
        if original_questions is not None:
            for query_id in set_:
                question_text, question_type = all_questions[query_id]
                if question_type == 'original':
                    original_set_question = question_text
                else:
                    perturbed_set_questions.append(question_text)
        num_changed_questions += sum([text != original_set_question for text in perturbed_set_questions])
        if verbose:
            print("===================")
            for query_id in set_:
                print(f"Question: {all_questions[query_id]}")
                print(f"Metrics: {all_instance_metrics[query_id]}")
                print(f"Consistency: {consistency}")

    global_consistency = np.mean(consistency_scores)

    percent_changed_questions = num_changed_questions / len(perturbed_questions) * 100
    print("\nMetrics on contrast sets:")
    print(f"Number of contrast sets: {len(contrast_sets)}")
    print(f"Max contrast set size: {max(set_sizes)}")
    print(f"Mean set size: {mean_size} (+/- {std_sizes})")
    print(f"Number of questions changed: {num_changed_questions} ({percent_changed_questions}%)")
    print("Consistency: {0:.2f}".format(global_consistency * 100))


if __name__ == "__main__":
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description="Evaluate Quoref predictions given contrast sets")
    parser.add_argument(
        "--original_gold_path",
        type=str,
        required=True,
        default="quoref-test-v0.1.json",
        help="location of the original test set with answers",
    )
    parser.add_argument(
        "--original_prediction_path",
        type=str,
        required=True,
        help="location of the file with predictions over the original test set",
    )
    parser.add_argument(
        "--perturbed_gold_path",
        type=str,
        required=True,
        help="location of the perturbed test set with answers",
    )
    parser.add_argument(
        "--perturbed_prediction_path",
        type=str,
        required=True,
        help="location of the file with predictions over the perturbed test set",
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="will show details of instances if set",
    )
    args = parser.parse_args()
    evaluate_contrast_sets(args.original_prediction_path,
                           args.original_gold_path,
                           args.perturbed_prediction_path,
                           args.perturbed_gold_path,
                           args.verbose)
