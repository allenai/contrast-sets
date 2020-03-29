""" Modified from the official evaluation script for v1.0 of the ROPES dataset to add consistency metric"""
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix((remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if str(qa['id']) not in predictions:
                    message = 'Unanswered question ' + str(qa['id']) + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = exact_match / total
    f1 = f1 / total

    return {'global_em': exact_match, 'global_f1': f1}

def evaluate_contrast(original_dataset, original_predictions, contrast_dataset, contrast_predictions):
    original_f1 = contrast_f1 = original_exact_match = contrast_exact_match  = total = consistency = 0
    for original_article, contrast_article in zip(original_dataset, contrast_dataset):
        for original_paragraph, contrast_paragraph in zip(original_article['paragraphs'], contrast_article['paragraphs']):
            for original_qa, contrast_qa in zip(original_paragraph['qas'], contrast_paragraph['qas']):
                total += 1
                if str(original_qa['id']) not in original_predictions:
                    message = 'Unanswered question ' + str(qa['id']) + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                original_ground_truths = list(map(lambda x: x['text'], original_qa['answers']))
                original_prediction = original_predictions[original_qa['id']]

                contrast_ground_truths = list(map(lambda x: x['text'], contrast_qa['answers']))
                contrast_prediction = contrast_predictions[contrast_qa['id']]
                original_exact_match += metric_max_over_ground_truths(
                    exact_match_score, original_prediction, original_ground_truths)

                contrast_exact_match += metric_max_over_ground_truths(
                    exact_match_score, contrast_prediction, contrast_ground_truths)

                original_f1 += metric_max_over_ground_truths(
                    f1_score, original_prediction, original_ground_truths)

                contrast_f1 += metric_max_over_ground_truths(
                    f1_score, contrast_prediction, contrast_ground_truths)
                
                consistency +=  metric_max_over_ground_truths(
                    exact_match_score, original_prediction, original_ground_truths) and metric_max_over_ground_truths(
                    exact_match_score, contrast_prediction, contrast_ground_truths)
                
    original_exact_match = original_exact_match / total
    original_f1 = original_f1 / total
    contrast_exact_match = contrast_exact_match / total
    contrast_f1 = contrast_f1 / total
    consistency = consistency / total
    
    return {'original_exact_match': original_exact_match,
            'original_f1': original_f1,
            'contrast_exact_match': contrast_exact_match,
            'contrast_f1': contrast_f1,
            'consistency': consistency}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation for ROPES contrast set')
    parser.add_argument('--original_dataset_path', help='Dataset path')
    parser.add_argument('--original_prediction_path', help='Prediction path')
    parser.add_argument('--contrast_dataset_path', help='Dataset path')
    parser.add_argument('--contrast_prediction_path', help='Prediction path')
    parser.add_argument("--output_path", help='Metrics path')

    args = parser.parse_args()
    with open(args.original_dataset_path) as original_dataset_path:
        original_dataset_json = json.load(original_dataset_path)
        original_dataset = original_dataset_json['data']
    with open(args.original_prediction_path) as original_prediction_path:
        original_predictions = json.load(original_prediction_path)

    with open(args.contrast_dataset_path) as contrast_dataset_path:
        contrast_dataset_json = json.load(contrast_dataset_path)
        contrast_dataset = contrast_dataset_json['data']
    with open(args.contrast_prediction_path) as contrast_prediction_path:
        contrast_predictions = json.load(contrast_prediction_path)
    metrics = evaluate_contrast(original_dataset, original_predictions, contrast_dataset, contrast_predictions)
    with open(args.output_path, "w", encoding="utf8") as outpath:
        json.dump(metrics, outpath)
