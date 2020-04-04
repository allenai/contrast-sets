import argparse
import json
from collections import defaultdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation for NLVR2 contrast set')
    parser.add_argument('--prediction-path', help='Prediction path')
    args = parser.parse_args()

    # prediction file is expected to be a text file with separate line per prediction,
    # with identifier and prediction (as 0/1) separated by a comma.
    # e.g.
    # test1-769-1-0-1,1
    # test1-769-1-0-2,0
    # test1-256-2-0-1,0
    # test1-256-2-0-2,1

    lines = list(open(args.prediction_path, 'rt'))
    predictions = [line.split(',') for line in lines]
    pred_per_identifier = {identifier: pred.strip() for identifier, pred in predictions}

    n_total = 0
    n_correct = 0
    correct_per_group = defaultdict(list)

    for line in open('contrast_set_nlvr2.jsonl', 'rt'):
        ex = json.loads(line)

        identifier = ex['identifier']
        group = ex['identifier'][:-2]
        gold_label = ex['label']
        correct = False

        n_total += 1

        if identifier in pred_per_identifier:
            pred = bool(pred_per_identifier[identifier])
            correct = pred == gold_label

            if correct:
                n_correct += 1
        else:
            # prediction not found
            pass

        correct_per_group[group].append(correct)

    acc = n_correct / n_total
    consistency = sum([all(g) for g in correct_per_group.values()]) / len(correct_per_group)
    print(f"Accuracy: {acc}")
    print(f"Consistency: {consistency}")
