import argparse
import csv
import json
import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from perspectrum_model import PerspectrumTransformerModel


def evaluate(model_dir, data_path, result_path, cuda=False, **kwargs):

    result = _evaluate_stance(model_dir, data_path, cuda)

    for key, val in result.items():
        print("{} = {}".format(key, val))

    result_dir = os.path.dirname(result_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(result_path, 'w') as fout:
        json.dump(result, fout)
        print("Results written to {}.".format(result_path))


def _evaluate_stance(model_dir, data_path, cuda):

    model = PerspectrumTransformerModel('roberta', model_dir, cuda=cuda)
    instances = _load_instances(data_path)

    # Skip instances where perspectives are not relevant to the claim at all
    relevant_instances = [ins for ins in instances if ins["original_stance_label"] != 'unk']
    print("Evaluating on {} examples...".format(len(relevant_instances)))

    original_claim_persp_pairs = [(ins['original_claim'], ins['perspective']) for ins in relevant_instances]
    contrast_claim_persp_pairs = [(ins['contrast_claim'], ins['perspective']) for ins in relevant_instances]

    original_logits = model.predict_batch(original_claim_persp_pairs)
    contrast_logits = model.predict_batch(contrast_claim_persp_pairs)

    # We trained the model with {0, 1} labels, so we use 0.5 as logits cutoff
    original_pred = [1 if logit > 0.5 else 0 for logit in original_logits]
    contrast_pred = [1 if logit > 0.5 else 0 for logit in contrast_logits]

    original_gold = [1 if ins['original_stance_label'] == 'pos' else 0 for ins in relevant_instances]
    contrast_gold = [1 if ins['contrast_stance_label'] == 'pos' else 0 for ins in relevant_instances]

    original_acc = accuracy_score(original_gold, original_pred)
    contrast_acc = accuracy_score(contrast_gold, contrast_pred)

    original_correct = np.equal(np.array(original_gold), np.array(original_pred))
    contrast_correct = np.equal(np.array(contrast_gold), np.array(contrast_pred))

    consistency_array = np.bitwise_and(original_correct, contrast_correct).astype(int)

    consistency = np.sum(consistency_array) / len(consistency_array)
    
    return {
        'original_f1': original_acc,
        'contrast_f1': contrast_acc,
        'consistency': consistency,
        'set_size': len(relevant_instances)
    }


def _load_instances(data_path):
    print("Loading examples from {}...".format(data_path))
    instances = []
    with open(data_path) as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            instances.append(row)

    return instances


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action='store_true',
                        help="Whether to use gpu for this")
    parser.add_argument("--task", default="stance", type=str,
                        help="task to evaluate on, either \'stance\' or \'relevance\'. ")
    parser.add_argument("--model_dir", default="model/stance_roberta", type=str,
                        help="directory containing pretrained model weights.")
    parser.add_argument("--data_path", default="perspectrum_contrast_sets.csv", type=str,
                        help="path to perspectrum_minimal_pairs.csv")
    parser.add_argument("--result_path", default="log/result.json", type=str,
                        help="path to log/write results to.")

    args = parser.parse_args()

    evaluate(**vars(args))
