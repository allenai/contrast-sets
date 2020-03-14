
def helper(arr):
    corr = 0
    for a in arr:
        if a == 1:
            corr += 1
    return 1.0*corr/len(arr)


if __name__ == "__main__":
    output_labels = {"BEFORE": 0, "AFTER": 1, "EQUAL": 2, "VAGUE": 3}
    with open('proposed_elmo_lr0.001.merged.output','r') as f:
        content = f.readlines()
        uid2result = {}
        perturb2result = {"order":[],"tense":[],"indicator":[],"other":[]}
        tmp_cnt = 0
        for line in content:
            line = line.strip()
            tmp = line.split(",")
            unit_id = tmp[0]
            gold_label = output_labels[tmp[1]]
            pred_label = int(tmp[2])

            if unit_id not in uid2result:
                uid2result[unit_id] = []
            if pred_label == gold_label:
                uid2result[unit_id].append(1)
            else:
                uid2result[unit_id].append(0)

            if "order" in line.lower() \
                    or "tense" in line.lower()\
                    or "indicator" in line.lower() or "connective" in line.lower()\
                    or "timex" in line.lower() or "word" in line.lower() or "verb" in line.lower():
                tmp_cnt += 1

            if "order" in line.lower():
                if pred_label == gold_label:
                    perturb2result["order"].append(1)
                else:
                    perturb2result["order"].append(0)
            if "tense" in line.lower():
                if pred_label == gold_label:
                    perturb2result["tense"].append(1)
                else:
                    perturb2result["tense"].append(0)
            if "indicator" in line.lower() or "connective" in line.lower():
                if pred_label == gold_label:
                    perturb2result["indicator"].append(1)
                else:
                    perturb2result["indicator"].append(0)
            if "timex" in line.lower() or "word" in line.lower() or "verb" in line.lower():
                if pred_label == gold_label:
                    perturb2result["other"].append(1)
                else:
                    perturb2result["other"].append(0)


    corr_cnt = 0
    cnt = 0
    for uid, result in uid2result.items():
        corr = True
        for r in result:
            if r == 0:
                corr = False
                break
        if corr:
            corr_cnt += 1
        cnt += 1
    print('Percentage of instance whose perturbations are all correctly predicted: %.2f%%' % (100.0*corr_cnt/cnt))

    print('Accuracy of appearance order change (#instance=%d/%d=%.2f): %.2f%%' % (len(perturb2result['order']), tmp_cnt, 1.0*len(perturb2result['order'])/tmp_cnt, 100.0*helper(perturb2result['order'])))
    print('Accuracy of tense change (#instance=%d/%d=%.2f): %.2f%%' % (len(perturb2result['tense']), tmp_cnt, 1.0*len(perturb2result['tense'])/tmp_cnt, 100.0*helper(perturb2result['tense'])))
    print('Accuracy of indicator change (#instance=%d/%d=%.2f): %.2f%%' % (len(perturb2result['indicator']), tmp_cnt, 1.0*len(perturb2result['indicator'])/tmp_cnt, 100.0 * helper(perturb2result['indicator'])))
    print('Accuracy of other changes (#instance=%d/%d=%.2f): %.2f%%' % (len(perturb2result['other']), tmp_cnt, 1.0*len(perturb2result['other'])/tmp_cnt, 100.0 * helper(perturb2result['other'])))