import sys
import conllu
import json
from collections import defaultdict
import IPython as ipy

def read_data(filename):
    data = open(filename).read()
    texts = [t for t in data.split("\n\n") if t.strip() != ""]
    trees = []
    for text in texts:
        trees += conllu.parse(text)
    return trees

def count_attachments(trees):
    labels = defaultdict(int)
    head_pos = defaultdict(int)
    total = 0
    for tree in trees:
        for token in tree:
            if token['upostag'] != 'ADP':
                continue
            total += 1
            head = tree[int(token['head'])-1]
            grand_head = tree[int(head['head'])-1]
            labels[head['deprel']] += 1
            head_pos[grand_head['upostag']] += 1
    print(f"total: {total}")
    return labels, head_pos


def compare_attachments(orig, edit):
    attachments = defaultdict(int)
    alterations = {}
    total = 0
    for i,otree in enumerate(orig):
        total += 1
        print(" ".join([f['form'] for f in otree]))
        etree = edit[i]
        print(" ".join([f['form'] for f in etree]))
        alt = False
        alt_adp = False
        for j,otoken in enumerate(otree):
            etoken = etree[j]
            if etoken['form'] != otoken['form']:
                if not alt:
                    alt = True
                    print("First altered token: {}-->{}".format(otoken['form'],etoken['form']))
            if otoken['upostag'] != 'ADP':
                continue
            ohead = otree[int(otoken['head'])-1]
            ehead = etree[int(etoken['head'])-1]
            ograndhead = otree[int(ohead['head'])-1]
            egrandhead = etree[int(ehead['head'])-1]
            if ohead['form'] == ehead['form'] and ograndhead['form'] == egrandhead['form']:
                continue
            else:
                print("First altered adposition: {0} {2}-->{1} {2}".format(ograndhead['form'],egrandhead['form'],otoken['form']))
                alterations[i] = (j, ograndhead['id']-1, egrandhead['id']-1)
                attachments[(ograndhead['upostag'],egrandhead['upostag'])] += 1
                break
    print(f"Total altered adpositions: {total}")
    return alterations,attachments


def compare_all_attachments(trees):
    alterations = defaultdict(list)
    for i,tree in enumerate(trees):
        print(" ".join([f['form'] for f in tree]))
        for j,token in enumerate(tree):
            if token['upostag'] != 'ADP':
                continue
            head = tree[int(token['head'])-1]
            grandhead = tree[int(head['head'])-1]
            alterations[i].append((j, grandhead['id']-1))
    return alterations


if __name__=="__main__":
    trees_orig = read_data(sys.argv[1])
    trees_edit = read_data(sys.argv[2])
    alter_file = sys.argv[3]
    
    label_distribution_orig, pos_distribution_orig = count_attachments(trees_orig)
    print("Original Stats:")
    print(", ".join([f"{key}:{value}" for key,value in sorted(label_distribution_orig.items())]))
    print(", ".join([f"{key}:{value}" for key,value in sorted(pos_distribution_orig.items())]))

    label_distribution_edit, pos_distribution_edit = count_attachments(trees_edit)
    print("Altered Stats:")
    print(", ".join([f"{key}:{value}" for key,value in sorted(label_distribution_edit.items())]))
    print(", ".join([f"{key}:{value}" for key,value in sorted(pos_distribution_edit.items())]))

    alterations,attachments = compare_attachments(trees_orig,trees_edit)
    print(alterations)
    print("")
    print("\n".join([f"{key}:{value}" for key,value in sorted(attachments.items())]))

    with open("alter_dict.json",'w') as f:
        f.write(json.dumps(alterations))

    orig_full_gold = compare_all_attachments(trees_orig)
    with open("orig_gold_dict.json",'w') as f:
        f.write(json.dumps(orig_full_gold))
        
    edit_full_gold = compare_all_attachments(trees_edit)
    with open("edit_gold_dict.json",'w') as f:
        f.write(json.dumps(edit_full_gold))
