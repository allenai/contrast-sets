import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()


# function to convert nltk tag to wordnet tag
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def tokenlist2PosAndLemma(toklist):
    try:
        pos = nltk.pos_tag(toklist)
    except:
        print()
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), pos)
    lemma = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemma.append(word)
        else:
            # else use the tag to lemmatize the token
            lemma.append(lemmatizer.lemmatize(word, tag).lower())

    assert len(pos)==len(lemma)==len(toklist)
    return pos, lemma


def bodygraph2xml(bodygraph):
    splitter = "///"
    sentences_all = bodygraph.split("<p>")
    sentences = []
    xml_str = ''
    # filter sentences out of context
    for s in sentences_all:
        if "<span" not in s:
            continue
        s = s.replace('</p>', '') \
            .replace("<span style='color:red;'>", '') \
            .replace("<span style='color:blue;'>", '') \
            .replace('</span>', '')
        sentences.append(s)
    assert len(sentences)<3

    if len(sentences) == 1:
        target_sentid = 0
        tokens = sentences[0].split()
        event_tok_id = []
        for i, tok in enumerate(tokens):
            if "<strong>" in tok:
                event_tok_id.append(i)
            tokens[i] = tokens[i].replace("<strong>",'').replace("</strong>",'')
            tokens[i] = tokens[i].strip()
        tokens = [x for x in tokens if x]
        assert len(event_tok_id)==2
        pos, lemma = tokenlist2PosAndLemma(tokens)
        for i, tok in enumerate(tokens):
            xml_str += tok + splitter + lemma[i] + splitter + pos[i][1] + splitter
            if i < event_tok_id[0]:
                xml_str += "B"
            elif i == event_tok_id[0]:
                xml_str += "E1"
            elif event_tok_id[0]<i<event_tok_id[1]:
                xml_str += "M"
            elif i == event_tok_id[1]:
                xml_str += "E2"
            else:
                xml_str += "A"
            xml_str += " "
    else:
        target_sentid = 1
        tokens1 = sentences[0].split()
        event1_tok_id = []
        for i, tok in enumerate(tokens1):
            if "<strong>" in tok:
                event1_tok_id.append(i)
            tokens1[i] = tokens1[i].replace("<strong>", '').replace("</strong>", '')
            tokens1[i] = tokens1[i].strip()
        tokens1 = [x for x in tokens1 if x]
        assert len(event1_tok_id) == 1
        pos1, lemma1 = tokenlist2PosAndLemma(tokens1)

        tokens2 = sentences[1].split()
        event2_tok_id = []
        for i, tok in enumerate(tokens2):
            if "<strong>" in tok:
                event2_tok_id.append(i)
            tokens2[i] = tokens2[i].replace("<strong>", '').replace("</strong>", '')
            tokens2[i] = tokens2[i].strip()
        tokens2 = [x for x in tokens2 if x]
        assert len(event2_tok_id) == 1
        pos2, lemma2 = tokenlist2PosAndLemma(tokens2)
        for i, tok in enumerate(tokens1):
            xml_str += tok + splitter + lemma1[i] + splitter + pos1[i][1] + splitter
            if i < event1_tok_id[0]:
                xml_str += "B"
            elif i == event1_tok_id[0]:
                xml_str += "E1"
            else:
                xml_str += "M"
            xml_str += " "

        for i, tok in enumerate(tokens2):
            xml_str += tok + splitter + lemma2[i] + splitter + pos2[i][1] + splitter
            if i < event2_tok_id[0]:
                xml_str += "M"
            elif i == event2_tok_id[0]:
                xml_str += "E2"
            else:
                xml_str += "A"
            xml_str += " "

    return target_sentid, xml_str



def row2xml_original(row):
    ret = '<SENTENCE UNITID="%s" DOCID="%s" SOURCE="E1" TARGET="E2" SOURCESENTID="%d" TARGETSENTID="%d"' \
          ' LABEL="%s" SENTDIFF="%d" PERTURB="">%s </SENTENCE>'
    unit_id = row['_unit_id']
    bodygraph = row['bodygraph']
    docid = row['docid']
    decision = row['decision'].strip()
    label = decision2label(decision)
    source_sentid = 0
    target_sentid, bodyxml = bodygraph2xml(bodygraph)
    sentdiff = target_sentid-source_sentid
    return ret % (unit_id, docid, source_sentid, target_sentid, label, sentdiff, bodyxml)


def row2xml_perturbed(row):
    ret = '<SENTENCE UNITID="%s" DOCID="%s" SOURCE="E1" TARGET="E2" SOURCESENTID="%d" TARGETSENTID="%d"' \
          ' LABEL="%s" SENTDIFF="%d" PERTURB="%s">%s </SENTENCE>'
    unit_id = row['_unit_id']
    bodygraph = row['modified bodygraph']
    docid = row['docid']
    decision = row['new decision'].strip()
    perturb = row['reason']
    label = decision2label(decision)
    source_sentid = 0
    target_sentid, bodyxml = bodygraph2xml(bodygraph)
    sentdiff = target_sentid-source_sentid
    return ret % (unit_id, docid, source_sentid, target_sentid, label, sentdiff, perturb, bodyxml)


def decision2label(decision):
    if decision == 'before':
        return "BEFORE"
    if decision == 'after':
        return "AFTER"
    if decision == 'simultaneous':
        return "EQUAL"
    if decision == 'vague':
        return "VAGUE"
    return "UNDEF" # shouldn't happen


if __name__ == "__main__":
    csv_fname = "Platinum_subset_minimal_pairs.csv"
    xml_fname_original = "Platinum_subset_original.xml"
    xml_fname_perturbed = "Platinum_subset_perturbed.xml"
    data = pd.read_csv(csv_fname)

    n = len(data)

    # get original annotations
    with open(xml_fname_original,'w') as f:
        f.write("<DATA>\n")
        prev_unit_id = -1
        for i, row in data.iterrows():
            if type(row['modified bodygraph']) is float:
                continue
            unit_id = row['_unit_id']
            if unit_id == prev_unit_id:
                continue
            f.write(row2xml_original(row))
            f.write("\n")
            prev_unit_id = unit_id
        f.write("</DATA>")

    # get perturbed annotations
    with open(xml_fname_perturbed,'w') as f:
        f.write("<DATA>\n")
        for i, row in data.iterrows():
            if type(row['modified bodygraph']) is float:
                continue
            f.write(row2xml_perturbed(row))
            f.write("\n")
        f.write("</DATA>")
