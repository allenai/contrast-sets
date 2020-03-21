import re
import json
import random
import hashlib
import argparse
import datetime
from collections import defaultdict


def get_answers(context):
    print("Enter answer spans below. You can copy text from the context and paste here.")
    print("Hit enter if you are done inputting all answer spans")
    new_answers = []
    current_span = None
    span_index = 0
    while current_span != '':
        current_span = input(f"Span {span_index + 1}: ")
        if current_span != '':
            # Important note: This will only find the first index of the answer.
            try:
                answer_start = context.index(current_span)
                new_answers.append({"text": current_span,
                                    "answer_start": answer_start})
            except ValueError:
                print("Could not find answer span in the context! Please try again.")
                continue
        span_index += 1
    return new_answers


def get_new_passage(context):
    while True:
        string_to_replace = input("Enter a unique string from the passage that you want to change: ")
        num_occurrences = len(re.findall(string_to_replace, context))
        if num_occurrences == 0:
            print("The string you entered does not occur in the passage. Please try again!")
        elif num_occurrences > 1:
            print("The string you entered is not unique. Please try again!")
        else:
            replacement = input("Enter a replacement: ")
            new_context = context.replace(string_to_replace, replacement)
            return new_context


def get_perturbed_info_for_article(datum):
    """
    Given paragraphs and question-answer pairs for an article, we can either get perturbed question-answer pairs,
    or perturbed contexts with new question-answer pairs. This method returns two lists for the two kinds of
    perturbations.
    """
    new_qas = []
    new_paragraphs = []
    end_session = False
    num_new_instances = 0
    for paragraph_index, paragraph_info in enumerate(datum["paragraphs"]):
        if end_session:
            break
        question_ids = {qa_info["id"] for qa_info in paragraph_info["qas"]}
        print("\nContext:")
        context = paragraph_info["context"]
        context_id = paragraph_info["context_id"]
        print(context)
        qa_indices = list(range(len(paragraph_info["qas"])))
        random.shuffle(qa_indices)
        for qa_index in qa_indices:
            qa_info = paragraph_info["qas"][qa_index]
            if "original_id" in qa_info:
                # This is a perturbed instance. Let's not perturb it further.
                continue
            original_id = qa_info["id"]
            question = qa_info['question']
            print(f"\nNEW QUESTION FOR THE ABOVE CONTEXT: {question}")
            print(f"Answers: {[a['text'] for a in qa_info['answers']]}")
            print("You can modify the question or the passage, skip to the next question, or exit.")
            response = input("Type a new question, hit enter to skip, type 'p' to edit passsage or 'exit' to end session: ")
            while len(response) > 0 and response.lower() != 'exit':
                if len(response) > 1:
                    perturbed_question = response.strip()
                    new_id = hashlib.sha1(f"{context} {perturbed_question}".encode()).hexdigest()
                    if new_id not in question_ids:
                        new_answers = get_answers(context)
                        if new_answers:
                            new_qa_info = {"question": perturbed_question,
                                           "id": new_id,
                                           "answers": new_answers,
                                           "original_id": original_id}
                            new_qas.append((paragraph_index, new_qa_info))
                            num_new_instances += 1
                    else:
                        print("This question exists in the dataset! Please try again.\n")
                elif response.lower() == "p":
                    perturbed_context = get_new_passage(context)
                    new_context_id = hashlib.sha1(perturbed_context.encode()).hexdigest()
                    print(f"New context: {perturbed_context}\n")
                    new_id = hashlib.sha1(f"{perturbed_context} {question}".encode()).hexdigest()
                    new_answers = get_answers(perturbed_context)
                    if new_answers:
                        new_qa_info = {"question": question,
                                       "id": new_id,
                                       "answers": new_answers,
                                       "original_id": original_id}
                        new_paragraph_info = {"context": perturbed_context,
                                              "qas": [new_qa_info],
                                              "context_id": new_context_id,
                                              "original_context_id": context_id}
                        new_paragraphs.append(new_paragraph_info)
                        num_new_instances += 1
                else:
                    print(f"Unrecognized input: {response}")
                print(f"\nSTILL MODIFYING THE SAME QUESTION: {question}")
                print(f"Answers: {[a['text'] for a in qa_info['answers']]}")
                print("You can modify the question or the passage, move on to the next question, or exit.")
                response = input("Type a new question, hit enter to move on, type 'p' to edit passsage or 'exit' to end session: ")
            if response.lower() == 'exit':
                end_session = True
                break
    return new_qas, new_paragraphs, end_session, num_new_instances


def add_perturbations(data):
    """
    Takes a dataset, queries user for perturbations, and adds them to the original dataset.
    """
    data_indices = list(range(len(data["data"])))
    random.shuffle(data_indices)
    num_new_instances = 0
    for datum_index in data_indices:
        datum = data["data"][datum_index]
        new_qa_info, new_paragraphs, end_session, num_instances = get_perturbed_info_for_article(datum)
        num_new_instances += num_instances
        for paragraph_index, qa_info in new_qa_info:
            datum["paragraphs"][paragraph_index]["qas"].append(qa_info)

        for paragraph_info in new_paragraphs:
            datum["paragraphs"].append(paragraph_info)

        if end_session:
            print(f"\nEnding session. You generated {num_new_instances} new instance(s). Thank you!")
            break


def get_perturbations(data):
    """
    Takes a dataset, queries user for perturbations, and returns a smaller dataset with just the perturbations.
    """
    perturbed_data = {"data": []}
    data_indices = list(range(len(data["data"])))
    random.shuffle(data_indices)
    num_new_instances = 0
    for datum_index in data_indices:
        datum = data["data"][datum_index]
        new_qa_info, new_paragraphs, end_session, num_instances = get_perturbed_info_for_article(datum)
        num_new_instances += num_instances
        if new_qa_info or new_paragraphs:
            paragraphs_info = defaultdict(lambda: {'qas': []})
            for paragraph_index, qa_info in new_qa_info:
                original_paragraph_info = datum['paragraphs'][paragraph_index]
                context_id = original_paragraph_info['context_id']
                paragraphs_info[context_id]['context'] = original_paragraph_info['context']
                paragraphs_info[context_id]['context_id'] = original_paragraph_info['context_id']
                paragraphs_info[context_id]['qas'].append(qa_info)
            for paragraph_info in new_paragraphs:
                paragraphs_info[paragraph_info['context_id']] = paragraph_info
            perturbed_data['data'].append({'title': datum['title'],
                                           'url': datum['url'],
                                           'paragraphs': list(paragraphs_info.values())})
        if end_session:
            print(f"\nEnding session. You generated {num_new_instances} new instance(s). Thank you!")
            break
    return perturbed_data


def main(args):
    input_filename = args.input
    data = json.load(open(input_filename))
    if args.output_perturbations_only:
        perturbed_data = get_perturbations(data)
    else:
        add_perturbations(data)
        perturbed_data = data
    filename_prefix = input_filename.split("/")[-1].split(".")[0]
    # Removing previous timestamp if any
    filename_prefix = re.sub('_2019[0-9]*$', '', filename_prefix)
    output_name_suffix = ''
    if '_perturbed' not in filename_prefix:
        output_name_suffix = '_perturbed'
    timestamp = re.sub('[^0-9]', '', str(datetime.datetime.now()).split('.')[0])
    # Will be written in current directory
    output_filename = f"{filename_prefix}{output_name_suffix}_{timestamp}.json"
    json.dump(perturbed_data, open(output_filename, "w"), indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='''Location of input file. If you want to continue your work from a
                        previous session, provide the output from that session as the input here. If
                        '--output-perturbations-only' is not set, the resulting dataset from this session will contain a
                        union of your perturbations from both sessions.''')
    parser.add_argument('--output-perturbations-only', dest='output_perturbations_only', action='store_true',
                        help='''If this flag is set, the output will not contain instances from the input file.
                        It will only have perturbations, but they will still have references to the original instances.''')
    args = parser.parse_args()
    main(args)
