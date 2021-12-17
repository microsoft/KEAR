# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
from tqdm import tqdm
import random
from collections import Counter
from pathlib import Path
import os
from add_knowledge import TripletFinder 
from collections import defaultdict

# freq_rel: we add a freq_rel field for each question, indicating which choice might not have the same relation with question concept as other choices.
# We filter out such answers in our prediction.

path_prefix = '..'
csqa_dir = '../data/'
input_version = 'csqa_ret_3datasets'
output_version = 'csqa_new'
input_files = ['train_data.json', 'dev_data.json', 'test_data.json']
file_paths = [os.path.join(csqa_dir, input_version, s) for s in input_files]
output_dir = os.path.join(csqa_dir, output_version)
Path(output_dir).mkdir(exist_ok=True, parents=True)
output_paths = [os.path.join(output_dir, s) for s in input_files]
cpnet_path = '{}/data/kear/most_edges_allweights.json'.format(path_prefix)
t_finder = TripletFinder(cpnet_path, weight_lb=0.0)
keys = ['A', 'B', 'C', 'D', 'E']
keys_mapping = {w:idx for idx, w in enumerate(keys)}

for fn, output_path in zip(file_paths, output_paths):
    with open(fn, 'r', encoding='utf-8') as json_file:
        data_list = json.load(json_file)
    output_data = []
    for q_data in tqdm(data_list):
        qc = q_data['question']['question_concept']
        q_text = q_data['question']['stem']
        correct_option_idx = keys_mapping[q_data.get('answerKey', 'A')]
        all_answers = []
        relation_counter = Counter()
        relation_mapping = defaultdict(list)
        opt_relations = {}
        q_data['question']['major_rel'] = None
        q_data['question']['major_rel_cnt'] = 0
        for o_idx, option in enumerate(q_data['question']['choices']):
            ac = option['text']
            triplets = t_finder.ground_find_triplet(qc, ac, find_answer_triplet=False)
            option['is_major_rel'] = False
            relation_counter.update([trip[1] for trip in triplets])
            opt_relations[ac] = triplets
            for trip in triplets:
                relation_mapping[trip[1]].append(ac)
            if o_idx != correct_option_idx:
                all_answers.append(ac)
            else:
                correct_answer = ac
        most_common_rels = relation_counter.most_common()
        if len(most_common_rels) == 0: # not found
            output_data.append(q_data)
            continue
        selected_rel = most_common_rels[0][0]
        com_rel_ans = set(relation_mapping[selected_rel])
        for option in q_data['question']['choices']:
            if option['text'] in com_rel_ans:
                option['is_major_rel'] = True
            this_triples = opt_relations[option['text']]
            option['frequent_triples'] = [triple for triple in this_triples if triple[1] == selected_rel]
        q_data['question']['major_rel'] = selected_rel
        q_data['question']['major_rel_cnt'] = most_common_rels[0][1]
        output_data.append(q_data)       
    if output_json:
        print('output length:', len(output_data))
        json.dump(output_data, open(output_path, 'w', encoding='utf-8'))
