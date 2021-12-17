# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import spacy
import Levenshtein
from collections import defaultdict, Counter
import os
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from meaning_resolver import MeaningResolver
import random

path_prefix = '..'

def generate_choice(ac_text, label, me_re=None):
    c_data = {
        'label': label,
        'text': ac_text,
        'triple': None,
        'surface': None,
        'weight': None,
        'ac_meaning': me_re.resolve_meaning_cached(ac_text) if me_re is not None else None
    }
    return c_data

class TripletFinder:
    
    def __init__(self, cpnet_graph_path='{}/data/kear/filtered_edges.json'.format(path_prefix), weight_lb=1.0, target_relations=None):
        self.weight_lb = weight_lb
        self.cpnet = json.load(open(cpnet_graph_path, encoding='utf-8'))
        self.nlp = spacy.load('en_core_web_sm')
        self.all_concepts = set(self.cpnet.keys())
        for c in self.cpnet:
            self.all_concepts.update(self.cpnet[c].keys())
        for w in self.nlp.Defaults.stop_words:
            if w in self.all_concepts:
                self.all_concepts.remove(w)
        self.all_concepts_list = list(self.all_concepts)
        self.target_relations = target_relations

    def lemma_first(self, qc, lemma_last=False):   
        # lemma_last: lemmatize last word instead of first
        words = self.nlp(qc)
        qc_words = [w.text for w in words]
        lemma_idx = -1 if lemma_last else 0
        lemma_word = words[lemma_idx].lemma_ if words[lemma_idx].lemma_ != '-PRON-' else words[lemma_idx].text
        if qc_words[lemma_idx] == lemma_word:
            return qc, qc_words
        else:
            qc_words[lemma_idx] = lemma_word
            qc_new = ' '.join(qc_words)
            return qc_new, qc_words

    def ground_find_triplet(self, qc, ac, find_answer_triplet=True):
        qc = qc.replace(' ', '_')
        if qc not in self.cpnet:
            self.cpnet[qc] = {}
        
        ac_lemma, _ = self.lemma_first(ac)
        ac = ac.replace(' ', '_')
        ac_lemma = ac_lemma.replace(' ', '_')
        if ac in self.cpnet[qc]:
            return self.find_triplet(qc, ac, find_answer_triplet)
        elif ac_lemma in self.cpnet[qc]:
            return self.find_triplet(qc, ac_lemma, find_answer_triplet)
        elif ac in self.cpnet:
            return self.find_triplet(qc, ac, find_answer_triplet)
        elif ac_lemma in self.cpnet:
            return self.find_triplet(qc, ac_lemma, find_answer_triplet)
        else:
            return []
    

    def find_triplet(self, qc, ac, find_answer_triplet=True):
        '''
        qc (if not None) and ac are guaranteed to be concepts in conceptnet; either ac in cpnet[qc], or ac in cpnet.
        '''
        retrieved_triplets = []
        if qc is not None and ac in self.cpnet[qc]:
            for rel, weight in self.cpnet[qc][ac].items():
                if self.target_relations is not None and rel not in self.target_relations:
                    continue
                if weight >= self.weight_lb:
                    retrieved_triplets.append((qc, rel, ac, weight)) # s, r, o, weight
            return [self.dash_to_space(tup) for tup in retrieved_triplets]
        if not find_answer_triplet:
            return []
        if len(retrieved_triplets) == 0:
            assert ac in self.cpnet
            for o in self.cpnet[ac]:
                for rel, weight in self.cpnet[ac][o].items():
                    if self.target_relations is not None and rel not in self.target_relations:
                        continue                    
                    if weight >= self.weight_lb:
                        retrieved_triplets.append((ac, rel, o, weight))
        return self.best_triplets(retrieved_triplets)

    def dash_to_space(self, tup):
        s, r, o = tup[:3]
        return (s.replace('_', ' '), r, o.replace('_', ' '))

    def best_triplets(self, triplet_list):
        N = len(triplet_list)
        if N == 1:
            return [self.dash_to_space(tuple(triplet_list[0][:3]))]
        rel_count = Counter([t[1] for t in triplet_list])
        all_weights = [(t, t[3] * N / rel_count[t[1]]) for t in triplet_list]
        all_weights = sorted(all_weights, key=lambda x: -x[1])
        res = []
        for idx in range(N):
            if all_weights[idx][1] == all_weights[0][1]:
                res.append(self.dash_to_space(all_weights[idx][0]))
            else:
                break
            
        return res            

if __name__ == '__main__':
    
    csqa_dir = '../data/'
    test_mode = False
    n_threads = 1 if test_mode else 32
    replace_key = 'choices'
    add_ac_meaning = False 
    input_files = ['train_data.json', 'dev_data.json', 'test_data.json']
    file_paths = [os.path.join(csqa_dir, 'csqa_ret_3datasets', s) for s in input_files]
    output_dir = os.path.join(csqa_dir, 'csqa_new')
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    output_paths = [os.path.join(output_dir, s) for s in input_files]

    t_finder = TripletFinder()
    me_re = MeaningResolver(wikdict_fn = '{}/data/yicxu/wiktionary_new/wik_dict.json'.format(path_prefix))
    max_num_relations = 5

    
    t_finder = TripletFinder()

    def add_triple(q_data):
        qc = q_data['question']['question_concept']
        if replace_key not in q_data['question']:
            q_data['question'][replace_key] = []
        for option in q_data['question'][replace_key]:
            ac = option['text']
            option['triple'] = []
            res = t_finder.ground_find_triplet(qc, ac)
            if len(res) > 0:
                option['triple'] = [res[0]]
            if add_ac_meaning:
                option['ac_meaning'] = me_re.resolve_meaning_cached(ac)

        return q_data


    for fn, output_path in zip(file_paths, output_paths):
        print(fn)
        output_data = []
        with open(fn, 'r', encoding='utf-8') as json_file:
            data_list = json.load(json_file)
        func_name = add_triple
        if n_threads == 1:
            for data in tqdm(data_list):
                output_data.append(func_name(data))
        else:
            with Pool(n_threads) as p:
                output_data = list(tqdm(p.imap(func_name, data_list, chunksize=32), total=len(data_list)))
        json.dump(output_data, open(output_path, 'w', encoding='utf-8'))