# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
from collections import defaultdict

useful_relations_more = ['AtLocation', 'LocatedNear', 'CapableOf', 'Causes', 
'CausesDesire', 'MotivatedByGoal', 'CreatedBy', 'Desires', 'Antonym', 
'DistinctFrom', 'HasContext', 'HasProperty', 'HasSubevent', 'HasPreRequisite', 'Entails',  
'InstanceOf', 'DefinedAs', 'MadeOf', 
'PartOf', 'HasA', 'SimilarTo', 'UsedFor']

useful_relations_less = ['CausesDesire', 'HasProperty', 'CapableOf', 'PartOf',
'AtLocation', 'Desires', 'HasPrerequisite', 'HasSubevent', 'Antonym', 'Causes']

def is_invalid(word):
    return len(word) <= 2 or len(word.split('_')) > 4
def filter_cpnet(useful_relations, output_name, filter_inconfident=False):
    print('output version', output_name)
    extracted_edges = {}
    with open('../data/kear/conceptnet.en.csv', encoding='utf-8') as cpnet:
        for line in cpnet:
            ls = line.strip().split('\t')
            rel = ls[0]
            subj = ls[1]
            obj = ls[2]
            weight = float(ls[3])
            if filter_inconfident and weight < 1.0:
                continue
            if rel not in useful_relations or is_invalid(subj) or is_invalid(obj):
                continue
            if subj not in extracted_edges:
                extracted_edges[subj] = {}
            if obj not in extracted_edges[subj]:
                extracted_edges[subj][obj] = defaultdict(int)
            extracted_edges[subj][obj][rel] = max(extracted_edges[subj][obj][rel], weight)

    json.dump(extracted_edges, open(f'../data/kear/{output_name}.json','w', encoding='utf-8'))    

filter_cpnet(useful_relations_more, 'most_edges_allweights', False)
filter_cpnet(useful_relations_less, 'filtered_edges', True)
