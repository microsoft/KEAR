# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from utils.feature import Feature


label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G':6, 'H': 7, None: -1}
rel_mapping = {
                'CausesDesire': "causes desire",
                'HasProperty': "has property",
                'CapableOf': 'capable of',
                'PartOf': 'part of',
                'AtLocation': 'at location',
                'Desires': 'desires',
                'HasPrerequisite': 'has prerequisite',
                'HasSubevent': 'has subevent',
                'Antonym': 'antonym',
                'Causes': 'causes',
}


def maybe_add(base_str, append_str, prepend=False, sep_word='[SEP]'):
    if base_str is None:
        return None if append_str is None else append_str
    if append_str is None:
        return base_str
    elif prepend:
        return f'{append_str} {sep_word} {base_str}'
    else:
        return f'{base_str} {sep_word} {append_str}'

class ConceptNetExample:
    max_len = 0
    all_lens = []
    def __init__(self, idx, choices, label = -1, append_descr=0, data=None):
        self.idx = idx
        self.texts = choices
        self.is_valid = True
        self.label = int(label)
        self.append_descr = append_descr
        self.data = data
    
    def __str__(self):
        return f"{self.idx} | {self.texts} | {self.label}"
          
    def tokenize_text(self, tokenizer, max_seq_length, vary_segment_id=False):
        
        def tokenize(texts):
            tokens = []
            for text_data in texts:
                token_data = {}
                for key in text_data:
                    token_data[key] = tokenizer.tokenize(text_data[key]) if isinstance(text_data[key], str) else text_data[key]
                tokens.append(token_data)
            return tokens
        self.tokens = tokenize(self.texts)
        
    @classmethod
    def load_from_json(cls, json_obj, append_answer_text=False, append_descr=0, append_triple=True, 
                       append_retrieval=0, sep_word='[SEP]',
                       append_frequent=0, frequent_thres=4):
        choices = json_obj['question']['choices']
        question_concept = json_obj['question'].get('question_concept', None)
                
        def mkinput(question_concept, choice, is_gt=False):
            out_data = {}
            if append_triple:
                triples_temp_alternate = maybe_add(None, question_concept, sep_word=sep_word)
                triples_temp_alternate = maybe_add(triples_temp_alternate, choice.get('answer_concept', choice['text']), sep_word=sep_word)
                if choice['triple']:
                    choice['triple'][0][1] = rel_mapping[choice['triple'][0][1]] 
                    triples = f' {sep_word} '.join([' '.join(trip) for trip in choice['triple']])
                    out_data['triples_temp'] = triples
                else:
                    out_data['triples_temp'] = triples_temp_alternate
            else:
                out_data['triples_temp'] = None
            out_data['is_freq_masked'] = 0
            if append_frequent and json_obj['question']['major_rel_cnt'] >= frequent_thres and (not choice['is_major_rel']):
                out_data['is_freq_masked'] = append_frequent
            if append_descr > 0:
                out_data['qc_meaning'] = json_obj['question']['qc_meaning']
                out_data['ac_meaning'] = choice['ac_meaning']
            else:
                out_data['qc_meaning'] = None
                out_data['ac_meaning'] = None
            if append_answer_text:
                out_data['question_text'] = '{} {}'.format(json_obj['question']['stem'], choice['text'])
            else:
                out_data['question_text'] = json_obj['question']['stem']           
            if append_retrieval > 0:
                retrieval_texts = choice['retrieval']                
                out_data['ac_meaning'] = maybe_add(out_data['ac_meaning'], ' '.join(retrieval_texts)) # add retrieval as a part of ac meaning
            return out_data

        texts = []
        for c_id, choice_data in enumerate(choices):
            is_gt = (label_dict[json_obj.get('answerKey', None) ] == c_id)
            texts.append(mkinput(question_concept, choice_data, is_gt))
        try:
            label =  label_dict[json_obj['answerKey']]
        except:
            label = -1
        if label is None:
            label = -1
        return cls(
            json_obj['initial_id'],
            texts,
            label,
            append_descr,
            json_obj,
        )


