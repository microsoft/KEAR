# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import spacy
import json
import string
import os

class MeaningResolver:

    def __init__(self, wikdict_fn = None, load_cache=True):
        self.nlp=spacy.load('en_core_web_sm')        
        self.wik_dict = json.load(open(wikdict_fn, encoding='utf-8'))
        self.cache_fn = wikdict_fn.replace('wik_dict.json', 'meaning_cache.json')
        if os.path.isfile(self.cache_fn) and load_cache:
            self.meaning_cache = json.load(open(self.cache_fn, encoding='utf-8'))
            print('loaded {} meanings'.format(len(self.meaning_cache)))
        else:
            self.meaning_cache = {}
        self.call_history = set()

    def lemma_first(self, qc):   
        words = self.nlp(qc)
        qc_words = [w.text for w in words]
        lemma_word = words[0].lemma_ if words[0].lemma_ != '-PRON-' else words[0].text
        if qc_words[0] == lemma_word:
            return qc, qc_words
        else:
            qc_words[0] = lemma_word
            qc_new = ' '.join(qc_words)
            return qc_new, qc_words

    def resolve_meaning_cached(self, c):
        c = c.replace('_', ' ')
        if c not in self.meaning_cache:
            self.call_history = set()
            self.meaning_cache[c] = self.resolve_meaning(c)
            if len(self.meaning_cache) % 1000 == 0:
                print('output cache to ', self.cache_fn)
                with open(self.cache_fn, 'w', encoding='utf-8') as fout:
                    json.dump(self.meaning_cache, fout)
        return self.meaning_cache[c]

    def resolve_meaning(self, qc, no_lemma=False):
        qc = qc.lower()
        if (qc, no_lemma) in self.call_history:
            # avoid loops; we simply return None if error occurs.
            print('return None because of call history: {}, {} for {}'.format(qc, no_lemma, self.call_history))
            return None
        self.call_history.add((qc, no_lemma))
        if qc == '':
            return None
        if not no_lemma:
            qc_new, _ = self.lemma_first(qc)
            qc = qc if (qc in self.wik_dict and qc_new not in self.wik_dict) else qc_new
            qc_new = qc.strip(string.punctuation+' ')
            if qc_new != qc:
                qc = qc_new
        if qc in self.wik_dict:
            for meaning in self.wik_dict[qc]:
                if 'senses' in meaning:
                    for sense in meaning['senses']:
                        if 'form_of' in sense or 'alt_of' in sense: # try to follow these links
                            form_str = 'form_of' if 'form_of' in sense else 'alt_of'
                            qc_new = sense[form_str][0]
                            if qc.lower() == qc_new.lower():
                                continue
                            if len(qc_new.split(' ')) <= 4:
                                return self.resolve_meaning(qc_new, no_lemma=True)
                        elif 'heads' in meaning and meaning['heads'][0].get('2', '') == 'verb form':
                            try_str = sense['glosses'][0].split(' of ')
                            if len(try_str) == 2:
                                qc_new = try_str[-1]
                                return self.resolve_meaning(qc_new, no_lemma=True)
                            else:
                                print('verb form failed:', meaning)             
                        if 'glosses' in sense:
                            mstr = '{}: {}'.format(qc, sense['glosses'][0])
                            return mstr
        qc_new, qc_words = self.lemma_first(qc)
        if qc_new in self.wik_dict and qc_new != qc:
            return self.resolve_meaning(qc_new)
        qc_new = ''.join(qc_words)
        if qc_new in self.wik_dict and qc_new != qc:
            return self.resolve_meaning(qc_new)    
        qc_new = ' '.join(qc.split(' ')[1:])                         
        res = self.resolve_meaning(qc_new)
        if res is not None:
            return res
        qc_new = qc.translate(str.maketrans('', '', string.punctuation))
        if qc_new in self.wik_dict and qc_new != qc:
            return self.resolve_meaning(qc_new)
        qc_new = qc.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        if qc_new in self.wik_dict and qc_new != qc:
            return self.resolve_meaning(qc_new)
