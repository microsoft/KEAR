# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from utils import _load_json
from .example import ConceptNetExample


def load_data(task, *args, **kwargs):
    if task == 'conceptnet':
        return _load_data_conceptnet(*args, **kwargs)    

def _load_data_conceptnet(file_name, type='json', append_answer_text=False, append_descr=False, append_triple=True):
    examples = []
    if type == 'json':
        for json_obj in _load_json(file_name):
            example = ConceptNetExample.load_from_json(json_obj, append_answer_text, append_descr, append_triple)
            examples.append(example)
    return examples