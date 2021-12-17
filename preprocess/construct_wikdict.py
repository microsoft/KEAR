# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
from collections import defaultdict

wik_file = '../data/kear/download-all-senses.json'
wik_dict = defaultdict(list)

with open(wik_file, 'r', encoding='utf-8') as f:
    counter = 0
    for line in f:
        if counter % 100000 == 0:
            print(counter)
        data = json.loads(line)
        data.pop('pronunciations', 'no')
        data.pop('lang', 'no')
        data.pop('translations', 'no')
        data.pop('sounds', 'no')
        wik_dict[data['word']].append(data)
        counter += 1

# join lower case with upper case
new_wik_dict = {}
for word in wik_dict:
    if word.lower() not in new_wik_dict:
        new_wik_dict[word.lower()] = wik_dict[word]
    elif word.lower() == word:
        new_wik_dict[word.lower()] = wik_dict[word] + new_wik_dict[word.lower()]
    else:
        new_wik_dict[word.lower()] = new_wik_dict[word.lower()] + wik_dict[word]

wik_dict = new_wik_dict

output_file = '../data/kear/wik_dict.json'
json.dump(wik_dict, open(output_file, 'w', encoding='utf-8'))