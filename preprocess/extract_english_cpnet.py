# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/INK-USC/MHGRN
import json
from tqdm import tqdm

conceptnet_path = '../data/kear/conceptnet-assertions-5.7.0.csv'
output_csv_path = '../data/kear/conceptnet.en.csv'
print('extracting English concepts and relations from ConceptNet...')
num_lines = sum(1 for line in open(conceptnet_path, 'r', encoding='utf-8'))
with open(conceptnet_path, 'r', encoding="utf8") as fin, \
        open(output_csv_path, 'w', encoding="utf8") as fout:
    for line in tqdm(fin, total=num_lines):
        toks = line.strip().split('\t')
        if toks[2].startswith('/c/en/') and toks[3].startswith('/c/en/'):
            """
            Some preprocessing:
                - Remove part-of-speech encoding.
                - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to 
                - Lowercase for uniformity.
            """
            rel = toks[1].split("/")[-1]
            head = toks[2].split("/")[3].lower()
            tail = toks[3].split("/")[3].lower()

            data = json.loads(toks[4])

            fout.write('\t'.join([rel, head, tail, str(data["weight"])]) + '\n')

print(f'extracted ConceptNet csv file saved to {output_csv_path}')
print()