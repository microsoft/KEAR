# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# optional: download conceptnet and wiktionary for preprocessing
# download conceptnet 5.7
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz -P ../data/kear/
gzip -d ../data/kear/conceptnet-assertions-5.7.0.csv.gz
python extract_english_cpnet.py
# extract cpnet with most edges or only a few relations
python extract_all.py

# download wiktionary
wget https://kaikki.org/dictionary/English/all-non-inflected-senses/kaikki.org-dictionary-English-all-non-inflected-senses.json -O ../data/kear/download-all-senses.json
python construct_wikdict.py


