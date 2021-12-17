# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel


run pip install --upgrade pip
run pip install --upgrade virtualenv
run pip install pandas
USER root
run apt-get update
run apt-get install -y vim
run pip install tensorflow
run pip install boto3
run pip install msgpack
run pip install spacy
run python -m spacy download en
run pip install nltk
run pip install sentencepiece

# install git
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

run pip install pyrouge
WORKDIR /workspace
run git clone https://github.com/andersjo/pyrouge.git
RUN rm -f pyrouge/tools/ROUGE-1.5.5/data/WordNet-2.0.exc.db
RUN pyrouge/tools/ROUGE-1.5.5/data/WordNet-2.0-Exceptions/buildExeptionDB.pl pyrouge/tools/ROUGE-1.5.5/data/WordNet-2.0-Exceptions pyrouge/tools/ROUGE-1.5.5/data/smart_common_words.txt pyrouge/tools/ROUGE-1.5.5/data/WordNet-2.0.exc.db
RUN pyrouge_set_rouge_path pyrouge/tools/ROUGE-1.5.5/

run apt-get update && apt-get install -y libxml-parser-perl

run pip install unidecode
RUN git clone https://github.com/NVIDIA/apex
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
RUN apt install -y libopenmpi-dev
RUN pip install mpi4py
run pip install networkx
run pip install stanford_openie
run pip install certifi
run update-ca-certificates --fresh
ENV SSL_CERT_DIR=/etc/ssl/certs
run apt-get update && apt-get -y upgrade
run apt-get install -y default-jdk

run pip install sklearn
run pip install tqdm
run pip install dgl-cu101
ENV DGLBACKEND=pytorch
run pip install http://www.jbox.dk/sling/sling-2.0.0-py3-none-linux_x86_64.whl
run pip install jsonlines
run apt install -y tmux
run apt install -y wget
run pip install wandb
run pip install jupyter
run pip install transformers==4.10.2
run pip install datasets
run pip install pytorch-lightning
run pip install packaging
run pip install rouge_score
run pip install sacrebleu

RUN pip install pip==9.0.0
RUN pip install ruamel.yaml==0.16 --disable-pip-version-check
RUN pip install --upgrade pip
RUN pip install ninja
RUN pip install deepspeed==0.4.5
RUN git clone --branch penhe/fix_lm_training https://github.com/microsoft/DeBERTa.git
RUN pip install ./DeBERTa