# Human Parity on CommonsenseQA: Augmenting Self-Attention with External Attention

This PyTorch package implements the KEAR model that surpasses human on the CommonsenseQA benchmark, as described in:

Yichong Xu, Chenguang Zhu, Shuohang Wang, Siqi Sun, Hao Cheng, Xiaodong Liu, Jianfeng Gao, Pengcheng He, Michael Zeng and Xuedong Huang<br/>
[Human Parity on CommonsenseQA: Augmenting Self-Attention with External Attention](https://arxiv.org/pdf/2012.04808.pdf)</br>
arXiv:2112.03254, 2021<br/>

The package also includes codes for our earilier DEKCOR model as in:

Yichong Xu∗, Chenguang Zhu∗, Ruochen Xu, Yang Liu, Michael Zeng and Xuedong Huang<br/>
[Fusing Context Into Knowledge Graph for Commonsense Question Answering](https://arxiv.org/pdf/2012.04808.pdf)</br>
Findings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL), 2021<br/>

Please cite the above papers if you use this code. 

## Results
This package achieves the state-of-art performance of 86.1% (single model), 89.4% (ensemble) on the [CommonsenseQA leaderboard](https://www.tau-nlp.org/csqa-leaderboard), surpassing the human performance of 88.9%.

## Quickstart 

1. pull docker: </br>
   ```> docker pull yichongx/csqa:human_parity```

2. run docker </br>
   ```> nvidia-docker run -it --mount src='/',target=/workspace/,type=bind yichongx/csqa:human_parity /bin/bash``` </br>
   ```> cd /workspace/path/to/repo``` </br>
    Please refer to the following link if you first use docker: https://docs.docker.com/

## Features

Our code supports flexible training of various models on multiple choice QA.

- Distributed training with Pytorch native DDP or [Deepspeed](https://www.deepspeed.ai/): see ``bash/task_train.sh``
- Pause and resume training at any step; use option ``--continue_train``
- Use any transformer encoders including ELECTRA, DeBERTa, ALBERT

## Preprocessing data
Pre-processed data is located at ```data/```.

We release codes for knowledge graph and dictionary external attention in ``preprocess/``

1. Download data</br>
   ```> cd preprocess```</br>
   ```> bash download_data.sh```</br>
2. Add ConceptNet triples and Wiktionary definitions to data</br>
   ```> python add_knowledge.py```</br>
3. We also add the most frequent relations in each question as a side information.</br>
   ```> python add_freq_rel.py ```</br>

## Training and Prediction
1. train a model</br>
   ```> bash bash/task_train.sh```
2. make prediction</br>
   ```> bash bash/task_predict.sh```
See ``task.py`` for available options.

## Running codes for DEKCOR

The current code is mostly compatible to run DEKCOR. To run the original DEKCOR code, please checkout tag ``DEKCOR`` to use the previous version.

by Yichong Xu</br>
yicxu@microsoft.com

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
