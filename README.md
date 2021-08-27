# Fusing Context Into Knowledge Graph for Commonsense Question Answering

This PyTorch package implements the DEKCOR model for Commonsense Question Answering, as described in:

Yichong Xu∗, Chenguang Zhu∗, Ruochen Xu, Yang Liu, Michael Zeng and Xuedong Huang<br/>
[Fusing Context Into Knowledge Graph for Commonsense Question Answering](https://aclanthology.org/2021.findings-acl.102.pdf)</br>
Findings of The 59th Annual Meeting of the Association for Computational Linguistics (ACL), 2021<br/>
[arXiv version](https://arxiv.org/pdf/2012.04808.pdf)

Please cite the above paper if you use this code. 

## Results
This package achieves the state-of-art performance of 80.7% (single model), 83.3% (ensemble) on the [CommonsenseQA leaderboard](https://www.tau-nlp.org/csqa-leaderboard).

## Quickstart 

1. pull docker: </br>
   ```> docker pull yichongx/csqa:acl2021```

2. run docker </br>
   ```> nvidia-docker run -it --mount src='/',target=/workspace/,type=bind yichongx/csqa:acl2021 /bin/bash``` </br>
    Please refer to the following link if you first use docker: https://docs.docker.com/

## Use the data
Pre-processed data is located at ```data/```.

## Use the code
1. train a model
   > bash bash/task_train.sh
2. make prediction
   > bash bash/task_predict.sh

## Notes and Acknowledgments
The code is developed based on KCR: https://github.com/jessionlin/csqa

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
