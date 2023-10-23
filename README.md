
# Improving Adversarial Robustness via Information Bottleneck Distillation (NeurIPS 2023 Poster)

## Introduction
  This paper studies the information bottleneck principle and proposes an Information Bottleneck Distillation approach.
  This specially designed, robust distillation technique utilizes prior knowledge obtained from a robust pre-trained model to boost information bottlenecks. 
  % shows that a specially designed robust distillation technique can boost information bottleneck, benefiting from the prior knowledge obtained from a robust pre-trained model.
  % Therefore, we present the Information Bottleneck Distillation (IBD) approach.
  Specifically, we propose two distillation strategies that align with the two optimization processes of the information bottleneck.
  Firstly, we use a robust soft-label distillation method to increase the mutual information between latent features and output prediction.
  Secondly, we introduce an adaptive feature distillation method that automatically transfers relevant knowledge from the teacher to the target student model, thereby reducing the mutual information between the input and latent features.


## Usage
### Installation
The training environment (PyTorch and dependencies) can be installed as follows:
```
cd IBD-master
pip install -r requirements.txt
```
### Train IBD
```
sh ./train.sh
```
### Evaluate IBD
```
sh ./eval.sh
```
