## Hostility Post Detection in Hindi

---
**NOTE**: We are currently in the process of updating the repo and scripts to resolve the issues along with some other changes as mentioned in [this](#updates-to-be-done) Section. The changes will be reflected pretty soon. Any other suggestions are warmly welcome.

---

Leveraging pre-trained Language models for Multidimensional Hostile post detection in Hindi - 3rd runner up at [CONSTRAINT 2021 Shared Task 2 - Hostile Post Detection in Hindi](https://constraint-shared-task-2021.github.io/), collocated with AAAI 2021.

This repo contains:
<ul>
  <li> Code for Models</li>
  <li> Trained models used in the final submission.</li>
  <li> Setup instructions to reproduce results from the paper.</li>
</ul>

Some important links: arxiv, poster.

In order to cite, use the following BiBTeX code:

```
@misc{kamal2021hostility,
      title={Hostility Detection in Hindi leveraging Pre-Trained Language Models}, 
      author={Ojasv Kamal and Adarsh Kumar and Tejas Vaidhya},
      year={2021},
      eprint={2101.05494},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

**Authors**: [Ojasv Kamal](https://github.com/kamalojasv181), [Adarsh Kumar](https://github.com/AdarshKumar712) and [Tejas Vaidhya](https://github.com/tejasvaidhyadev)

## Overview

### Abstract

Hostile content on social platforms is ever increasing. This has led to the need for proper detection of hostile posts so that appropriate action can be taken to tackle them. Though a lot of work has been done recently in the English Language to solve the problem of hostile content online, similar works in Indian Languages are quite hard to find. This paper presents a transfer learning based approach to classify social media (i.e Twitter, Facebook, etc.) posts in Hindi Devanagari script as Hostile or Non-Hostile. Hostile posts are further analyzed to determine if they are Hateful, Fake, Defamation, and Offensive. This paper harnesses attention based pre-trained models fine-tuned on Hindi data with Hostile-Non hostile task as Auxiliary and fusing its features for further sub-tasks classification. Through this approach, we establish a robust and consistent model without any ensembling or complex pre-processing. We have presented the results from our approach in CONSTRAINT-2021 Shared Task on hostile post detection where our model performs extremely well with <b> 3rd runner up</b> in terms of Weighted Fine-Grained F1 Score

### Key Contributions
<ol>
  <li> We fine-tuned transformer based pre-trained, Hindi Language Models fordomain-specific contextual embeddings, which are further used in ClassificationTasks.</li>
  <li> We incorporate the fine-tuned hostile vs. non-hostile detection model as anauxiliary model, and fuse it with the features of specific subcategory models(pre-trained models) of hostility category, with further fine-tuning.</li>
</ol>

Refer our [paper](https://arxiv.org/abs/2101.05494) for complete details.

## Dependencies


| Dependency | Version | Installation Command |
| ---------- | ------- | -------------------- |
| Python     | 3.8     | `conda create --name covid_entities python=3.8` and `conda activate covid_entities` |
| PyTorch, cudatoolkit    | >=1.5.0, 10.1   | `conda install pytorch==1.5.0 cudatoolkit=10.1 -c pytorch` |
| Transformers (Huggingface) | 3.5.1 | `pip install transformers==3.5.1` |
| Scikit-learn | >=0.23.1 | `pip install scikit-learn==0.23.1` |
| Pandas | 0.24.2 | `pip install pandas==0.24.2` |
| Numpy | 1.18.5 | `pip install numpy==1.18.5` |
| Emoji | 0.6.0 | `pip install emoji==0.6.0` |
| Tqdm | 4.48.2| `pip install tqdm==4.48.2` |


## Setup Instruction 
Will be included soon

## Trained Models
Our model weights used in the submission have been [released now](https://github.com/kamalojasv181/Hostility-Detection-in-Hindi-Posts/releases/tag/v0.0.1).

## Model Performance

Performance of our best model, i.e. Auxiliary Indic Bert on the Test Dataset.

| Approach | Hostile | Defamation | Fake | Hate | Offensive | Weighted |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|Baseline Model|0.8422|0.3992|0.6869|0.4926|0.4198|0.542|
|Indic-Bert Auxiliary Model|0.9583|0.42|0.7741|0.5725|0.6120|0.6250|

Please note that due to the skewed and biased dataset, along with model here not being perfectly reproducible, slight variations in final state of the model may lead the results to vary by 0.02-0.03 f1 scores (as has been observed by us in different runs for same model). Refer [this](https://pytorch.org/docs/stable/notes/randomness.html#:~:text=Completely%20reproducible%20results%20are%20not,even%20when%20using%20identical%20seeds.) for more details.

## Updates to be done
- [x] Resolve issues with main_multitask_learning.py
- [x] Some Minor changes in code and functions
- [ ] Add Setup Instructions
- [x] Add corrected code for csv file generation
- [ ] Colab Notebook on Usage

## Miscellaneous
<ul>
  <li> In case of any issues or any query, please contact <a href="mailto:kamalojasv181@gmail.com?">Ojasv Kamal</a></li>
</ul>

