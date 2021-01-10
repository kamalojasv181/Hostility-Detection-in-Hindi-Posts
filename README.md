## Hostility Post Detection in Hindi

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
To be added
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

Refer our paper (#TODO add link) for complete details.

## Dependencies

## Setup Instruction 

## Trained Models

## Model Performance

## Miscellaneous

