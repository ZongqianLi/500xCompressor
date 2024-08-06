# ♟️ 500xCompressor: Generalized Prompt Compression for Large Language Models

<p align="center">
  <b>Content</b>
</p>

<p align="center">
  <a href="#news">🚀News</a> •
  <a href="#todo">✏️Todo</a> •
  <a href="#introduction">✨Introduction</a>
</p>

<p align="center">
  <a href="#corpus">📚Corpus</a> •
  <a href="#dataset">🤗Dataset</a> •
  <a href="#model">🤗Models</a>
</p>

<p align="center">
  <a href="#download">💾 Download</a> •
  <a href="#citation">📌Citation</a> •
  <a href="#license">🔖License</a>
</p>

<p align="center">
  <b>Links</b>
</p>

<p align="center">
  <a href="">Project Page</a> •
  <a href="https://huggingface.co/spaces/ZongqianLi/SolarCellBERT">Demo Page</a> •
  <a href="">Paper</a>
</p>

<div id="news">&nbsp;</div>



## 🚀 News

- **[2024.08.06]** The paper was uploaded to Arxiv.

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="todo">&nbsp;</div>



## ✏️ Todo

- [ ] Datasets and models were uploaded to Huggingface but are not open to the public.

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="introduction">&nbsp;</div>



## ✨ Introduction

**500xCompressor** is a **prompt compression method** that could compresss a maximum of **500** natural language tokens into only **1** special token. This compressed token could **regenerate** the original text or be used for **question answering (QA)**.

Initially, 500xCompressor was pretrained on the **Arxiv Corpus**, followed by fine-tuning on the **ArxivQA dataset**, and subsequently evaluated on various **strictly unseen** and **classical** **QA** datasets.

500xCompressor has several **features and advantages**:
- **Small additional parameters:** only **0.3% extra** parameters are added to the LLM
- **Zero-shot usage:** the compressed tokens can be used by the original LLM **without being finetuned**
- **High compression ratio:** from **6x** to **480x**
- **Generalization ability:** could compress any **unseen** text and be used for **unseen** datasets in downstream tasks
- **Retained capabilities:** **62.26-72.89%** of LLM abilities compared to using non-compressed prompts

This research gave several **insights**:
- **Not** all the compressed tokens are **equally** utilized
- **K V values** have significant advantages over **embeddings** in preserving information at high compression ratios
- Natural language prompts are **highly compressive**
- **Fine-grained complex** information could be compressed and retrieved exactly as well

Here is an example:

<p align="left">
  <img src="./Figures/cover_figure_2.png" width="40%">
</p>

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="corpus">&nbsp;</div>



## 📚 Arxiv Corpus

This is a collection of **Arxiv abstracts**.

- **Train:** 2353924 items, based on Arxiv abstracts before 07/2023
- **Validation:** 3000 items, based on Arxiv abstracts during 01-04/2024
- **Test:** 2500 items, based on Arxiv abstracts during 01-04/2024

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="dataset">&nbsp;</div>



## 🤗 ArxivQA Dataset

This is an **extractive QA dataset** created based on the abstracts of Arxiv papers.

- **Train:** 250000 items, based on Arxiv abstracts before 07/2023
- **Validation:** 1000 items, based on Arxiv abstracts before 07/2023
- **Test:** 1000 items, based on Arxiv abstracts during 01-04/2024

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="model">&nbsp;</div>



## 🤗 500xCompressor

### Quick use for regeneration:
```
# codes
```
```
# output
```

<div>&nbsp;</div>

### Quick use for QA:
```
# codes
```
```
# output
```

<div>&nbsp;</div>

### Training process: 

The compression model was pretrained on the **Arxiv Corpus** for **regenerating** the original text according to the compressed tokens. Then, it was finetuned on the **ArxivQA Dataset** for **answering the questions** based on the compressed tokens.

<p align="left">
  <img src="./Figures/mechanism.png" width="100%">
</p>

<div>&nbsp;</div>

### Performance: 

The compression models were evaluated on various **strictly unseen** and **classic** QA benchmarks.

<p align="left">
  <img src="./Figures/results.png" width="80%">
</p>

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="download">&nbsp;</div>



## 💾 Download

The models are the **LORA** parameters for finetuning LLaMa-3-8b-Instruct. **Regeneration** means pretraining the compression model to regenerate the original text based on the compressed tokens. **QA** means finetuning the compression model for extractive QA based on the compressed tokens. **500->X** means 500 tokens in the original text are compressed into X special token.
- [ArxivQA Dataset](https://huggingface.co/datasets/ZongqianLi/ArxivQA)
- [Ours: 500xCompressor Regeneration & QA (500->16, 500->4, 500->1) (2*3 models)](https://huggingface.co/collections/ZongqianLi/500xcompressor-66b24b2db2efe5732539a3d3)
- [Baselines: ICAE Regeneration & QA (500->16, 500->4, 500->1) (2*3 models)](https://huggingface.co/collections/ZongqianLi/icae-66b250fdc40442c79e9fb88c)

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="citation">&nbsp;</div>



## 📌 Citation

```
cite
```

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="license">&nbsp;</div>



## 🔖 License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the [LICENSE]([LICENSE](https://creativecommons.org/licenses/by/4.0/deed.en)) for details.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)



