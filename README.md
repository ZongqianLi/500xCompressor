# 🎓 500xCompressor: Generalized and Non-retrieval Prompt Compression for Large Language Models

<p align="center">
  <b>Content</b>
</p>

<p align="center">
  <a href="#news">🚀News</a> •
  <a href="#todo">✏️Todo</a> •
  <a href="#introduction">✨Introduction</a>
</p>

<p align="center">
  <a href="#corpus">📚Corpora</a> •
  <a href="#dataset">🤗Datasets</a> •
  <a href="#model">🤗Models</a> •
  <a href="#algorithm">🖥️Algorithm</a>
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
- **[2024.06.26]** Create the github page!

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="todo">&nbsp;</div>



## ✏️ Todo
- [x] Create the github page!

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="introduction">&nbsp;</div>



## ✨ Introduction
Prompt compression is crucial for enhancing inference speed, reducing costs, and improving user experience. However, current methods face challenges such as low compression ratios and potential data leakage during evaluation. To address these issues, we propose 500xCompressor, a method that compresses extensive natural language contexts into a minimum of one single special token. The 500xCompressor introduces approximately 0.3% additional parameters and achieves compression ratios ranging from 6x to 480x. It is designed to compress any text, answer various types of questions, and could be utilized by the original large language model (LLM) without requiring fine-tuning. Initially, 500xCompressor was pretrained on the Arxiv Corpus, followed by fine-tuning on the ArxivQA dataset, and subsequently evaluated on strictly unseen and classical question answering (QA) datasets. The results demonstrate that the LLM retained 62.26-72.89% of its capabilities compared to using non-compressed prompts. This study also shows that not all the compressed tokens are equally utilized and that K V values have significant advantages over embeddings in preserving information at high compression ratios. The highly compressive nature of natural language prompts, even for fine-grained complex information, suggests promising potential for future applications and further research into developing a new LLM language.

<p align="left">
  <img src="./Figures/cover_figure_2.png" width="30%">
</p>

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="corpus">&nbsp;</div>



## 📚 Arxiv Corpus

This is a collections of Arxiv abstracts.

- **Train:** 2353924 items, based on Arxiv abstracts before 07/2023
- **Validation:** 3000 items, based on Arxiv abstracts during 01-04/2024
- **Test:** 2500 items, based on Arxiv abstracts during 01-04/2024

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="dataset">&nbsp;</div>



## 🤗 ArxivQA Dataset

This is an extractive QA dataset created based on the abstracts of Arxiv papers.

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

### Quick use for QA:
```
# codes
```
```
# output
```

### Training process: 

The compression model was pretrained on the **Arxiv Corpus** for **regenerating** the original text according to the compressed tokens. Then, it was finetuned on the **ArxivQA Dataset** for **answering the questions** based on the compressed tokens.

<p align="left">
  <img src="./Figures/mechanism.png" width="100%">
</p>

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
```
license
```

