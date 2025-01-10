# Systematic-Evaluation-of-Single-Cell-Foundation-Models

**Our article is now published on WSDM Day 2025.**

We systemly evaluate three state-of-the-art single-cell foundation models—**scGPT**, **GeneFormer**, and **scFoundation**—on cell-type classification tasks. We evaluated the models using five datasets: **Myeloid (Mye.)**, **hPancreas**, and **Multiple Sclerosis (M.S.)**, **Covid-19**, **Lung-Kim** under standard **Fine-tuning** and **Few-shot** learning scenarios. We also selected two baseline models: **Linear Regression (LR)** and **XGBoost** as our benchmarks. 

In the extended experiments, we added training results from **Scratch** and **Zero-Shot** and tried various combinations of classification layers.

## Single Cell Foundation Models

- **[scGPT](https://github.com/bowang-lab/scGPT)** a foundation model for single-cell biology, trained on over 33 million cells using a generative pretrained transformer. scGPT extracts critical biological insights and excels in tasks like cell type annotation, multi-batch integration, and gene network inference through transfer learning.
- **[scFoundation](https://github.com/biomap-research/scFoundation/tree/main)**  learned specific gene expressions across 19,264 common human genes which is a 100M-parameter pretrained model based on xTrimoGene, trained on over 50 million human single-cell transcriptomics data
- **[GeneFormer](https://github.com/jkobject/geneformer)** is a foundation transformer model pretrained on a large-scale corpus of ~30 million single-cell transcriptomes to enable context-aware predictions in settings with limited data in network biology.

## Model Comparison

| **Aspect**               | **scGPT**                                | **GeneFormer**                            | **scFoundation**                           |
|---------------------------|------------------------------------------|-------------------------------------------|--------------------------------------------|
| **Architecture**          | GPT-style decoder-only transformer      | BERT-style encoder-only transformer       | Transformer Encoder-Decoder based xTrimoGene |
| **Model Size**            | 85M parameters                          | 220M parameters                           | 119M parameters                            |
| **Input Format**          | Gene expression counts as tokens        | Gene expression values mapped to vocabulary tokens | Raw continuous value of gene expression counts |
| **Pretraining Data**      | 10M human cells across multiple tissues | 5.5M human cells from GTEx                | 50M human cells (heart, liver, etc.)       |
| **Training Objective**    | Gene expression value prediction from gene-prompt and cell-prompt | Masked gene prediction                    | Next-gene prediction with metadata prediction |
| **Vocabulary**            | 60K tokens                              | 29K tokens                                | 963B tokens                                |
| **Context Length**        | 1,200 tokens                            | 2,048 tokens                              | 19,266 genes                               |

## Data Availability

As provided by the scGPT authors:
- **Multiple Sclerosis (M.S.) dataset**: [[Download Link](https://drive.google.com/drive/folders/1Qd42YNabzyr2pWt9xoY4cVMTAxsNBt4v)]
- **Myeloid (Mye.) dataset**: [[Download Link](https://drive.google.com/drive/folders/1VbpApQufZq8efFGakW3y8QDDpY9MBoDS)]
- **hPancreas dataset**: [[Download Link](https://drive.google.com/drive/folders/1s9XjcSiPC-FYV3VeHrEa7SeZetrthQVV)]

Extended experiments
- **Covid-19**: [[Download Link](https://drive.google.com/drive/folders/1jSPoPunGQOmd71vDsK0FS7UvmDhGdhQS)]
- **Lung-Kim**: [[Download Link](https://drive.google.com/drive/folders/1gbfO7VqxCOkfzgHAih6hO88zFv6pd8wO)]


## Key Findings

- **scFoundation** consistently outperforms the other models.
- **GeneFormer** often underperforms, sometimes yielding results worse than baseline models.
- **scFoundation** demonstrates strong generalization on out-of-distribution (OOD) data, a capability lacking in baseline models.

## Significance

Our work highlights the potential of foundation models for addressing complex biomedical challenges, especially in scenarios where models are trained on one population but deployed on another.

For further details, feel free to contact us or check back after the repository is updated.
