# Systematic-Evaluation-of-Single-Cell-Foundation-Models

**Our article is now published on WSDM Day 2025.**

We systemly evaluate three state-of-the-art single-cell foundation models—**scGPT**, **GeneFormer**, and **scFoundation**—on cell-type classification tasks. We evaluated the models using five datasets: **Myeloid**, **human Pancreas**, and **Multiple Sclerosis**, **Covid-19**, **Lung-Kim** under standard **fine-tuning** and **few-shot** learning scenarios. We also selected two baseline models: **Linear Regression (LR)** and **XGBoost** as our benchmarks. In the extended experiments, we added training results from **Scratch** and **Zero-Shot** and tried various combinations of classification layers.

## Single Cell Foundation Models
- **[scFoundation](https://github.com/biomap-research/scFoundation/tree/main)**  learned specific gene expressions across 19,264 common human genes which is a 100M-parameter pretrained model based on xTrimoGene, trained on over 50 million human single-cell transcriptomics data.
- **[GeneFormer](https://github.com/jkobject/geneformer)** is a foundation transformer model pretrained on a large-scale corpus of ~30 million single-cell transcriptomes to enable context-aware predictions in settings with limited data in network biology.
- **[scGPT](https://github.com/bowang-lab/scGPT)** a foundation model for single-cell biology, trained on over 33 million cells using a generative pretrained transformer. scGPT extracts critical biological insights and excels in tasks like cell type annotation, multi-batch integration, and gene network inference through transfer learning.

## Key Findings

- **scFoundation** consistently outperforms the other models.
- **GeneFormer** often underperforms, sometimes yielding results worse than baseline models.
- **scFoundation** demonstrates strong generalization on out-of-distribution (OOD) data, a capability lacking in baseline models.

## Significance

Our work highlights the potential of foundation models for addressing complex biomedical challenges, especially in scenarios where models are trained on one population but deployed on another.

For further details, feel free to contact us or check back after the repository is updated.
