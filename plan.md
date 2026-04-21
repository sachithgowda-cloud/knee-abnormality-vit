# EEEM068 — Vision Transformer Knee Abnormality Classification
## Project Plan

---

## Overview

**Module:** EEEM068 Applied Machine Learning  
**University:** University of Surrey  
**Task:** Train a ViT-Small model to classify knee MRI scans into 3 categories  
**Classes:** Normal · ACL Tear · Meniscal Tear  
**Dataset:** MRNet (Stanford, 1370 MRI exams)  
**Tools:** Python · PyTorch · Google Colab · VS Code (Mac)

---

## Marks Breakdown

| Component | Weight |
|---|---|
| Technical Report | 30% (30 marks) |
| Functionality / Code | 30% (18 marks) |
| Code Quality | 20% (12 marks) |
| Oral Exam / Viva | 40% |

---

## Phase 1 — Setup & Environment

- [ ] Create Kaggle account and download MRNet dataset (~5GB zip)
- [ ] Upload dataset to Google Drive
- [ ] Open Google Colab, set runtime to **GPU (T4)**
- [ ] Mount Google Drive in Colab
- [ ] Install dependencies

```bash
pip install timm torch torchvision tensorboard numpy matplotlib scikit-learn
```

- [ ] Download SiT-S pretrained weights from https://github.com/Sara-Ahmed/SiT
- [ ] Set up VS Code locally with Jupyter extension (for editing notebooks)

---

## Phase 2 — Data Pipeline

- [ ] Explore dataset structure (train/valid folders → axial, coronal, sagittal planes)
- [ ] Start with **sagittal plane only**
- [ ] Write custom PyTorch `Dataset` class to load `.npy` files
- [ ] Extract **2 slices per 3D volume** to create 2D images
- [ ] Formulate as **multi-class classification** (3 classes)
- [ ] Resize images from `128×128` → `224×224`
- [ ] Apply data augmentation on training set:
  - Random X and Y translations
  - Random rotations (−20° to +20°)
  - Random scaling
- [ ] Create DataLoaders with appropriate batch size
- [ ] Optionally split training set further for hyperparameter tuning

---

## Phase 3 — Model Architecture

- [ ] Load ViT-Small via `timm` library
- [ ] Load SiT-S pretrained ImageNet checkpoint
- [ ] Remove default classification head
- [ ] Add new 3-class classification head
- [ ] Set **differential learning rates**:
  - Lower LR for pretrained ViT backbone
  - Higher LR for new classification head

---

## Phase 4 — Training

- [ ] Define loss function (CrossEntropyLoss)
- [ ] Choose optimiser (AdamW recommended for ViT)
- [ ] Set up TensorBoard logging for training progress
- [ ] Train model and monitor:
  - Training loss
  - Validation loss
  - Validation accuracy
- [ ] Tune hyperparameters:
  - Batch size
  - Number of epochs
  - Learning rate (backbone vs head)
  - Weight decay
- [ ] Save best model checkpoint

---

## Phase 5 — Evaluation

- [ ] Calculate **top-1 accuracy** on validation set
- [ ] Generate **confusion matrix** (visualise clearly)
- [ ] Calculate clinical metrics:
  - Sensitivity (recall)
  - Specificity
  - F1-score
  - ROC-AUC
- [ ] Plot ROC curves per class

---

## Phase 6 — Visualisation & Interpretation

- [ ] Generate **class attention maps** (class token → image token attention)
- [ ] Overlay attention maps on MRI images
- [ ] Observe whether the model localises the correct injury region
- [ ] Test model on **publicly available internet MRI images**:
  - Search: "ACL tear sagittal plane MRI"
  - Compare performance vs MRNet validation images
  - Discuss why predictions differ

---

## Phase 7 — Report Writing (IEEE Double Column, 5 pages)

| Section | Marks | Notes |
|---|---|---|
| Abstract | 5 | Brief summary of task, method, results |
| Introduction | 5 | Clinical context, problem motivation |
| Literature Review | 5 | Minimum 5 papers — ViT, MRI classification, transfer learning |
| Methodology | 7 | Dataset, preprocessing, model, training setup |
| Experiments | 5 | Results tables, confusion matrix, attention maps, hyperparameter analysis |
| Conclusion & Future Work | 3 | Key findings, limitations, next steps |

**Additional pages (optional):** extra visualisations + references only  
**Every figure/table must be referenced** in the main text by number

---

## Phase 8 — Oral Exam / Viva Prep (40%)

- [ ] 5-minute presentation covering the full pipeline
- [ ] 15-minute Q&A — each group member questioned individually
- [ ] Make sure **everyone** can explain:
  - How the ViT works (patches, attention, class token)
  - Why you chose specific hyperparameters
  - What the attention maps show
  - Ethical considerations (dataset bias, clinical reliability)

---

## Key Resources

| Resource | Link |
|---|---|
| MRNet Dataset | https://www.kaggle.com/datasets/cjinny/mrnet-v1 |
| SiT Pretrained Weights | https://github.com/Sara-Ahmed/SiT |
| timm Library | https://github.com/huggingface/pytorch-image-models |
| Google Colab | https://colab.research.google.com |
| TensorBoard Docs | https://www.tensorflow.org/tensorboard |

---

## Ethical Considerations to Address in Report

- Dataset bias (all MRIs from Stanford Medical Center — limited demographic diversity)
- Model reliability and false negative risk in clinical settings
- Role of AI as a decision-support tool, not a replacement for clinicians
- Importance of interpretability (attention maps) for clinical trust

---

## Group Responsibilities (to divide among 4–5 members)

| Task | Owner |
|---|---|
| Data pipeline & preprocessing | |
| Model architecture & loading weights | |
| Training loop & hyperparameter tuning | |
| Evaluation metrics & confusion matrix | |
| Attention map visualisation | |
| Report writing | All |
| Presentation | All |

---

*Last updated: April 2026*