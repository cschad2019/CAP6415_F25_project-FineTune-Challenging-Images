# Fine-tuning a Small Model for Challenging Test Images (CAP6415 F25)

**Student:** Cale Schad (z23503021)  
**Project:** Individual Project #8 — *Fine-tune a small model for challenging test images*  

## Abstract
We train a small CNN (< 50 layers; ResNet-18) on CIFAR-10 to obtain a baseline, identify the lowest-precision (most challenging) class from the confusion matrix, and fine-tune the model using class-focused sampling and mild augmentations. We report before/after precision for the target class, overall accuracy, curves, and qualitative samples. Code is fully reproducible via `requirements.txt` (or `env.yml`) and deterministic seeds.

## Repository Layout
