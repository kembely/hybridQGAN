# Hybrid QGAN

[[Dataset: CICDDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html)]
[[Dataset: CICIoMT2024](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html)]

Code for the research paper **"Unsupervised Hybrid Quantum-Classical Generative Adversarial Network-Based Intrusion Detection System for 6G-enabled Internet of Things"** by Ivette Kembely Carrera Manosalvas, Paulo Freitas de Araujo-Filho, Georges Kaddoum, and Hadi Otrok. This paper was published in IEEE TNSM: https://ieeexplore.ieee.org/document/11780743 

This repository provides the experimental pipeline, including preprocessing the CICDDoS2019 and CICIoMT2024 datasets, unsupervised adversarial training, and attack detection evaluation.

Please contact me at ivette-kembely.carrera-manosalvas.1@ens.etsmtl.ca for more information regarding our code and paper. 

---
## Requirements

python=3.10.13
numpy=1.26.4
pandas=2.2.2
scikit-learn=1.5.1
pytorch=1.12.1
matplotlib=3.8.4
scipy=1.14.1
pennylane=0.36.0
pennylane-lightning=0.36.0
optuna=3.6.1

## Scripts 

### Dataset Preprocessing

- Scripts/dataPreparation_CICDDoS2019.py
- Scripts/dataPreparation_CICIoMT2024.py

  - Loads and cleans dataset files
  - Creates and saves training, validation, and testing sets as numpy tensors (`trainingSet_40features.npy`, `validationSet_40features.npy`, and `testingSet_40features.npy).

- Scripts/datasetClasses_IoT.py

  - Normalizes and prepares data loaders for using datasets,

### Model Definition

- Scripts/qganModel.py

  - Defines the hybrid generator and classical discriminator architectures.

### Model Training

- Scripts/trainingcicddos2019.py 
- Scripts/trainingCICIoMT2024.py 

  - Train the HybridQGAN model using the benign training and validation sets.

### Evaluation & Metrics

- Scripts/qidsCICDDoS2019.py
- Scripts/qidsCICIoMT2024.py 

  - Evaluate the trained discriminator checkpoint against the testing sets.
  - Compute detection performance metrics such as accuracy, precision, recall, F1, and AUC-ROC scores and plot ROC curves (`.png` and `.pdf`).

- Scripts/qualityMetrics.py

  - Evaluate JSD, WSD, and TSTR metrics 

### Others 

- Script/utilFunctions.py

  - Auxiliary functions used in the project

## Experiments
- Saved models and results are stored in the following folder:
  - qgan_CICIoMT2024 and qgan_CICDDoS2019 folder contains:
    - checkpointModels folder with saved HQGAN models
    - IDScores folder with intrusion detection scores and inference times.
    - Results folder with model results plots
    - comparison_results folder contains the comparison plots against baselines (AUCROC and detection times)
    - trainingPlots folder with history plots during training (AUCROC, losses, etc.)

  - baselines_results
    - contains results of related works: FID-GAN, iForest, RUIDS, and LSTM/TCN/Self-attention


## Citation

If you use this codebase or find our framework helpful in your research, please cite our paper:

```bibtex
@article{hybrid_qgan,
  author={Carrera Manosalvas, Ivette Kembely and De Araujo-Filho, Paulo Freitas and Kaddoum, Georges and Otrok, Hadi},
  journal={IEEE Transactions on Network and Service Management}, 
  title={Unsupervised Hybrid Quantum-Classical Generative Adversarial Network-Based Intrusion Detection System for 6G-enabled Internet of Things}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TNSM.2026.3747752}
}
```
