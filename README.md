# Introduction to Machine Learning – Coursework (CS-233, EPFL)

This repository contains two machine learning projects completed as part of the course **CS-233 – Introduction to Machine Learning** at EPFL.

The goal of these projects is to implement and compare classical ML models on a tabular dataset, and to design and train deep learning models on a medical image dataset.

---

## Repository structure

- `classical/`  
  Classical machine learning on a tabular **heart disease** dataset.
  - `main.py` – entry point for the experiments
  - `src/data.py` – data loading and preprocessing
  - `src/methods/` – implementations of k-NN, logistic regression and k-means
  - `src/utils.py` – helper functions (normalisation, metrics, etc.)

- `deep/`  
  Deep learning on the **DermaMNIST** skin-lesion dataset.
  - `main.py` – training and evaluation loop
  - `src/data.py` – data loading for DermaMNIST
  - `src/methods/deep_network.py` – MLP and CNN architectures
  - `src/utils.py` – helper functions (metrics, number of classes, etc.)

- `requirements.txt` – Python dependencies

---

## Classical ML project (heart disease)

In the `classical/` directory:

- Implementation of **k-nearest neighbours**, **logistic regression** and **k-means** *from scratch* (no high-level ML libraries).
- Full pipeline on a heart disease dataset:
  - preprocessing and normalisation,
  - train/validation/test splits,
  - hyperparameter tuning and model selection,
  - comparison of model performance and bias–variance behaviour.

This project focuses on understanding the behaviour of classical models and the effect of model complexity, regularisation and validation strategy.

---

## Deep learning project (DermaMNIST)

In the `deep/` directory:

- Implementation of **MLP** and **CNN** models using **PyTorch**.
- Training on the **DermaMNIST** skin-lesion classification dataset.
- Use of:
  - train/validation/test splits,
  - hyperparameter tuning (learning rate, architecture, regularisation),
  - analysis of overfitting and regularisation.

The goal is to compare simple fully-connected networks with convolutional architectures and to observe the impact of depth and capacity on performance.

---

## How to run

Clone the repository and install the dependencies:

```bash
git clone https://github.com/pierre-emery/introduction_ML.git
cd introduction_ML
pip install -r requirements.txt
Classical ML (heart disease)
From the root of the repository:

bash
Copy code
cd classical
python main.py    # adjust arguments if needed, see the end of main.py for options
Deep learning (DermaMNIST)
bash
Copy code
cd deep
python main.py    # adjust arguments if needed, see the end of main.py for options