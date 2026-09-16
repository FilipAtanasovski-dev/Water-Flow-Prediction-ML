# Hydraulic Throughput Prediction with CNNs

A convolutional neural network for predicting the **hydraulic throughput of porous materials** from 40×40 binary image representations.

This project was developed as part of a Machine Learning in Science assignment at **TU/e (Eindhoven University of Technology)** and demonstrates the application of deep learning to a physics-based prediction problem.

## Overview

The input consists of 40×40 binary maps representing porous materials:

* `0` → impermeable to water
* `1` → permeable to water

The model learns to predict the corresponding hydraulic throughput from these spatial patterns.

The dataset contains **1,660 samples**, with each sample represented as a 40×40 image.

## What I Worked With

* Python
* PyTorch
* Convolutional Neural Networks (CNNs)
* NumPy
* Matplotlib
* GPU-accelerated training
* Regression
* Model validation and early stopping
* Data augmentation
* Symmetry-aware machine learning
* Scientific machine learning

## Key Concepts

### Convolutional Neural Network

A CNN was used because the input represents a spatial structure rather than a conventional tabular dataset. Convolutional layers allow the model to learn local and higher-level spatial patterns within the porous material.

### Physical Scaling

The input represents areas (`m²`), while hydraulic throughput is represented in `m⁴`. To make the learning problem more homogeneous, the training targets were transformed using a square-root scaling before training.

### Symmetry

The problem contains **8 relevant spatial symmetries**, consisting of rotations and reflections. These transformations preserve the underlying physical structure and therefore should not change the predicted hydraulic throughput.

The project explored several ways of incorporating these symmetries:

* Data augmentation
* Disambiguation
* Hard-wiring symmetries into the network

Hard-wiring applies the symmetry transformations to an input, evaluates the variants using the same network, and averages the resulting predictions.

## Training

The model was trained using:

* **100 maximum epochs**
* Learning rate: `5 × 10⁻⁵`
* RMSPE-based loss
* GPU acceleration
* Early stopping
* Separate training and validation datasets

The training procedure keeps the model state corresponding to the best validation loss and stops when validation performance no longer improves.

## Evaluation

Training and validation losses were monitored throughout training to evaluate generalization and identify potential overfitting.

The final model was evaluated independently on both the training and validation datasets.

The resulting model achieved a relatively low RMSPE, while the comparison between training and validation performance indicated no substantial overfitting according to the project's evaluation.

## Why This Project Matters

This project demonstrates how I approach machine learning problems beyond simply training a model:

**Physical problem → mathematical formulation → symmetry analysis → data preparation → model design → GPU training → validation → evaluation**

It combines software implementation with mathematical and Physics reasoning, particularly in situations where domain knowledge can be incorporated directly into a machine-learning system.

## Project Structure

```text
.
├── README.md
├── ...
└── ...
```

The repository contains the implementation used to preprocess the data, construct the CNN, train the model, and evaluate its performance.

## Skills Demonstrated

**Machine Learning**

* Neural network design
* CNNs
* Regression
* Loss-function selection
* Validation
* Early stopping
* Generalization analysis

**Scientific Computing**

* Translating physical properties into ML constraints
* Dimensional analysis
* Exploiting problem symmetries
* Physics-informed reasoning

**Software & Tools**

* Python
* PyTorch
* NumPy
* Matplotlib
* GPU computing

## Academic Context

Developed at **Eindhoven University of Technology (TU/e)** as part of *Machine Learning in Science*.

Group project by:

* Gabriel Leite Savegnago
* Robin Chung
* Filip Atanasovski

The implementation and report were developed collaboratively.
