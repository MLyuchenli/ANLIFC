# ANLIFC
Core Implementation of "Anti-Noisy-Labeling AUC Maximization Learning for Fuzzy Classification on Imbalanced Data"

This repository provides a **reference MATLAB implementation** of the training
procedure for the ANLI-FC classifier described in:

> [Anti-Noisy-Labeling AUC Maximization Learning for Fuzzy Classification on Imbalanced Data]

The goal is to clarify how the β-weighted MAUC surrogate loss is implemented
and how it is optimized over the fuzzy rule consequents and antecedents.

## Contents

- `fitANLIFC.m`  
  Core training function implementing the β-weighted MAUC surrogate with
  one-vs-one class pairs and four components (correct / half-correct / wrong
  terms), optimized with an gradient-based update.

## Usage

In MATLAB:

```matlab
X = randn(200, 10);        % features (N x D)
Y = randi([1,3], 200, 1);  % labels in {1,...,M}

options = struct();
options.beta = 0.10;
options.numRules = 20;

model = fitANLIFC(X, Y, options.numRules, options);
