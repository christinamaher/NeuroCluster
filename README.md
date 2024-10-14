# NeuroCluster
A Python pipeline for non-parametric cluster-based permutation testing for electrophysiological signals related to computational cognitive model variables.

**Motivation**: Time-varying, continuous latent variables from computational cognitive models enable model-based neural analyses that identify how cognitive processes are encoded in the brain, informing the development of neural-inspired algorithms. However, current statistical methods for linking these variables to electrophysiological signals are limited, which may hinder the understanding of neurocomputational mechanisms. To address this methodological limitation, we propose a multivariate linear regression that leverages non-parametric cluster-based permutation testing strategy.

*This repository is in collaboration with Alexandra Fink (co-first author, Saez Lab, ISMMS). Manuscript in preparation.* 

## Installation

```
conda create --name neurocluster_env pip requests git python=3.12.4
conda activate neurocluster_env
pip install git+https://github.com/aliefink/NeuroCluster.git
```

## Updating

```
pip install --upgrade git+pip install git+https://github.com/aliefink/NeuroCluster.git

```

## Quick Start 

```NeuroCluster_template.ipynb```: Notebook for running cluster-based permutation test to detect encoding of continuous behavioral signals/latent model parameters.

Here is a schematic of the NeuroCluster workflow: 
![neurocluster workflow](https://github.com/christinamaher/NeuroCluster/blob/main/workflow/workflow.png)
