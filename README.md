# NeuroCluster
A Python pipeline for non-parametric cluster-based permutation testing for electrophysiological signals related to computational cognitive model variables.

**Motivation**: Time-varying, continuous latent variables from computational cognitive models enable model-based neural analyses that identify how cognitive processes are encoded in the brain, informing the development of neural-inspired algorithms. However, current statistical methods for linking these variables to electrophysiological signals are limited, which may hinder the understanding of neurocomputational mechanisms. To address this methodological limitation, we propose a multivariate linear regression that leverages non-parametric cluster-based permutation testing strategy.

This repository contains code and example data for performing NeuroCluster. 

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

```/notebooks/NeuroCluster_template.ipynb```: This Jupyter notebook uses example data stored within ```/data/``` directory to perform Neurocluster for one example electrode. 

Below is a schematic of the NeuroCluster workflow and quick summary. A more detailed step-by-step summary follows. By following these steps, researchers can identify significant time-frequency clusters and assess their statistical validity using non-parametric methods! 

![neurocluster workflow](https://github.com/christinamaher/NeuroCluster/blob/main/workflow/workflow.png)

**Summary of Workflow:** 

1. **Initialize Analysis**: Create the ```TFR_Cluster_Test``` object.

2. **Perform Regression**: Extract beta coefficients and t-statistics.

3. **Identify Clusters**: Find the largest significant clusters in the TFR data.

4. **Generate Null Distribution**: Permute data to create a null cluster distribution.

5. **Compute Significance**: Compare observed clusters against the null distribution.

6. **Visualize and Save Results**: Plot and save all key outputs for interpretation. 

# NeuroCluster Single Electrode Workflow - *detailed overview* 

## **Step 1: Create TFR_Cluster_Test Object**

```cluster_test = NeuroCluster.TFR_Cluster_Test(tfr_data, predictor_data, target_var, demo_channel, alternative='two-sided')```

**Explanation**:

**Purpose**: Initialize a ```TFR_Cluster_Test``` object.

**Inputs**:

```tfr_data```: Time-frequency representation data.

```predictor_data```: Data for the independent variable(s) (e.g., behavioral regressors).

```target_var```: Dependent variable of interest.This variable will be permuted to compute non-parametric p-value.

```demo_channel```: The channel to analyze (e.g., electrode name).

```alternative```: Specifies the type of test ('two-sided', 'greater', or 'less').

**Output**: A ```TFR_Cluster_Test``` object ready for subsequent analysis. 

## **Step 2: Run TFR Regression and Threshold t-statistics**

```betas, tstats = cluster_test.tfr_regression()```

```tstat_threshold = cluster_test.threshold_tfr_tstat(tstats)```

**Explanation:**

**```tfr_regression()```**: 

* **Purpose**: Performs a time-frequency regression analysis to compute:

* ```betas```: Beta coefficients for the predictor of interest.

* ```tstats```: t-statistics for each pixel in the time-frequency representation (TFR).

**```threshold_tfr_tstat()```**: 

* **Purpose**: Determines which t-statistics are significant based on a critical t-value.

* **Output**: A thresholded t-statistic matrix where non-significant values are removed.

## **Step 3: Identify Largest Clusters**

```max_cluster_data = cluster_test.max_tfr_cluster(tstats, max_cluster_output='all')```

**Explanation**:

* **Purpose**: Identifies the largest contiguous clusters of significant t-statistics.

* **Inputs**: 

* ```tstats```:  t-statistics matrix.

* ```max_cluster_output```: Specifies the type of output ('all' for full cluster details).

* **Output**: 

* ```max_cluster_data```: Contains the maximum cluster statistics and their corresponding time-frequency indices.

## **Step 4: Compute Null Distribution**

```null_cluster_distribution = cluster_test.compute_null_cluster_stats(num_permutations=100)```

**Explanation**:

* **Purpose**: Creates a null distribution of maximum cluster statistics by permuting the data.

* **Inputs**:

* ```num_permutations```: Number of permutations to generate the null distribution.

* **Outputs**:

* ```null_cluster_distribution```: A distribution of maximum cluster statistics under the null hypothesis.

## **Step 5: Compute non-parametric p-value**

**Explanation**

* **Purpose**: Calculates the statistical significance of the observed clusters.

* **Inputs**:

* ```max_cluster_data```: Data for the largest observed cluster(s).

* ```null_cluster_distribution```: Null distribution of cluster statistics.

* **Output**:

* ```cluster_pvalue```: Non-parametric p-value for the observed cluster(s).

## **Step 6 (optional): Generate and save plots**

```beta_plot, tstat_plot, cluster_plot, max_cluster_plot, null_distribution_plot = NeuroCluster.plot_neurocluster_results(betas, cluster_test, max_cluster_data, null_cluster_distribution, tstats, tstat_threshold, cluster_pvalue)```
    
```output_directory = f'{results_dir}/{demo_channel}_{target_var}'```

```NeuroCluster.create_directory(output_directory)```

```NeuroCluster.save_plot_to_pdf(beta_plot, output_directory, 'beta_plot.png')```

```NeuroCluster.save_plot_to_pdf(tstat_plot, output_directory, 'tstat_plot.png')```

```NeuroCluster.save_plot_to_pdf(cluster_plot, output_directory, 'cluster_plot.png')```

```NeuroCluster.save_plot_to_pdf(max_cluster_plot, output_directory, 'max_cluster_plot.png')```

```NeuroCluster.save_plot_to_pdf(null_distribution_plot, output_directory, 'null_distribution_plot.png')```

**Explanation:**

* **```plot_neurocluster_results()```:**

* **Purpose**: Generates visualizations for each step of the analysis:

* ```beta_plot```: Visualizes beta coefficients.

* ```tstat_plot```: Displays the t-statistics matrix.

* ```cluster_plot```: Shows significant clusters.

* ```max_cluster_plot```: Highlights the maximum observed cluster.

* ```null_distribution_plot```: Plots the null distribution of cluster statistics.


* **```create_directory()```:** Ensures the output directory exists.

* **```save_plot_to_pdf()```:** Saves the generated plots to the specified directory in ```.png``` format.
