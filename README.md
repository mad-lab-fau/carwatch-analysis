# carwatch-analysis

[![PyPI](https://img.shields.io/pypi/v/carwatch-analysis)](https://pypi.org/project/carwatch-analysis/)
![GitHub](https://img.shields.io/github/license/mad-lab-fau/carwatch-analysis)
[![Lint](https://github.com/mad-lab-fau/carwatch-analysis/actions/workflows/lint.yml/badge.svg)](https://github.com/mad-lab-fau/carwatch-analysis/actions/workflows/lint.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/mad-lab-fau/carwatch-analysis)

A python project for the analysis of data from the *CARWatch* study, aimed to assess the influence of the inner clock on the Cortisol Awakening Response.

This package contains various helper functions to work with the dataset (including [`tpcp`](https://github.com/mad-lab-fau/tpcp) `Dataset` representations) and to process data. Additionally, it contains different analysis experiments performed with the dataset.

The repository is structured as follows:


## Repository Structure

The repository is structured as follows:

```bash
├── carwatch_analysis/                                      # carwatch-analysis Python package
└── experiments/                                            # Folder with conducted analysis experiments; each experiment has its own subfolder
    ├── 2021_bhi/                                           # Analysis for the 2021 BHI Conference Paper (see below)
    │   ├── data/                                           # Processed data and extracted parameters
    │   ├── notebooks/                                      # Notebooks for data processing, analysis and plotting
    │   │   ├── analysis/
    │   │   │   ├── bhi_2021/
    │   │   │   │   ├── BHI2021_IMU.ipynb
    │   │   │   │   └── BHI2021_Saliva.ipynb
    │   │   │   ├── madconf_2021_02/
    │   │   │   │   └── Plots_MaD_Conf_2021_02.ipynb
    │   │   │   ├── IMU_Analysis.ipynb
    │   │   │   └── Saliva_Analysis.ipynb
    │   │   ├── classification/
    │   │   │   └── Chronotype_Classification.ipynb         # Classification experiments to classify chronotype based on movements during sleep extracted from IMU data
    │   │   ├── data_processing/
    │   │   │   ├── IMU_Feature_Extraction_Pipeline.ipynb
    │   │   │   ├── Questionnaire_Processing.ipynb
    │   │   │   ├── Saliva_Processing.ipynb
    │   │   │   └── Sleep_IMU_Questionnaire_Merge.ipynb
    │   │   └── pipeline_visualization/
    │   │       └── Pipeline_Visualization.ipynb
    │   ├── exports/                                        # Exported features, per subject, as well as merged and cleaned
    │   │   ├── features/
    │   │   ├── sleep_endpoints/
    │   └── results/                                        # Notebooks for data processing, analysis and plotting
    └── 2022_pnec/                                           # Analysis for the 2022 PNEC Paper (see below)

```

* `carwatch_analysis`: Subfolder containing Python library with helper functions specific for this repository
* `notebooks`: Subfolder containing Jupyter Notebooks used for data processing, merging of data sources, statistical analysis and plot creation. The folder consists of the following subfolders:
    * `analysis`: Notebooks for statistical analysis of saliva and sleep data:
        * `Saliva_Analysis.ipynb` and `Sleep_Analysis.ipynb`: General analysis notebooks
        * `mad_conf_2021_02`: Folder with notebooks used to create results and plots (example plots as well as result plots) for the 2021 Spring MaD Conf (16.02.2021)
        * `bhi_2021`: Folder with notebooks used to create results and plots paper "Assessing the Influence of the Inner Clock on the Cortisol Awakening Response and Pre-Awakening Movement", presented at BHI 2021
    * `classification`: Notebooks for classification tasks, e.g. for the attempt to predict the chronotype based on IMU patterns during sleep (`Chronotype_Classification.ipynb`).
    * `data_processing`: Notebooks to clean, process and extract relevant data:
        * `IMU_Feature_Extraction_Pipeline.ipynb`: Notebook for IMU feature extraction and sleep endpoint computation. Results are exported as per-subject files in the folders `../exports/features` and `../exports/sleep_endpoints`. Concatenated results are exported to the following files:  
        **Outputs**:
            * `../exports/imu_features_complete.csv`: IMU features characterizing pre-awakening movement. All features listed here are computed for two different ways of assessing wakeup time: `imu` and `selfreport` as well as for two different time intervals: for the whole night (`total`), the last hour before awakening (`last_hour`), and the last 30min before awakening (`last_30min`).
                * Number of static moments (larger than 60s) (`ss_max`, `ss_max_60`)
                * Position of the largest static moment relative to night (`ss_max_position`)
                * Mean duration of static moments (larger than 60s) (`ss_mean`, `ss_mean_60`)
                * Median duration of static moments (larger than 60s) (`ss_median`, `ss_median_60`)
                * Total number of static moments (larger than 60s) (`ss_number`, `ss_number_60`)
                * Skewness of static moments (larger than 60s) (`ss_skewness`, `ss_skewness_60`)
                * Standard deviation of static moments (larger than 60s) (`ss_std`, `ss_std_60`)
            * `../exports/imu_sleep_endpoints_complete.csv`: Sleep endpoints computed from IMU data, such as:
                * Sleep and Wake Onset (`sleep_onset`, `wake_onset`)
                * Total Sleep Time (`total_sleep_time`)
                * Bed Interval Start and End (`major_rest_period_start`, `major_rest_period_end`)
                * Sleep Efficiency (`sleep_efficiency`)
                * Sleep Onset and Getup Latency (`sleep_onset_latency`, `getup_latency`)
                * Wake Time after Sleep Onset (`wake_after_sleep_onset`)
                * Number of Wake Bouts (`number_wake_bouts`)
        * `Questionnaire_Processing.ipynb`: Notebook for extracting relevant information from questionnaire data, such as chronotype (assessed by MEQ), ideal bedtime range, and self-reported sleep endpoints (bedtime, sleep onset, wake onset, etc.).  
        **Output**:
            * `../exports/questionnaire_chronotype_bedtimes.csv`
        * `Sleep_IMU_Questionnaire_Merge.ipynb`: Notebook for merging sleep endpoints (from IMU data) with chronotype and self-assessed sleep endpoints (from questionnaire data). Missing sleep endpoints (due to missing or corrupted IMU data) will be imputed by self-assessed sleep endpoints. Also, the notebook computes whether subjects went to bed within the ideal bedtime according to their chronotype (by computing whether the sensor-assessed bedtime lies within the questionnaire-assessed ideal bedtime range).  
        **Note**: For merging data the two files `../exports/imu_sleep_endpoints_complete.csv` and `../exports/questionnaire_chronotype_bedtimes.csv` are required.  
            **Output**:
            * `../exports/imu_questionnaire_merged.csv`
        * `Saliva_Processing.ipynb`: Notebook for cleaning and processing saliva data, as well as computing saliva features (area under the curve, slope, maximum increase, etc.).  
            **Note**: For cleaning saliva data information from the merged sleep endpoints (`../exports/imu_questionnaire_merged.csv`) are required.  
            **Output**:
            * `../exports/cortisol_features_cleaned.csv`
            * `../exports/cortisol_samples_cleaned.csv`
    * `playgrounds`: Notebooks used for prototyping.

* `exports`: Exported data, plots, and intermediate processing results. The folder consists of the following subfolders:
    * `features`: Extracted IMU features - exported as one file per subject
    * `sleep_endpoints`: Computed sleep endpoints - exported as one file per subject  
    * `sleep_plots`: Per-subject and -night IMU plots for visual inspection of correct sleep endpoint computation  
    *Note:* These files are ignored by `.gitignore` and thus not included in the repository.
* `results`: Result plots and statistical results from the notebooks in the `analysis` folder, stored in different subfolders.
