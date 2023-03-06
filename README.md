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
├── carwatch_analysis/                                      # `carwatch-analysis` Python package with helper functions
└── experiments/                                            # Folder with conducted analysis experiments; each experiment has its own subfolder
    ├── 00_general/                                         # General analysis of the dataset
    │   ├── exports/                                        # Processed data and extracted parameters
    │   ├── notebooks/                                      # Notebooks for data processing, analysis, plotting, etc. in subfolder
    │   └── results/                                        # results of analysis
    ├── 2021_car_inner_clock_bhi/                           # Analysis for the 2021 BHI Conference Paper (see below)
    │   ├── exports/
    │   ├── notebooks/
    │   │   ├── analysis/
    │   │   ├── classification/
    │   │   ├── data_cleaning/
    │   │   ├── pipeline_visualization/
    │   │   └── paper_path.json                             # json file with path to paper folder as 'pather_path' argument. used to export figures and latex tables
    │   └── results/
    │   │   ├── plots/
    │   │   └── statistics/
    └── 2023_car_sampling_pnec/                             # Analysis for the 2023 PNEC Paper (see below)
        ├── exports/
        ├── notebooks/
        │   ├── analysis/
        │   ├── data_cleaning/
        │   ├── data_processing/
        │   ├── plotting/
        │   └── paper_path.json                             # json file with path to paper folder as 'pather_path' argument. used to export figures and latex tables
        └── results/
            ├── plots/
            ├── statistics/
            └── ...
```

## Installation
### Dataset
In order to run the code, first download the CARWatch Dataset, e.g. from [OSF](https://osf.io/c35f2/). Then, create a file named `config.json` in the `/experiments` folder with the following content:
```json
{
    "base_path": "<path-to-dataset>"
}
```
This config file is parsed by all notebooks to extract the path to the dataset.   
**NOTE**: This file is ignored by git because the path to the dataset depends on the local configuration!

### Code
If you want to use this package to reproduce the analysis results then clone the repository and install the 
package via [poetry](https://python-poetry.org). For that, open a terminal and run the following commands:

```bash
git clone git@github.com:mad-lab-fau/carwatch-analysis.git
cd carwatch-analysis
poetry install # alternative: pip install .
```

This creates a new python venv in the `carwatch-analysis/.venv` folder. Next, register a new IPython kernel for the venv:
```bash
cd carwatch-analysis
poetry run poe register_ipykernel
```

Finally, go to the `experiments` folder and run the Jupyter Notebooks for the specific experiment you want to reproduce (see below).


## Experiments

### General (`00_general`)
This "experiment" contains general data processing, cleaning, and feature extraction.

#### Folder Structure
It consists of the following subfolders:
* `notebooks`:
    * `data_processing`: Notebooks to process data and extract relevant features:
        * `IMU_Feature_Extraction_Pipeline.ipynb`: Notebook for IMU feature extraction and sleep endpoint computation. Results are exported as per-subject files in the subject-specific folders of the dataset (`sleep/<subject_id>/processed`).
        * `Questionnaire_Processing.ipynb`: Notebook for extracting relevant information from questionnaire data, such as chronotype (assessed by MEQ), ideal bedtime range, and self-reported sleep endpoints (bedtime, sleep onset, wake onset, etc.).  
        * `Saliva_Processing.ipynb`: Notebook for cleaning and processing saliva data, as well as computing saliva features (area under the curve, slope, maximum increase, etc.).  
        * `Sleep_IMU_Questionnaire_Merge.ipynb`: Notebook for merging sleep endpoints (from IMU data) with chronotype and self-assessed sleep endpoints (from questionnaire data). Missing sleep endpoints (due to missing or corrupted IMU data) will be imputed by self-assessed sleep endpoints. Also, the notebook computes whether subjects went to bed within the ideal bedtime according to their chronotype (by computing whether the sensor-assessed bedtime lies within the questionnaire-assessed ideal bedtime range). The results are saved in the `CARWatch` dataset.
    * `data_cleaning`: Notebooks to clean processed data:
        * `IMU_Cleaning.ipynb`: Cleaning of sleep endpoints and static moment features.  
            **Note**: For cleaning saliva data information from the merged sleep information (`../exports/sleep_information_merged.csv`) are required.  
            **Outputs**:
            * `../exports/imu_sleep_endpoints_cleaned.csv`: Sleep endpoints computed from IMU data, such as:
                * Sleep and Wake Onset (`sleep_onset`, `wake_onset`)
                * Total Sleep Time (`total_sleep_time`)
                * Bed Interval Start and End (`major_rest_period_start`, `major_rest_period_end`)
                * Sleep Efficiency (`sleep_efficiency`)
                * Sleep Onset and Getup Latency (`sleep_onset_latency`, `getup_latency`)
                * Wake Time after Sleep Onset (`wake_after_sleep_onset`)
                * Number of Wake Bouts (`number_wake_bouts`)
            * `../exports/imu_static_moment_features_cleaned.csv`: IMU features characterizing pre-awakening movement. All features listed here are computed for two different ways of assessing wakeup time: `imu` and `selfreport` as well as for two different time intervals: for the whole night (`total`), the last hour before awakening (`last_hour`), and the last 30min before awakening (`last_30min`).
                * Number of static moments (larger than 60s) (`ss_max`, `ss_max_60`)
                * Position of the largest static moment relative to night (`ss_max_position`)
                * Mean duration of static moments (larger than 60s) (`ss_mean`, `ss_mean_60`)
                * Median duration of static moments (larger than 60s) (`ss_median`, `ss_median_60`)
                * Total number of static moments (larger than 60s) (`ss_number`, `ss_number_60`)
                * Skewness of static moments (larger than 60s) (`ss_skewness`, `ss_skewness_60`)
                * Standard deviation of static moments (larger than 60s) (`ss_std`, `ss_std_60`)
        * `Saliva_Cleaning.ipynb`: Cleaning of saliva features.  
            **Note**: For cleaning saliva data information from the merged sleep information (`../exports/sleep_information_merged.csv`) are required.  
            **Outputs**:
            * `../exports/cortisol_samples_cleaned.csv`: Raw cortisol values after cleaning
            * `../exports/cortisol_features_cleaned.csv`: Cortisol features after cleaning
    * `analysis`: Notebooks for statistical analysis of saliva and IMU/sleep data
    * `classification`: Notebooks for classification experiments, in particular for the attempt to predict the chronotype based on IMU patterns during sleep (`Chronotype_Classification.ipynb`)
* `exports`: Files exported from data processing and cleaning steps:
    * `cortisol_samples_cleaned.csv`
    * `cortisol_features_cleaned.csv`
    * `imu_sleep_endpoints_cleaned.csv`
    * `imu_static_moment_features_cleaned.csv`
* `results`: Plots and statistical results from the analysis steps


#### Notebook Processing Order
To run the data processing and analysis pipeline, we recommend the following order:
1. `data_processing`: `IMU_Feature_Extraction_Pipeline.ipynb` | `Questionnaire_Processing.ipynb` | `Saliva_Processing.ipynb` (*no specific order*)
2. `data_merge`: `Sleep_IMU_Questionnaire_Merge.ipynb` (`Sleep_IMU_Questionnaire_Merge.ipynb` requires exports from the notebooks in the `data_processing` step, so it needs to be run afterwards)
3. `data_cleaning`: `IMU_Cleaning.ipynb` | `Saliva_Cleaning.ipynb` (*no specific order*)
4. `analysis`: `General_Information.ipynb` | `IMU_Analysis.ipynb` | `Saliva_Analysis.ipynb` (*no specific order*)


### 2021 BHI Paper (`2021_car_inner_clock_bhi`)

This analysis was performed for the [paper](https://ieeexplore.ieee.org/abstract/document/9508529): Richer, R., Küderle, A., Dörr, J., Rohleder, N., & Eskofier, B. M. (2021, July). Assessing the Influence of the Inner Clock on the Cortisol Awakening Response and Pre-Awakening Movement. In *2021 IEEE EMBS International Conference on Biomedical and Health Informatics (BHI)* (pp. 1-4). IEEE.

#### Folder Structure

This experiment contains the following subfolders:
* `notebooks`: 
    * `data_cleaning`: Notebooks for cleaning saliva and IMU/sleep data
    * `analysis`: Notebooks for statistical analysis of saliva and IMU/sleep data
    * `pipeline_visualization`: Notebooks for creating plots illustrating the single data processing steps
* `exports`: Files exported from data processing and cleaning steps (*Note*: These results are the same as in the *General* analysis, but included here for consistency):
    * `cortisol_samples_cleaned.csv`
    * `cortisol_features_cleaned.csv`
    * `imu_sleep_endpoints_cleaned.csv`
    * `imu_static_moment_features_cleaned.csv`
* `results`: Plots and statistical results from the analysis steps

#### Notebook Processing Order
To run the data processing and analysis pipeline, we recommend the following order:

0. general `data_processing`: First, run the data processing (`data_processing`) and data merge (`data_merge`) notebook of the general data processing and analysis pipeline (folder `00_general`) to create the export files that are used for further cleaning and processing
1. `data_cleaning`: `IMU_Cleaning.ipynb` | `Saliva_Cleaning.ipynb` (*no specific order*)
2. `analysis`: `BHI2021_IMU.ipynb` | `BHI2021_Saliva.ipynb` (*no specific order*)



### 2023 PNEC Paper (`2023_car_sampling_pnec`)

This analysis was performed for the [paper](https://doi.org/10.1016/j.psyneuen.2023.106073): Richer, R., Abel, L., Küderle, A., Eskofier, B. M., Rohleder, N. (2023). CARWatch – A smartphone application for improving the accuracy of cortisol awakening response sampling. *Psychoneuroendocrinology*, 151, 106073.



#### Folder Structure

This experiment contains the following subfolders:
* `notebooks`: 
    * `data_processing`:
        * `Saliva_Processing_All_Reporting_Types.ipynb`: Notebook computing relevant saliva features for the different reporting types (naive, self-report, app, etc.)
    * `data_cleaning`:
        * `Cortisol_Cleaning.ipynb`: Notebook for cleaning data from missing data, outlier, etc.
    * `analysis`: Notebooks for statistical analysis of sampling time and saliva data
        * `General_Information.ipynb`: General information about the dataset reported in the paper (demographics, descriptive information, etc.)
        * `Sample_Time_Analysis.ipynb`: Analysis of sampling and awakening times for different reporting types
        * `Time_Unit_Digit_Analysis.ipynb`: Analysis of time unit digits for different reporting types
        * `Cortisol_Analysis.ipynb`: Analysis of differences in derived CAR measures for different reporting types
    * `plotting`: Notebooks to create plots used in presentations, graphical abstracts, etc.
* `exports`: Files exported from data processing and cleaning steps (*Note*: These results are the same as in the *General* analysis, but included here for consistency):
    * `cortisol_samples_cleaned.csv`
    * `cortisol_samples_processed_all_reporting_types.csv`
    * `cortisol_features_cleaned.csv`
    * `cortisol_features_processed_all_reporting_types.csv`
* `results`: Plots and statistical results from the analysis steps

#### Notebook Processing Order
To run the data processing and analysis pipeline, we recommend the following order:

0. general `data_processing`: First, run the data processing (`data_processing`) and data merge (`data_merge`) notebook of the general data processing and analysis pipeline (folder `00_general`) to create the export files that are used for further cleaning and processing
1. `data_processing`: `Saliva_Processing_All_Reporting_Types.ipynb`
2. `data_cleaning`: `Cortisol_Cleaning.ipynb`
3. `analysis`: `General_Information.ipynb` | `Sample_Time_Analysis.ipynb` | `Time_Unit_Digits_Analysis.ipynb` | `Cortisol_Analysis.ipynb` (*no specific order*)
