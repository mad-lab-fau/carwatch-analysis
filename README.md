# CARWatch Analysis

This repository contains code for the analysis data collected in the *CARWatch* study, aimed to assess the influence of the inner clock on the Cortisol Awakening Response.

The repository is structured as follows:

* `carwatch_analysis`: Subfolder containing Python library with helper functions specific for this repository
* `notebooks`: Subfolder containing Jupyter Notebooks used for data processing, merging of data sources, statistical analysis and plot creation. The folder consists of the following subfolders:
    * `analysis`: Notebooks for statistical analysis of saliva and sleep data (`Saliva_Analysis.ipynb` and `Sleep_Analysis.ipynb`)
    * `classification`: Notebooks for classification tasks, e.g. for the attempt to predict the chronotype based on IMU patterns during sleep (`Chronotype_Classification.ipynb`).
    * `data_processing`: Notebooks to clean, process and extract relevant data:
        * `IMU_Feature_Extraction_Pipeline.ipynb`: Notebook for IMU feature extraction and sleep endpoint computation. Results are exported as per-subject files in the folders `../exports/features` and `../exports/sleep_endpoints`. Concatenated results are exported to the following files:  
            **Outputs**:
            * `../exports/imu_features_complete.csv`
            * `../exports/imu_sleep_endpoints_complete.csv`
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
    * `mad_conf_2021_02`: Notebooks used to create reults and plots (example plots as well as result plots) for the 2021 Spring MaD Conf (16.02.2021)

* `exports`: Exported data, plots, and intermediate processing results. The folder consists of the following subfolders:
    * `features`: Extracted IMU features - exported as one file per subject  
        *Note:* These files are ignored by `.gitignore` and thus not included in the repository since they are only intermediate processing results. The concatenated results of all subjects are stored in the file `imu_features_complete.csv`.
    * `sleep_endpoints`: Compute sleep endpoints - exported as one file per subject  
        *Note:* These files are ignored by `.gitignore` and thus not included in the repository since they are only intermediate processing results. The concatenated results of all subjects are stored in the file `imu_sleep_endpoints_complete.csv`.
    * `sleep_plots`: Per-subject and -night IMU plots for visual inspection of correct sleep endpoint computation; </br> *Note:* These files are ignored by `.gitignore` and thus not included in the repository.
    * `plots`: Result plots exported by the analysis Notebooks
