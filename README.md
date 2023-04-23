## Reproduction of Learning Tasks for Multitask Learning: Heterogeneous Patient Populations in the ICU

The code in this repository reproduces the models described in the paper *Learning Tasks for Multitask Learning: Heterogenous Patient Populations in the ICU* (KDD 2018). https://doi.org/10.1145/3219819.3219930

### File Description

In this repository, there are four files to run: 

1. SAPII & Code_status.ipynb, which extracts data from the mimic-iii database to create the code_status.csv and saps.csv files and can be used for data exploration.

2. preprocess.py, which pre-processes the data and creates the X.h5 and static.csv files. 

3. generate_clusters.py, which trains a sequence-to-sequence autoencoder on patient timeseries data to produce a dense representation, and then fits a Gaussian Mixture Model to the samples in this new space. 

4. run_mortality_prediction.py, which contains methods to preprocess data, as well as train and run a predictive model to predict in-hospital mortality after a certain point, given patients' physiological timeseries data. 

### Instructions

Python package dependencies:
```
os
sys
argparse
numpy
pandas
tensorflow
keras
sklearn
pickle
```

#### Data Download

1. Download MIMIC-III Clinical Database v1.4 from https://physionet.org/content/mimiciii/1.4/ and unzip the folder at the top level of this repository.

#### Pre-processing

1. Download the SAPII & Code_status.ipynb file as a .py file to create the 'saps.csv' and 'code_status.csv files'. Place the generated files in the 'data/' folder at the top level of this repository. The query is sourced from https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii. 

2. Run the preprocess.py file to create the 'X.h5' and 'static.csv' files, which will be outputted into the 'data/' folder using the following command via the terminal in the folder where this repository is cloned. 

    ```
    python preprocess.py
    ```

#### Cohort Discovery
1. Run the generate_cluster.py file to discover the patient cohorts using the following command via the terminal in the folder where this repository is cloned.

     ```
    python generate_cluster.py
    ```

#### Models
1. Run the python run_mortality_prediction.py via the terminal in the folder where this repository is cloned. The model ran can be specified by command line arguments --model_type {'GLOBAL', 'MULTITASK'}.
2. Define cohort type by specifying '--cohorts {'careunit', 'custom'}'.Indicates whether to use original cart units or clusters from GMM model for prediction.
3. Specify cluster file names in argument '--cohort_filepath'.
    ```
    python run_mortality_prediction.py
    ```
 

### Data

Without any modification, this code assumes that you have the following files in a 'data/' folder: 
1. X.h5: an hdf file containing one row per patient per hour. Each row should include the columns {'subject_id', 'icustay_id', 'hours_in', 'hadm_id'} along with any additional features.
2. static.csv: a CSV file containing one row per patient. Should include {'subject_id', 'hadm_id', 'icustay_id', 'gender', 'age', 'ethnicity', 'first_careunit'}.
3. saps.csv: a CSV file containing one row per patient. Should include {'subject_id', 'hadm_id', 'icustay_id', 'sapsii'}. This data is found in the saps table in MIMIC III.
4. code_status.csv: a CSV file containing one row per patient. Should include {'subject_id', 'hadm_id', 'icustay_id', 'timecmo_chart', 'timecmo_nursingnote'}. This data is found in the code_status table of MIMIC III.

### References

Original source code:
- https://github.com/mit-ddig/multitask-patients

Data Preprocessing Reference code: 
- https://github.com/DLH2022-Team39/multitask-patients
- https://github.com/MLforHealth/MIMIC_Extract
- https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii
