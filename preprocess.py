#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


MIMIC_DIR = 'mimic-iii-clinical-database-1.4/'
    
# Variables used for prediction as stated in the paper
pred_vars = [
    'Anion gap',
    'Bicarbonate',
    'blood pH',
    'Blood urea nitrogen',
    'Chloride',
    'Creatinine',
    'Diastolic blood pressure',
    'Fraction inspired oxygen',
    'Glascow coma scale total',
    'Glucose',
    'Heart rate',
    'Hematocrit',
    'Hemoglobin',
    'INR',
    'Lactate',
    'Magnesium',
    'Mean blood pressure',
    'Oxygen saturation',
    'Partial thromboplastin time',
    'Phosphate',
    'Platelets',
    'Potassium',
    'Prothrombin time',
    'Respiratory rate',
    'Sodium',
    'Systolic blood pressure',
    'Temperature',
    'Weight',
    'White blood cell count',
]

# Lab tests the paper uses as predictive variables
labevents_items = [
    'Anion Gap',
    'Bicarbonate',
    'pH',
    'Urea Nitrogen',
    'Chloride',
    'Creatinine',
    'Glucose',
    'Hematocrit',
    'Hemoglobin',
    'Lactate',
    'Magnesium',
    'Oxygen Saturation',
    'Phosphate',
    'Platelet Clumps','Platelet Count','Platelet Smear',
    'Potassium',
    'Sodium',
    'WBC Count',
]

# Feature names in chartevents
chart_features = [
    'Hematocrit',
    'Hemoglobin',
    'Platelets',
    'Chloride',
    'Creatinine',
    'Glucose',
    'Magnesium',
    'Potassium',
    'Sodium',
    'Potassium',
    'Platelets',
    'Magnesium',
    'Platelets',
    'Anion gap',
    'Prothrombin time',
    'Creatinine',
    'Magnesium',
    'Hemoglobin',
    'Heart Rate',
    'INR',
    'Admission Weight (lbs.)',
    'Daily Weight',
    'Admission Weight (Kg)',
    'Respiratory Rate',
    'Heart Rate',
    'PH (Venous)',
    'BUN',
    'GCS Total',
    'Phosphorous',
    'Lactic Acid'
    'Non Invasive Blood Pressure diastolic',
    'Arterial Blood Pressure diastolic',
    'Pulmonary Artery Pressure diastolic',
    'ART BP Diastolic',
    'Manual Blood Pressure Diastolic Left',
    'Manual Blood Pressure Diastolic Right',
    'Aortic Pressure Signal - Diastolic',
    'Temperature Fahrenheit',
    'Temperature Celsius'
    'Arterial Blood Pressure systolic',
    'Pulmonary Artery Pressure systolic',
    'Non Invasive Blood Pressure systolic',
    'ART BP Systolic',
    'O2 saturation pulseoxymetry',
    'Prothrombin time',
    'WBC',
    'Inspired O2 Fraction',
    'Arterial Blood Pressure mean',
    'Non Invasive Blood Pressure mean',
    'ART BP mean',
    'Arterial CO2 Pressure',
    'TCO2 (calc) Arterial',
    'TCO2 (calc) Venous',
    'Venous CO2 Pressure',
]


def get_charts(nrows=None):
    def get_item_map():
        d_items = pd.read_csv(MIMIC_DIR + 'D_ITEMS.csv.gz')
        item_map = d_items.set_index('itemid')['label'].to_dict()
        item_map = { x[0]:x[1] for x in item_map.items()                 if isinstance(x[1],str) }
        return item_map


    # Read chartevents
    charts_types = {
        "subject_id": np.int32,
        "hadm_id": np.int32,
        "icustay": np.float32,
        "itemid": np.int32,
        "cgid": np.float32,
        # "VALUE": float,
        "valuenum": np.float32,
        "warning": np.float32,
        "error": np.float32,
    }
    charts_times = ["charttime"]
    usecols = list(charts_types.keys()) + charts_times
    # charts = pd.read_csv(MIMIC_DIR + 'CHARTEVENTS.csv.gz', 
    #         dtype=charts_types, parse_dates=charts_times, nrows=nrows,
    #         usecols=usecols)
    print('pd.read_pickle chartevents ...', flush=True)
#     charts = pd.read_pickle(MIMIC_DIR + 'CHARTEVENTS_pandas.pkl')
    charts = pd.read_csv(MIMIC_DIR + 'CHARTEVENTS.csv')
    print('done.')

    # Filter for only chart events the paper uses
#     item_map = get_item_map()
#     chart_feature_keys = [k for k,v in item_map.items() if v in chart_features]
#     charts = charts.loc[charts.ITEMID.isin(chart_feature_keys),:]
    
#     # Lower the column names to match the paper's code
#     charts.columns = [x.lower() for x in charts.columns]
   
    # Create hours_in feature
    icustays = pd.read_csv(MIMIC_DIR + 'ICUSTAYS.csv', 
            parse_dates=['intime'])
    charts = charts.join(icustays.groupby('subject_id')['intime'].min(), 
            on='subject_id')
    # Convert 'charttime' and 'intime' columns to datetime format
    charts['charttime'] = pd.to_datetime(charts['charttime'])
    charts['intime'] = pd.to_datetime(charts['intime'])

    # Calculate the time difference in hours
    charts['hours_in'] = (charts['charttime'] - charts['intime']).dt.total_seconds() / 3600

#     charts['hours_in'] = charts['hours_in'] / np.timedelta64(1, 'h')
    charts['hours_in'] = charts['hours_in'].round()
    
    # Drop rows corresponding to previous stays
    charts = charts.loc[charts.hours_in>0]
    charts.hours_in = charts.hours_in.astype(np.int32)
    
    # Drop unused and reorder
    INDEX_COLS = ['subject_id', 'icustay_id', 'hours_in', 'hadm_id']
    other_cols = [x for x in charts.columns if x not in INDEX_COLS]
    cols = INDEX_COLS + other_cols
    charts = charts[cols]

    
    # Average all measurements per subject per stay per lab test per hour
    charts = charts.groupby(
            ['subject_id','icustay_id', 'hours_in','hadm_id','itemid']).mean()
    
    # Convert column multindex to flat
    charts = charts.unstack()
    charts.columns = ['_'.join([str(y) for y in x]) for             x in charts.columns.to_flat_index()]
    charts = charts.reset_index()
    return charts


def get_labevents():
    # Read labevents
    labitems_types = {
        "subject_id": np.int32,
        "hadm_id": np.float32,
        "itemid": np.int32,
        "value": str,
        "valuenum": np.float32,
    }
    labitems_times = ["charttime"]
    usecols = list(labitems_types.keys()) + labitems_times
    labevents = pd.read_csv(MIMIC_DIR + 'LABEVENTS.csv',
            dtype=labitems_types, parse_dates=labitems_times,
            usecols=usecols)
    
    # Add labitem description
    d_labitems = pd.read_csv(MIMIC_DIR + 'D_LABITEMS.csv')
    lab_map = d_labitems.set_index('itemid')['label'].to_dict()
    lab_item_nums = [k for k,v in lab_map.items() if v in labevents_items]

    # Filter only relevant lab_items
    labevents[labevents.itemid.isin(lab_item_nums)]

    # Create hours_in column
    icustays = pd.read_csv(MIMIC_DIR + 'ICUSTAYS.csv', 
            parse_dates=['intime'])
    labevents = labevents.join(icustays.groupby('subject_id')['intime'].min(),
            on='subject_id')
    labevents['hours_in'] = labevents.charttime-labevents.intime
    labevents['hours_in'] = labevents['hours_in'] / np.timedelta64(1, 'h')
    labevents['hours_in'] = labevents['hours_in'].round()
    labevents = labevents.drop(columns=['intime', 'charttime'])

    # Drop rows corresponding to previous stays
    labevents = labevents.loc[labevents.hours_in>0]
    labevents.hours_in = labevents.hours_in.astype(np.int32)

    # Average all measurements per subject per stay per lab test per hour
    labevents = labevents.groupby(
            ['subject_id', 'hours_in','hadm_id','itemid']).mean()

    # Convert column multindex to flat
    labevents = labevents.unstack()
    labevents.columns = ['_'.join([str(y) for y in x]) for x in labevents.columns.to_flat_index()]
    labevents = labevents.reset_index()
    
    # Lower the column names to match the paper's code
    labevents.columns = [x.lower() for x in labevents.columns]
    return labevents


def get_static():
    # Read admissions data
    admissions_cols = ['ethnicity', 'deathtime', 'dischtime']
    admissions = pd.read_csv(MIMIC_DIR + 'ADMISSIONS.csv')
    admissions = admissions.set_index('subject_id')[admissions_cols]
    
    # Read patients data
    patients_cols = ['gender', 'dob']
    patients = pd.read_csv(MIMIC_DIR + 'PATIENTS.csv', parse_dates=['dob'])
    patients = patients.set_index('subject_id')[patients_cols]

    # Read icustay data
    icustays_cols = ['hadm_id', 'icustay_id', 'first_careunit', 'intime']
    icustays = pd.read_csv(MIMIC_DIR + 'ICUSTAYS.csv',
            parse_dates=['intime'])
    icustays = icustays.set_index('subject_id')[icustays_cols]

    # Join tables on subject_id
    static = icustays.join(admissions,how='left').join(patients, how='left')
    static = static.reset_index().drop_duplicates()

    # Sort by subject_id, intime, and drop all but first hospital stay
    # as stated in the paper
    static = static.sort_values(by=['subject_id', 'intime'])
    static = static.drop_duplicates(subset='subject_id', keep='first')

    # Calculate age

    # Convert 'intime' and 'dob' columns to datetime format
    static['intime'] = pd.to_datetime(static['intime'])
    static['dob'] = pd.to_datetime(static['dob'])

    # Calculate the age in years
    static['age'] = static.apply(lambda row: relativedelta(row['intime'], row['dob']).years, axis=1)

 
    # Lower the column names to match the paper's code
    static.columns = [x.lower() for x in static.columns]
    return static


def make_X(static, nrows=None):
    charts = get_charts(nrows=nrows)
    print('got charts')
    labevents = get_labevents()
    print('got labevents')

    # Join charts and labevents dataframes on subject, hours_in
    labevents = labevents.set_index(['subject_id','hours_in','hadm_id'])
    X = charts.set_index(
            ['subject_id','icustay_id', 'hours_in','hadm_id']).join(labevents)

    # Drop columns with all NA's and reset index
    X = X.dropna(axis=1,how='all').reset_index()
    
    # Convert to smaller width on some columns
    cols = ['subject_id', 'hours_in', 'hadm_id', 'icustay_id']
    for col in cols:
        X[col] = X[col].astype(np.int32)

    # Drop patients less than 15 years of age as stated in the paper
    X = X.set_index(cols).join(
            static.set_index(['subject_id','icustay_id', 'hadm_id'])['age'])
    X = X[X.age>=15].drop(columns='age')

    # Write out
    X.to_hdf('X.h5', key='X', mode='w')


if __name__ == "__main__":
    static = get_static()
    static.to_csv('static.csv', index=False)
    print('df saved.')
    #make_X(static, nrows=100_000_000)
    make_X(static)

