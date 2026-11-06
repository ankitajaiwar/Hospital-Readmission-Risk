"""Column definitions for the UCI 'Diabetes 130-US Hospitals (1999-2008)' dataset."""

ID_COLS = ["encounter_id", "patient_nbr"]

MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]

CATEGORICAL_COLS = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "payer_code", "medical_specialty", "max_glu_serum",
    "A1Cresult", "change", "diabetesMed", "diag_1", "diag_2", "diag_3",
]

TARGET_RAW = "readmitted"          # values: "NO", ">30", "<30"
TARGET = "readmit_30d"             # 1 if "<30"

REQUIRED_COLS = ID_COLS + NUMERIC_COLS + CATEGORICAL_COLS + MED_COLS + [TARGET_RAW]

# Discharge dispositions after which a readmission cannot happen
# (expired or discharged to hospice). Standard exclusion in readmission studies.
EXCLUDED_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

AGE_BANDS = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
             "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]

# Attributes used for the subgroup (fairness) audit. They are NOT model features.
AUDIT_COLS = ["race", "gender", "age"]
