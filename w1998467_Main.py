import pandas as pd

#reading the patients and admissions csv
patients = pd.read_csv("patients.csv/patients.csv")
admissions = pd.read_csv("admissions.csv/admissions.csv")

patients = patients[["subject_id", "gender", "anchor_age"]]
admissions = admissions[["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"]]

print(patients.head(10))
print(admissions.head(10))

#merging the patients.csv and admission.csv based on subject_id
df = admissions.merge(patients, on="subject_id", how="left")
print(df.head())

#reading the d_icd_diagnoses csv to identify MI related ICD codes
d_icd = pd.read_csv("d_icd_diagnoses.csv/d_icd_diagnoses.csv", dtype={"icd_code": "string"})

#filtering only MI-related ICD-9 and ICD-10 codes
icd9_mi = d_icd[(d_icd["icd_version"] == 9) & (d_icd["icd_code"].str.startswith("410")) & (d_icd["icd_code"].str.len() == 5)]
icd10_mi = d_icd[(d_icd["icd_version"] == 10) &(d_icd["icd_code"].str.startswith(("I21", "I22")))]

#combining ICD-9 and ICD-10 MI codes into a single list
MI_ICD_Codes = pd.concat([icd9_mi, icd10_mi])["icd_code"].unique().tolist()
print("Myocardial Infarction ICD codes:", MI_ICD_Codes)

#reading the diagnoses_icd zip file in chunks
diagnoses_icd = "diagnoses_icd.csv.gz"
mi_rows = [] #list to store MI related diagnosis rows from each chunk

for chunk in pd.read_csv(
    diagnoses_icd,
    compression="gzip",
    chunksize=200_000,  #200,000 records at a time
    dtype={"icd_code": "string"}
):
    #filtering and storing only MI related rows
    mi_chunk = chunk[chunk["icd_code"].isin(MI_ICD_Codes)]
    mi_rows.append(mi_chunk)

#combining all MI related chunks to a single dataframe
mi_diagnoses = pd.concat(mi_rows, ignore_index=True)

print(mi_diagnoses.head())
print("MI diagnosis rows:", mi_diagnoses.shape[0])
print("Unique patients:", mi_diagnoses["subject_id"].nunique())
print("Unique admissions:", mi_diagnoses["hadm_id"].nunique())
print(mi_diagnoses.shape)

MI_finaldf = df.merge(
    mi_diagnoses,
    on=["subject_id", "hadm_id"],
    how="inner"
)

print(MI_finaldf.head())
print(MI_finaldf.isna().sum())
print(MI_finaldf.shape)

#below is to print - remove # when needed if not it will print everytime I run it
#mi_diagnoses.to_csv("mi_diagnoses.csv", index=False)
#MI_finaldf.to_csv("mi_combined.csv", index=False)

#######################################################################################################
import numpy as np

#removing duplicated rows/admissions - for every diagnosis per patient, a row is created - have to remove
rows_before = MI_finaldf.shape[0]

#counting unique admission records #can even use "nunique" as before
unique_adm_before = MI_finaldf[["subject_id", "hadm_id"]].drop_duplicates().shape[0]

print("\nBEFORE")
print("Rows:", rows_before)
print("Unique admissions:", unique_adm_before)

#dropping duplicate admission rows
MI_finaldf = MI_finaldf.drop_duplicates(subset=["subject_id", "hadm_id"]).reset_index(drop=True)

rows_after = MI_finaldf.shape[0]
unique_adm_after = MI_finaldf[["subject_id", "hadm_id"]].drop_duplicates().shape[0]

print("\nAFTER")
print("Rows:", rows_after)
print("Unique admissions:", unique_adm_after)


#converting admission and discharge time to datetime format
MI_finaldf["admittime"] = pd.to_datetime(MI_finaldf["admittime"])
MI_finaldf["dischtime"] = pd.to_datetime(MI_finaldf["dischtime"])

#calculating length of stay (LOS) in days
MI_finaldf["los_days"] = (
    (MI_finaldf["dischtime"] - MI_finaldf["admittime"])
    .dt.total_seconds() / (60 * 60 * 24)
)
MI_finaldf["los_days"] = MI_finaldf["los_days"].round(1)

#creating LOS categories:
MI_finaldf["los_cat"] = np.where(
    MI_finaldf["los_days"] < 7,
    "< 7 days",
    "≥ 7 days"
)
print(MI_finaldf[["admittime", "dischtime", "los_days", "los_cat"]].head())

print("\n",MI_finaldf["anchor_age"].describe())

#creating age group bands
MI_finaldf["age_group"] = pd.cut(
    MI_finaldf["anchor_age"],
    bins=[18, 45, 60, 74, 120],
    labels=["18–45", "46–60", "61–74", "75+"],
    right=True,
    include_lowest=True
)

# Check age group distribution
print("n/", MI_finaldf["age_group"].value_counts(dropna=False))

#quick check
print(MI_finaldf.head(10))


#creating day-of-week variable from admission time (Monday=0, Sunday=6) - check
MI_finaldf["admit_dayofweek"] = MI_finaldf["admittime"].dt.dayofweek

#creating weekday vs weekend tag
MI_finaldf["admit_weekend"] = MI_finaldf["admit_dayofweek"].apply(
    lambda x: "Weekend" if x >= 5 else "Weekday"
)

#weekday vs weekend count check
print(MI_finaldf["admit_weekend"].value_counts())

#quick check with actual day names
tmp = MI_finaldf[["admittime"]].copy()
tmp["day_name"] = MI_finaldf["admittime"].dt.day_name()
tmp["dayofweek"] = MI_finaldf["admittime"].dt.dayofweek
print("\n",tmp.head(15))


print("\n", MI_finaldf["admission_type"].value_counts())

#removing extra spaces in admission type
MI_finaldf["admission_type_clean"] = (
    MI_finaldf["admission_type"].str.strip()
)

#Emergency/Unplanned and Non-Emergency/Planned admission types
emergency_types = [
    "URGENT",
    "EW EMER.",
    "DIRECT EMER.",
    "EU OBSERVATION",
    "OBSERVATION ADMIT",
    "DIRECT OBSERVATION",
    "AMBULATORY OBSERVATION"
]

planned_types = [
    "ELECTIVE",
    "SURGICAL SAME DAY ADMISSION"
]

MI_finaldf["admission_group"] = MI_finaldf["admission_type_clean"].apply(
    lambda x: "Emergency / Unplanned"
    if x in emergency_types
    else "Non-Emergency / Planned"
)

print(MI_finaldf["admission_group"].value_counts())

#quick check
print(MI_finaldf[["admission_type", "admission_group"]].drop_duplicates().sort_values("admission_type"))


#checking for prior MI admissions
MI_finaldf = MI_finaldf.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
MI_finaldf["prior_mi"] = MI_finaldf.groupby("subject_id").cumcount().gt(0).map({True: "Y", False: "N"})

#Quick check
print(MI_finaldf["prior_mi"].value_counts(dropna=False))
print(MI_finaldf.head())





#Other Diagnoses Flag
dicdOther = pd.read_csv("d_icd_diagnoses.csv/d_icd_diagnoses.csv")
dicdOther["long_title_otherD"] = dicdOther["long_title"].str.lower()

#retrieving keywords to get other diagnoses affecting MI
include_otherD = [
    "smoking", "tobacco",
    "hypertension",
    "diabetes",
    "obesity", "overweight",
    "hyperlipidemia",
    "chronic kidney disease"
]
exclude_otherD = ["screen", "food", "pregnan"]

#filtering relevant ICD codes affecting MI
candidate_dic = dicdOther[
    dicdOther["long_title_otherD"].str.contains("|".join(include_otherD), na=False) &
    ~dicdOther["long_title_otherD"].str.contains("|".join(exclude_otherD), na=False)
]

#separating ICD9 and ICD10 versions
code_key_set = set(candidate_dic["icd_version"].astype(str) + "|" + candidate_dic["icd_code"].astype(str))

#restricting to MI admissions only
mi_hadm_ids = set(MI_finaldf["hadm_id"])
hadm_to_codes = {}

#reading diagnoses file in chunks to identify other diagnoses
for chunk in pd.read_csv(
    "diagnoses_icd.csv.gz",
    compression="gzip",
    chunksize=200_000,
    usecols=["hadm_id", "icd_code", "icd_version"],
    dtype={"icd_code": "string"}
):
    chunk = chunk[chunk["hadm_id"].isin(mi_hadm_ids)]
    if chunk.empty:
        continue

    chunk["code_key"] = chunk["icd_version"].astype(str) + "|" + chunk["icd_code"].astype(str)
    chunk = chunk[chunk["code_key"].isin(code_key_set)]
    if chunk.empty:
        continue

    for hadm_id, keys in chunk.groupby("hadm_id")["code_key"]:
        hadm_to_codes.setdefault(hadm_id, set()).update(keys)

#adding the variable as a column to the table
MI_finaldf["other_dn_codes"] = MI_finaldf["hadm_id"].map(
    lambda x: ",".join(sorted(hadm_to_codes[x])) if x in hadm_to_codes else np.nan
)
MI_finaldf["other_diag"] = np.where(MI_finaldf["other_dn_codes"].notna(), "Y", "N")

#quick check
print("\n",MI_finaldf["other_diag"].value_counts())


#sorting by patient and admission time and getting the next admission time 
MI_finaldf = MI_finaldf.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
MI_finaldf["next_admittime"] = MI_finaldf.groupby("subject_id")["admittime"].shift(-1)

#calculating days from discharge to next MI admission
MI_finaldf["days_to_readmit"] = (MI_finaldf["next_admittime"] - MI_finaldf["dischtime"]).dt.days

#readmission risk categories
MI_finaldf["readmission_risk"] = np.select(
    [
        MI_finaldf["days_to_readmit"].between(0, 30, inclusive="both"),
        MI_finaldf["days_to_readmit"].between(31, 60, inclusive="both"),
        (MI_finaldf["days_to_readmit"] > 60) | (MI_finaldf["days_to_readmit"].isna())
    ],
    ["High", "Medium", "Low"],
    default="Low"
)

#quick check
print(MI_finaldf["readmission_risk"].value_counts())
print("\n", MI_finaldf.head())


#MI_finaldf.to_csv("MI_final_dataset.csv", index=False)
#MI_finaldf.to_csv(r"C:\Users\User\Downloads\MI_final_dataset.csv", index=False)
print(MI_finaldf.shape)

#################################################################################################
#Modelling

#predictors
predictor_cols = [
    "age_group",
    "gender",
    "admission_group",
    "admit_weekend",
    "prior_mi",
    "other_diag"
]

#targets
target_cols = [
    "los_cat",
    "readmission_risk"
]

#building a separate dataframe for modelling
Finalmodel_df = MI_finaldf[predictor_cols + target_cols + ["subject_id"]].copy()

#quick checks
print("Rows:", Finalmodel_df.shape[0])
print("Columns:", Finalmodel_df.shape[1])
print("\nMissing values per column:")
print(Finalmodel_df.isna().sum())

print("\nTarget distributions:")
print("\nLOS category:")
print(Finalmodel_df["los_cat"].value_counts(dropna=False))
print("\nReadmission risk:")
print(Finalmodel_df["readmission_risk"].value_counts(dropna=False))
print(Finalmodel_df.head(10))

#Finalmodel_df.to_csv("Finalmodel_df.csv", index=False)
#Finalmodel_df.to_csv(r"C:\Users\User\Downloads\MI_final_model_dataset.csv", index=False, encoding="utf-8-sig")


from sklearn.model_selection import train_test_split

#spliting unique patients into train and test dfs (80/20)
train_patients, test_patients = train_test_split(Finalmodel_df["subject_id"].unique(), test_size=0.2, random_state=42)

#train/test dataframes using patient split
train_df = Finalmodel_df[Finalmodel_df["subject_id"].isin(train_patients)].copy()
test_df  = Finalmodel_df[Finalmodel_df["subject_id"].isin(test_patients)].copy()

#dropping subject_id from features and creating x y variables for LOS and readmission
X_train = train_df.drop(columns=["los_cat", "readmission_risk", "subject_id"])
y_train_los, y_train_readmit = train_df["los_cat"], train_df["readmission_risk"]
X_test  = test_df.drop(columns=["los_cat", "readmission_risk", "subject_id"])
y_test_los,  y_test_readmit  = test_df["los_cat"],  test_df["readmission_risk"]

#quick checks
print("Train rows:", X_train.shape[0], "| Test rows:", X_test.shape[0])
print("Train patients:", train_df["subject_id"].nunique(), "| Test patients:", test_df["subject_id"].nunique())
print("Patient overlap:", len(set(train_df["subject_id"]) & set(test_df["subject_id"])))


from sklearn.preprocessing import OneHotEncoder

#categorical predictor columns
cat_cols = ["age_group", "gender", "admission_group", "admit_weekend", "prior_mi", "other_diag"]

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

#applying encoder on train and test data
X_train_enc = encoder.fit_transform(X_train[cat_cols])
X_test_enc  = encoder.transform(X_test[cat_cols])

print("X_train_enc:", X_train_enc.shape, "| X_test_enc:", X_test_enc.shape)
print(pd.DataFrame(X_train_enc[:20].toarray(), columns=encoder.get_feature_names_out(cat_cols)).head(20))

#checking for class imbalance
print(y_train_los.value_counts(normalize=True))
print(y_test_los.value_counts(normalize=True))

############################################################################################################

#Modeling and evaluation 1 - Logistic regression for LOS

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

logreg_los = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)  

#training the model on training data
logreg_los.fit(X_train_enc, y_train_los)

#predictions on the test set
y_pred_los = logreg_los.predict(X_test_enc)
y_prob_los = logreg_los.predict_proba(X_test_enc)[:, 1] #(for ROC-AUC)

#evaluation
print("Accuracy (Logistic Regression - LOS):", round(accuracy_score(y_test_los, y_pred_los), 4))
print("\nClassification Report (Logistic Regression - LOS):\n", classification_report(y_test_los, y_pred_los))
print("Confusion Matrix (Logistic Regression - LOS):\n", confusion_matrix(y_test_los, y_pred_los))
roc_los = roc_auc_score((y_test_los == "≥ 7 days").astype(int), y_prob_los)
print("ROC-AUC (Logistic Regression - LOS):", round(roc_los, 4))

#Modeling and evaluation 2 - Decision Tree for LOS

from sklearn.tree import DecisionTreeClassifier

decisiontree_los = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)

decisiontree_los.fit(X_train_enc, y_train_los)

y_pred_dt = decisiontree_los.predict(X_test_enc)
y_prob_dt = decisiontree_los.predict_proba(X_test_enc)[:, 1]

print("\n", "Accuracy (Decision Tree - LOS):", round(accuracy_score(y_test_los, y_pred_dt), 4))
print("\nClassification Report (Decision Tree - LOS):\n", classification_report(y_test_los, y_pred_dt))
print("Confusion Matrix (Decision Tree - LOS):\n", confusion_matrix(y_test_los, y_pred_dt))
roc_dt = roc_auc_score((y_test_los == "≥ 7 days").astype(int), y_prob_dt)
print("ROC-AUC (Decision Tree - LOS):", round(roc_dt, 4))

#Modeling and evaluation 3 - Random Forest for LOS

from sklearn.ensemble import RandomForestClassifier

randomforest_los = RandomForestClassifier( n_estimators=200, max_depth=8, class_weight="balanced",random_state=42)

randomforest_los.fit(X_train_enc, y_train_los)

y_pred_rf, y_prob_rf = randomforest_los.predict(X_test_enc), randomforest_los.predict_proba(X_test_enc)[:, 1]

print("\n", "Accuracy (Random Forest - LOS):", round(accuracy_score(y_test_los, y_pred_rf), 4))
print("\nClassification Report (Random Forest - LOS):\n", classification_report(y_test_los, y_pred_rf))
print("Confusion Matrix (Random Forest - LOS):\n", confusion_matrix(y_test_los, y_pred_rf))
print("ROC-AUC (Random Forest - LOS):", round(roc_auc_score((y_test_los == "≥ 7 days").astype(int), y_prob_rf), 4))

#Modeling and evaluation 4 - Gradient boosting for LOS

from sklearn.ensemble import GradientBoostingClassifier

gradientboosting_los = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)

gradientboosting_los.fit(X_train_enc, y_train_los)

y_pred_gb, y_prob_gb = (gradientboosting_los.predict(X_test_enc),gradientboosting_los.predict_proba(X_test_enc)[:, 1])

print("Accuracy (Gradient Boosting - LOS):", round(accuracy_score(y_test_los, y_pred_gb), 4))
print("\nClassification Report (Gradient Boosting - LOS):\n", classification_report(y_test_los, y_pred_gb))
print("Confusion Matrix (Gradient Boosting - LOS):\n", confusion_matrix(y_test_los, y_pred_gb))
print("ROC-AUC (Gradient Boosting - LOS):", round(roc_auc_score((y_test_los == "≥ 7 days").astype(int), y_prob_gb), 4))

#Modeling and evaluation 5 - Cat boosting for LOS

from catboost import CatBoostClassifier

#class weights (balanced) - without extra imports
_counts = y_train_los.value_counts()
w_short = float(_counts.max() / _counts.get("< 7 days", 1))
w_long  = float(_counts.max() / _counts.get("≥ 7 days", 1))

catboost_los = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    class_weights=[w_short, w_long],   # order matches classes: ["< 7 days", "≥ 7 days"]
    random_seed=42,
    verbose=False
)

catboost_los.fit(X_train_enc, y_train_los)

y_pred_cb = catboost_los.predict(X_test_enc)
y_prob_cb = catboost_los.predict_proba(X_test_enc)[:, 1]

print("\n", "Accuracy (CatBoost - LOS):", round(accuracy_score(y_test_los, y_pred_cb), 4))
print("\nClassification Report (CatBoost - LOS):\n", classification_report(y_test_los, y_pred_cb))
print("Confusion Matrix (CatBoost - LOS):\n", confusion_matrix(y_test_los, y_pred_cb))
print("ROC-AUC (CatBoost - LOS):", round(roc_auc_score((y_test_los == "≥ 7 days").astype(int), y_prob_cb), 4))

#Modeling and evaluation 6 - Suppoirt Vector Machine(SVM) for LOS

from sklearn.svm import SVC

svm_los = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)

svm_los.fit(X_train_enc, y_train_los)

y_pred_svm = svm_los.predict(X_test_enc)
y_prob_svm = svm_los.predict_proba(X_test_enc)[:, 1]

print("\n", "Accuracy (SVM - LOS):", round(accuracy_score(y_test_los, y_pred_svm), 4))
print("\nClassification Report (SVM - LOS):\n", classification_report(y_test_los, y_pred_svm))
print("Confusion Matrix (SVM - LOS):\n", confusion_matrix(y_test_los, y_pred_svm))
print("ROC-AUC (SVM - LOS):", round(roc_auc_score((y_test_los == "≥ 7 days").astype(int), y_prob_svm), 4))
