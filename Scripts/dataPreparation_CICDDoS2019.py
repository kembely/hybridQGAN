import os
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from utilFunctions import dataLoad
from typing import Mapping, Sequence
np.random.seed(123)

TRAFFIC_CATEGORY_MAPPING = {
    "BENIGN": 0,
    "Portmap": 1,
    "Syn": 2,
    "UDP": 3,
    "DrDoS_UDP": 3,
    "UDPLag": 4,
    "UDP-lag": 4,
    "MSSQL": 5,
    "DrDoS_MSSQL": 5,
    "NetBIOS": 6,
    "DrDoS_NetBIOS": 6,
    "LDAP": 7,
    "DrDoS_LDAP": 7,
    "WebDDoS": 8,
    "TFTP": 9,
    "DrDoS_SSDP": 10,
    "DrDoS_SNMP": 11,
    "DrDoS_NTP": 12,
    "DrDoS_DNS": 13,
}

CATEGORY_NAMES = {
    0: "BENIGN",
    1: "Portmap",
    2: "Syn",
    3: "UDP",
    4: "UDPLag",
    5: "MSSQL",
    6: "NetBIOS",
    7: "LDAP",
    8: "WebDDoS",
    9: "TFTP",
    10: "SSDP",
    11: "SNMP",
    12: "NTP",
    13: "DNS",
}

CICDDOS_40_FEATURE_MAP  = {
    "Flow ID": "Flow_ID",
    "Source IP": "Source_IP",
    "Source Port": "Source_Port",
    "Destination IP": "Destination_IP",
    "Destination Port": "Destination_Port",
    "Protocol": "Protocol",
    "Flow Duration": "FlowDuration",
    "Total Fwd Packets": "Total_Fwd_Packets",
    "Total Backward Packets": "Total_Backward_Packets",
    "Total Length of Fwd Packets": "Total_Length_of_Fwd_Packets",
    "Total Length of Bwd Packets": "Total_Length_of_Bwd_Packets",
    "Fwd Packet Length Max": "Fwd_Packet_Length_Max",
    "Fwd Packet Length Min": "Fwd_Packet_Length_Min",
    "Fwd Packet Length Mean": "Fwd_Packet_Length_Mean",
    "Fwd Packet Length Std": "Fwd_Packet_Length_Std",
    "Bwd Packet Length Max": "Bwd_Packet_Length_Max",
    "Bwd Packet Length Min": "Bwd_Packet_Length_Min",
    "Bwd Packet Length Mean": "Bwd_Packet_Length_Mean",
    "Bwd Packet Length Std": "Bwd_Packet_Length_Std",
    "Flow Bytes/s": "Flow_Bytes/s",
    "Flow Packets/s": "Flow_Packets/s",
    "Flow IAT Mean": "Flow_IAT_Mean",
    "Flow IAT Std": "Flow_IAT_Std",
    "Flow IAT Max": "Flow_IAT_Max",
    "Flow IAT Min": "Flow_IAT_Min",
    "Fwd IAT Total": "Fwd_IAT_Total",
    "Fwd IAT Mean": "Fwd_IAT_Mean",
    "Fwd IAT Std": "Fwd_IAT_Std",
    "Fwd IAT Max": "Fwd_IAT_Max",
    "Fwd IAT Min": "Fwd_IAT_Min",
    "Bwd IAT Total": "Bwd_IAT_Total",
    "Bwd IAT Mean": "Bwd_IAT_Mean",
    "Bwd IAT Std": "Bwd_IAT_Std",
    "Bwd IAT Max": "Bwd_IAT_Max",
    "Bwd IAT Min": "Bwd_IAT_Min",
    "Fwd PSH Flags": "Fwd_PSH_Flags",
    "Fwd Header Length": "Fwd_Header_Length",
    "Bwd Header Length": "Bwd_Header_Length",
    "Fwd Packets/s": "Fwd_Packets/s",
    "Bwd Packets/s": "Bwd_Packets/s",
    "label": "label",
    "category": "category",}

# Remove missing values and spaces in columns from dataframe ------------
def clean_numerical_records(df):
    df.columns = df.columns.str.strip()
    # Remove missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how="any", inplace=True)
    return df

# Encode Labels in numbers ----------------------------------------------
def encode_traffic_labels(df, label_col):
    target_col = label_col.strip()
    if target_col not in df.columns:
        raise KeyError(f"The feature '{target_col}' is not in the DataFrame.")
    
    cat_series = (
        df[target_col]
        .astype(str)
        .str.strip()
        .map(TRAFFIC_CATEGORY_MAPPING)
        .fillna(-1)
        .astype(int)
    )
    df["label"] = np.where(cat_series == 0, 0, 1)
    df["category"] = cat_series
    return df


# Sort samples in dataframe per timestamp ------------------------------------------
def sort_dataframe_by_Timestamp(df):
    df = df.sort_values(by='Timestamp')
    return df

# Convert categorical features to numerical ----------------------------------------
def encode_categorical_features(df, columns= ("Flow_ID", "Source_IP", "Destination_IP")):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes.astype(np.int64)
    return df

# Print the number of occurrences of each label and category in dataframe ----------
def get_dataframe_Stats(df):
    cat_counts = df["category"].value_counts().to_dict()
    lbl_counts = df["label"].value_counts().to_dict()
    normal_count = int(lbl_counts.get(0, 0))
    attack_count = int(lbl_counts.get(1, 0))

    primary_counts = {
        name: int(cat_counts.get(cat_id, 0))
        for cat_id, name in CATEGORY_NAMES.items()
        if cat_id != 0 and cat_id <= 7
    }

    sum_primary_attacks = sum(primary_counts.values())
    others_count = max(0, attack_count - sum_primary_attacks)
    print(f"Normal: {normal_count:d}  Attack: {attack_count:d}")
        
    detail_msg = " ".join(f"{name}: {cnt:d}" for name, cnt in primary_counts.items())
    print(f"Normal: {normal_count:d} {detail_msg} Others: {others_count:d}")

# Print the number of occurrences of each label and category in numpy file ---------
def datasetStatsCICDDoS2019(dataset):
    labels = dataset[:, -2].astype(int).ravel()
    categories = dataset[:, -1].astype(int).ravel()
    cat_counts = np.bincount(categories, minlength=14)

    bin_counts = np.bincount(labels, minlength=2)
    normal_count = int(bin_counts[0])
    total_attacks = int(bin_counts[1])

    primary_counts = {
        name: int(cat_counts[cat_id])
        for cat_id, name in CATEGORY_NAMES.items()
        if cat_id != 0 and cat_id <=7
    }

    sum_primary_attacks = sum(primary_counts.values())
    others_count = max(0, total_attacks - sum_primary_attacks)

    print(f"Normal: {normal_count:d}  Attack: {total_attacks:d}")
        
    detail_msg = " ".join(f"{name}: {cnt:d}" for name, cnt in primary_counts.items())
    print(f"Normal: {int(cat_counts[0]):d} {detail_msg} Others: {others_count:d}")


# Load dataset files, clean missing values, convert labels, and create dataframes --
def preprocessData(training=True, save=True):
    path = os.getcwd() + "/datasets/CICDDoS2019/"
    folder_name = "01-12/" if training else "03-11/"
    # Training
    Files_List = os.listdir(path + folder_name)    # Training Day
    df_BENIGN = []
    benign_chunks = []
    attack_chunks = {cat_id: [] for cat_id in range(1, 8)}
    for i in range(len(Files_List)-3):
        filePath = path + folder_name + Files_List[i]
        df = pd.read_csv(filePath, sep=",")
        df = clean_numerical_records(df) 
        df = encode_traffic_labels(df, label_col="Label")
        subset_BENIGN = df.loc[(df.category == 0)]
        benign_chunks.append(subset_BENIGN)
        for cat_id in range(1, 8):
            subset_attack = df.loc[df["category"] == cat_id]
            if not subset_attack.empty:
                attack_chunks[cat_id].append(subset_attack)
        del df
        gc.collect()
    df_BENIGN = pd.concat(benign_chunks, ignore_index=True) if benign_chunks else pd.DataFrame()
    df_attacks_dict = {
                    cat_id: pd.concat(chunks, ignore_index=True)
                    for cat_id, chunks in attack_chunks.items()
                    if chunks
        }            
    if training:
        df_training, df_validation = constructTrainingValidationSets(df_BENIGN, df_attacks_dict)
        df_training = sort_dataframe_by_Timestamp(df_training)
        df_validation = sort_dataframe_by_Timestamp(df_validation)
        get_dataframe_Stats(df_training)
        get_dataframe_Stats(df_validation)
        if(save == 1):
            df_training.to_pickle(path + "sets/df_training.pkl")
            df_validation.to_pickle(path + "sets/df_validation.pkl")
    else:
        # Testing
        df_testing = constructTestingSet(df_BENIGN, df_attacks_dict)
        df_testing = sort_dataframe_by_Timestamp(df_testing)
        get_dataframe_Stats(df_testing)
        if save:
            df_testing.to_pickle(path + "sets/df_testing.pkl")

# Construct training and validation sets -- --------------------------------
def constructTrainingValidationSets(df_BENIGN, df_attacks):
    msk = np.random.rand(len(df_BENIGN)) < 0.8
    df_training = df_BENIGN[msk]
    df_validation_benign = df_BENIGN[~msk]

    # Extracción de DataFrames de ataque
    sampled_attacks = []
    for cat_id in sorted(df_attacks.keys()):
        df_atk = df_attacks[cat_id]
        if df_atk.empty:
            continue
        # Muestreo sobre el pool completo original
        sampled_attacks.append(df_atk.sample(n=11342))

    df_validation = pd.concat([df_validation_benign] + sampled_attacks, ignore_index=True)
    df_validation.sample(frac=1)
    return df_training, df_validation

# Construct testing set --------------------------------------------------------
def constructTestingSet(df_BENIGN, df_attacks):
    quotas={
    "Portmap": 8022,
    "Syn": 8021,
    "UDP": 8021,
    "UDPLag": 1873,
    "MSSQL": 8021,
    "NetBIOS": 8021,
    "LDAP": 8021,
    }
    selected_subsets = [df_BENIGN.sample(n=50000)]
    for key, df_atk in df_attacks.items():
        if df_atk.empty:
            continue
        target_n = quotas.get(key, len(df_atk))
        selected_subsets.append(df_atk.sample(n=target_n))

    df_testing = pd.concat(selected_subsets, ignore_index=True)

    return df_testing

# Load dataframe and save it as a numpy array --------------------------------------
def createDataset(datasetName, save=True):
    df = pd.read_pickle(os.getcwd() + "/datasets/CICDDoS2019/sets/df_" + datasetName + ".pkl")
    dataset = df.to_numpy()
    datasetStatsCICDDoS2019(dataset)
    if save:
        np.save(os.getcwd() + "/datasets/CICDDoS2019/sets/" + datasetName + "Set.npy", dataset)

# Load dataframe and print statistics of each class ---------------------------
def checkDataframe(datasetName):
    df = pd.read_pickle(os.getcwd() + "/datasets/CICDDoS2019/sets/df_" + datasetName + ".pkl")
    get_dataframe_Stats(df)
    return df

# Load numpy array and print statistics of each class ------------------------------
def checkDataset(datasetName):
    dataset = dataLoad(os.getcwd() + "/datasets/CICDDoS2019/sets/" + datasetName + ".npy")
    datasetStatsCICDDoS2019(dataset)
    return dataset

# Clean features names and encode categorical features-------------------------------
def extract_and_format_features(df, feature_map = None, categorical_cols = ("Flow_ID", "Source_IP", "Destination_IP")):
    mapping = feature_map or CICDDOS_40_FEATURE_MAP
    df.columns = df.columns.str.strip()
    available_cols = [c for c in mapping.keys() if c in df.columns]
    df = df[available_cols].rename(columns=mapping)

    # Encode categorical features
    result = encode_categorical_features(df, columns=categorical_cols)
    return result.reset_index(drop=True)

# Select features and create subset files --------------------------------------------
def select_features_and_export( datasetName, save_numpy = False, feature_map= None):
    rootdir = os.getcwd() + "/datasets/CICDDoS2019/sets/"
    df = pd.read_pickle(rootdir + "df_" + datasetName + ".pkl")
    df_processed = extract_and_format_features(df, feature_map=feature_map)

    get_dataframe_Stats(df_processed)

    if save_numpy:
        npy_path = rootdir + datasetName+ "Set_40features.npy"
        np.save(npy_path, df_processed.to_numpy(dtype=np.float64))

    return df_processed


if __name__ == '__main__':
    #preprocessData(training=True)
    #createDataset(datasetName="training")
    #createDataset(datasetName="validation")

    preprocessData(training=False)
    createDataset(datasetName="testing")

    #dfTraining = select_features_and_export(datasetName="training", save_numpy=True)
    #dfValidation = select_features_and_export(datasetName="validation", save_numpy=True)
    #dfTesting = select_features_and_export(datasetName="testing", save_numpy=True)

    #datasetTraining = checkDataset("trainingSet_40features")
    #datasetValidation = checkDataset("validationSet_40features")
    #datasetTesting = checkDataset("testingSet_40features")
