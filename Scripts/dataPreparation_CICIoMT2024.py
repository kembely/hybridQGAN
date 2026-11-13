import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from utilFunctions import dataLoad
import gc
from typing import Mapping
np.random.seed(123)

CICIOMT2024_34_FEATURES = (
    "Header_Length",
    "Protocol_Type",
    "Rate",
    "Srate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "TCP",
    "UDP",
    "ARP",
    "ICMP",
    "IPv",
    "LLC",
    "Tot_sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot_size",
    "IAT",
    "Number",
    "Magnitue",
    "Radius",
    "Covariance",
    "Variance",
    "Weight",
    "label",
    "category",
)

CICIOMT2024_GROUPS = {
    "ARP_Spoofing": [1],
    "MQTT-DDOS": [2, 3],
    "MQTT_DOS": [4, 5],
    "MQTT_Malformed_Data": [6],
    "Recon": [7, 8, 9, 10],
    "DDOS": [11, 12, 13, 14],
    "DOS": [15, 16, 17, 18],
}
CICIOMT2024_CATEGORY_MAPPING = {
    "Benign": 0,
    "ARP_Spoofing": 1,
    "MQTT-DDoS-Connect_Flood": 2,
    "MQTT-DDoS-Publish_Flood": 3,
    "MQTT-DoS-Connect_Flood": 4,
    "MQTT-DoS-Publish_Flood": 5,
    "MQTT-Malformed_Data": 6,
    "Recon-OS_Scan": 7,
    "Recon-Ping_Sweep": 8,
    "Recon-Port_Scan": 9,
    "Recon-VulScan": 10,
    "TCP_IP-DDoS-ICMP": 11,
    "TCP_IP-DDoS-SYN": 12,
    "TCP_IP-DDoS-TCP": 13,
    "TCP_IP-DDoS-UDP": 14,
    "TCP_IP-DoS-ICMP": 15,
    "TCP_IP-DoS-SYN": 16,
    "TCP_IP-DoS-TCP": 17,
    "TCP_IP-DoS-UDP": 18,
}

CICIOMT2024_VAL_ATTACK_IDS = (
    1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
)

#Quotas for validation dataset with whole benign_data
CICIOMT2024_FULL_VAL_QUOTAS = {
    1: 11342,   # ARP_Spoofing
    2: 11342,   # MQTT-DDoS-Connect_Flood
    3: 11342,   # MQTT-DDoS-Publish_Flood
    4: 11342,   # MQTT-DoS-Connect_Flood
    5: 11342,   # MQTT-DoS-Publish_Flood
    6: 5130,    # MQTT-Malformed_Data
    7: 11342,   # Recon-OS_Scan
    9: 11342,   # Recon-Port_Scan
    10: 2173,   # Recon-VulScan
    11: 11342,  # TCP_IP-DDoS-ICMP
    12: 11342,  # TCP_IP-DDoS-SYN
    13: 11342,  # TCP_IP-DDoS-TCP
    14: 11342,  # TCP_IP-DDoS-UDP
    15: 11342,  # TCP_IP-DoS-ICMP
    16: 11342,  # TCP_IP-DoS-SYN
    17: 11342,  # TCP_IP-DoS-TCP
    18: 11342,  # TCP_IP-DoS-UDP
}

CICIOMT2024_TEST_QUOTAS = {
    1: 1744,   # ARP_Spoofing
    2: 2350,   # MQTT-DDoS-Connect_Flood
    3: 2350,   # MQTT-DDoS-Publish_Flood
    4: 2350,   # MQTT-DoS-Connect_Flood
    5: 2350,   # MQTT-DoS-Publish_Flood
    6: 1747,   # MQTT-Malformed_Data
    7: 2350,   # Recon-OS_Scan
    8: 186,    # Recon-Ping_Sweep (incluida en test)
    9: 2350,   # Recon-Port_Scan
    10: 1034,  # Recon-VulScan
    11: 2350,  # TCP_IP-DDoS-ICMP
    12: 2350,  # TCP_IP-DDoS-SYN
    13: 2349,  # TCP_IP-DDoS-TCP
    14: 2349,  # TCP_IP-DDoS-UDP
    15: 2349,  # TCP_IP-DoS-ICMP
    16: 2350,  # TCP_IP-DoS-SYN
    17: 2350,  # TCP_IP-DoS-TCP
    18: 2349,  # TCP_IP-DoS-UDP
}
# Remove missing values and space in columns from dataframe ------------------------
def clean_numerical_records(df):
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(axis=0, how="any", inplace=True)
    return df
# ----------------------------------------------------------------------------------
# Sort samples in dataframe per timestamp ------------------------------------------
def sortDataframePerTimestamp(df):
    df = df.sort_values(by = 'Timestamp')
    return df
# ----------------------------------------------------------------------------------

# Convernt names of labels to numbers ----------------------------------------------
def encode_labels(df, filename, mapping = None, strict = True):
    """
    'label' (0: Normal, 1: Acttack) and 'category' (0..18) based on filename.
    """
    category_map = mapping or CICIOMT2024_CATEGORY_MAPPING
    base_name = filename

    category_id = next(
        (
            cat_id
            for pattern, cat_id in sorted(
                category_map.items(), key=lambda item: len(item[0]), reverse=True
            )
            if pattern in base_name
        ),
        -1,
    )

    if category_id == -1:
        if strict:
            raise ValueError(f"Unknown class in file: '{base_name}'")
        label_id = -1
    else:
        label_id = 0 if category_id == 0 else 1

    df["label"] = np.int32(label_id)
    df["category"] = np.int32(category_id)
    return df
# ----------------------------------------------------------------------------------

# Construct training and validation sets dataframes with subset of the data to reduce training time --------------------------------
def constructTrainingValidationSets(df_attacks, df_benign, benign_total = 10000, train_ratio = 0.7, attack_samples_per_class = 588,
                                    included_attack_ids = CICIOMT2024_VAL_ATTACK_IDS):
    sampled_benign = df_benign.sample(n=benign_total)
    msk = np.random.rand(len(sampled_benign)) < train_ratio
    df_training = sampled_benign[msk].reset_index(drop=True)
    df_validation_benign = sampled_benign[~msk].reset_index(drop=True)

    sampled_attacks = []
    for cat_id in included_attack_ids:

        source = df_attacks.get(cat_id)
        df_atk = (
            pd.concat(source, ignore_index=True)
            if isinstance(source, list)
            else source
        )
        sampled_attacks.append(df_atk.sample(n=attack_samples_per_class))

    # 3. Consolidación de validación
    df_validation = pd.concat(
        [df_validation_benign] + sampled_attacks,
        ignore_index=True,
    )

    df_validation.sample(frac=1)

    return df_training, df_validation
# ----------------------------------------------------------------------------------

# Construct training and validation sets dataframes with whole benign data--------------------------------
def constructFullTrainingValidationSets(
    df_benign, df_attacks, train_ratio = 0.7, quotas = CICIOMT2024_FULL_VAL_QUOTAS):
  
    msk = np.random.rand(len(df_benign)) < train_ratio
    df_training = df_benign[msk].reset_index(drop=True)
    df_validation_benign = df_benign[~msk].reset_index(drop=True)

    sampled_attacks = []
    for cat_id, target_n in quotas.items():
        source = df_attacks.get(cat_id)
        if source is None:
            raise KeyError(f"{cat_id}  is not in df_attacks.")

        df_cat = (
            pd.concat(source, ignore_index=True)
            if isinstance(source, list)
            else source
        )

        sampled_attacks.append(df_cat.sample(n=target_n))

        del df_cat
        gc.collect()

    df_validation = pd.concat(
        [df_validation_benign] + sampled_attacks,
        ignore_index=True,
    )
    del df_validation_benign, sampled_attacks
    gc.collect()

    df_validation.sample(frac=1)

    return df_training, df_validation


def constructTestingSet(df_benign, df_attacks, benign_quota = 37607, quotas = CICIOMT2024_TEST_QUOTAS):
    benign_pool = (
        pd.concat(df_benign, ignore_index=True)
        if isinstance(df_benign, list)
        else df_benign
    )
    selected_subsets = [benign_pool.sample(n=benign_quota)]

    if isinstance(df_benign, list):
        del benign_pool
        gc.collect()

    for cat_id in sorted(quotas.keys()):
        target_n = quotas[cat_id]
        source = df_attacks.get(cat_id)
        if source is None:
            raise KeyError(f"Category {cat_id} is not in df_attacks.")

        df_atk = (
            pd.concat(source, ignore_index=True)
            if isinstance(source, list)
            else source
        )

        selected_subsets.append(df_atk.sample(n=target_n))

        if isinstance(source, list):
            del df_atk
            gc.collect()

    df_testing = pd.concat(selected_subsets, ignore_index=True)
    del selected_subsets
    gc.collect()

    return df_testing

# ----------------------------------------------------------------------------------

# Print the number of occurrences of each label and category in dataframe ----------
def get_data_frame_Stats(df):
    cat_counts = df["category"].value_counts().to_dict()
    lbl_counts = df["label"].value_counts().to_dict()
    normal_count = int(lbl_counts.get(0, 0))
    attack_count = int(lbl_counts.get(1, 0))
    
    group_counts = {
        name: int(sum(cat_counts.get(cat_id, 0) for cat_id in indices))
        for name, indices in CICIOMT2024_GROUPS.items()
    }

    print(f"Normal: {normal_count:d}  Attack: {attack_count:d}")
        
    detail_msg = " ".join(
        f"{name}: {cnt:d}" for name, cnt in group_counts.items()
        )
    print(f"Normal: {int(cat_counts.get(0, 0)):d} {detail_msg}")


# Print the number of occurrences of each label and category in numpy file ---------
def datasetStatsCICIoMT2024(dataset):
    labels = dataset[:, -2].astype(int).ravel()
    categories = dataset[:, -1].astype(int).ravel()
    cat_counts = np.bincount(categories, minlength=19)
   
    bin_counts = np.bincount(labels, minlength=2)
    normal_count = int(bin_counts[0])
    attack_count = int(bin_counts[1])

    group_counts = {
        name: int(np.sum(cat_counts[indices]))
        for name, indices in CICIOMT2024_GROUPS.items()
    }
    print(f"Normal: {normal_count:d}  Attack: {attack_count:d}")
        
    detail_msg = " ".join(
            f"{name}: {cnt:d}" for name, cnt in group_counts.items()
        )
    print(f"Normal: {int(cat_counts[0]):d} {detail_msg}")


def _collect_ciciomt2024_chunks(folder_path, chunksize = 250_000):

    # Mantiene el mismo orden de recorrido que os.listdir original
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    class_chunks = {cat_id: [] for cat_id in range(19)}

    for idx, filename in enumerate(csv_files, 1):
        file_path = folder_path + filename
        print(f"[{idx}/{len(csv_files)}] Processing: {filename}...")

        for chunk in pd.read_csv(file_path, sep=",", chunksize=chunksize, low_memory=False):
            chunk = clean_numerical_records(chunk)
            if chunk.empty:
                continue

            chunk = encode_labels(chunk, filename=filename, strict=True)
            cat_id = int(chunk["category"].iloc[0])
            class_chunks[cat_id].append(chunk)

        gc.collect()

    return class_chunks
#--------------------------------------------------------------------------------------

# Load dataset files, clean missing values, convert labels, and create dataframes ----
def preprocessData(training = True, save = True, full_benign = True, chunksize = 250_000):
    base_dir = os.getcwd() + "/datasets/CICIoMT2024/"
    is_training = bool(training)
    sub_folder = "train/" if is_training else "test/"

    class_chunks = _collect_ciciomt2024_chunks(
        base_dir + sub_folder,
        chunksize=chunksize,
    )

    df_benign = (
        pd.concat(class_chunks[0], ignore_index=True)
        if class_chunks[0]
        else pd.DataFrame()
    )
    del class_chunks[0]
    gc.collect()

    if is_training:
        if full_benign:
            df_training, df_validation = constructFullTrainingValidationSets(
                df_benign=df_benign,
                df_attacks=class_chunks,
            )
        else:
            df_training, df_validation = constructTrainingValidationSets(
                df_benign=df_benign,
                df_attacks=class_chunks,
            )

        del df_benign, class_chunks
        gc.collect()

        # 4. Estandarización de columnas y reinicio limpio de índices
        df_training = fixFeaturesNames(df_training).reset_index(drop=True)
        df_validation = fixFeaturesNames(df_validation).reset_index(drop=True)

        get_data_frame_Stats(df_training)
        get_data_frame_Stats(df_validation)

        if bool(save):
            
            df_training.to_pickle(base_dir + "sets/df_training.pkl")
            df_validation.to_pickle(base_dir + "sets/df_validation.pkl")

        return df_training, df_validation

    else:
        #Testing
        df_testing = constructTestingSet(
            df_benign=df_benign,
            df_attacks=class_chunks,
        )
        del df_benign, class_chunks
        gc.collect()

        df_testing = fixFeaturesNames(df_testing).reset_index(drop=True)

        print("\n Testing stats:")
        get_data_frame_Stats(df_testing)

        if bool(save):
            df_testing.to_pickle(base_dir + "sets/df_testing.pkl")

        return df_testing 
# ----------------------------------------------------------------------------------

# Load dataframe and save it as a numpy array --------------------------------------
def createDataset(datasetName, save=True):
    df = pd.read_pickle(os.getcwd() + "/datasets/CICIoMT2024/sets/df_" + datasetName + ".pkl")
    dataset = df.to_numpy()
    datasetStatsCICIoMT2024(dataset)
    if(save):
        np.save(os.getcwd() + "/datasets/CICIoMT2024/sets/" + datasetName + "Set.npy", dataset)
# ----------------------------------------------------------------------------------

# Load dataframe and print statistics of each class --------------------------------
def checkDataframe(datasetName):
    df = pd.read_pickle(os.getcwd() + "/datasets/CICIoMT2024/sets/df_" + datasetName + ".pkl")
    get_data_frame_Stats(df)
    return df
# ----------------------------------------------------------------------------------

# Load numpy array and print statistics of each class ------------------------------
def checkDataset(datasetName):
    dataset = dataLoad(os.getcwd() + "/datasets/CICIoMT2024/sets/" + datasetName + ".npy")
    datasetStatsCICIoMT2024(dataset)
    return dataset
# ----------------------------------------------------------------------------------

# Select features and create subset files ------------------------------------------
def select_features_and_export(datasetName, save_numpy = False, features = CICIOMT2024_34_FEATURES):
    rootdir = os.getcwd() + "/datasets/CICIoMT2024/sets/"
    df = pd.read_pickle(rootdir + "df_" + datasetName + ".pkl")

    target_cols = list(features)
    missing_cols = [c for c in target_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in df_{datasetName}.pkl: {missing_cols}")

    # Selección directa y reinicio de índice sin copias intermedias
    df_selected = df[target_cols].reset_index(drop=True)
    del df
    gc.collect()

    get_data_frame_Stats(df_selected)
    if save_numpy:
        npy_path = rootdir + datasetName+ "Set_34features.npy"
        np.save(npy_path, df_selected.to_numpy(dtype=np.float64))
    return df_selected

# ----------------------------------------------------------------------------------

# Fix names of features that are not correct in original dataset files -------------
def fixFeaturesNames(df):
    df.rename(columns={"Protocol Type":"Protocol_Type"}, inplace=True)
    df.rename(columns={"Tot sum":"Tot_sum"}, inplace=True)
    df.rename(columns={"Tot size":"Tot_size"}, inplace=True)
    return df
# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    
    #preprocessData(training=True)
    #createDataset(datasetName="training")
    #createDataset(datasetName="validation")

    preprocessData(training=False)
    createDataset(datasetName="testing", save=1)

    #dfTraining = select_features_and_export(datasetName="training", save_numpy=True)
    #dfValidation = select_features_and_export(datasetName="validation", save_numpy=True)
    dfTesting = select_features_and_export(datasetName="testing", save_numpy=True)

    datasetTraining = checkDataset("trainingSet")
    datasetValidation = checkDataset("validationSet")
    datasetTesting = checkDataset("testingSet")
    