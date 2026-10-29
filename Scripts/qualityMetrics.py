import numpy as np
import torch
import random
import os 
import seaborn as sns
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.spatial.distance import jensenshannon
from qganModel import HQuantumGenerator
from scipy.stats import entropy, wasserstein_distance
from datasetClasses_IoT import Dataset_IoT
from datasetClasses_IoT import ToTensor_IoT
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_recall_curve

#Calclulate  Jenshen Shannon divergence
def calculate_jsd(reales, generados, n_bins=73):
    min_val = min(np.min(reales), np.min(generados))
    max_val = max(np.max(reales), np.max(generados))
    range_bins = (min_val, max_val)
    p, _ = np.histogram(reales, bins=n_bins, range=range_bins, density=False)
    q, _ = np.histogram(generados, bins=n_bins, range=range_bins, density=False)
    p = p + 1e-6
    q = q + 1e-6
    p = p / np.sum(p)
    q = q / np.sum(q)
    js_div = jensenshannon(p, q)**2

    print("AVG_JSD:", js_div)

def calculate_wsd(real_features, synthetic_features):
    """
    Calculate JSD y RMSE to compare synthetic data quality.
    """
    jsd_list = []
    rmse_per_feature = []
    ws_scores = []
    
    # Loop for each feature
    for i in range(real_features.shape[1]):
        r_feat = real_features[:, i]
        s_feat = synthetic_features[:, i]
        # Wasserstein (EMD)
        ws_scores.append(wasserstein_distance(r_feat, s_feat))

    print("Avg_WSD", np.mean(ws_scores))
    

def run_ocsvm_tstr(data_fake, data_real_test, y_test):
    
    ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    
    # Training OCSVM with synthetic from Hybrid QGAN or real training data
    ocsvm.fit(data_fake)
    
    # Get OCSVM scores
    scores_ocsvm = ocsvm.decision_function(data_real_test)

    # Finde best F1-score
    precision, recall, thresholds = precision_recall_curve(y_test, scores_ocsvm)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_ocsvm = np.max(f1_scores)

    print(f"OCSVM Best F1-score: {best_f1_ocsvm:.4f}")
    preds_ocsvm = ocsvm.predict(data_real_test)
    
    # Metrics
    auc_ocsvm = roc_auc_score(y_test, scores_ocsvm)
    if auc_ocsvm < 0.5:
        auc_ocsvm = 1 - auc_ocsvm
    print(f"OCSVM AUC-ROC: {auc_ocsvm:.4f}")
    


def quality_metrics_main(datasetName):
    rootPath = os.getcwd() + "/Experiments/qgan_" + datasetName +"/"  
    
    config_params = { "seed": 123,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "shuffle": False,
            "n_workers": 0}
    trainingPaths = { "dataPath": "/datasets/" + datasetName + "/sets/",
                 "TransformsPath": rootPath + "dataTransformations/",
                 "trainingPlotsPath": rootPath + "trainingPlots/",
                 "modelRepoPath": rootPath + "checkpointModels/modelCheckPoint"}

    # Printing device availability -----------------------------------------------------
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")

    # For reproducibility -------------------------------------------------------------
    random.seed(config_params['seed'])
    torch.manual_seed(config_params['seed'])
    # ----------------------------------------------------------------------------------

    if datasetName == "CICIoMT2024":
        trainingFileName = "trainingSet_34features"
        testingFileName = "testingSet_34features"
        n_features=34
        n_qubits = 5
        n_layers = 4
        batch_size = 64
        num_eval = 134757
        n_bins = 103

    else:
        trainingFileName = "trainingSet_40features"
        testingFileName = "testingSet_40features" 
        n_features=40
        n_qubits = 3
        n_layers = 3
        batch_size = 512
        num_eval = 45145
        n_bins = 73

    generator= HQuantumGenerator(device= config_params["device"], num_features=n_features, n_qubits=n_qubits, n_layers=n_layers)
    generator.load_state_dict(torch.load(trainingPaths["modelRepoPath"] + "_G.pt"))

    # Get training -------------------------------------------------
    dataset = Dataset_IoT(dataPath=trainingPaths["dataPath"], datasetName=datasetName, fileName=trainingFileName, isTraining=False, doNormalization=1,
                                noAttacks=1, transformationsPath=trainingPaths["TransformsPath"], transformation=transforms.Compose([ ToTensor_IoT() ]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=config_params["shuffle"], num_workers=config_params["n_workers"], drop_last=True)


    # Loading testing data -----------------------------------------------------------------
    datasetTesting = Dataset_IoT(dataPath=trainingPaths["dataPath"], datasetName=datasetName, fileName=testingFileName, isTraining=False, doNormalization=1,
                            noAttacks=0, transformationsPath=trainingPaths["TransformsPath"], transformation=transforms.Compose([ ToTensor_IoT() ]))
    
    dataloaderTesting = DataLoader(datasetTesting, batch_size=batch_size, shuffle=config_params["shuffle"], num_workers=config_params["n_workers"], pin_memory=False, drop_last=True)
    # -----------------------------------------------------------------------------------
   
    
    noise = torch.randn(num_eval, n_qubits, device=config_params["device"]) * np.pi/2
    fake_features = generator(noise).detach().cpu().numpy()

    # Evaluate JSD_WSD Metrics ---------------------------------------------------------------
    calculate_jsd( dataloader.dataset.dataset[:, :-2], fake_features, n_bins=n_bins)
    calculate_wsd(dataloader.dataset.dataset[:, :-2], fake_features)

    #Evaluate TSTR metrics -------------------------------------------------------------------
    y_test_transformed = np.where(dataloaderTesting.dataset.dataset[:, -2] == 0, 1, -1)
    print("Real data")
    run_ocsvm_tstr(dataloader.dataset.dataset[:, :-2], dataloaderTesting.dataset.dataset[:, :-2], y_test_transformed)
    print("Synthetic data from HQGAN")
    run_ocsvm_tstr(fake_features, dataloaderTesting.dataset.dataset[:, :-2], y_test_transformed)


if __name__ == '__main__':

    rootPath= os.getcwd() + "/Experiments/" 
    #datasetName = "CICDDoS2019"
    datasetName = "CICIoMT2024"
    # Main function
    quality_metrics_main(datasetName=datasetName)





