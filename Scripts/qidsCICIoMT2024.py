import os
import random
import numpy as np
import torch
from datasetClasses_IoT import Dataset_IoT
from datasetClasses_IoT import ToTensor_IoT
from torchvision import transforms
from torch.utils.data import DataLoader
import time
from dataPreparation_CICIoMT2024 import datasetStatsCICIoMT2024
from torchinfo import summary
from qganModel import fully_Classical_Discriminator2024
from qganModel import HQuantumGenerator
from utilFunctions import evaluateDetectionPerformance
from utilFunctions import computeLatencyStats


# Intrusion Detection Class ----------------------------------------------------------
class HQIDS():
    def __init__(self, device, dataloaderTesting, pathModel, pathSaveIDScores, pathSaveResults, batch_size):
        self.device = device
        self.dataloader = dataloaderTesting
        self.batch_size = batch_size
        self.pathSaveIDScores = pathSaveIDScores
        self.pathSaveResults = pathSaveResults + str("D") + "_"
        self.discriminator = fully_Classical_Discriminator2024(num_features=34).to(device)
        self.generator= HQuantumGenerator(device=self.device, num_features=34, n_qubits=5, n_layers=4)

        self.discriminator.load_state_dict(torch.load(pathModel + "_D.pt"))
        self.generator.load_state_dict(torch.load(pathModel + "_G.pt"))
        # Load and print parameter statistics of trained model
        self.discriminator.eval()
        summary(self.discriminator)
        summary(self.generator)

    # Get labels and categories---------------------------------------------------------
    def getLabels_categories(self, sample):
        labels = sample["labels"].float().view(-1).cpu().detach().numpy()
        categories = sample["category"].float().view(-1).cpu().detach().numpy()
        return labels, categories
    # ----------------------------------------------------------------------------------

    # Compute discrimination score ------------------------------------------------------
    def discriminationScore(self, sample):
        data = sample["features"].float().to(self.device)
        Dscore = self.discriminator(data)
        Dscore = Dscore.view(-1).cpu().detach().numpy()
        IdScore = 1 - Dscore
        return IdScore
    # ----------------------------------------------------------------------------------

    # Compute and save intrusion detection scores ----------------------------------------
    def evaluate_and_export_metrics(self):
        print("Testing and Saving Discrimination Scores ........")
        collectedLabels = []
        collectedCats = []
        collectedDetectionTimes = []
        collectedDScores = []
        with torch.no_grad():
            for sample in self.dataloader:
                # Get labels and categories
                labels, cats = self.getLabels_categories(sample)
                collectedLabels.append(labels)
                collectedCats.append(cats)
                # Compute Discrimination Losses
                tbegin = time.time()
                Dscore = self.discriminationScore(sample)
                collectedDetectionTimes.append(time.time() - tbegin)
                collectedDScores.append(Dscore)
        # Save labels and categories
        all_labels = np.vstack(collectedLabels).reshape(-1)
        all_categories = np.vstack(collectedCats).reshape(-1)
        allLabelsCategories = np.vstack((all_labels, all_categories))
        np.save(self.pathSaveIDScores + "labelsCats.npy", allLabelsCategories)   #before labels.npy
        
        # Save detection scores and detection times
        all_dScores = np.vstack(collectedDScores).reshape(-1)
        np.save(self.pathSaveIDScores + "D_scores.npy", all_dScores) # before losse_D
        all_detectionTimes = np.vstack(collectedDetectionTimes).reshape(-1)
        np.save(self.pathSaveIDScores + "detectionTimes_D.npy", all_detectionTimes)
        
        print("Testing and Saving Disc. Scores ........ Finished.")
        evaluateDetectionPerformance(all_labels, all_categories, all_dScores, self.pathSaveResults, 200, "CICIoMT2024") # EER evaluation
        #evaluateDetectionPerformance(all_labels, all_categories, all_dScores, self.pathSaveResults, 200, "CICIoMT2024", "Y") # Youden's Index evaluation

        detectionTimes_mean, detectionTimes_std = computeLatencyStats(all_detectionTimes)
        print(f'Detection Times Mean: {detectionTimes_mean:.6f} s ')
        print(f'Detection Times Std:  {detectionTimes_std:.6f} s ')
    # ----------------------------------------------------------------------------------


def ID_main():
    rootPath = os.getcwd() + "/Experiments/qgan_CICIoMT2024/" 
    datasetName = "CICIoMT2024" 
    testingFileName = "testingSet_34features" 

    config_params = { "seed": 123,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "shuffle": False,
            "n_workers": 0,
            "batch_size": 1}
    trainingPaths = { "dataPath": "/datasets/" + datasetName + "/sets/",
                 "TransformsPath": rootPath + "dataTransformations/",
                 "trainingPlotsPath": rootPath + "trainingPlots/",
                 "modelRepoPath": rootPath + "checkpointModels/modelCheckPoint",
                 "IDScoresPath": rootPath + "IDScores/",
                 "ResultsPath": rootPath + "Results/"}

    # Creating directories if they don't exist-------------------------------------------
    if not os.path.exists(trainingPaths["IDScoresPath"]):
            os.makedirs(trainingPaths["IDScoresPath"])
    if not os.path.exists(trainingPaths["ResultsPath"]):
            os.makedirs(trainingPaths["ResultsPath"])
    
    # Printing device availability -----------------------------------------------------
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")
    # For reproducibility -------------------------------------------------------------
    random.seed(config_params['seed'])
    torch.manual_seed(config_params['seed'])
    # ----------------------------------------------------------------------------------

    # Loading testing data -----------------------------------------------------------------
    datasetTesting = Dataset_IoT(dataPath=trainingPaths["dataPath"], datasetName=datasetName, fileName=testingFileName, isTraining=False, doNormalization=1,
                            noAttacks=0, transformationsPath=trainingPaths["TransformsPath"], transformation=transforms.Compose([ ToTensor_IoT() ]))
    
    dataloaderTesting = DataLoader(datasetTesting, batch_size=config_params["batch_size"], shuffle=config_params["shuffle"], num_workers=0, pin_memory=False, drop_last=True)
    # ----------------------------------------------------------------------------------

    # Load Trained Model ---------------------------------------------------------------
    trainedQGAN = HQIDS(device=config_params["device"], dataloaderTesting=dataloaderTesting, batch_size=config_params["batch_size"], pathModel=trainingPaths["modelRepoPath"], 
                        pathSaveIDScores=trainingPaths["IDScoresPath"], pathSaveResults=trainingPaths["ResultsPath"])
    # ----------------------------------------------------------------------------------

    # Compute and save anomaly detection losses ----------------------------------------
    trainedQGAN.evaluate_and_export_metrics()
    # ----------------------------------------------------------------------------------

if __name__ == '__main__':

    rootPath= os.getcwd() + "/Experiments/qgan_CICIoMT2024/" 

    # Main function
    ID_main()
