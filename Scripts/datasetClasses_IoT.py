import os
import random
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from utilFunctions import dataLoad
from dataPreparation_CICDDoS2019 import datasetStatsCICDDoS2019  # when running QGAN with CICDDoS2019
from dataPreparation_CICIoMT2024 import datasetStatsCICIoMT2024 # when running QGAN with CICIoMT2024


class Dataset_IoT(torch.utils.data.Dataset):
      def __init__(self, dataPath, datasetName, fileName, isTraining, doNormalization, noAttacks, transformationsPath, transformation):
            self.isTraining = isTraining
            self.transformationPaths = transformationsPath
            self.path = os.getcwd() + dataPath + fileName + ".npy"
            self.dataset = dataLoad(self.path)

            #Print dataset statistics
            if datasetName=="CICDDoS2019": 
                  datasetStatsCICDDoS2019(self.dataset)
            elif datasetName == "CICIoMT2024":
                  datasetStatsCICIoMT2024(self.dataset)
            else:
                  raise ValueError(f"Dataset not supported: {datasetName}")
      
            # Use only normal data
            if (noAttacks == 1):
                  self.dataset = np.delete(self.dataset, np.where(self.dataset[:,-2:-1] != 0)[0], axis=0)

            # Data normalization
            if (doNormalization == 1):
                  self.dataNormalization1()
                  self.dataNormalization2()

            # Transformation ToTensorIoT
            self.transformation = transformation

      def dataNormalization1(self):
            datasetFeatures = self.dataset[:, :-2]
            if(self.isTraining == True):      
                  transf_std_scale = StandardScaler().fit(datasetFeatures)
                  joblib.dump(transf_std_scale, self.transformationPaths + "transf_normalization.pkl")
            else: 
                  transf_std_scale = joblib.load(self.transformationPaths + "transf_normalization.pkl")
            datasetFeatures = transf_std_scale.transform(datasetFeatures)
            self.dataset[:, :-2] = datasetFeatures
      
      # Normalization (-1,1)
      def dataNormalization2(self):
            datasetFeatures = self.dataset[:, :-2]
            if(self.isTraining == True):      
                  transf_minmax_scale = MinMaxScaler((-1, 1)).fit(datasetFeatures)
                  joblib.dump(transf_minmax_scale, self.transformationPaths + "transf_minmax_scale.pkl")
            else: 
                  transf_minmax_scale = joblib.load(self.transformationPaths + "transf_minmax_scale.pkl")

            datasetFeatures = transf_minmax_scale.transform(datasetFeatures)
            self.dataset[:, :-2] = datasetFeatures

      # --------------------------------------------------------------------------------------------------------------

      def __len__(self):
            return len(self.dataset)

      def __getitem__(self, idx):
            if torch.is_tensor(idx):
                  idx = idx.tolist()
            sample = {  "features": self.dataset[idx, :-2],
                        "labels": self.dataset[idx, -2:-1],
                        "category": self.dataset[idx, -1:]
                        }
            if self.transformation:
                  sample = self.transformation(sample)
            return sample

# Convert ndarrays in sample to Tensors
class ToTensor_IoT(object):
      def __call__(self, sample):
            transformedSample = { "features": torch.from_numpy(sample["features"]),
                              "labels": torch.from_numpy(sample["labels"]),
                              "category": torch.from_numpy(sample["category"])
                  }
            return transformedSample

