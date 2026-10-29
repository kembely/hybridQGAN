import os
import random
import numpy as np
import torch
from datasetClasses_IoT import Dataset_IoT
from datasetClasses_IoT import ToTensor_IoT
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.utils as vutils
from utilFunctions import EarlyStopping
from utilFunctions import selectOptimizer
import matplotlib.pyplot as plt
import optuna
import joblib
from sklearn.metrics import roc_auc_score
from qganModel import fully_Classical_Discriminator2019
from qganModel import HQuantumGenerator
from time import time
from utilFunctions import plot_training_progress_dual_losses
from utilFunctions import plot_training_progress_metric_perEpoch
from utilFunctions import computeLatencyStats
from utilFunctions import evaluateDetectionPerformance
from statistics import mean
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import scipy
from scipy.optimize import minimize


# Setting seed for reproducibility-----------------------------------------------------------------------------------------
seed = 123 
random.seed(seed)
torch.manual_seed(seed)
# --------------------------------------------------------------------------------------------------------------

# Define hybrid quantum-classical GAN class ---------------------------------------------------------------------------------------------
class hybridQGAN():
    def __init__(self, n_features, training_data, validation_data, device,
                        batch_size, lr_disc, lr_gen, opt_disc, opt_gen, n_qubits, n_layers):
        
        # Hybrid generator and classical discriminator
        self.generator = HQuantumGenerator(device=device, num_features=n_features, n_qubits=n_qubits, n_layers=n_layers).to(device)
        self.discriminator = fully_Classical_Discriminator2019(num_features=n_features).to(device)
   
        self.dataloaderTraining = training_data                                # Training data dataloader
        self.dataloaderValidation = validation_data                             # Validation data dataloader
        self.device = device                                                    # Device used
        self.batch_size = batch_size                                            # Batch size
        self.opt_d = opt_disc(self.discriminator.parameters(), lr=lr_disc)      # Classical discriminator learning rate
        self.opt_g = opt_gen(self.generator.parameters(), lr=lr_gen)            # Hybrid generator learning rate
        self.n_qubits = n_qubits                                                # Total number of qubits / N_q
        self.q_depth = n_layers                                                 # Depth of the parameterised quantum circuit / L
        
    # Step training with one batch of data samples -------------------------------------
    def train_step_batch(self, real_data): 
        #Discriminator training
        
        # Noise sample from a Gaussian distribution 
        noise = torch.randn(self.batch_size, self.n_qubits, device=self.device) * np.pi/2
        fake_data = self.generator(noise)
        self.discriminator.zero_grad()
        errD, Wloss = self.wassertein_loss(real_data, fake_data.detach(), penalty=1)

        # Propagate discriminator gradients
        errD.backward()
        self.opt_d.step()

        #Generator training
        self.generator.zero_grad()
        outD_fake_output = self.discriminator(fake_data).view(-1)
        errG= -torch.mean(outD_fake_output)
            
        # Propagate generator gradients
        errG.backward()
        self.opt_g.step()
        return errD, errG, Wloss

    # Calculate wassertein loss per batch ----------------------------------------------
    def wassertein_loss(self, real_data, synthetic_data, penalty=0):
        gp=0
        outD_real = self.discriminator(real_data).view(-1)
        outD_fake = self.discriminator(synthetic_data).view(-1)
        if penalty==1:
            gp = self.gradient_penalty(real_data, synthetic_data)
        Dreal= torch.mean(outD_real) 
        Dfake= torch.mean(outD_fake)
        return  Dfake - Dreal + gp, Dreal - Dfake

    # Calculate gradiente penalty per batch ----------------------------------------------
    def gradient_penalty(self, real_data, synthetic_data, gp_lambda=0.1):
        batch_size = real_data.size()[0]
        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data).to(self.device)
        interpolated = alpha * real_data + (1 - alpha) * synthetic_data
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)
    
        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]
        
        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return gp_lambda * ((gradients_norm - 1) ** 2).mean()
    
    
    # Train one epoch ------------------------------------------------------------------
    def train_epoch(self):
        errD_lbatch, errG_lbatch, errW_lbatch = [], [], []
        for batch, data in enumerate(self.dataloaderTraining):
            # Data for training the discriminator
            data = data["features"]
            real_data = data.to(torch.float32).to(self.device)          
            errD, errG, wloss = self.train_step_batch(real_data)        
            errD_lbatch.append(errD.detach().item())
            errG_lbatch.append(errG.detach().item())
            errW_lbatch.append(wloss.detach().item())
        errD_epoch = np.mean(errD_lbatch)
        errG_epoch = np.mean(errG_lbatch)
        errW_epoch = np.mean(errW_lbatch)
        return errG_epoch, errD_epoch, errW_epoch


    # Validate model---------- -----------------------------------------------------
    def compute_validate_error(self, path, i, isTraining=1):
        self.discriminator.eval()                       # setting mode for evaluation
        labelList = []
        catList = []
        discScoreList = []
        detectiontimeList = []
        with torch.no_grad():
            for batch, sample in enumerate(self.dataloaderValidation):
                data = sample["features"].float().to(self.device)
                label = sample["labels"].float().to(self.device)
                cat = sample["category"].float().to(self.device)
                label = label.view(-1).cpu().detach().numpy()
                tbefore = time()
                dscore = self.discriminator(data)   
                tafter = time ()
                cat = cat.view(-1).cpu().detach().numpy()
                catList.append(cat)
                labelList.append(label)
                detectiontimeList.append((tafter - tbefore)/len(data))
                dscore = dscore.view(-1).cpu().detach().numpy()
                discScoreList.append(1-dscore)
            detectionTime2array= np.array(detectiontimeList)
            detectionTime_mean, detectionTime_std = computeLatencyStats(detectionTime2array)
            labelL = np.vstack(labelList).reshape(-1)
            dScoreL = np.vstack(discScoreList).reshape(-1)
            catL = np.vstack(catList).reshape(-1)
            AUCROC_score = roc_auc_score(labelL, dScoreL)
            error = 1 - AUCROC_score
            if i==200:  
                evaluateDetectionPerformance(labelL, catL, dScoreL, path, i, "CICDDoS2019", "EER")
        if isTraining == 1:
            self.discriminator.train()                 # back to train mode
        return error, AUCROC_score, detectionTime_mean, detectionTime_std
# ------------------------------------------------------------------------------

def training_main(trial, optuna=True):
    patience = 15
    opt_disc = selectOptimizer("optim.Adam") 
    opt_gen = selectOptimizer("optim.Adam") 
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    if optuna==True:
        config_param = { "seed": 123,
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                "shuffle": True,
                "n_workers": 2,
                "batch_size": trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512]),
                "n_epochs": 100,
                "n_qubits": trial.suggest_int("n_qubits", 2, 6, log=True),
                "q_depth": trial.suggest_int("q_depth", 2, 4, log=True),  
                "lr_d": trial.suggest_float("lr_d", 1e-4, 1e-1, log=True),   
                "lr_g": trial.suggest_float("lr_g", 1e-4, 1e-1, log=True),    
                "patience": patience,
                "opt_d": opt_disc, 
                "opt_g": opt_gen } 
    else:
        config_param = { "seed": 123,
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                "shuffle": True,
                "n_workers": 2,
                "batch_size": 512, 
                "n_epochs": 100, 
                "n_qubits": 3,
                "q_depth": 3, 
                "lr_d": 0.004,    
                "lr_g": 0.004, 
                "patience": patience,
                "opt_d": opt_disc, 
                "opt_g": opt_gen } 
        
    # Paths -------------------------------------------------------------------------
    trainingPaths = { "dataPath": "/datasets/" + "CICDDoS2019" + "/sets/",
                "TransformsPath": expRootPath + "dataTransformations/",
                "trainingPlotsPath": expRootPath + "trainingPlots/" + str(config_param["n_qubits"])+ "_" + str(config_param["q_depth"]) + "_"+ str(config_param["lr_d"]) + "_"+ str(config_param["lr_g"])+ "/",
                "modelRepoPath": expRootPath + "checkpointModels/" + str(config_param["n_qubits"])+ "_" + str(config_param["q_depth"]) + "_"+ str(config_param["lr_d"]) + "_"+ str(config_param["lr_g"]) + "/"}
    
    # Show device availability----------------------------------------------------------
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")

    # For reproduciblility -------------------------------------------------------------
    seed = 123
    random.seed(seed)
    torch.manual_seed(seed)

    # Create directories if they don't exist -------------------------------------------
    if not os.path.exists(trainingPaths["TransformsPath"]):
        os.makedirs(trainingPaths["TransformsPath"])
    if not os.path.exists(trainingPaths["trainingPlotsPath"]):
        os.makedirs(trainingPaths["trainingPlotsPath"])
    if not os.path.exists(trainingPaths["modelRepoPath"]):
        os.makedirs(trainingPaths["modelRepoPath"])
    
    # Setting training and validation data -------------------------------------------------
    datasetTraining = Dataset_IoT(dataPath=trainingPaths["dataPath"],datasetName="CICDDoS2019", fileName="trainingSet_40features", isTraining=True, doNormalization=1,
                            noAttacks=1, transformationsPath=trainingPaths["TransformsPath"], transformation=transforms.Compose([ ToTensor_IoT() ]))

    datasetValidation = Dataset_IoT(dataPath=trainingPaths["dataPath"], datasetName="CICDDoS2019", fileName="validationSet_40features", isTraining=False, doNormalization=1,
                        noAttacks=0, transformationsPath=trainingPaths["TransformsPath"], transformation=transforms.Compose([ ToTensor_IoT() ]))
  
    dataloaderTraining = DataLoader(datasetTraining, batch_size=config_param["batch_size"], shuffle=config_param["shuffle"], num_workers=config_param["n_workers"], drop_last=True)
    dataloaderValidation = DataLoader(datasetValidation, batch_size=config_param["batch_size"], shuffle=config_param["shuffle"], num_workers=config_param["n_workers"], drop_last=True)
    
    nfeatures = datasetTraining.dataset.shape[1] - 2

    #Setting hybrid QGAN model
    hqgan = hybridQGAN(n_features=nfeatures, training_data=dataloaderTraining, validation_data=dataloaderValidation, device=config_param["device"], batch_size=config_param["batch_size"],
                lr_disc=config_param["lr_d"], lr_gen=config_param["lr_g"], opt_disc=config_param["opt_d"], opt_gen=config_param["opt_g"],
                n_qubits=config_param["n_qubits"], n_layers= config_param["q_depth"])

    #Call early stopping mechanism -----------------------------------------------------
    eStopping = EarlyStopping(patience=config_param["patience"], verbose=True, minChange=0.0001, path=trainingPaths["modelRepoPath"] + "modelCheckPoint")

    # Training Loop --------------------------------------------------------------------
    errG_epochList = []
    errD_epochList = []
    errW_epochList = []
    trainingTime_epochList = []
    valError_epochList = []
    valAucrocScore_epochList= []

    start = time()
    for i in range(config_param["n_epochs"]):
        t_start = time()
        errG_epoch, errD_epoch, errW_epoch = hqgan.train_epoch()
        ttEpoch = time() - t_start
        trainingTime_epochList.append(ttEpoch)
        errG_epochList.append(errG_epoch)
        errD_epochList.append(errD_epoch)
        errW_epochList.append(errW_epoch)
        # Calculate validation error after training each epoch -------------------------
        valErrorEpoch, aurocScoreEpoch, detecTimes_mean, detecTimes_std = hqgan.compute_validate_error(path=trainingPaths["trainingPlotsPath"], i=i)
        valError_epochList.append(valErrorEpoch)
        valAucrocScore_epochList.append(aurocScoreEpoch)
        # Print Losses and statistics each 10 rounds -----------------------------------
        if i % 10 == 0:
            print(f'Iteration: {i}, Discriminator Loss: {errD_epoch:0.3f}, Generator Loss: {errG_epoch:0.3f}')
            print(f'Validation Error epoch: {valErrorEpoch:.6f}, Training Time Epoch: {ttEpoch:.6f}')
            print(f'AUCROC epoch: {aurocScoreEpoch:.6f}', f'Estimated Detection Time epoch: {detecTimes_mean:.6f}')
            print("")
        
        # Verify early stopping rule ---------------------------------------------------
        eStopping(valErr=valErrorEpoch, mDisc=hqgan.discriminator, mGen=hqgan.generator)
        if eStopping.eStop:
            print("Stopping at Epoch:", i , "Best Score Validation Error:", eStopping.bestScore, "ValidationAurocValue:", 1 + eStopping.bestScore)
            break
    # ----------------------------------------------------------------------------------
    print("")
    print(f'Total Training Time: {np.sum(trainingTime_epochList):.6f}')
    print(f'Mean Training Time per Epoch: {np.mean(trainingTime_epochList):.6f}')
    minValErrorEpoch = min(valError_epochList)
    print(f'Min validation error epoch {minValErrorEpoch:-6f}')

    # Create training progress images----------------------------------------------------------------------------------
    plot_training_progress_dual_losses(errG_epochList, errD_epochList, trainingPaths["trainingPlotsPath"])
    plot_training_progress_metric_perEpoch(errW_epochList, trainingPaths["trainingPlotsPath"], "WassersteinLoss", "Wloss","royalblue")
    plot_training_progress_metric_perEpoch(valAucrocScore_epochList, trainingPaths["trainingPlotsPath"], "AUCROC_Scores", "AUCROC", "r")
    
    # Compute statistics for best model ------------------------------------------------
    bestModel = hybridQGAN(n_features=nfeatures, training_data=dataloaderTraining, validation_data=dataloaderValidation, device=config_param["device"], 
                                        batch_size=config_param["batch_size"], lr_disc=config_param["lr_d"], lr_gen=config_param["lr_g"], opt_disc=config_param["opt_d"], 
                                        opt_gen=config_param["opt_g"], n_qubits=config_param["n_qubits"], n_layers=config_param["q_depth"])
    bestModel.discriminator.load_state_dict(torch.load(trainingPaths["modelRepoPath"] + "/3_3_0.004_0.004/modelCheckPoint" + "_D.pt"))           # Load hqgan discriminator trained
    
    print("")
    print("Testing on best model with earlystopping........")
    bestValErrorEpoch, bestAurocValueEpoch, bestDetectionTimeMean, bestDetectionTimeStd= bestModel.compute_validate_error(path=trainingPaths["trainingPlotsPath"], i=200, isTraining=0) 
    print(f'Best Validation Error : {bestValErrorEpoch:.6f}')
    print(f'Best AUCROC : {bestAurocValueEpoch:.6f}', f'Best Estimated Detection Time: {bestDetectionTimeMean:.6f}')
    
    return minValErrorEpoch    


if __name__ == '__main__':    
    
    expRootPath =  os.getcwd() + "/Experiments/qgan_CICDDoS2019/"  #Setting root training path
    optunaOn = 0
    if not os.path.exists(expRootPath):
        os.makedirs(expRootPath)
    if(optunaOn == 1):                                             # Training QGAN optimizing hyperparameters with Optuna 
        sampler = optuna.samplers.TPESampler(seed=seed)
        optuna_trial = optuna.create_study(sampler=sampler, direction='minimize')
        optuna_trial.optimize(func=training_main, n_trials=20)
        joblib.dump(optuna_trial, expRootPath + "checkpointModels/optunaOptimizationResults.pkl")
        print(optuna_trial.best_trial)
        print(optuna_trial.best_params)
    else:
        training_main(0, False)                                    # Training QGAN without optimizing hyperparameters with Optuna
