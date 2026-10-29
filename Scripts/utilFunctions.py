import os
import numpy as np
import torch
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from pylab import savefig
from IPython.display import clear_output
from scipy.stats import ecdf  # Requerido: SciPy >= 1.11


# Plot GLoss and DiscLoss progress ------------------------------------------------------------------------------
def plot_training_progress_dual_losses(lossG, lossD, pathSavePlot):
    # Don't plot if not enough data
    if len(lossG) < 2:
        return
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 7))
    prop = fm.FontProperties(fname='times_new_roman.ttf', size='large')
    # Generator Loss
    ax1.set_title("Loss During Training", fontproperties=prop, size=22)
    ax1.plot(lossG, label="Generator loss", color="royalblue", linewidth=1)
    ax1.plot(lossD, label="Discriminator loss", color="r", linewidth=1)
    ax1.legend(loc="best", fontsize=16)
    ax1.set_xlabel("Epoch", fontproperties=prop, size=18)
    ax1.set_ylabel("Loss", fontproperties=prop, size=18)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid(True)
    #plt.show()
    plt.tight_layout() # Adjusts plot to ensure everything fits without overlap

    if not os.path.exists(pathSavePlot):
        os.makedirs(pathSavePlot)
    plt.savefig(pathSavePlot + "G_D_lossEpoch.png", bbox_inches='tight')
    plt.savefig(pathSavePlot + "G_D_lossEpoch.pdf", bbox_inches='tight')

# Plot any metric progress during training-------------------------------------------------------------------------
def plot_training_progress_metric_perEpoch(train_metric, pathSavePlot, name, labely, color):
    # Don't plot if not enough data
    if len(train_metric) < 2:
        return
    clear_output(wait=True)
    fig, (ax1) = plt.subplots(1, 1, figsize=(18, 9))
    prop = fm.FontProperties(fname='times_new_roman.ttf', size='large')
    # Generator Loss
    ax1.set_title(name, fontproperties=prop, size=14)
    ax1.plot(train_metric, label=labely, color=color)
    ax1.legend(loc="best")
    ax1.set_xlabel("Epoch", fontproperties=prop, size=14)
    ax1.set_ylabel(labely, fontproperties=prop, size=14)
    ax1.grid()
    #plt.show()
    if not os.path.exists(pathSavePlot):
        os.makedirs(pathSavePlot)
    plt.savefig(pathSavePlot + name +".png")
    plt.savefig(pathSavePlot + name +".pdf")

# Plot AUC ROC on validation or test data ---------------------------------------------------------------------------
def plot_AUCROC(fpr, tpr, pathSavePlot, epoch, label=" ", linestyle=':', marker='o', color='r', markersize=6):
    fprSampled = []
    tprSampled = []
    fprSampled.append(fpr[0])
    tprSampled.append(tpr[0])
    for i in range(1, fpr.size):
        d = pow(( pow(fprSampled[-1] - fpr[i], 2) + pow(tprSampled[-1] - tpr[i], 2) ), 0.5)
        if( d > 0.07 ):
            fprSampled.append(fpr[i])
            tprSampled.append(tpr[i])
    ownFont = fm.FontProperties(fname='times_new_roman.ttf', size='large')
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.plot(fprSampled, tprSampled, label=label, linestyle=linestyle, marker=marker, color=color, linewidth=1, markersize=markersize)
    ax.set_xlabel('False positive rate', fontproperties=ownFont, size=14)
    ax.set_ylabel('True positive rate', fontproperties=ownFont, size=14)
    ax.set_title('ROC Curves', fontproperties=ownFont, size=14)
    ax.legend(loc='best', prop=ownFont, fontsize=25)
    if not os.path.exists(pathSavePlot):
        os.makedirs(pathSavePlot)
    plt.savefig(pathSavePlot + str(epoch) + "aucRoc.png")
    plt.savefig(pathSavePlot +  str(epoch) +"aucRoc.pdf")


# Compute the Equal Error Rate (EER) ---------------------------------------------------------------------------
def computeEER(fpr, tpr, thresholds):
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_thresh = interp1d(fpr, thresholds)(eer)
    return eer, eer_thresh

# Compute Youden Index Criterion
def computeWeightedYoudenIndex(fpr, tpr, thresholds, weight=0.5):
    sensitivity = tpr
    specificity = 1 - fpr
    j_weighted = 2 * (weight * sensitivity + (1 - weight)* specificity) - 1
    idx = np.argmax(j_weighted)
    return thresholds[idx]

def computeYoudenIndex(fpr, tpr, thresholds):
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]

# Compute detection rates and plot ROC ---------------------------------------------
def evaluateDetectionPerformance(labels, categories, DScores, pathSaveResults, epoch, dataset, thresholdCriterion="EER"):
    # Print and plot AUCROC
    aucrocValue = roc_auc_score(labels, DScores)
    print(f'AUCROC: {aucrocValue:.6f}')
    fpr, tpr, thresholds = roc_curve(labels, DScores, pos_label=1)

    #Drawing AUCROC plot
    plot_AUCROC(fpr, tpr, pathSaveResults, epoch, label="AUC = " + str('%.4f'%(aucrocValue)), linestyle=':', marker='s', color='#A65E34', markersize=4)
    
    # Compute detection rates for the equal error rate or the youden index threshold
    if thresholdCriterion=="EER":
        print("Calculating EER threshold")
        eer, eer_thresh = computeEER(fpr, tpr, thresholds)
    else:
        print("Calculating Youden Index threshold")
        eer_thresh = computeWeightedYoudenIndex(fpr,tpr,thresholds, 0.9)
    
    predictions = DScores
    predictions[predictions >= eer_thresh] = 1
    predictions[predictions < eer_thresh] = 0
    print("Threshold:", eer_thresh)

    #Calculate detection metrics
    acc, pre, rec, f1, cf_matrix, TP, TN, FN, FP = computeDetectionMetrics(labels, predictions)
    print(f'Accuracy: {acc:.7f} ' + f'Precision: {pre:.7f} ' + f'Recall: {rec:.7f} ' + f'F1-Score: {f1:.7f}')
    print(f'TP: {TP:.7f} ' + f'TN: {TN:.7f} ' + f'FN: {FN:.7f} ' + f'FP: {FP:.7f}')
    
    #Creating Confusion Matrix
    drawConfusionMatrix(labels, predictions, pathSaveResults, epoch)
    
    detRate_normal = TN/(TN + FP)
    detRate_attack = TP/(TP + FN)
    print(f'DetectionRate_normal: {detRate_normal:.7f}')
    print(f'DetectionRate_attack: {detRate_attack:.7f}')

    if dataset == "CICDDoS2019":
        detectionRatesCategories_CICDDoS2019(categories, predictions) #CICDDoS2019 dataset
    elif dataset == "CICIoMT2024": 
        detectionRatesCategories_CICIoMT2024(categories, predictions) #CICIoMT2024 dataset
    print("")

# Compute detection rates per category ------------------------------------------------
def detectionRatesCategories_CICDDoS2019(categories, predictions):
    CICDDOS_LABELS = {
    0: "Normal",
    1: "Portmap",
    2: "Syn",
    3: "UDP",
    4: "UDPLag",
    5: "MSSQL",
    6: "NetBIOS",
    7: "LDAP",
    }
    cats = np.asarray(categories, dtype=int).ravel()
    preds = np.asarray(predictions, dtype=int).ravel()
    num_classes = 14
    class_counts = np.bincount(cats, minlength=num_classes)
    total_attacks = int(np.sum(class_counts[1:num_classes]))
    others_attacks = int(np.sum(class_counts[8:num_classes]))

    detected = np.zeros(num_classes, dtype=np.float64)
    detected[0] = np.count_nonzero((cats == 0) & (preds == 0))
    attack_mask = preds != 0
    if np.any(attack_mask):
        attack_detected = np.bincount(cats[attack_mask], minlength=num_classes)
        detected[1:] = attack_detected[1:]

    # 3. Tasas de detección individuales (evitando división por cero)
    rates = np.zeros(num_classes, dtype=np.float64)
    valid_mask = class_counts > 0
    rates[valid_mask] = detected[valid_mask] / class_counts[valid_mask]

    # 4. Tasa global sobre las 7 clases principales evaluadas
    primary_detected_sum = np.sum(detected[1:8])
    primary_total_sum = np.sum(class_counts[1:8])
    rate_total_attacks = (
        float(primary_detected_sum / primary_total_sum) if primary_total_sum > 0 else 0.0
    )

    print(f"Normal: {class_counts[0]:d} | Attack: {total_attacks:d}")
        
    detail_counts = " | ".join(
            f"{name}: {class_counts[idx]:d}" for idx, name in CICDDOS_LABELS.items()
        )
    print(f"{detail_counts} | Others: {others_attacks:d}")

    print(f"Normal: {rates[0]:.6f} | Attacks: {rate_total_attacks:.6f}")
        
    detail_rates = " | ".join(
            f"{name}: {rates[idx]:.6f}" for idx, name in CICDDOS_LABELS.items()
        )
    print(detail_rates)


# --------------------------------------------------------------------------------------------------------------
def detectionRatesCategories_CICIoMT2024(categories, predictions):
    CICIOMT_GROUPS = {
    "ARP_Spoofing": [1],
    "MQTT-DDOS": [2, 3],
    "MQTT-DOS": [4, 5],
    "MQTT_Malformed_Data": [6],
    "Recon": [7, 8, 9, 10],
    "DDoS": [11, 12, 13, 14],
    "DoS": [15, 16, 17, 18],
    }
    cats = np.asarray(categories, dtype=int).ravel()
    preds = np.asarray(predictions, dtype=int).ravel()
    num_classes = 19
    class_counts = np.bincount(cats, minlength=num_classes)
    total_attacks = int(np.sum(class_counts[1:num_classes]))

    detected = np.zeros(num_classes, dtype=np.float64)
    detected[0] = np.count_nonzero((cats == 0) & (preds == 0))

    attack_mask = preds != 0
    if np.any(attack_mask):
        attack_detected = np.bincount(cats[attack_mask], minlength=num_classes)
        detected[1:] = attack_detected[1:]

    # 3. Tasa de detección por clase individual (división protegida contra cero)
    rates = np.divide(
        detected,
        class_counts,
        out=np.zeros_like(detected, dtype=np.float64),
        where=class_counts > 0,
    )

    sum_detected_attacks = np.sum(detected[1:num_classes])
    rate_total_attacks = (
        float(sum_detected_attacks / total_attacks) if total_attacks > 0 else 0.0
    )

    # 5. Agrupación por familias de ataque (Mantiene el cálculo macro original)
    group_counts = {}
    group_rates = {}

    for group_name, class_indices in CICIOMT_GROUPS.items():
        group_counts[group_name] = int(np.sum(class_counts[class_indices]))
        # Promedio macro de las tasas de detección individuales del grupo
        group_rates[group_name] = float(np.mean(rates[class_indices]))

    # Tasa específica de ReconPing (índice 8) aislada en el reporte original
    recon_ping_rate = float(rates[8])

    print(f"Normal: {class_counts[0]:d}  Attack: {total_attacks:d}")
        
    counts_summary = " ".join(
            f"{grp}: {group_counts[grp]:d}" for grp in CICIOMT_GROUPS
        )
    print(f"Normal: {class_counts[0]:d} {counts_summary}")

    print(f"Normal: {rates[0]:.6f}  Attack: {rate_total_attacks:.6f}")
        
    rates_summary = " ".join(
            f"{grp}: {group_rates[grp]:.6f}" for grp in CICIOMT_GROUPS
        )
    print(f"Normal: {rates[0]:.6f} ReconPing: {recon_ping_rate:.6f} {rates_summary}")


# Compute Metrics ----------------------------------------------------------------------------------------------
def computeDetectionMetrics(labels, predictions):
    cf_matrix = confusion_matrix(labels, predictions)   # Positive = 0; Negative = 1
    TP = cf_matrix[1][1]
    FN = cf_matrix[1][0]
    FP = cf_matrix[0][1]
    TN = cf_matrix[0][0]
    acc = accuracy_score(labels, predictions)
    pre = precision_score(labels, predictions)
    rec = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return acc, pre, rec, f1, cf_matrix, TP, TN, FN, FP

def drawConfusionMatrix(labels, predictions, pathSave, iter):
    cf_matrix = confusion_matrix(labels, predictions) 
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix)
    disp.plot()
    if not os.path.exists(pathSave):
        os.makedirs(pathSave)
    plt.savefig(pathSave + str(iter) + "confusionMatrix.png")
    plt.savefig(pathSave + str(iter) + "confusionMatrix.pdf")

def computeLatencyStats(detectionTimes):
    # 5 highest and lowest detection times are removed to avoid outliers
    sorted_times = np.sort(detectionTimes)
    trimmed_times = sorted_times[5:-5]
    detectionTimes_mean = np.mean(trimmed_times)
    detectionTimes_std = np.std(trimmed_times)
    return detectionTimes_mean, detectionTimes_std


# Choose Optimizer ------------------------------------------------------------------------------
def selectOptimizer(optimizer):
    if(optimizer == "optim.Adam"):
        return optim.Adam
    elif(optimizer == "optim.SGD"):
        return optim.SGD
    elif(optimizer == "optim.RMSprop"):
        return optim.RMSprop
    else:
        return optimizer

# Load Data ---------------------------------------------------------------------------------------------
def dataLoad(path):
    old_dataload = np.load
    np.load = lambda *args,**kwargs: old_dataload(*args, allow_pickle=True, **kwargs)
    new_dataload = np.load(path)
    np.load = old_dataload
    return new_dataload

# Early Stopping Class-----------------------------------------------------------------------------------
class EarlyStopping:
    """Stop training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, minChange=0, path='checkpoint.pt', trace_func=print):
        """ Args:
            patience (int): Waiting count after last time validation loss improved. Default: 5
            verbose (bool): If True, prints a message when validation loss improve. Default: False
            minChange (float): Minimum change to qualify as an improvement. Default: 0
            path (str): Path for the improved model to be saved. Default: 'checkpoint.pt'
            trace_func (function): trace print function. Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.bestScore = None
        self.eStop = False
        self.valErrMin = np.Inf
        self.minChange = minChange
        self.path = path
        self.trace_func = trace_func
    
    def __call__(self, valErr, mDisc, mGen):
        score = -valErr
        if self.bestScore is None:
            self.bestScore = score
            self.save_checkpoint(valErr, mDisc, mGen)
        elif score <= self.bestScore + self.minChange:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.eStop = True
        else:
            self.bestScore = score
            self.save_checkpoint(valErr, mDisc, mGen)
            self.counter = 0

    def save_checkpoint(self, valErr, mDisc, mGen):
        '''Save model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.valErrMin:.6f} --> {valErr:.6f}).  Saving model ...')
        torch.save(mDisc.state_dict(), self.path + "_D.pt")
        torch.save(mGen.state_dict(), self.path + "_G.pt")

        self.valErrMin = valErr