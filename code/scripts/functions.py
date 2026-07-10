import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn.metrics import confusion_matrix
import pandas as pd

import numpy as np

def helperPrecisionRecall(confmat, Sort='classic'):
    """
    Parameters
    ----------
    confmat : array 
        confusion matrix where N depends on the 'Sort' parameter
    Sort : string
        Allows to control the expected categories. The default is 'classic', 
        are allowed : 'classic', 'chf', 'nsr1', 'nsr2'
        
    Returns
    -------
    precision : array 
        precision 
    recall : array 
        recall
    f1_score : array 
        f1_score

    Calculate precision, recall and f1_score from a dynamically sized confusion matrix
    """
    # Détermination du nombre de classes attendues en fonction du paramètre Sort
    if Sort == 'classic':
        n_classes = 3  # Classes attendues : ARR, CHF, NSR
    elif Sort in ['chf', 'nsr1']:
        n_classes = 2  # Classes attendues : CHF, NSR
    elif Sort == 'nsr2':
        n_classes = 2  # Classes attendues : ARR, NSR
    else:
        raise ValueError("Unsupported arguments for Sort ; the supported arguments are : 'classic', 'chf', 'nsr1', 'nsr2'")

    confmat = np.asarray(confmat, dtype=float)

    # Vérification pour s'assurer que la taille de la matrice correspond au nombre de classes
    if confmat.shape[0] != n_classes or confmat.shape[1] != n_classes:
        raise ValueError(f"Pour Sort='{Sort}', la matrice de confusion doit être de taille ({n_classes}, {n_classes}). Taille reçue : {confmat.shape}.")

    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1_score = np.zeros(n_classes)

    # La boucle s'adapte désormais à n_classes plutôt qu'à un '3' codé en dur
    for i in range(n_classes):
        
        total_pred = np.sum(confmat[:, i])
        if total_pred > 0:
            precision[i] = confmat[i, i] / total_pred * 100
        else:
            precision[i] = 0.0

        total_true = np.sum(confmat[i, :])
        if total_true > 0:
            recall[i] = confmat[i, i] / total_true * 100
        else:
            recall[i] = 0.0

        if (precision[i] + recall[i]) > 0:
            f1_score[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1_score[i] = 0.0

    return np.array(precision), np.array(recall), np.array(f1_score)

def Classifier_eq(x,y,model,percentage=0.8,Sort='classic'):
    """

    Parameters
    ----------
    x  : list
        signals on which to perform the classification
    y : list
        labels of the signals
    model : sklearn model
        model used for the classification
    percentage : float, optional
        percentage of each categories of data to use set to 80% by default
    Sort : string    
        Allows to control the expected categories. The default is 'classic', 
        are allowed : 'classic', 'chf', 'nsr1', 'nsr2'

    Returns
    -------
    y_predict : array
        labels predict by the model
    Y_test : array
        real labels 
    err_rate : float
        error rate number of wrong prediction on number of prediction
        
    Use function Random_Data to create training and test variables randomly 
    from a dataset, apply the selected classifier model return the prediction, 
    the test used and the corresponding error rate 
    """
    X_train, Y_train, X_test, Y_test = Random_Data(x,y,percentage)
    
    model.fit(X_train, Y_train)#train it
    model.score(X_train, Y_train)

    y_predict = model.predict(X_test)
    
    err_rate = (y_predict != Y_test).mean()

    if Sort == 'classic':
        class_names = ['A', 'C', 'N']  # Classes attendues : ARR, CHF, NSR
    elif Sort in ['chf', 'nsr1']:
        class_names = ['C', 'N']  # Classes attendues : CHF, NSR
    elif Sort == 'nsr2':
        class_names = ['A', 'N']  # Classes attendues : ARR, NSR
    else:
        raise ValueError("Unsupported arguments for Sort ; the supported arguments are : 'classic', 'chf', 'nsr1', 'nsr2'")
    
    cf_matrix = confusion_matrix(y_true=Y_test,y_pred=y_predict,labels=class_names)
    precision,recall,f1_score = helperPrecisionRecall(cf_matrix,Sort)
    
    return y_predict,Y_test,err_rate,precision,recall,f1_score  

def Classifier_eq_iter(x,y,model,percentage=0.8,iter=100,Sort='classic') :
    """
    

    Parameters
    ----------
    x  : list
        signals on which to perform the classification
    y : list
        labels of the signals
    model : sklearn model
        model used for the classification
    percentage : float, optional
        percentage of each categories of data to use set to 80% by default
    iter : int, optional
        Number of Classifier_eq iterations. The default is 100.
    Sort : string
        Allows to control the expected categories. The default is 'classic', 
        are allowed : 'classic', 'chf', 'nsr1', 'nsr2'

    Returns
    -------
    float
        mean of Classifier_eq error rates
    PRTable : pandas.DataFrame
        mean of the precion, recall and f1 score displayed in a pandas DataFrame
    
    Run Classifier_eq a certain number of times 
    and return the mean of the error rates, the precion, recall and f1 score

    """
    Err,Pr,Rc,F1=[],[],[],[]
    for k in range(iter) :
        _,_,err,precision,recall,f1=Classifier_eq(x,y,model,percentage,Sort)
        Err.append(err)
        Pr.append(precision)
        Rc.append(recall)
        F1.append(f1)
    
    if Sort == 'classic':
        class_names = ['A', 'C', 'N']  # Classes attendues : ARR, CHF, NSR
    elif Sort in ['chf', 'nsr1']:
        class_names = ['C', 'N']  # Classes attendues : CHF, NSR
    elif Sort == 'nsr2':
        class_names = ['A', 'N']  # Classes attendues : ARR, NSR
    else:
        raise ValueError("Unsupported arguments for Sort ; the supported arguments are : 'classic', 'chf', 'nsr1', 'nsr2'")
            
    Pr=np.mean(Pr,axis=0)
    Rc=np.mean(Rc,axis=0)
    F1=np.mean(F1,axis=0)
    PRTable = pd.DataFrame({"Precision": Pr,"Recall": Rc,"F1_Score": F1},index=class_names)

    return np.mean(Err),PRTable

def Import_Data(Path_data,Path_atome,Sort='classic') :
    """
    Parameters
    ----------
    Path_data : string
        path to the file ECGData.mat which must be downloaded from Physionet
    Path_atome : string
        path to the file where the atomes and activation times are stored. 
        The Atomes.py file shall be run and his result download before using 
        this function you may use the Atomes.mat file instead
    Sort : string
        Allows to control the expected categories. The default is 'classic', 
        are allowed : 'classic', 'chf', 'nsr1', 'nsr2'
    
    Returns
    -------
    X : list
        activation times for each atomes for each signal
    y : list
        labels of X 
    signaux : list
        raw ecg signals
    D : list
        atomes of the signal
    
    Import and organise datas in usable way

    """
    
    mat=scipy.io.loadmat(Path_data)
    xdata=mat['ECGData']
    signaux=xdata[0][0][0]
    ARR=96#number or arr type signals
    CHF=ARR+30#CHF-ARR : number of chf type signals
    
    mat2=scipy.io.loadmat(Path_atome)
    z_arr=mat2['z_arr']#activation times of the atomes for each signal types
    z_chf=mat2['z_chf']
    z_nsr=mat2['z_nsr']
    d_arr=mat2['d_arr']
    d_chf=mat2['d_chf']
    d_nsr=mat2['d_nsr']
    
    D=[d_arr,d_chf,d_nsr]
    Z=[z_arr,z_chf,z_nsr]
    
    X=[]
    for i in range(96) :
        X.append(list(z_arr[0][i])+list(z_arr[1][i])+list(z_arr[2][i]))
    for i in range(30) :
        X.append(list(z_chf[0][i])+list(z_chf[1][i])+list(z_chf[2][i]))
    for i in range(36) :
        X.append(list(z_nsr[0][i])+list(z_nsr[1][i])+list(z_nsr[2][i]))
    n_sample=len(signaux)
    y= np.array(["" for j in range(n_sample)])
    if Sort == 'classic' :
        y[:ARR],y[ARR:CHF],y[CHF:]="ARR","CHF","NSR"
    elif Sort == 'chf' :
        y[:ARR],y[ARR:CHF],y[CHF:]="CHF","CHF","NSR"
    elif  Sort== 'nsr1' :
        y[:ARR],y[ARR:CHF],y[CHF:]="NSR","CHF","NSR"
    elif  Sort == 'nsr2' :
        y[:ARR],y[ARR:CHF],y[CHF:]="ARR","NSR","NSR"
    else :
        raise ValueError("Unsupported arguments for Sort ; the supported arguments are : 'classic', 'chf', 'nsr1', 'nsr2'")
    
    return X,y,signaux,D,Z

def Random_Data(x,y,percentage=0.8) :
    """
    

    Parameters
    ----------
    x : list
        signals to organize
    y : list
        labels of x
    percentage : float, optional
        Percentage of each categories of data to use for training. The default is 0.8.

    Returns
    -------
    X_train : array
        randomly chosen trainig datas 
    Y_train : array
        X_train corresponding labels
    X_test : array
        randomly chosen test datas
    Y_test :array
        X_test corresponding labels
    """
    
    ARR=95
    CHF=ARR+30

    x_arr,x_chf,x_nsr=x[:ARR],x[ARR:CHF],x[CHF:]
    y_arr,y_chf,y_nsr=y[:ARR],y[ARR:CHF],y[CHF:]
    n_arr,n_chf,n_nsr=len(x_arr),len(x_chf),len(x_nsr)
    rd_arr,rd_chf,rd_nsr=np.random.permutation(n_arr),np.random.permutation(n_chf),np.random.permutation(n_nsr)
    nt_arr,nt_chf,nt_nsr=int(percentage*n_arr),int(percentage*n_chf),int(percentage*n_nsr)

    x_arr,x_chf,x_nsr=np.array(x_arr)[rd_arr],np.array(x_chf)[rd_chf],np.array(x_nsr)[rd_nsr]
    x_arr,x_chf,x_nsr=list(x_arr),list(x_chf),list(x_nsr)
    y_arr,y_chf,y_nsr=y_arr[rd_arr],y_chf[rd_chf],y_nsr[rd_nsr]
    y_arr,y_chf,y_nsr=list(y_arr),list(y_chf),list(y_nsr)
    X_train=np.array(x_arr[:nt_arr]+x_chf[:nt_chf]+x_nsr[:nt_nsr])
    Y_train=np.array(y_arr[:nt_arr]+y_chf[:nt_chf]+y_nsr[:nt_nsr])
    X_test=np.array(x_arr[nt_arr:]+x_chf[nt_chf:]+x_nsr[nt_nsr:])
    Y_test=np.array(y_arr[nt_arr:]+y_chf[nt_chf:]+y_nsr[nt_nsr:])

    r = np.random.permutation(len(Y_test))
    r2 = np.random.permutation(len(Y_train))
    X_test, Y_test=X_test[r], Y_test[r]
    X_train, Y_train = X_train[r2], Y_train[r2]
    
    return X_train, Y_train, X_test, Y_test  


def reconstruire_signaux(z_activ, d_activ):
     """
     

     Parameters
     ----------
     z_activ : array of size (n_atoms, n_trials, n_times_valid)
         activation times of a signal's atomes
     d_activ : array of size (n_atoms, n_times_atom)
         atomes of a signal

     Returns
     -------
     X_reconstruit : array (n_trials,n_times_reconstructed)
         rebuilt signa

     """

     n_atoms, n_trials, n_times_valid = z_activ.shape
     n_times_atom = d_activ.shape[1]
     

     n_times_reconstructed = n_times_valid + n_times_atom - 1
     
     X_reconstruit = np.zeros((n_trials, n_times_reconstructed))
     

     for i in range(n_trials):

         for k in range(n_atoms):

             X_reconstruit[i] += np.convolve(z_activ[k, i, :], d_activ[k, :], mode='full')
             
     return X_reconstruit
 
def find_peak(signal,seuil=2) :
    """
    parameters : 
    - signal : studied signal
    - seuil : threshold above which we consider the peaks 

    returns : 
    - activ : list
        list of maximas indices 
    """
    CHF=126
    sgn_test=signal
    n_mes=len(signal)
    activ=[]
    for j in range(n_mes) :
        if sgn_test[j] > seuil :
            k,zone=0,[]
            while j+k<n_mes and sgn_test[j+k] > seuil :
                zone.append(sgn_test[j+k])
                k+=1
            activ.append(j+zone.index(np.max(np.array(zone))))
        j+=1
    return activ

def extract_features(signals) : 
    """
    

    Parameters
    ----------
    signals : array or list
        signals on which to extract features

    Returns
    -------
    signals_features : array 
        features of the signals

    """
    
    signals_features =[]
    
    for signal in signals:
        # features temporelles de base ---
        energy = np.linalg.norm(signal, ord=2)
        var = np.var(signal)
        skewness = scipy.stats.skew(signal, axis=0, bias=True)
        kurtosis = scipy.stats.kurtosis(signal, axis=0, fisher=True, bias=True)
        ptp = np.ptp(signal)
        
        # Features Temporelles Avancées ---
        # Taux de passage par zéro (ZCR) 
        signal_centre = signal - np.mean(signal)
        zcr = np.sum(np.diff(np.sign(signal_centre)) != 0) / len(signal)
        
        # Indicateurs de distribution (Quantiles et Écart Interquartile)
        q25 = np.percentile(signal, 25)
        q75 = np.percentile(signal, 75)
        iqr = q75 - q25
        
        # Dynamique des pentes (pression/vitesse du signal via la dérivée)
        pente = np.diff(signal)
        var_pente = np.var(pente)
        energy_pente = np.linalg.norm(pente, ord=2)
        
        #  Features Fréquentielles (FFT)
        fft_vals = np.abs(np.fft.rfft(signal)) 
        fft_energy = np.sum(fft_vals**2)
        
        # Fréquence moyenne 
        freqs = np.arange(len(fft_vals))
        mean_freq = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-8)
        
        # Les 3 amplitudes spectrales les plus fortes 
        top3_amps = np.sort(fft_vals)[-3:]
        
        features=[
            energy, var, skewness, kurtosis, ptp,zcr, q25, q75, iqr,var_pente, energy_pente,fft_energy, mean_freq,top3_amps[0], top3_amps[1], top3_amps[2]
        ]
    
        signals_features.append(np.array(features))
    
    return np.array(signals_features)
    