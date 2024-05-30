#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zlib

import nltk
nltk.download('punkt')


from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import re

import numpy as np
#import cupy as np

from sklearn.decomposition import KernelPCA
import csv

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import TruncatedSVD
import random
# import seaborn as sns
import os.path as path
import os
# import matplotlib
# import matplotlib.font_manager
# import matplotlib.pyplot as plt # graphs plotting
# import Bio
# from Bio import SeqIO # some BioPython that will come in handy
#matplotlib inline

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


# from matplotlib import rc
# # for Arial typefont
# matplotlib.rcParams['font.family'] = 'Arial'

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from pandas import DataFrame

from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import confusion_matrix

from numpy import mean

import seaborn as sns

import itertools
from itertools import product

## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# matplotlib.rcParams['mathtext.fontset'] = 'cm'

## for LaTeX typefont
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

## for another LaTeX typefont
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# rc('text', usetex = True)

print("done")


# In[2]:


# In[13]:

data_path = "/olga-data1/Sarwan/Dataset/Lungs_Cancer/"



seq_data = np.load(data_path + "Lungs_Cancer_Sequences_901.npy", allow_pickle=True)
attribute_data = np.load(data_path + "Lungs_Cancer_attributes_901.npy", allow_pickle=True)




attr_new = []
for i in range(len(attribute_data)):
    aa = str(attribute_data[i]).replace("[","")
    aa_1 = aa.replace("]","")
    aa_2 = aa_1.replace("\'","")
    #aa_2 = aa_1.replace("-","")
    attr_new.append(aa_2)

unique_hst = list(np.unique(attr_new))

int_hosts = []
for ind_unique in range(len(attr_new)):
    variant_tmp = attr_new[ind_unique]
    ind_tmp = unique_hst.index(variant_tmp)
    int_hosts.append(ind_tmp)
    
print("Attribute data preprocessing Done")

# X = np.array(seq_data[:])
# y = np.array(int_hosts[:])

# X = np.array(seq_data[0:100])
# y = np.array(int_hosts[0:100])


# In[3]:


seq_data[0]


# In[4]:


def clean_and_tokenize(sequence):
    # Remove non-alphabetic characters and tokenize
    cleaned_tokens = [token for token in word_tokenize(re.sub(r'[^a-zA-Z]', ' ', sequence)) if token.isalpha()]
    return cleaned_tokens


def encode(sequence):
    tokens = clean_and_tokenize(sequence)

    vectorizer = CountVectorizer()
    numerical_representation = vectorizer.fit_transform([' '.join(tokens)])

    # Convert the numerical representation to a string
    encoded_sequence = ' '.join(map(str, numerical_representation.toarray().flatten()))

    return encoded_sequence


def compress(encoded_sequence):
    # Gzip compression
    compressed_data = zlib.compress(encoded_sequence.encode("utf-8"))
    return compressed_data



def incremental_encoding_distance_matrix(sequences):
    distance_matrix = []
    k_val=3

    counter=1
    for s1 in sequences:
        print("i: ",counter,"/",len(sequences))
        counter = counter + 1
        Es1 = ""
        Cs1 = b""
        Ls1 = 0
        D_local = []
        D_local_2 = []
        for s2 in sequences:
            Es2 = ""
            Cs2 = b""
            Ls2 = 0

            # Incremental encoding and compression for s1
            for i in range(len(s1) - k_val + 1):
                chunk_s1 = s1[i:i+k_val]
                Es1 += encode(chunk_s1)
                Cs1 = compress(Es1)
                Ls1 = len(Cs1)

                # Incremental encoding and compression for s2
                for j in range(len(s2) - k_val + 1):
                    chunk_s2 = s2[j:j+k_val]
                    Es2 += encode(chunk_s2)
                    Cs2 = compress(Es2)
                    Ls2 = len(Cs2)

                    # Incremental concatenation
                    concatenated_seq = s1[:i+1] + s2[:j+1]
                    Es1s2 = encode(concatenated_seq)
                    Cs1s2 = compress(Es1s2)
                    Ls1s2 = len(Cs1s2)

                    # Incremental NCD calculation
                    NCD = (Ls1s2 - min(Ls1, Ls2)) / max(Ls1, Ls2)
                    D_local.append(NCD)

            scalar_value = np.mean(D_local)  # or np.sum(D_local)
            D_local_2.append(scalar_value)
        distance_matrix.append(D_local_2)
#         distance_matrix.append(D_local)

    return distance_matrix




# In[5]:


# Example usage
# sequences = ["abcdeabcdeabcdeabcdeabcde", "fghijfghijfghijfghijfghij", "klmnoklmnoklmnoklmnoklmno", 
#              "pqrstpqrstpqrstpqrstpqrstpqrstpqrst", "uvwxyuvwxyuvwxyuvwxyuvwxyuvwxyuvwxyuvwxyuvwxyuvwxy"]

sequences = seq_data[:]

result_distance_matrix = incremental_encoding_distance_matrix(sequences)
# print(result_distance_matrix)


# In[6]:


result_distance_matrix = np.array(result_distance_matrix)

for i in range(len(result_distance_matrix)):
    for j in range(len(result_distance_matrix)):
        if i==j:
            result_distance_matrix[i,j] = 0
        temp = (result_distance_matrix[i,j] + result_distance_matrix[j,i])/2
        result_distance_matrix[i,j] = temp
        result_distance_matrix[j,i] = temp


# In[7]:


result_distance_matrix.shape


# In[8]:


def gaussian_kernel(distance_matrix, sigma):
    similarity_matrix = np.exp(-distance_matrix**2 / (2 * sigma**2))
    return similarity_matrix

# Sigma parameter for the Gaussian kernel
sigma = 1.0

# Apply Gaussian kernel
kernel_matrix = gaussian_kernel(result_distance_matrix, sigma)


# In[9]:


kernel_matrix.shape


# In[10]:


from sklearn.decomposition import KernelPCA
#############  Kernel PCA ################
transformer = KernelPCA(n_components=500, kernel='precomputed')
X_transformed = transformer.fit_transform(kernel_matrix)
X_transformed.shape
#############  Kernel PCA ################

X = np.array(X_transformed)
# y = np.array(int_hosts)

X.shape


# In[18]:


np.save(data_path + "Lungs_Cancer_901_kmers_compression_After_Kernel_PCA.npy",X)

print("PCA Done!!")

# # Classification Function

# In[12]:


# In[4]
def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc
    
    
    check = pd.DataFrame(roc_auc_dict.items())
    return mean(check)

def svm_fun_kernel(X_train,y_train,X_test,y_test,kernel_mat):
    import time
    
    start = timeit.default_timer()
    

#     clf = svm.SVC()
    clf = svm.SVC(kernel=kernel_mat)
    
    #Train the model using the training sets
    clf.fit(kernel_mat, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("SVM Kernel Time : ", time_final)
    
    svm_acc = metrics.accuracy_score(y_test, y_pred)
#     print("SVM Accuracy:",svm_acc)
    
    svm_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("SVM Precision:",svm_prec)
    
    svm_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("SVM Recall:",svm_recall)

    svm_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("SVM F1 Weighted:",svm_f1_weighted)
    
    svm_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("SVM F1 macro:",svm_f1_macro)
    
    svm_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("SVM F1 micro:",svm_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix SVM : \n", confuse)
#     print("SVM Kernel Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
#    print(macro_roc_auc_ovo[1])
    check = [svm_acc,svm_prec,svm_recall,svm_f1_weighted,svm_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)
    
# In[5]
##########################  SVM Classifier  ################################
def svm_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()
    
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("SVM Time : ", time_final)
    
    svm_acc = metrics.accuracy_score(y_test, y_pred)
#     print("SVM Accuracy:",svm_acc)
    
    svm_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("SVM Precision:",svm_prec)
    
    svm_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("SVM Recall:",svm_recall)

    svm_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("SVM F1 Weighted:",svm_f1_weighted)
    
    svm_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("SVM F1 macro:",svm_f1_macro)
    
    svm_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("SVM F1 micro:",svm_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix SVM : \n", confuse)
#     print("SVM Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
#    print(macro_roc_auc_ovo[1])
    check = [svm_acc,svm_prec,svm_recall,svm_f1_weighted,svm_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)
    


# In[5]
##########################  NB Classifier  ################################
def gaus_nb_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()
    
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("NB Time : ", time_final)


    NB_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Gaussian NB Accuracy:",NB_acc)

    NB_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB Precision:",NB_prec)
    
    NB_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB Recall:",NB_recall)
    
    NB_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB F1 weighted:",NB_f1_weighted)
    
    NB_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Gaussian NB F1 macro:",NB_f1_macro)
    
    NB_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Gaussian NB F1 micro:",NB_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix NB : \n", confuse)
#     print("NB Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    check = [NB_acc,NB_prec,NB_recall,NB_f1_weighted,NB_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)

# In[5]
##########################  MLP Classifier  ################################
def mlp_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()
    
    # Feature scaling
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)  
    X_test_2 = scaler.transform(X_test)


    # Finally for the MLP- Multilayer Perceptron
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
    mlp.fit(X_train, y_train)


    y_pred = mlp.predict(X_test_2)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("MLP Time : ", time_final)
    
    MLP_acc = metrics.accuracy_score(y_test, y_pred)
#     print("MLP Accuracy:",MLP_acc)
    
    MLP_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("MLP Precision:",MLP_prec)
    
    MLP_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("MLP Recall:",MLP_recall)
    
    MLP_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("MLP F1:",MLP_f1_weighted)
    
    MLP_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("MLP F1:",MLP_f1_macro)
    
    MLP_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("MLP F1:",MLP_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix MLP : \n", confuse)
#     print("MLP Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [MLP_acc,MLP_prec,MLP_recall,MLP_f1_weighted,MLP_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)

# In[5]
##########################  knn Classifier  ################################
def knn_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()
    
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("knn Time : ", time_final)

    knn_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Knn Accuracy:",knn_acc)
    
    knn_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Knn Precision:",knn_prec)
    
    knn_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Knn Recall:",knn_recall)
    
    knn_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Knn F1 weighted:",knn_f1_weighted)
    
    knn_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Knn F1 macro:",knn_f1_macro)
    
    knn_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Knn F1 micro:",knn_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix KNN : \n", confuse)
#     print("KNN Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [knn_acc,knn_prec,knn_recall,knn_f1_weighted,knn_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)

# In[5]
##########################  Random Forest Classifier  ################################
def rf_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()
    
    # Import the model we are using
    from sklearn.ensemble import RandomForestClassifier
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators = 100)
    # Train the model on training data
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("RF Time : ", time_final)

    fr_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Random Forest Accuracy:",fr_acc)
    
    fr_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Random Forest Precision:",fr_prec)
    
    fr_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Random Forest Recall:",fr_recall)
    
    fr_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Random Forest F1 weighted:",fr_f1_weighted)
    
    fr_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Random Forest F1 macro:",fr_f1_macro)
    
    fr_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Random Forest F1 micro:",fr_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix RF : \n", confuse)
#     print("RF Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [fr_acc,fr_prec,fr_recall,fr_f1_weighted,fr_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)

# In[5]
    ##########################  Logistic Regression Classifier  ################################
def lr_fun(X_train,y_train,X_test,y_test):
    import time
    
    start = timeit.default_timer()

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("LR Time : ", time_final)

    LR_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Logistic Regression Accuracy:",LR_acc)
    
    LR_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Precision:",LR_prec)
    
    LR_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Recall:",LR_recall)
    
    LR_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression F1 weighted:",LR_f1_weighted)
    
    LR_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Logistic Regression F1 macro:",LR_f1_macro)
    
    LR_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Logistic Regression F1 micro:",LR_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix LR : \n", confuse)
#     print("LR Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [LR_acc,LR_prec,LR_recall,LR_f1_weighted,LR_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)


def fun_decision_tree(X_train,y_train,X_test,y_test):
    from sklearn import tree
    import time
    
    start = timeit.default_timer()


    
    clf = tree.DecisionTreeClassifier()    
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    stop = timeit.default_timer()
    time_final = stop - start
    print("DT Time : ", time_final) 
    
    dt_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Logistic Regression Accuracy:",LR_acc)
    
    dt_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Precision:",LR_prec)
    
    dt_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Recall:",LR_recall)
    
    dt_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression F1 weighted:",LR_f1_weighted)
    
    dt_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Logistic Regression F1 macro:",LR_f1_macro)
    
    dt_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Logistic Regression F1 micro:",LR_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix DT : \n", confuse)
#     print("DT Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [dt_acc,dt_prec,dt_recall,dt_f1_weighted,dt_f1_macro,macro_roc_auc_ovo[1],time_final]
    return(check)


# In[13]:


import timeit

# print("Accuracy   Precision   Recall   F1 (weighted)   F1 (Macro)   F1 (Micro)   ROC AUC")
svm_table = []
gauu_nb_table = []
mlp_table = []
knn_table = []
rf_table = []
lr_table = []
dt_table = []


from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit

y = np.array(int_hosts)

# path_tmp = "E:/RA/ISBRA_Adversarial_Attack/Dataset/"
# X_train = np.array(np.load(path_tmp + "String_kernel_Original_k3_m0_Alphabet_Size4_trial_1.npy"))
# X_test = np.array(np.load("E:/RA/ISBRA_Adversarial_Attack/Dataset/String_kernel_" + kernel_name + ".npy"))

# y_train = np.array(int_hosts)
# y_test = np.array(int_hosts)

total_splits = 5

sss = ShuffleSplit(n_splits=total_splits, test_size=0.3)

for t in range(total_splits):
    sss.get_n_splits(X, y)
    train_index, test_index = next(sss.split(X, y)) 

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


#     start = timeit.default_timer()
    gauu_nb_return = gaus_nb_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("NB Time : ", stop - start) 

#     start = timeit.default_timer()
    mlp_return = mlp_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("MLP Time : ", stop - start) 

#     start = timeit.default_timer()
    knn_return = knn_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("KNN Time : ", stop - start) 

#     start = timeit.default_timer()
    rf_return = rf_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("RF Time : ", stop - start) 

#     start = timeit.default_timer()
    lr_return = lr_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("LR Time : ", stop - start) 

#     start = timeit.default_timer()
    dt_return = fun_decision_tree(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("DT Time : ", stop - start) 

#     start = timeit.default_timer()
    svm_return = svm_fun(X_train,y_train,X_test,y_test)
#     stop = timeit.default_timer()
#     print("SVM Time : ", stop - start) 

    gauu_nb_table.append(gauu_nb_return)
    mlp_table.append(mlp_return)
    knn_table.append(knn_return)
    rf_table.append(rf_return)
    lr_table.append(lr_return)
    dt_table.append(dt_return)
    svm_table.append(svm_return)

    svm_table_final = DataFrame(svm_table, columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
    gauu_nb_table_final = DataFrame(gauu_nb_table, columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
    mlp_table_final = DataFrame(mlp_table, columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
    knn_table_final = DataFrame(knn_table, columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
    rf_table_final = DataFrame(rf_table, columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
    lr_table_final = DataFrame(lr_table, columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])

    dt_table_final = DataFrame(dt_table, columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])


# In[14]:


#taking average of all k-fold performance values
final_mean_mat = []

final_mean_mat.append(np.transpose((list(svm_table_final.mean()))))
final_mean_mat.append(np.transpose((list(gauu_nb_table_final.mean()))))
final_mean_mat.append(np.transpose((list(mlp_table_final.mean()))))
final_mean_mat.append(np.transpose((list(knn_table_final.mean()))))
final_mean_mat.append(np.transpose((list(rf_table_final.mean()))))
final_mean_mat.append(np.transpose((list(lr_table_final.mean()))))
final_mean_mat.append(np.transpose((list(dt_table_final.mean()))))

final_avg_mat = DataFrame(final_mean_mat,columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","ROC AUC","Runtime (Sec.)"], 
                          index=["SVM","NB","MLP","KNN","RF","LR","DT"])

print(final_avg_mat)


# In[15]:


#taking average of all k-fold performance values
final_mean_mat = []

final_mean_mat.append(np.transpose((list(svm_table_final.max()))))
final_mean_mat.append(np.transpose((list(gauu_nb_table_final.max()))))
final_mean_mat.append(np.transpose((list(mlp_table_final.max()))))
final_mean_mat.append(np.transpose((list(knn_table_final.max()))))
final_mean_mat.append(np.transpose((list(rf_table_final.max()))))
final_mean_mat.append(np.transpose((list(lr_table_final.max()))))
final_mean_mat.append(np.transpose((list(dt_table_final.max()))))

final_avg_mat = DataFrame(final_mean_mat,columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","ROC AUC","Runtime (Sec.)"], 
                          index=["SVM","NB","MLP","KNN","RF","LR","DT"])

print(final_avg_mat)


# In[16]:


#taking average of all k-fold performance values
final_mean_mat = []

final_mean_mat.append(np.transpose((list(svm_table_final.std()))))
final_mean_mat.append(np.transpose((list(gauu_nb_table_final.std()))))
final_mean_mat.append(np.transpose((list(mlp_table_final.std()))))
final_mean_mat.append(np.transpose((list(knn_table_final.std()))))
final_mean_mat.append(np.transpose((list(rf_table_final.std()))))
final_mean_mat.append(np.transpose((list(lr_table_final.std()))))
final_mean_mat.append(np.transpose((list(dt_table_final.std()))))

final_avg_mat = DataFrame(final_mean_mat,columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","ROC AUC","Runtime (Sec.)"], 
                          index=["SVM","NB","MLP","KNN","RF","LR","DT"])

print(final_avg_mat)

