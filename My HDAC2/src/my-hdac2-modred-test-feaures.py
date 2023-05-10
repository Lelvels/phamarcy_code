from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors

import pandas as pd
import numpy as np
import ast
import math

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

pcc_point = 0.9

modred_des_fp = '../output/modred_descriptors_out.xlsx'
test_dataset_fp = '../data_for_modeling/filter_data/v1/HDAC2_test.csv'
train_dataset_fp = '../data_for_modeling/filter_data/v1/HDAC2_train.csv'
features_data_fp = '../output/shapes_and_features/'+str(pcc_point)+"_shapes_and_features.xlsx"
result_fp = "../output/model_result/"+str(pcc_point)+"_Ket qua danh gia mo hinh HDAC2.xlsx"

def get_train_test_modred_des():
    # read from file
    print("[+] Read modred descriptors for HDAC2 " + modred_des_fp)
    train_modred_descriptors = pd.read_excel(modred_des_fp, sheet_name="train_modred_descriptors")
    test_mordred_descriptors = pd.read_excel(modred_des_fp, sheet_name="test_modred_descriptor")
    train_modred_descriptors = pd.DataFrame(train_modred_descriptors)
    test_mordred_descriptors = pd.DataFrame(test_mordred_descriptors)
    #filter data
    train_np = np.array(train_modred_descriptors)
    test_np = np.array(test_mordred_descriptors)
    for (row, col), value in np.ndenumerate(train_np):
        if not (value.__class__ in [int, float, np.float64, np.float32, np.int64, np.int32]):
            train_np[row, col] = 0
        
    for (row, col), value in np.ndenumerate(test_np):
        if not (value.__class__ in [int, float, np.float64, np.float32, np.int64, np.int32]):
            test_np[row, col] = 0
    train_modred_descriptors = pd.DataFrame(train_np, columns=train_modred_descriptors.columns)
    test_mordred_descriptors = pd.DataFrame(test_np, columns=test_mordred_descriptors.columns)
    all_mordred_descriptors = pd.concat([test_mordred_descriptors, train_modred_descriptors], ignore_index=False)
    print("[+] Finish gettting modred des")
    return train_modred_descriptors, test_mordred_descriptors, all_mordred_descriptors

def get_train_test_y():
    print("[+] Getting y data")
    test_dataset = pd.read_csv(test_dataset_fp)
    train_dataset = pd.read_csv(train_dataset_fp)
    y_train = np.array(train_dataset['FINAL_LABEL'])
    y_test = np.array(test_dataset['FINAL_LABEL'])
    y_all = np.append(y_train, y_test)
    print("[+] Finish getting y data")
    return y_train, y_test, y_all

def get_features_list():
    print("[+] Getting feature list from " + features_data_fp)
    features_data = pd.read_excel(features_data_fp, sheet_name='Sheet1')
    features_strings = features_data['Features']
    list_of_features = []
    for features_string in features_strings:
        list_of_features.append(ast.literal_eval(features_string))
    print("[+] Finish getting feature list")
    return list_of_features

def model_evaluation_calculation(cm):
    tp = cm[0][0]; tn = cm[1][1]; fp = cm[0][1]; fn = cm[1][0]
    ac = (tp+tn)/(tp+tn+fp+fn)
    se = tp/(tp+fn)
    sp = tn/(tn+fp)
    mcc = (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return ac, se, sp, mcc

def model_10_cv(model, X_train, y_train, X_test, y_test):
    X_Total = np.concatenate((X_train, X_test), axis=0)
    y_Total = np.concatenate((y_train, y_test), axis=0)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    ac_score = cross_val_score(model, X_Total, y_Total, scoring='accuracy', cv=cv, n_jobs=-1)
    ten_ac_score = ac_score.mean()
    return ten_ac_score

def model_evaluation(model, X_train, y_train, X_test, y_test):
    #10-fold-cross-validation
    ten_cv_ac = model_10_cv(model, X_train, y_train, X_test, y_test)
    #test_set
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ac, se, sp, mcc = model_evaluation_calculation(cm)
    #auc
    y_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_proba)
    return ten_cv_ac, ac, se, sp, mcc, auc_score
    

def knn_evaluation(X_train, y_train, X_test, y_test):
    knn_des = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn_des.fit(X_train, y_train)
    ten_cv_ac, ac, se, sp, mcc, auc_score = model_evaluation(knn_des, X_train, y_train, X_test, y_test)
    return ten_cv_ac, ac, se, sp, mcc, auc_score

def svm_evaluation(X_train, y_train, X_test, y_test):
    svm_des = SVC(kernel='rbf', probability=True, random_state=0)
    svm_des.fit(X_train, y_train)
    ten_cv_ac, ac, se, sp, mcc, auc_score = model_evaluation(svm_des, X_train, y_train, X_test, y_test)
    return ten_cv_ac, ac, se, sp, mcc, auc_score

def rf_evaluation(X_train, y_train, X_test, y_test):
    rf_des = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    rf_des.fit(X_train, y_train)
    ten_cv_ac, ac, se, sp, mcc, auc_score = model_evaluation(rf_des, X_train, y_train, X_test, y_test)
    return ten_cv_ac, ac, se, sp, mcc, auc_score

def xgboost_evaluation(X_train, y_train, X_test, y_test):
    bst_des = XGBClassifier(n_estimators=100, objective='binary:logistic')
    bst_des.fit(X_train, y_train)
    ten_cv_ac, ac, se, sp, mcc, auc_score = model_evaluation(bst_des, X_train, y_train, X_test, y_test)
    return ten_cv_ac, ac, se, sp, mcc, auc_score

def sc_standard(X_train, X_test, features):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train_np = sc.fit_transform(X_train)
    X_test_np = sc.transform(X_test)
    X_train = pd.DataFrame(X_train_np, columns = features)
    X_test = pd.DataFrame(X_test_np, columns = features)
    return X_train, X_test
    

def main():
    train_modred_descriptors, test_mordred_descriptors, _ = get_train_test_modred_des()
    y_train, y_test, _ = get_train_test_y()
    list_of_features = get_features_list()
    evaluation_data = {
        "Number of features": [],
        "Model name": [],
        "Ten fold cross-validation": [],
        "Test-set Accuracy": [],
        "Test-set Sensitivity": [],
        "Test-set Specificity": [],
        "Test-set MCC": [],
        "Test-set AUC": []
    }
    
    
    evaluation_df = pd.DataFrame(evaluation_data)
    prev_features_number = 0
    
    for features in list_of_features:
        features_number = len(features)
        if(features_number != prev_features_number and features_number>0):
            prev_features_number = features_number
            print("[+] Starting evaluation for number " + str(features_number))
            X_train = train_modred_descriptors[features]
            X_test = test_mordred_descriptors[features]
            X_train, X_test = sc_standard(X_train, X_test, features)
            #Model evaluation data
            knn_ten_ac, knn_ac, knn_se, knn_sp, knn_mcc, knn_auc = knn_evaluation(X_train, y_train, X_test, y_test)    
            rf_ten_ac, rf_ac, rf_se, rf_sp, rf_mcc, rf_auc = rf_evaluation(X_train, y_train, X_test, y_test)
            xg_ten_ac, xg_ac, xg_se, xg_sp, xg_mcc, xg_auc = xgboost_evaluation(X_train, y_train, X_test, y_test)
            svm_ten_ac, svm_ac, svm_se, svm_sp, svm_mcc, svm_auc = svm_evaluation(X_train, y_train, X_test, y_test)
            #Add data
            knn_new_row = {
                "Number of features": features_number,
                "Model name": "K-nearest neighbor",
                "Ten fold cross-validation": knn_ten_ac,
                "Test-set Accuracy": knn_ac,
                "Test-set Sensitivity": knn_se,
                "Test-set Specificity": knn_sp,
                "Test-set MCC": knn_mcc,
                "Test-set AUC": knn_auc
            }
            rf_new_row = {
                "Number of features": features_number,
                "Model name": "Random forest",
                "Ten fold cross-validation": rf_ten_ac,
                "Test-set Accuracy": rf_ac,
                "Test-set Sensitivity": rf_se,
                "Test-set Specificity": rf_sp,
                "Test-set MCC": rf_mcc,
                "Test-set AUC": rf_auc
            }
            xg_new_row = {
                "Number of features": features_number,
                "Model name": "XgBoost 2",
                "Ten fold cross-validation": xg_ten_ac,
                "Test-set Accuracy": xg_ac,
                "Test-set Sensitivity": xg_se,
                "Test-set Specificity": xg_sp,
                "Test-set MCC": xg_mcc,
                "Test-set AUC": xg_auc
            }
            svm_new_row = {
                "Number of features": features_number,
                "Model name": "SVM (RBF-kernel)",
                "Ten fold cross-validation": svm_ten_ac,
                "Test-set Accuracy": svm_ac,
                "Test-set Sensitivity": svm_se,
                "Test-set Specificity": svm_sp,
                "Test-set MCC": svm_mcc,
                "Test-set AUC": svm_auc
            }
            evaluation_df = evaluation_df.append(knn_new_row, ignore_index=True)
            evaluation_df = evaluation_df.append(rf_new_row, ignore_index=True)
            evaluation_df = evaluation_df.append(xg_new_row, ignore_index=True)
            evaluation_df = evaluation_df.append(svm_new_row, ignore_index=True)
            print("[+] Finish evaluation")
        else:
            print("[+] Skip with " + str(features_number))
    #Write this to file
    print("[+] Write to file " + result_fp)
    evaluation_df.to_excel(result_fp, index=False)
if __name__ == "__main__":
    main()