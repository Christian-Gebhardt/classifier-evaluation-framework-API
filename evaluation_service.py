from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

''' @author Christian Gebhardt, christian.gebhardt@uni-bayreuth.de
This is the evaluation_service.py file, all computations that are needed for API for the evaluation framework of classifiers (github link)
are done here.
'''

# Split dataset in features matrix X and label vector y
# @params dataset (as matrix, notice that it is assumed that labels are in the last column!)
# @returns X (numpy array) and y (numpy array)
def split_X_y(dataset):
    X = dataset[:, :-1]
    y = dataset[:, -1]
    return (X, y)

# Initialize Scikit Learn classifiers
# @params keyword clf for each classifier
# @returns instance of classifier or -1 if non existent
# @see https://scikit-learn.org/stable/supervised_learning.html
def map_classifier(clf, settings):
    print("Classifer {0} with settings {1} created.".format(clf, settings))
    return {
        "sgd": lambda: SGDClassifier(**settings),
        "gnb": lambda: GaussianNB(**settings),
        "dct": lambda: DecisionTreeClassifier(**settings),
        "rfo": lambda: RandomForestClassifier(**settings),
        "nnm": lambda: MLPClassifier(**settings),
           }.get(clf, -1)

# Calculate metrics score
# @params keyword mtc for each metric, vectors y_true (numpy array), y_pred (numpy array)
# @returns metrics score
# @see https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
def evaluate_metric(mtc, y_true, y_pred):
    return {
        "acc": lambda:  accuracy_score(y_true, y_pred),
        "acc_m": lambda: accuracy_score(y_true, y_pred),
        "fme": lambda: f1_score(y_true, y_pred),
        "prc": lambda: precision_score(y_true, y_pred),
        "rcl": lambda: recall_score(y_true, y_pred),
        "auc": lambda: roc_auc_score(y_true, y_pred),
        "fme_ma": lambda: f1_score(y_true, y_pred, average='macro'),
        "prc_ma": lambda: precision_score(y_true, y_pred, average='macro'),
        "rcl_ma": lambda: recall_score(y_true, y_pred, average='macro'),
        "fme_mi": lambda: f1_score(y_true, y_pred, average='micro'),
        "prc_mi": lambda: precision_score(y_true, y_pred, average='micro'),
        "rcl_mi": lambda: recall_score(y_true, y_pred, average='micro'),
        "lgl": lambda: log_loss(y_true, y_pred),
        "bri": lambda: brier_score_loss(y_true, y_pred),
    }.get(mtc, -1)()

# Evaluate own classifier
# @params metrics, y_true, y_pred, cnf_matrix (toggle on/off), clf_report (toggle on/off)
# @returns evaluation for each metric, cnf_matrix and classification report for own classifier
# @see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html?highlight=confusion%20matrix#,
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification%20report#
def evaluate_input(metrics, y_true=None, y_pred=None, cnf_matrix=True, clf_report=True, metricType="qualitative"):

    evaluation = {}

    for mtc in metrics:
        evaluation[mtc] = {}
        clf_reports = {}
        cnf_matrices = {}
        metric = {"metric": mtc}
        # Own classifier
        if y_true is not None and y_pred is not None:
            score = evaluate_metric(mtc, y_true, y_pred)
            metric["score_clf_own"] = score
            if cnf_matrix and metricType=="qualitative":
                cnf_matrices["cnf_matrix_own"] = confusion_matrix(y_true, y_pred).tolist()
            if clf_report and metricType=="qualitative":
                target_names = ["class {0}".format(i) for i in np.unique(y_true)]
                clf_reports["clf_report_own"] = classification_report(y_true, y_pred, output_dict=True, target_names=target_names)
            evaluation[mtc] = metric

    for v in evaluation.values():
        print(v)

    return (list(evaluation.values()), cnf_matrices, clf_reports)

# Compare classifiers on dataset with k-fold cross validation
# @params clfs (selected comparison classifiers), metrics, dataset, k (number of k-fold cross validation), train_indices (numpy array),
# test_indices (numpy array), y_pred (numpy array, results of own classifier, all y_pred vectors stacked)
# @returns 
# @see also https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html?highlight=stratified%20k%20fold#sklearn.model_selection.StratifiedKFold
def compare_clfs(clfs, metrics, dataset, k, train_indices, test_indices, classifier_settings, y_pred_arr=None):
    X, y = split_X_y(dataset)
    classifier_dict = {}
    evaluation_dict = {}
    clf_scores_dict = {}
    # Own classifier
    if y_pred_arr is not None:
        evaluation_dict["own"] = {}
        for mtc in metrics:
            evaluation_dict["own"][mtc] = []
        for i in range(k):
            y_true = y[test_indices[i]]
            y_pred = y_pred_arr[i]
            for mtc in metrics:
                score = evaluate_metric(mtc, y_true, y_pred)
                evaluation_dict["own"][mtc].append(score)
            
    for clf in clfs:
        clf_call = map_classifier(clf, classifier_settings[clf])
        if clf_call != -1:
            classifier_dict[clf] = clf_call()
            evaluation_dict[clf] = {}
            for mtc in metrics:
                evaluation_dict[clf][mtc] = []
    for i in range(k):
        for clf_key in classifier_dict.keys():
            classifier_dict.get(clf_key).fit(X[train_indices[i]], y[train_indices[i]])
            y_true = y[test_indices[i]]
            y_pred = classifier_dict[clf_key].predict(X[test_indices[i]])
            for mtc in metrics:
                score = evaluate_metric(mtc, y_true, y_pred)
                evaluation_dict[clf_key][mtc].append(score)

    for mtc in metrics:
        clf_scores_dict["average_{0}".format(mtc)] = { "name": "average_{0}".format(mtc) } 
        for clf in evaluation_dict.keys():
            clf_scores_dict["average_{0}".format(mtc)]["score_clf_{0}".format(clf)] = sum(evaluation_dict[clf][mtc]) / len(evaluation_dict[clf][mtc])

    return (evaluation_dict, list(clf_scores_dict.values()))

# Calculate data for roc curve plot
# @params y_true (true labels), y_proba (probabilities of labels shape (labels, n_classes)), n_classes (number of unique classes)
# @returns roc curve data (fpr, tpr, thresholds) and auc for each class
# @see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
def plot_multiclass_roc(y_true, y_proba, n_classes):
    # structures
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_true, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_test_dummies[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    roc = []
    for i in range(n_classes):
        roc_class = []
        for j in range(min(len(fpr[i]), len(tpr[i]))):
            roc_value = {"fpr": fpr[i][j], "tpr": tpr[i][j], "threshold": thresholds[i][j]}
            roc_class.append(roc_value)
        roc.append(roc_class)
    return {"roc": roc, "roc_auc": roc_auc}

# Generate 5x2 cross validation indices
# @params feature matrix X and label vector y
# @returns a tuple containing np arrays of train and test indices
def generate_5x2cv(X, y):
    train_indices = []
    test_indices = []
    for i in range(5):
        skf = StratifiedKFold(n_splits=2, shuffle=True)
        skf.get_n_splits(X, y)
        train_index, test_index = skf.split(X, y)
        train_indices.append(train_index[0])
        test_indices.append(test_index[0])
        train_indices.append(train_index[1])
        test_indices.append(test_index[1])
    return (train_indices, test_indices)
