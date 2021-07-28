from flask import Flask, request, send_file
from flask_cors import CORS
from evaluation_service import *
import pandas as pd
import numpy as np
from tempfile import TemporaryFile

''' 
@author Christian Gebhardt, christian.gebhardt@uni-bayreuth.de
This is the main file of the API for the evaluation framework for classifiers (github link).
app.py manages all incoming HTTP-Requests for http://127.0.0.1:5000/ (local host), when the server is running.
It serves all json data responses to the webapp and acts as an interface between the webapp and
evaluation_service.py.
'''

# Init app
app = Flask(__name__)

# Cross origin seetings for local host
CORS(app)

# Greeting at base url
@app.route("/")
def greeting():
    return "<h1>Welcome to evaluation API.</h1>"


# Evaluation for own classifier
# @params y_true.npy file (true labels), y_pred.npy file (predicted labels), metrics
# @returns json response containing: results (for given metrics), cnf matrix
# and classification report
@app.route("/evaluate", methods=['GET', 'POST'])
def evaluate():
    # Case own classifier is selected, check if all necessary params are in HTTP request
    if 'y_true' in request.files and 'y_pred' in request.files and request.form.getlist('metrics[]') and request.form.get("metricType"): 
        y_true_file = TemporaryFile()
        y_pred_file = TemporaryFile()
        metrics = request.form.getlist('metrics[]')
        request.files['y_true'].save(y_true_file)
        request.files['y_pred'].save(y_pred_file)
        _ = y_true_file.seek(0) # Only needed here to simulate closing & reopening file
        y_true = np.load(y_true_file)
        _ = y_pred_file.seek(0) # Only needed here to simulate closing & reopening file
        y_pred = np.load(y_pred_file)
        
        # Case qualitative metrics
        if request.form.get("metricType") == "qualitative":
            results, cnf_matrices, clf_reports = evaluate_input(y_true=y_true, y_pred=y_pred, metrics=metrics)
            return {'results': results, 'cnf_matrices': cnf_matrices, 'clf_reports': clf_reports}
        # Case probabilistic metrics
        else:
            results, cnf_matrices, clf_reports = evaluate_input(y_true=y_true, y_pred=y_pred, metrics=metrics, metricType="probabilistic")
            # Case ROC image
            if request.form.get("roc"):
                roc_dict = plot_multiclass_roc(y_true=y_true, y_proba=y_pred, n_classes=len(np.unique(y_true)))
                return {'results': results, 'cnf_matrices': cnf_matrices, 'clf_reports': clf_reports, 'roc_analysis': roc_dict}
            # Case no ROC image
            else:
                return {'results': results, 'cnf_matrices': cnf_matrices, 'clf_reports': clf_reports}
    # Case wrong params
    else: 
        return { "message": "Incorrect parameters, mandatory params are: y_true.npy, y_pred.npy, metrics (list)"}

# Comparison with other classifiers by k-fold cross validation
# @params y_pred.npy file (predicted labels of own classifier on cross validation), train_test_indices.npz file (indices for cross validation),
# dataset.csv file, metrics, classifiers
# @returns json response containing: evaluation (averages on cross validation), results (detailed results of cross validation)
@app.route("/compare", methods=['GET', 'POST'])
def compare():
    # Check if all necessary params are in HTTP request
    if 'y_pred' in request.files and 'train_test_indices' in request.files and request.form.getlist('metrics[]') and \
    'dataset' in request.files and request.form.getlist('classifiers[]'):
        y_pred_file = TemporaryFile()
        metrics = request.form.getlist('metrics[]')
        request.files['y_pred'].save(y_pred_file)
        _ = y_pred_file.seek(0) # Only needed here to simulate closing & reopening file
        y_pred = np.load(y_pred_file)
        dataset_file = TemporaryFile()
        request.files['dataset'].save(dataset_file)
        _ = dataset_file.seek(0) # Only needed here to simulate closing & reopening file
        dataset = pd.read_csv(dataset_file).to_numpy()
        comp_clfs= request.form.getlist('classifiers[]')

        train_test_indices_file = TemporaryFile()
        request.files['train_test_indices'].save(train_test_indices_file)
        _ = train_test_indices_file.seek(0) # Only needed here to simulate closing & reopening file
        train_test_indices = np.load(train_test_indices_file)

        train_indices = train_test_indices['train_indices']
        test_indices = train_test_indices['test_indices']

        k = len(train_indices)

        evaluation, results = compare_clfs(comp_clfs, metrics, dataset, k, train_indices, test_indices, y_pred)
        return { "evaluation": evaluation, "results": results }
        # Case wrong params
    else:
        return { "message": "Incorrect parameters, mandatory params are: y_pred.npy, train_test_indices.npz, dataset.csv, metrics (list), classifiers (list)"}

# Not as feature in webapp, maybe in later versions.
@app.route("/generateIndices", methods=['GET', 'POST'])
def generateIndices():
    if 'dataset' in request.files:
        dataset_file = TemporaryFile()
        request.files['dataset'].save(dataset_file)
        _ = dataset_file.seek(0) # Only needed here to simulate closing & reopening file
        dataset = pd.read_csv(dataset_file).to_numpy()
        X, y = split_X_y(dataset)
        train_indices, test_indices = generate_5x2cv(X, y)
        data = np.hstack((train_indices, test_indices))
        file = TemporaryFile()
        np.save(file, data)
        _ = file.seek(0)
        return send_file(file, "train_test_indices.npy")
    return "TODO"

# Run Server
if __name__ == '__main__':
    app.run(debug=True)
