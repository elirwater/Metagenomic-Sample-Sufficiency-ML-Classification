# Runs all main functions

import label_classes
import handler_csv
import results
from sklearn import preprocessing
import run_samples
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Gathers feature data from QC_Dump csv and puts it into a 2D array
feature_data = handler_csv.qc_dump_csv_handler('PGLibraryQC_dump_20200717.tsv')


# Runs all label set creations, training set pruning, and ML model classification and creates results CSV
def run_all(number_runs, sort_by, normalization):
    # Runs non-numpy structured ML algorithms
    results1 = label_classes.QCDumpLabels(feature_data, 'PGLibraryQC_dump_20200717.tsv', number_runs).run_ml(normalization)
    results2 = label_classes.QCDBLabels(feature_data, 'Hi-C QC DB - hicqc.20181113.0.csv', number_runs).run_ml(normalization)
    final_results = [results1, results2]
    out = results.create_csv(final_results, sort_by, True)
    return out

run_all(2, 'Withholding Score', 'PGLibraryQC_dump_20200717.tsv')





# Runs all models against every available label set, with-holding percentage, etc.
# Runs the initialization of results csv, graphs (both ROC and PvR), and saves the top classifier to the disk
def initialize(normalization, number_runs, ranking, label_set):
    # Creates training set used for precision vs. recall calculation
    label_object = label_classes.QCDumpLabels(feature_data, label_set, 1)
    if normalization:
        features = preprocessing.normalize(label_object.list_features())
        labels = label_object.list_labels()
    else:
        features = label_object.list_features()
        labels = label_object.list_labels()

    df = run_all(number_runs, ranking, normalization)

    top_classifier = df.iloc[0][2]
    results.save_top_model(features, labels, top_classifier)

    classifier_specs_list = []
    for i in range(3):
        holding_list = []
        for classifier_specs in df.iloc[i][2:5]:
            holding_list.append(classifier_specs)
        classifier_specs_list.append(holding_list)

        results.precision_recall(features, labels, holding_list[0], holding_list[1], holding_list[2])


# For running ideas without everything having to run
def quick_run():
    feature_data = handler_csv.parse_input_samples('PGLibraryQC_dump_20200717.tsv')
    label_object = label_classes.QCDumpLabels(feature_data, 'PGLibraryQC_dump_20200717.tsv', 1)
    features = preprocessing.normalize(label_object.list_features())
    labels = label_object.list_labels()

    results.save_top_model(features, labels)







