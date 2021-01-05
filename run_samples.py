# For running a series of samples that you want to know the classification of
# Will return a CSV containing the predict labels of your input samples
# For now, the format of the input samples must match the format of the QC_Dump_Data sample format
# structured in a TSV format, label_classes is which file you want to choose as your label classes
import pandas as pd
import handler_csv
import pickle
from sklearn import preprocessing
import numpy as np

def run_samples(input_sample_tsv):
    samples = handler_csv.parse_input_samples(input_sample_tsv)

    features = []
    for x in samples:
        features.append(x[1:])
    features = preprocessing.normalize(features)

    # Loads top classifier from disk, requires running the initialize function in main
    clf = pickle.load(open('best_model.sav', 'rb'))
    predictions = clf.predict(features)
    probabilities = clf.predict_proba(features)

    holding_val = 0

    fields = ['Dimension match', 'Sample Timestamp', 'Classification', 'Confidence of Classification']
    out_arr = np.array([fields])
    for prediction in predictions:

        if prediction == 3:
            prediction = "SUFFICIENT"
        elif prediction == 2:
            prediction = "MIXED RESULTS"
        elif prediction == 1:
            prediction = "INSUFFICIENT"
        elif prediction == 0:
            prediction = "LOW SIGNAL"
        else:
            prediction = "UNKNOWN"

        sample_name = samples[holding_val][0]

        prediction_prob = max(probabilities[holding_val]) * 100

        dimension_match = "1"  # For matching the input dimensions of the panda dataframe
        row_arr = np.array([[dimension_match, sample_name, prediction, prediction_prob]], dtype=object)
        out_arr = np.concatenate((out_arr, row_arr))
        holding_val += 1

    df = pd.DataFrame(data=out_arr[1:, 1:], index=out_arr[1:, 0], columns = out_arr[0, 1:])
    df.to_csv('sample_results.csv', index=False)








