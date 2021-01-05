import pandas as pd
from sklearn.preprocessing import label_binarize, normalize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import statistics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import numpy as np
from sklearn.metrics import average_precision_score
import random
import pickle
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

# Creates the out CSV for the results of running each model
# Results can be found in a file called ml_results.csv
# Called by run_all in the main module
def create_csv(out_data_all, sort_by, show_descriptions):
    fields = ['Holding', 'Label Set', 'Number of Runs', 'ML Model', 'Withholding Score', 'Percentage Witheld',
              'Average Accuracy Percentage', 'Standard Deviation', 'Variance', 'Run Time (Seconds)', 'Normalization',
              'Average Precision']
    info_for_fields = [
        'Which csv the labels used in training came from',
        'Number of times each classifier with tuned parameters was run, for classifiers with no variation, only 1 run',
        'Specific classifier used',
        'Percentage correct score of the classifier in relation to samples witheld from fitting (randomly withheld)',
        'Percentage of samples witheld from fitting, to be used to evaluate the accuracy of the model',
        'Percentage correct score of the classifier in relation to all samples being re-run through fitted model, subject to '
        'extreme model over-fitting',
        'Standard deviation of the Average Accuracy Percentage over the number of runs',
        'Variance of the Average Accuracy Percentage over the number of runs',
        'Amount of time each individual classifier took to run (not perfect accuracy)',
        'Whether the features are being normalized or not (True=Normalized)'
    ]


    numpy_data = np.array([fields])
    for row in out_data_all:
        for ml_run in row:
            ml_run.insert(0, "test") # For matching the index dimensionality
            arr_row = np.array([ml_run], dtype=object)
            numpy_data = np.concatenate((numpy_data, arr_row))
    df = pd.DataFrame(data=numpy_data[1:, 1:], index=numpy_data[1:, 0], columns=numpy_data[0, 1:])
    df = df.sort_values(sort_by, ascending=False)
    if show_descriptions:
        df.iloc[0] = info_for_fields
    df.to_csv('ml_results.csv', index=False)  # df.iloc[0], for grabbing a specific row

    return df.iloc[1:4]


# Loops through each classifier, re-fitting it everytime in order to get a sense of the average prediction
# accuracy, along with the standard deviation and variance of these average prediction accuracies
def calculate_accuracy(classifier, normalized_features, real_samples, labels, runs, withholding_percentage):
    percentage_correct_list = list()
    while runs > 0:
        classifier.fit(normalized_features, labels)
        # All prediction are compared against all ground-truth labels (ideally would have a different dataset for this)
        y_hat = classifier.predict(real_samples)
        total_correct = 0
        counter = 0

        for prediction in y_hat:
            if labels[counter] == prediction:
                total_correct += 1
                counter += 1
            else:
                counter += 1

        percentage_correct = total_correct / len(labels) * 100
        percentage_correct_list.append(percentage_correct)
        runs -= 1
    average_accuracy_percentage = sum(percentage_correct_list) / len(percentage_correct_list)
    try:
        sd = statistics.stdev(percentage_correct_list)
        var = statistics.variance(percentage_correct_list)
    except statistics.StatisticsError:
        sd = None
        var = None

    withholding_score = calculate_accuracy_witholding(normalized_features, labels, classifier, withholding_percentage)
    return_list = [average_accuracy_percentage, sd, var, withholding_score, withholding_percentage]

    return return_list


# Calculates the score of a given ML model by withholding a certain percentage of the data for training
# and then using that withheld data for evaluating the accuracy of the classifier
def calculate_accuracy_witholding(x_normalized, y_labels, classifier, witholding_percentage):
    x_train, x_test, y_train, y_test = train_test_split(x_normalized, y_labels, test_size=witholding_percentage,
                                                        random_state=0)
    # 40% of the data is held out of training so we can test it independently (for evaluating the classifier)
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    return score


# Calculates the micro-averaged precision for a given classifier, without graphing it
def precision_recall_scores(x_normalized, y_labels, clf, perc_witheld):
    # Only works for Random Forest
    print("Running precision vs recall calculations for", clf)
    Y = label_binarize(y_labels, classes=[0, 1, 2, 3, 10])
    n_classes = Y.shape[1]
    X_train, X_test, Y_train, Y_test = train_test_split(x_normalized, Y, test_size=perc_witheld, random_state=0)

    classifier = OneVsRestClassifier(clf)
    classifier.fit(X_train, Y_train)
    y_score = classifier.predict_proba(X_test)
    classifier.fit(X_train, Y_train)

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes - 1):
        class_list = ["LOW SIGNAL", "INSUFFICIENT", "MIXED RESULTS", "SUFFICIENT"]
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], lw=2, label=class_list[i].format(i))

    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")
    return average_precision["micro"]


# Graphs the precison vs. recall curve for a given classifier and training set
def precision_recall(x_normalized, y_labels, input_classifier, witholding_score, perc_witheld):
    # Only works for Random Forest
    Y = label_binarize(y_labels, classes=[0, 1, 2, 3, 10])
    n_classes = Y.shape[1]
    X_train, X_test, Y_train, Y_test = train_test_split(x_normalized, Y, test_size=perc_witheld, random_state = 0)

    classifier = OneVsRestClassifier(input_classifier)
    classifier.fit(X_train, Y_train)
    y_score = classifier.predict_proba(X_test)
    classifier.fit(X_train, Y_train)

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes - 1):
        class_list = ["LOW SIGNAL", "INSUFFICIENT", "MIXED RESULTS", "SUFFICIENT"]
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], lw=2, label=class_list[i].format(i))

    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    title_text = "Precision vs. Recall Curve for " + str(classifier)[30:]
    plt.title(title_text, fontsize=11, ha='center')
    txt1 = "Witholding Percentage: " + str(perc_witheld * 100) + "%"
    txt2 = "Witholding Score: " + str(witholding_score)
    final_txt = txt1 + ", " + txt2
    plt.text(0.63, .05, final_txt, ha='center', fontsize=7.2)

    s1 = str(classifier)[30:]
    s1 = s1.split()[0]
    s1 = s1.replace("(", "_")
    s1 = s1.replace("=", "_")
    # For Randomizing and making each unique out graph save under a different name
    out_save_name = s1[0: len(s1) - 1] + "_" + str(int(witholding_score * 100) + random.randint(0, 50))
    print("Graph has been generated under the name: pvr" + out_save_name)
    plt.savefig('pvr' + out_save_name)

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average Precision Score for ' + str(classifier)[30:])
    final_txt2 = 'Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"])
    plt.text(.5, .05, final_txt2, ha='center')

    print("Graph has been generated under the name: aps" + out_save_name)
    plt.savefig('aps' + out_save_name)


def roc_curve_grapher(x_normalized, y_labels, clf):
    print("Running ROC Curve Calculations for", clf)

    trainX, testX, trainy, testy = train_test_split(x_normalized, y_labels, test_size=0.5, random_state=2)
    clf.fit(trainX, trainy)
    y_score = clf.predict_proba(testX)
    positive_outcome_prob = y_score[:, 1]

    fpr, tpr, _ = roc_curve(testy, positive_outcome_prob)
    roc_auc = auc(fpr, tpr)

    ns_probs = [0 for _ in range(len(positive_outcome_prob))]
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, positive_outcome_prob)

    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for RandomForest Classifier')
    plt.legend(loc="lower right")
    plt.show()

    return_list = ['No Skill: ROC AUC=%.3f' % (ns_auc), 'Logistic: ROC AUC=%.3f' % (lr_auc)]
    return return_list


# Re-classifies all input samples, and returns any sample that was given a different prediction
# then it's ground truth label
def compare_predictions(ground_truth, features, labels):
    normalized_features = []
    for x in features:
        normalized_features.append(x[1:])

    index = 0
    trainX, testX, trainy, testy = train_test_split(normalized_features, labels, test_size=0.2, random_state=2)

    clf = RandomForestClassifier(n_estimators=500, max_depth=None, criterion='entropy',
                                           min_samples_split=2, random_state=0, max_features='log2', min_samples_leaf=2)

    clf.fit(trainX, trainy)

    predictions = clf.predict(normalized_features)

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

        ground_truth[index].append(prediction)
        index +=1

    fields = ['Holder', 'Sample ID', 'Ground Truth', 'Predicted']
    out_list = np.array([fields])

    out_important_list = np.array([fields])

    for sample in ground_truth:
        sample_name = sample[0]
        true_label = sample[1]
        prediction = sample[2]

        if true_label != prediction:
            dimension_match = "1"  # For matching the input dimensions of the panda dataframe
            row_arr = np.array([[dimension_match, sample_name, true_label, prediction]], dtype=object)
            out_list = np.concatenate((out_list, row_arr))
        if true_label == 'INSUFFICIENT' and prediction == 'SUFFICIENT':
            dimension_match = "1"  # For matching the input dimensions of the panda dataframe
            row_arr2 = np.array([[dimension_match, sample_name, true_label, prediction]], dtype=object)
            out_important_list = np.concatenate((out_important_list, row_arr2))
        if true_label == "SUFFICIENT" and prediction == "INSUFFICIENT":
            dimension_match = "1"  # For matching the input dimensions of the panda dataframe
            row_arr3 = np.array([[dimension_match, sample_name, true_label, prediction]], dtype=object)
            out_important_list = np.concatenate((out_important_list, row_arr3))


    df = pd.DataFrame(data=out_list[1:, 1:], index=out_list[1:, 0], columns=out_list[0, 1:])
    df.to_csv('discrepancies.csv', index=False)
    df2 = pd.DataFrame(data=out_important_list[1:, 1:], index=out_important_list[1:, 0], columns=out_important_list[0, 1:])
    df2.to_csv('important_discrepancies.csv', index=False)









    # need to normalize features!!!!!


# Saves the best trained model to the disk
def save_top_model(features, y_labels, input_classifier):
    clf = input_classifier.fit(features, y_labels)

    #filename = 'best_model.sav'
    #pickle.dump(clf, open(filename, 'wb'))



