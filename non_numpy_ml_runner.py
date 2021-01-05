# Runs ML part of project
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import time
import results


# Runs through each ML model type, responsible for running all class methods and functions in this file
# Called by the label_classes run_ml() methods
def run_ml_algorithms(X_features, Y_labels, dataset, runs, normalization):

    if normalization:
        normalized_x = preprocessing.normalize(X_features)
        out_data = MLModels(X_features, normalized_x, Y_labels, dataset, runs, normalization).parameterLoop()
    else:
        out_data = MLModels(X_features, X_features, Y_labels, dataset, runs, normalization).parameterLoop()
    # Note: Lowest SD and Highest accuracy: SDG(p='l1'), non-normalized data, but normalized data much smaller SD

    return out_data


# Implements all ML Models with their various tuned classifiers
class MLModels:
    def __init__(self, feature_set, normalized_features, labels, label_name, runs, normalization):
        self.normalization = normalization
        self.percentage_witheld = [0.3, 0.4, 0.5]
        self.feature_set = feature_set
        self.normalized_features = normalized_features
        self.labels = labels
        self.label_name = label_name
        self.runs = [runs, 1, 1]  # Caps runs of svc and rfc because there will be no variation
        self.sdg = [#SGDClassifier(loss="hinge", penalty="l2", max_iter=100),
                    #SGDClassifier(loss="hinge", penalty="l2", max_iter=100000),
                    #SGDClassifier(loss="hinge", penalty="l2", max_iter=1000000000),
                    #SGDClassifier(loss="hinge", penalty="l1", max_iter=1000),
                    #SGDClassifier(loss="modified_huber", penalty="l2", max_iter=1000),
                    #SGDClassifier(loss="log", penalty="l2", max_iter=1000)
            ]
        self.svc = [#SVC(gamma='scale', C=1.0)
            ]
        self.rfc = [RandomForestClassifier(n_estimators=10, max_depth=None,
                                           min_samples_split=2, random_state=0),
                    RandomForestClassifier(n_estimators=700, max_depth=None,
                                           min_samples_split=2, random_state=0),
                    RandomForestClassifier(n_estimators=500, max_depth=None, criterion='entropy',
                                           min_samples_split=2, random_state=0, max_features='log2', min_samples_leaf=2)
                    ]
        self.classifiers = [self.sdg, self.svc, self.rfc]

    # Loops over training set with adjusted hyper-parameters
    def parameterLoop(self):
        out_data = []
        run_holding_val = 0
        for classifier_list in self.classifiers:
            runs = self.runs[run_holding_val]
            for classifier in classifier_list:
                for percentage in self.percentage_witheld:
                    print("Running", classifier, "on", self.label_name, "labels with", percentage * 100,
                          "% Witheld for fitting")
                    start = time.monotonic()
                    out = results.calculate_accuracy(classifier, self.normalized_features, self.feature_set, self.labels,runs,
                                             percentage)
                    avg_precision = results.precision_recall_scores(self.normalized_features, self.labels, classifier,
                                                                    percentage)

                    end = time.monotonic()
                    total = end - start
                    out_data.append([self.label_name, runs, classifier, out[3], percentage, out[0], out[1], out[2],
                                     total, self.normalization, avg_precision])
            run_holding_val += 1
        return out_data


# For very quickly running all model types to test certain things
class QuickRunMLModels:

    def __init__(self, feature_set, normalized_features, labels, label_name, runs, normalization):
        self.feature_set = feature_set
        self.normalized_features = normalized_features
        self.labels = labels
        self.label_name = label_name
        self.runs = 1
        self.sdg = [SGDClassifier(loss="hinge", penalty="l2", max_iter=100)]
        self.svc = [SVC(gamma='scale', C=1.0)]
        self.rfc = [RandomForestClassifier(n_estimators=10, max_depth=None)]
        self.classifiers = [self.sdg, self.svc, self.rfc]

    def parameterLoop(self):
        out_data = []
        for classifier_list in self.classifiers:
            for classifier in classifier_list:
                out = results.calculate_accuracy(classifier, self.normalized_features, self.feature_set, self.labels, self.runs, 0.4)
                total = 0
                out_data.append([self.label_name, self.runs, classifier, out[3], 0.4, out[0], out[1], out[2],
                                     total, True])
        return out_data



















