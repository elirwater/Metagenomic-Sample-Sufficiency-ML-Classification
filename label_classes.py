# Stores classes for the various label libraries
from abc import ABC, abstractmethod
import csv
import numpy as np
import classification_ml_runner


# Represents an interface for all the different label sets being used
class FeatureLabelsInterface(ABC):

    @abstractmethod
    def csv_handler(self):
        pass

    @abstractmethod
    def list_features(self):
        pass

    @abstractmethod
    def list_labels(self):
        pass

    def run_ml(self, normalization):
        pass


# Implements the collection of labels from a specific data set and runs these labels against their feature data
class QCDumpLabels(FeatureLabelsInterface):

    def __init__(self, feature_array, csv_name, runs):
        self.feature_array = feature_array
        self.csv_name = csv_name
        self.runs = runs

    # Collects labels for each sample
    def csv_handler(self):
        out_array = []
        with open(self.csv_name, newline='') as tsv_file:
            reader = csv.DictReader(tsv_file, dialect='excel-tab')

            for row in reader:
                row_dict = dict()
                if row['judgment'] == "PASS" or row['judgment'] == "SUFFICIENT":
                    row_dict[row['sample_timestamp']] = 3
                elif row['judgment'] == "MIXED RESULTS":
                    row_dict[row['sample_timestamp']] = 2
                elif row['judgment'] == "INSUFFICIENT" or row['judgment'] == "FAIL":
                    row_dict[row['sample_timestamp']] = 1
                elif row['judgment'] == "LOW SIGNAL":
                    row_dict[row['sample_timestamp']] = 0
                else:
                    row_dict[row['sample_timestamp']] = 10
                out_array.append(row_dict)
        return out_array

    # Creates the feature list to be used in training
    def list_features(self):
        out_feature_array = []
        feature_array = self.feature_array
        for sample in feature_array:
            out_feature_array.append(sample[2:])
        return out_feature_array
    
    # Creates a list of all labels corresponding with features
    def list_labels(self):
        out_label_array = []
        label_array = self.csv_handler()
        for sample in label_array:
            out_label_array.append(list(sample.values())[0])
        return out_label_array

    # Runs ML models from non_numpy_ml_runner python file
    def run_ml(self, normalization):
        results = classification_ml_runner.run_ml_algorithms(self.list_features(), self.list_labels(), self.csv_name,
                                                             self.runs, normalization)
        return results

    # For ROC graphing using binary classification
    # Could also use Sklearn's binarize method
    def generate_binary_labels(self):
        out_array = []
        with open(self.csv_name, newline='') as tsv_file:
            reader = csv.DictReader(tsv_file, dialect='excel-tab')

            for row in reader:
                row_dict = dict()
                if row['judgment'] == "PASS" or row['judgment'] == "SUFFICIENT":
                    row_dict[row['sample_timestamp']] = 1
                else:
                    row_dict[row['sample_timestamp']] = 0
                out_array.append(row_dict)

        out_label_array = []
        for sample in out_array:
            out_label_array.append(list(sample.values())[0])

        return out_label_array


# Implements the collection of labels from a specific data set and runs these labels against their feature data
class QCDBLabels(FeatureLabelsInterface):

    def __init__(self, feature_array, csv_name, runs):
        self.feature_array = feature_array
        self.csv_name = csv_name
        self.out_label_array = []
        self.runs = runs

    # Collects labels for each sample
    def csv_handler(self):
        #  WARNING -- DATA MUST BE STORED AS A CSV
        out_array = []
        with open(self.csv_name, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                row_dict = dict()
                try:
                    row_dict[row['sample_timestamp']] = int(row['we_like_it'])
                except ValueError:
                    row_dict[row['sample_timestamp']] = 10
                out_array.append(row_dict)
        return out_array

    # Creates the feature list to be used in training, weeding out features without labels from the dataset
    def list_features(self):
        feature_array = self.feature_array
        out_feature_array = []
        label_array = self.csv_handler()
        for row in feature_array:
            for sample in label_array:
                sample_key = list(sample.keys())[0]
                if row[0] == sample_key:
                    out_feature_array.append(row[2:])
                    self.out_label_array.append(list(sample.values())[0])
        return out_feature_array

    # Creates a list of all labels corresponding with features
    def list_labels(self):
        return self.out_label_array

    # Runs ML models from non_numpy_ml_runner python file
    def run_ml(self, normalization):
        features = self.list_features()
        results = classification_ml_runner.run_ml_algorithms(features, self.list_labels(), self.csv_name,
                                                             self.runs, normalization)
        return results


# Unused --- Issue with matching samples to labels, currently un-referenced and unimplemented
# Also not enough label samples to train a good model
class PostMortem2019Labels(FeatureLabelsInterface):
    def __init__(self, feature_array, csv_name):
        self.feature_array = feature_array
        self.csv_name = csv_name

    def csv_handler(self):
        #  WARNING -- DATA MUST BE STORED AS A CSV
        out_array = []
        with open(self.csv_name, newline='') as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                row_dict = dict()
                try:
                    row_dict[row['Name']] = int(row['Customer Satisfaction'])
                except ValueError:
                    row_dict[row['Name']] = 0
                out_array.append(row_dict)
        return out_array


    def create_training_set(self):
        feature_array = self.feature_array
        label_array = self.csv_handler()
        out_feature_array = []
        out_label_array = []

        test_list = list()

        for row in feature_array:
            sample_name = row[1]
            identifiers = sample_name.split("_")
            try:
                identifier1 = identifiers[0].lower()
                identifier2 = identifiers[1].lower()
                if len(identifier1) > 2:
                    for sample in label_array:
                        label_name = list(sample.keys())[0].lower()
                        if identifier1 in label_name and identifier2 in label_name:
                            test_list.append([identifier1, identifier2, label_name])

                            out_feature_array.append(row[2:])
                            out_label_array.append(list(sample.values())[0])

            except IndexError:
                identifier1 = identifiers[0]

        return out_feature_array, out_label_array






