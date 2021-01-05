import csv


# Collects and prunes all feature data used for training ML models
# This data is the feature set, and will be run against different label sets with matching sample-id's
def qc_dump_csv_handler(tsv_name):
    out_2d_array = []
    # WARNING -- DATA MUST BE STORED AS A TSV
    with open(tsv_name, newline='') as tsv_file:
        # Allows reader to be a dict reader where you can pull out individual excel tabs, in dict fashion
        reader = csv.DictReader(tsv_file, dialect='excel-tab')

        # Create a file of the "useful" rows for parsing
        # This can be accessed and changed if other information in original file is deemed useful
        kept_headers_list = keep_which_headers('KeptFeaturesHeaders.csv')

        # Should be noted that null fields are treated as 0s, might consider changing this if
        # important for different models
        # Also this is hard-coded so if a different type of feature-data file is used, this will
        # be completed useless
        for row in reader:
            row_array = [row['sample_timestamp'], row['sample']]
            for header in kept_headers_list:
                val = row[header]
                try:
                    if "%" in val:
                        final_conversion = float(val[:-1]) / 100
                    elif "" == val:
                        final_conversion = 0
                    elif "," in val:
                        final_conversion = float(val.replace(",", ""))
                    elif val == 0:
                        final_conversion = 0
                    elif val.lower() == "nan":
                        final_conversion = 0

                    else:
                        final_conversion = float(val)
                except ValueError:
                    final_conversion = 0

                row_array.append(final_conversion)
            out_2d_array.append(row_array)

    return out_2d_array


# Creates a list of the headers you want to keep on the QCDumpCSV file
def keep_which_headers(kept_headers_csv):
    kept_headers_list = []
    with open(kept_headers_csv, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter='\t')
        for row in reader:
            kept_headers_list.append(row[0])
    return kept_headers_list


# For parsing new input samples that will be classified
def parse_input_samples(tsv_name):
    out_2d_array = []

    with open(tsv_name, newline='') as tsv_file:
        reader = csv.DictReader(tsv_file, dialect='excel-tab')
        kept_headers_list = keep_which_headers('KeptFeaturesHeaders.csv')
        for row in reader:
            row_array = [row['sample_timestamp']]
            for header in kept_headers_list:
                val = row[header]
                try:
                    if "%" in val:
                        final_conversion = float(val[:-1]) / 100
                    elif "" == val:
                        final_conversion = 0
                    elif "," in val:
                        final_conversion = float(val.replace(",", ""))
                    elif val == 0:
                        final_conversion = 0
                    elif val.lower() == "nan":
                        final_conversion = 0

                    else:
                        final_conversion = float(val)
                except ValueError:
                    final_conversion = 0

                row_array.append(final_conversion)
            out_2d_array.append(row_array)

    return out_2d_array


# Returns the sample ID and label from the QC Dump
def sample_and_label(tsv_name):
    out_2d_array = []

    with open(tsv_name, newline='') as tsv_file:
        reader = csv.DictReader(tsv_file, dialect='excel-tab')
        kept_headers_list = keep_which_headers('KeptFeaturesHeaders.csv')
        for row in reader:
            row_array = [row['sample_timestamp']]
            judgment = row['judgment']

            if judgment == "PASS" or judgment == "SUFFICIENT":
                row_array.append("SUFFICIENT")
            elif judgment == "INSUFFICIENT" or judgment == "FAIL":
                row_array.append("INSUFFICIENT")
            elif judgment == '':
                row_array.append("UNKNOWN")
            else:
                row_array.append(judgment)

            out_2d_array.append(row_array)

    return out_2d_array

