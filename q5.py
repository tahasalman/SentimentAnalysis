import logging
from data_processor import Data
from Classifiers.gnb import GaussianNaiveBayesClassifier
from Classifiers.lsvc import LinearSupportVectorClassifier
from Classifiers.decision_tree import DecisionTree

logging.basicConfig(filename="q3.log",level=logging.INFO)

def logging_wrapper(func):
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.exception("There was an exception {} in function {}".format(str(e),str(func)))
    return inner

@logging_wrapper
def run_naive_bayes_gaussian():
    OUTPUT_PATH = "Outputs/IMDB/naive-bayes-fbow-out.txt"
    f = open(OUTPUT_PATH, "w")

    TRAINING_DATA_PATH = "Data/FrequencyBOW/IMDB-train"
    VALIDATION_DATA_PATH = "Data/FrequencyBOW/IMDB-valid"
    TESTING_DATA_PATH = "Data/FrequencyBOW/IMDB-test"

    f.write("Loading Frequency Bag-Of-Words Representation for Training Data\n")
    training_data_x = Data.read_x_array(TRAINING_DATA_PATH+"-X.csv")
    training_data_y = Data.read_y_array(TRAINING_DATA_PATH + "-Y.csv")

    f.write("Initializing Gaussian Naive Bayes Class Classifier with training data\n")
    gnb = GaussianNaiveBayesClassifier(
        training_data_x=training_data_x,
        training_data_y=training_data_y
    )

    f.write("Finding best variance smoothing value for Gaussian Naive Bayes Model\n")
    f.write("Loading validation data\n")
    validation_data_x = Data.read_x_array(VALIDATION_DATA_PATH+"-X.csv")
    validation_data_y = Data.read_y_array(VALIDATION_DATA_PATH+"-Y.csv")

    best_params,results = gnb.find_best_params(validation_data_x,validation_data_y,n_jobs=1)
    f.write("The best variance smoothing value found was {}\n".format(best_params["var_smoothing"]))
    f.write("\nPerformance metrics for all var_smoothing values tested:\n\n")

    index = 0
    while (index < 100 and index < len(results['params'])):
        f.write("var_smoothing: {} --> {}\n".format(
            results['params'][index]['var_smoothing'],
            results['mean_test_score'][index]
        ))
        index += 1

    f.write("\n\nInitializing and training a Gaussian Naive Bayes Model with best hyper-parameters\n")
    gnb = GaussianNaiveBayesClassifier(training_data_x,training_data_y)
    gnb.initialize_classifier(best_params['var_smoothing'])
    gnb.train()

    testing_data_x = Data.read_x_array(TESTING_DATA_PATH + "-X.csv")
    testing_data_y = Data.read_y_array(TESTING_DATA_PATH + "-Y.csv")

    f.write("Finding F1-Measure for different datasets\n")
    f1_train = gnb.get_f1_measure(training_data_x,training_data_y)
    f1_valid = gnb.get_f1_measure(validation_data_x,validation_data_y)
    f1_test = gnb.get_f1_measure(testing_data_x,testing_data_y)

    f.write("The F1-Measure on training data is {}\n".format(f1_train))
    f.write("The F1-Measure on validation data is {}\n".format(f1_valid))
    f.write("The F1-Measure on testing data is {}\n".format(f1_test))

    f.close()

@logging_wrapper
def run_linear_svm():
    OUTPUT_PATH = "Outputs/IMDB/linear-svm-fbow-out.txt"
    f = open(OUTPUT_PATH, "w")

    TRAINING_DATA_PATH = "Data/FrequencyBOW/IMDB-train"
    VALIDATION_DATA_PATH = "Data/FrequencyBOW/IMDB-valid"
    TESTING_DATA_PATH = "Data/FrequencyBOW/IMDB-test"

    f.write("Loading Frequency Bag-Of-Words Representation for Training Data\n")
    training_data_x = Data.read_x_array(TRAINING_DATA_PATH + "-X.csv")
    training_data_y = Data.read_y_array(TRAINING_DATA_PATH + "-Y.csv")

    f.write("Initializing Linear Support Vector Classifier with training data\n")
    lsvc = LinearSupportVectorClassifier(
        training_data_x=training_data_x,
        training_data_y=training_data_y
    )


    f.write("Loading validation data\n")
    validation_data_x = Data.read_x_array(VALIDATION_DATA_PATH + "-X.csv")
    validation_data_y = Data.read_y_array(VALIDATION_DATA_PATH + "-Y.csv")


    f.write("Finding the best hyper-parameters:\n")
    best_params,best_score,results = lsvc.find_best_params(
        validation_data_x,
        validation_data_y,
        n_jobs=1)

    f.write("The best hyper-parameters are as follows: \n")
    f.write("C: {}\t| tol: {} with an F1-Measure of {}\n\n".format(
        best_params['C'],best_params['tol'],best_score
    ))

    f.write("\nPerformance metrics for the first 100 hyper-parameters_tested:\n\n")
    index=0
    while(index<100 and index<len(results['params'])):
        f.write("C: {}\t| tol: {} --> {}\n".format(
            results['params'][index]['C'],
            results['params'][index]['tol'],
            results['mean_test_score'][index]
    ))
        index+=1

    f.write("\n\nInitializing and training a Linear Support Vector Classifier with C={} and tol={} \n".format(best_params['C'],best_params['tol']))
    best_C = float(best_params['C'])
    best_tol = float(best_params['tol'])
    lsvc = LinearSupportVectorClassifier(training_data_x, training_data_y)
    lsvc.initialize_classifier(tol=best_tol,C=best_C)
    lsvc.train()

    testing_data_x = Data.read_x_array(TESTING_DATA_PATH + "-X.csv")
    testing_data_y = Data.read_y_array(TESTING_DATA_PATH + "-Y.csv")

    f.write("Finding F1-Measure for different datasets\n")
    f1_train = lsvc.get_f1_measure(training_data_x, training_data_y)
    f1_valid = lsvc.get_f1_measure(validation_data_x, validation_data_y)
    f1_test = lsvc.get_f1_measure(testing_data_x, testing_data_y)

    f.write("The F1-Measure on training data with C={} and tol={} is {}\n".format(best_C,best_tol, f1_train))
    f.write("The F1-Measure on validation data with C={} and tol={} is {}\n".format(best_C,best_tol, f1_valid))
    f.write("The F1-Measure on testing data with C={} and tol={} is {}\n".format(best_C,best_tol, f1_test))

    f.close()


@logging_wrapper
def run_decision_tree():
    OUTPUT_PATH = "Outputs/IMDB/decision-tree-fbow-out.txt"
    f = open(OUTPUT_PATH, "w")

    TRAINING_DATA_PATH = "Data/FrequencyBOW/IMDB-train"
    VALIDATION_DATA_PATH = "Data/FrequencyBOW/IMDB-valid"
    TESTING_DATA_PATH = "Data/FrequencyBOW/IMDB-test"

    f.write("Loading Frequency Bag-Of-Words Representation for Training Data\n")
    training_data_x = Data.read_x_array(TRAINING_DATA_PATH + "-X.csv")
    training_data_y = Data.read_y_array(TRAINING_DATA_PATH + "-Y.csv")

    f.write("Initializing Decision Tree Classifier with training data\n")
    dt = DecisionTree(
        training_data_x=training_data_x,
        training_data_y=training_data_y
    )


    f.write("Loading validation data\n")
    validation_data_x = Data.read_x_array(VALIDATION_DATA_PATH + "-X.csv")
    validation_data_y = Data.read_y_array(VALIDATION_DATA_PATH + "-Y.csv")


    f.write("Finding the best hyper-parameters:\n")
    best_params,best_score,results = dt.find_best_params(
        validation_data_x,
        validation_data_y,
        n_jobs=1)

    f.write("The best hyper-parameters are as follows: \n")
    f.write("max_depth: {}\t| min_samples_split: {}\t| min_samples_leaf: {}\t| max_features: {} with an average F1-Measure of {}\n\n".format(
        best_params['max_depth'],best_params['min_samples_split'],best_params['min_samples_leaf'],best_params['max_features'],best_score
    ))

    f.write("\nPerformance metrics for the first 100 hyper-parameters_tested:\n\n")
    index=0
    while(index<100 and index<len(results['params'])):
        f.write("max_depth: {}\t| min_samples_split: {}\t| min_samples_leaf: {}\t| max_features: {} --> {}\n".format(
            results['params'][index]['max_depth'],
            results['params'][index]['min_samples_split'],
            results['params'][index]['min_samples_leaf'],
            results['params'][index]['max_features'],
            results['mean_test_score'][index]
    ))
        index+=1

    f.write("\n\nInitializing and training a Decision Tree Classifier with the best parameters \n")
    dt = DecisionTree(training_data_x, training_data_y)
    dt.initialize_classifier(max_depth=best_params['max_depth'],
                             min_samples_split=best_params['min_samples_split'],
                             min_samples_leaf=best_params['min_samples_leaf'],
                             max_features=best_params['max_features'])
    dt.train()

    testing_data_x = Data.read_x_array(TESTING_DATA_PATH + "-X.csv")
    testing_data_y = Data.read_y_array(TESTING_DATA_PATH + "-Y.csv")

    f.write("Finding F1-Measure for different datasets\n")
    f1_train = dt.get_f1_measure(training_data_x, training_data_y)
    f1_valid = dt.get_f1_measure(validation_data_x, validation_data_y)
    f1_test = dt.get_f1_measure(testing_data_x, testing_data_y)

    f.write("The F1-Measure on training data with these parameters is {}\n".format(f1_train))
    f.write("The F1-Measure on validation data with these parameters is {}\n".format(f1_valid))
    f.write("The F1-Measure on testing data with these parameters is {}\n".format(f1_test))

    f.close()

if __name__ == "__main__":
    logging.info("Starting Q3\n")
    run_naive_bayes_gaussian()
    logging.info("Done with GNB\n")
    run_linear_svm()
    logging.info("Done with LSVM\n")
    run_decision_tree()
    logging.info("Done with Q3\n")