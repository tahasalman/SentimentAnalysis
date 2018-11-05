import logging
import numpy as np
from data_processor import Data
from Classifiers.simple_classifiers import RandomClassifier
from Classifiers.simple_classifiers import MajorityClassClassifier
from Classifiers.simple_classifiers import PerformanceTester
from Classifiers.bnb import BernoulliNaiveBayesClassifier
from Classifiers.lsvc import LinearSupportVectorClassifier

logging.basicConfig(filename="q2.log",level=logging.INFO)

def logging_wrapper(func):
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.exception("There was an exception {} in function {}".format(str(e),str(func)))
    return inner


@logging_wrapper
def run_random_classifier():
    OUTPUT_PATH = "Outputs/yelp/random-classifier-out.txt"
    f = open(OUTPUT_PATH,"w")

    TRAINING_DATA_PATH = "Data/Processed/yelp-train.txt"
    TESTING_DATA_PATH = "Data/Processed/yelp-test.txt"

    f.write("Initializing Random Classifier with training data from the path {}\n".format(TRAINING_DATA_PATH))
    rc = RandomClassifier(TRAINING_DATA_PATH)

    f.write("Making Predictions on the data from the path {}\n".format(TESTING_DATA_PATH))
    predictions = rc.classify(TESTING_DATA_PATH)

    p_tester = PerformanceTester(predictions)
    f1s = p_tester.get_F1_score(TESTING_DATA_PATH)
    f.write("The F1 score for this random classifier is {}\n".format(f1s))

    f.close()


@logging_wrapper
def run_majority_class_classifier():
    OUTPUT_PATH = "Outputs/yelp/majority-class-classifier-out.txt"
    f = open(OUTPUT_PATH, "w")

    TRAINING_DATA_PATH = "Data/Processed/yelp-train.txt"
    TESTING_DATA_PATH = "Data/Processed/yelp-test.txt"

    f.write("Initializing Majority Class Classifier with training data from the path {}\n".format(TRAINING_DATA_PATH))
    mcc = MajorityClassClassifier(TRAINING_DATA_PATH)

    f.write("Making Predictions on the data from the path {}\n".format(TESTING_DATA_PATH))
    predictions = mcc.classify(TESTING_DATA_PATH)

    p_tester = PerformanceTester(predictions)
    f1s = p_tester.get_F1_score(TESTING_DATA_PATH)
    f.write("The F1 score for this Majority Class Classifier is {}\n".format(f1s))

    f.close()


@logging_wrapper
def run_naive_bayes_bernoulli():
    OUTPUT_PATH = "Outputs/yelp/naive-bayes-bbow-out.txt"
    f = open(OUTPUT_PATH, "w+")

    TRAINING_DATA_PATH = "Data/BinaryBOW/yelp-train"
    VALIDATION_DATA_PATH = "Data/BinaryBOW/yelp-valid"
    TESTING_DATA_PATH = "Data/BinaryBOW/yelp-test"

    f.write("Loading Binary Bag-Of-Words Representation for Training Data\n")
    training_data_x = Data.read_x_array(TRAINING_DATA_PATH+"-X.csv")
    training_data_y = Data.read_y_array(TRAINING_DATA_PATH + "-Y.csv")

    f.write("Initializing Bernoulli Naive Bayes Class Classifier with training data\n")
    bnb = BernoulliNaiveBayesClassifier(
        training_data_x=training_data_x,
        training_data_y=training_data_y
    )

    f.write("Finding best alpha value for Naive Bayes Bernoulli Model\n")
    f.write("Loading validation data\n")
    validation_data_x = Data.read_x_array(VALIDATION_DATA_PATH+"-X.csv")
    validation_data_y = Data.read_y_array(VALIDATION_DATA_PATH+"-Y.csv")


    alpha_start = 0
    alpha_stop = 1
    num_intervals = 100
    alpha_vals = np.linspace(start=alpha_start,stop=alpha_stop,num=num_intervals)
    f.write("Testing {} alpha values between {} and {}:\n".format(num_intervals,alpha_start,alpha_stop))


    best_params,results = bnb.find_best_params(validation_data_x,validation_data_y,alpha_vals,n_jobs=10)
    f.write("The best alpha value found was {}\n".format(best_params["alpha"]))
    f.write("\nPerformance metrics for all alpha values tested:\n\n")

    output = ""
    for i in range(0,len(alpha_vals)):
        output+= "Alpha value: " + str(alpha_vals[i]) + " --> F1-Score: " + str(results['mean_test_score'][i]) + "\n"
    f.write(output)

    f.close()


@logging_wrapper
def run_linear_svm():
    OUTPUT_PATH = "Outputs/yelp/linear-svm-bbow-out.txt"
    f = open(OUTPUT_PATH, "w")

    TRAINING_DATA_PATH = "Data/BinaryBOW/yelp-train"
    VALIDATION_DATA_PATH = "Data/BinaryBOW/yelp-valid"
    TESTING_DATA_PATH = "Data/BinaryBOW/yelp-test"

    f.write("Loading Binary Bag-Of-Words Representation for Training Data\n")
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
        n_jobs=10)

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

    print(best_score)

    f.close()


if __name__ == "__main__":
    # run_random_classifier()
    # run_majority_class_classifier()
    # run_naive_bayes_bernoulli()
    run_linear_svm()