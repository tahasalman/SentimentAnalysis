import logging
import numpy as np
from data_processor import Data
from Classifiers.simple_classifiers import RandomClassifier
from Classifiers.simple_classifiers import MajorityClassClassifier
from Classifiers.simple_classifiers import PerformanceTester
from Classifiers.bnb import BernoulliNaiveBayesClassifier

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



    alpha_vals = np.linspace(start=0,stop=1,num=100)
    f.write("Testing 200 alpha values between 0 and 1:\n")

    #sorted_alpha_vals = bnb.sort_alpha_vals(alpha_vals,validation_data_x,validation_data_y)
    # f.write(
    #     "The best alpha value is {} with an F1-Measure of {}".format(sorted_alpha_vals[0][0], sorted_alpha_vals[0][1]))

    best_params,results = bnb.find_best_params(validation_data_x,validation_data_y,alpha_vals)
    f.write("The best alpha value found was {}\n".best_params["alpha"])
    f.write("\nPerformance metrics for all alpha values tested:\n\n")

    output = ""
    for i in range(0,len(alpha_vals)):
        output+= "Alpha value: " + str(alpha_vals[i]) + " --> F1-Score: " + str(results['mean_test_score'][i]) + "\n"
    f.write(output)




if __name__ == "__main__":
    run_random_classifier()
    run_majority_class_classifier()
    run_naive_bayes_bernoulli()
