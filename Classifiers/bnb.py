from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import PredefinedSplit
import sys
sys.path.append("../")
from data_processor import Data

class BernoulliNaiveBayesClassifier():
    def __init__(self,training_data_x,training_data_y,alpha=1.0):
        self.training_data_x = training_data_x
        self.training_data_y = training_data_y
        self.initialize_classifier(alpha)

    def initialize_classifier(self,alpha=1.0):
        self.classifier = BernoulliNB(alpha)

    def train(self):
        self.classifier.fit(self.training_data_x,self.training_data_y)

    def predict(self,sample_x):
        return self.classifier.predict(sample_x)

    def get_f1_measure(self,sample_x,sample_y):
        predictions = self.predict(sample_x)
        score = f1_score(
            y_true=sample_y,
            y_pred=predictions,
            average="micro"
        )
        return score

    def find_best_alpha_old(self,alpha_list,valid_data_x, valid_data_y):
        alpha_dict = {}

        for alpha in alpha_list:
            self.initialize_classifier(alpha)
            self.train()
            alpha_dict[alpha] = self.get_f1_measure(valid_data_x,valid_data_y)

        alpha_vals = []
        for val in sorted(alpha_dict, key=alpha_dict.get, reverse=True):
            alpha_vals.append((val, alpha_dict[val]))

        return alpha_vals


    def find_best_params(self,validation_data_x,validation_data_y,alpha_vals,n_jobs=1):

        merged_x = Data.merge_arrays(self.training_data_x, validation_data_x)
        merged_y = Data.merge_arrays(self.training_data_y, validation_data_y)
        test_fold = []

        for i in range(0,len(self.training_data_y)):
            test_fold.append(1)
        for i in range(0,len(validation_data_y)):
            test_fold.append(0)

        cv = PredefinedSplit(test_fold)

        param = {"alpha": alpha_vals}
        gs = GridSearchCV(
            estimator=BernoulliNB(),
            scoring='f1_micro',
            param_grid=param,
            n_jobs=n_jobs,
            cv=cv
        )

        gs.fit(merged_x,merged_y)

        best_params = gs.best_params_
        results = gs.cv_results_
        return best_params,results

