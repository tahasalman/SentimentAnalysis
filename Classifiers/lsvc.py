from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import PredefinedSplit
import sys
sys.path.append("../")
from data_processor import Data

class LinearSupportVectorClassifier():
    def __init__(self,training_data_x,training_data_y):
        self.training_data_x = training_data_x
        self.training_data_y = training_data_y
        self.initialize_classifier()

    def initialize_classifier(self,penalty='l2',loss='squared_hinge',dual=True,tol=0.0001,C=1.0,):
        self.classifier = LinearSVC(penalty=penalty,loss=loss,dual=dual,tol=tol,C=C)

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

    def find_best_params(self,validation_data_x,validation_data_y,n_jobs=1,param_grid=[]):
        if not param_grid:
            param_grid = self.get_param_grid_default()

        merged_x = Data.merge_arrays(self.training_data_x, validation_data_x)
        merged_y = Data.merge_arrays(self.training_data_y, validation_data_y)
        test_fold = []

        for i in range(0,len(self.training_data_y)):
            test_fold.append(1)
        for i in range(0,len(validation_data_y)):
            test_fold.append(0)

        cv = PredefinedSplit(test_fold)

        gs = GridSearchCV(
            estimator=LinearSVC(),
            scoring='f1_micro',
            param_grid=param_grid,
            n_jobs=n_jobs,
            cv=cv
        )

        gs.fit(merged_x,merged_y)

        best_params = gs.best_params_
        best_score = gs.best_score_
        results = gs.cv_results_
        return best_params,best_score,results


    def get_param_grid_default(self):
        dual_val = True
        if len(self.training_data_x) > len(self.training_data_x[0]): #if number of samples > number of features
            dual_val = False

        tol_vals = []
        current_tol = 0.00000001
        while(current_tol<1):
            tol_vals.append(current_tol)
            current_tol *= 10

        c_vals = []
        current_c_val = 0.001
        while(current_c_val<=10000):
            c_vals.append(current_c_val)
            current_c_val *= 10

        param_grid = {"penalty": ['l1', 'l2'],
                      "loss": ["hinge","squared_hinge"],
                      "dual": [dual_val],
                      "tol" : [tol_vals],
                      "C" : c_vals
                      }

        return param_grid