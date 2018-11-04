import random
from sklearn.metrics import f1_score

def read_data(data_path, encoding="utf-8"):
    data = []
    with open(data_path, "r", encoding=encoding) as f:
        data = f.readlines()

    return data

class Classifier():
    def __init__(self,training_data_path,classes=[]):
        if classes:
            self.classes = classes
        else:
            self.classes = self.set_classes(training_data_path)

    def set_classes(self,data_path):
        classes = []
        data = read_data(data_path)
        for line in data:
            temp = line.split(" ")
            cl = (temp[-1].split("\t")[1]).strip('\n')
            if cl not in classes:
                classes.append(cl)
        return classes


class RandomClassifier(Classifier):

    def classify(self,data_path):
        data = read_data(data_path)
        predictions = []
        for line in data:
            predictions.append(self.predict_class())
        return predictions

    def predict_class(self):
        rint = random.randint(0,len(self.classes)-1)
        return self.classes[rint]


class MajorityClassClassifier(Classifier):
    def __init__(self, training_data_path, classes=[]):
        super().__init__(training_data_path, classes)
        self.class_frequencies = self.set_class_frequencies(training_data_path)

    def classify(self,data_path):
        predictions = []
        data = read_data(data_path)
        for line in data:
            predictions.append(self.predict_class())

        return predictions

    def set_class_frequencies(self,data_path):
        class_dict = {}
        data = read_data(data_path)

        for line in data:
            temp = line.split(" ")
            cl = (temp[-1].split("\t")[1]).strip('\n')
            if cl in class_dict:
                class_dict[cl] = class_dict[cl] + 1
            else:
                class_dict[cl] = 1

        class_frequencies = []
        for cl in sorted(class_dict,key=class_dict.get,reverse=True):
            class_frequencies.append((cl,class_dict[cl]))

        return class_frequencies

    def predict_class(self):
        return self.class_frequencies[0][0]


class PerformanceTester():
    def __init__(self,predictions):
        self.predictions=predictions

    def get_F1_score(self,data_path=None,results=[]):
        if data_path:
            results = PerformanceTester.get_actual_results(data_path)
        score = f1_score(
            y_true=results,
            y_pred=self.predictions,
            average="micro"
        )
        return score


    @staticmethod
    def get_actual_results(data_path):
        results = []
        data = read_data(data_path)
        for line in data:
            temp = line.split(" ")
            cl = (temp[-1].split("\t")[1]).strip('\n')
            results.append(cl)
        return results
