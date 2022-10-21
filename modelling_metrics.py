from sklearn.feature_extraction import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import pandas as pd



def LogisticRegressionClassifier( X_train_text, training_labels, X_test_text, testing_labels):
    print(len(X_train_text), len(X_test_text))
    print(len(training_labels), len(testing_labels))
    training_labels = pd.DataFrame({'values': training_labels})
    testing_labels = pd.DataFrame({'values': testing_labels})
    map_lables = {'positive': 0, 'negative': 1}
    training_labels = training_labels['values'].map(map_lables)
    testing_labels = testing_labels['values'].map(map_lables)
    logreg = LogisticRegression(C=C, solver=solver, multi_class='multinomial', random_state=17, n_jobs=4)
    logreg.fit(X_train_text, training_labels)
    X_train_predict = logreg.predict(X_train_text)
    X_test_predict = logreg.predict(X_test_text)
    print(X_train_predict[:5])
    return X_train_predict, X_test_predict, training_labels, testing_labels

def compute_metrics(X_train_predict, X_test_predict, trainY, testY, metrics):
        training_acc_score = accuracy_score(X_train_predict, trainY)
        testing_acc_score = accuracy_score(X_test_predict, testY)

        training_precision_score = precision_score(X_train_predict, trainY)
        testing_precision_score = precision_score(X_test_predict, testY)

        training_recall_score = recall_score(X_train_predict, trainY)
        testing_recall_score = recall_score(X_test_predict, testY)

        training_f1_score = f1_score(X_train_predict, trainY)
        testing_f1_score = f1_score(X_test_predict, testY)

        if metrics == 'all':
            return (training_acc_score, 
                    testing_acc_score,
                    training_precision_score,
                    testing_precision_score,
                    training_recall_score,
                    testing_recall_score,
                    training_f1_score,
                    testing_f1_score)
        elif metrics == 'acc':
            print('*************************************')
            print(training_acc_score, testing_acc_score)
            print('****************************************')
            return (training_acc_score, 
                    testing_acc_score)
        elif metrics == 'precision':
            print('*************************************')
            print(training_acc_score, testing_acc_score)
            print('****************************************')
            return (training_precision_score, 
                    testing_precision_score)
        elif metrics == 'recall':
            return (training_recall_score, 
                    testing_recall_score)
        elif metrics == 'f1':
            return (training_f1_score, 
                    testing_f1_score)