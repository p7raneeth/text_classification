from sklearn.feature_extraction import *
from sklearn.linear_model import LogisticRegression

def LogisticRegressionClassifier(C, solver, X_train_text, training_labels):
    logreg = LogisticRegression(C=C, solver=solver, multi_class='multinomial', random_state=17, n_jobs=4)
    logreg.fit(X_train_text, training_labels)
    X_test_predict = logreg.predict(X_test_vect_avg)