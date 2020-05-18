from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression

from gensim.models import word2vec

import our_metrics

TRAIN_FILE = Path("raw_data/GunViolence/train.tsv")
DEV_FILE = Path("raw_data/GunViolence/dev.tsv")
TEST_FILE = Path("raw_data/GunViolence/test.tsv")

LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# These frames/labels correspond to
# 1) Gun or 2nd Amendment rights
# 2) Gun control/regulation
# 3) Politics
# 4) Mental health
# 5) School or public space safety
# 6) Race/ethnicity
# 7) Public opinion
# 8) Society/culture
# 9) Economic consequences


def load_data_file(data_file):
    """Load newsframing data

    Returns
    -------
    tuple
        First element is a list of strings(headlines)
        If `data_file` has labels, the second element
        will be a list of labels for each headline.
        Otherwise, the second element will be None.
    """
    print("Loading from {} ...".format(data_file.name), end="")
    text_col = "news_title"
    theme1_col = "Q3 Theme1"

    with open(data_file) as f:
        df = pd.read_csv(f, sep="\t")
        X = df[text_col].tolist()
        y = None
        if theme1_col in df.columns:
            y = df[theme1_col].tolist()

        print(
            "loaded {} lines {} labels ... done".format(
                len(X), "with" if y is not None else "without"
            )
        )

        print(len(X))
        print(len(y))
    return (X, y)


def build_naive_bayes():
    """

    Returns
    -------
        Pipeline
        An sklearn Pipeline
    """
    nb_pipeline = None
    ##### Write code here
    nb_pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', ComplementNB())
    ])

    ##### End of your work ######
    return nb_pipeline


def build_logistic_regr():
    """

    Returns
    -------
        Pipeline
        An sklearn Pipeline
    """
    logistic_pipeline = None
    ##### Write code here #######
    logistic_pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', LogisticRegression())
    ])
    ##### End of your work ######
    return logistic_pipeline


def build_own_pipeline() -> Pipeline:
    """

    Returns
    -------
        Pipeline
        An sklearn Pipeline
    """
    clf = svm.LinearSVC(C=2, loss='hinge')
    vect = TfidfVectorizer(ngram_range=(1, 2))

    pipeline = None
    ##### Write code here #######
    pipeline = Pipeline([
        ('vect', vect),
        ('tfidf', TfidfTransformer()),
        ('clf', clf)
    ])
    ##### End of your work ######
    return pipeline


def output_predictions(pipeline):
    """Load test data, predict using given pipeline, and write predictions to file.

    The output must be named "predictions.tsv" and must have the following format.
    Here, the first three examples were predicted to be 7,2,3, and the last were
    predicted to be 6,6, and 2.

    Be sure not to permute the order of the examples in the test file.

        7
        2
        3
        .
        .
        .
        6
        6
        2

    """
    ##### Write code here #######
    X_train, y_train_true = load_data_file(TRAIN_FILE)
    X_dev, y_dev_true = load_data_file(DEV_FILE)
    X_test, y_test_true = load_data_file(TEST_FILE)

    #train pipeline with dev and train file
    pipeline.fit(X=X_train, y=y_train_true)
    pipeline.fit(X=X_dev, y=y_dev_true)

    y_pred_test = pipeline.predict(X=X_test)

    df = pd.DataFrame(y_pred_test)
    with open('predictions.tsv', 'w'):
        df.to_csv('predictions.tsv', sep='\t', index=False, header=False)
    ##### End of your work ######

def main():
    X_train, y_train_true = load_data_file(TRAIN_FILE)
    X_dev, y_dev_true = load_data_file(DEV_FILE)

    bayes_pipeline = build_naive_bayes()
    logistic_pipeline = build_logistic_regr()
    own_pipeline = build_own_pipeline()

    for name, pipeline in (
        ["Naive Bayes", bayes_pipeline,],
        ["Logistic Regression", logistic_pipeline,],
        ["Own Pipeline", own_pipeline,]
    ):
        if pipeline is not None:

            ##### Write code here #######
            pipeline.fit(X=X_train, y=y_train_true)
            y_pred_dev = pipeline.predict(X=X_dev)

            mac_pre = our_metrics.precision(y_dev_true, y_pred_dev, 'macro', LABELS)
            mac_rec = our_metrics.recall(y_dev_true, y_pred_dev, 'macro', LABELS)

            mic_pre = our_metrics.precision(y_dev_true, y_pred_dev, 'micro', LABELS)
            mic_rec = our_metrics.recall(y_dev_true, y_pred_dev, 'micro', LABELS)
            print(name)
            print('macro_precision = ', mac_pre)
            print('macro_recall = ', mac_rec)
            print('micro_precision = ', mic_pre)
            print('micro_recall = ', mic_rec)
            ##### End of your work ######

    output_predictions(own_pipeline)


if __name__ == "__main__":
    main()
