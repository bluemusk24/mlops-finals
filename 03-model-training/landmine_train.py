# import the libraries

import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# import numpy as np


# Best hyperparameters of Support Vector Machine Classifier
SVC_best_params = {
    "C": 239.7017845360123,
    "kernel": "poly",
    "degree": 3,
    "gamma": "scale",
    "coef0": 9.717139430035742,
    "tol": 0.6958712596862648,
    "cache_size": 139,
    "decision_function_shape": "ovo",
}


# data preparation

df = pd.read_csv("Mine_Dataset.csv")

df = df.rename(
    columns={"V": "voltage", "H": "height", "S": "soil_types", "M": "mine_types"}
)

df["mine_types"] = df["mine_types"].replace({1: 0, 2: 1, 3: 2, 4: 3, 5: 4})

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.reset_index(drop=True)

num_variables = ["voltage", "height"]
cat_variable = ["soil_types"]
df[cat_variable] = df[cat_variable].astype(str)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# Training function


def train(df_train, y_train, **SVC_best_params):
    train_dicts = df_train[cat_variable + num_variables].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)

    model_SVC = SVC(**SVC_best_params)
    model_SVC.fit(X_train, y_train)

    return dv, model_SVC


# Prediction function:


def predict(df, dv, model):
    dicts = df[cat_variable + num_variables].to_dict(orient="records")

    X = dv.transform(dicts)
    y_pred = model.predict(X)

    return y_pred


# Evaluation function of auc_score for individual target


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):

    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:

        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # using scikit-learn method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


# applying both train and predict functions

dv, model_SVC = train(
    df_full_train, df_full_train["mine_types"].values, **SVC_best_params
)
y_pred = predict(df_test, dv, model_SVC)

y_test = df_test["mine_types"].values

accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy of model_SVC : {accuracy}")
print("\n")
auc = roc_auc_score_multiclass(y_test, y_pred)
print(f"auc of model_SVC : {auc}")


# #### Save the SVC model as file in a model folder

with open("models/model_SVC.bin", "wb") as f_out:
    pickle.dump((dv, model_SVC), f_out)


print("the model is saved to the folder models")
