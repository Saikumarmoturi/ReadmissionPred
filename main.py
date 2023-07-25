# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time
import pandas as pd
import numpy as np
import seaborn as sns
import unicorn as unicorn
import uvicorn as uvicorn
from fastapi import FastAPI
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def calc_prevalence(y_actual):
    return (sum(y_actual)/len(y_actual))


from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

app = FastAPI()

@app.get("/")
async def root():
   return {"message": "Hello World"}
def calc_specificity(y_actual, y_pred, thresh):
    # calculates specificity
    return sum((y_pred < thresh) & (y_actual == 0)) / sum(y_actual == 0)


def print_report(y_actual, y_pred, thresh):
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    specificity = calc_specificity(y_actual, y_pred, thresh)
    print('AUC:%.3f' % auc)
    print('accuracy:%.3f' % accuracy)
    print('recall:%.3f' % recall)
    print('precision:%.3f' % precision)
    print('specificity:%.3f' % specificity)
    print('prevalence:%.3f' % calc_prevalence(y_actual))
    print(' ')
    return auc, accuracy, recall, precision, specificity

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

def buildModels(X_train_tf, y_train, X_valid_tf, y_valid,df_test):
    thresh = 0.5
    knn = KNeighborsClassifier(n_neighbors=100)
    knn.fit(X_train_tf, y_train)
    y_train_preds = knn.predict_proba(X_train_tf)[:, 1]
    y_valid_preds = knn.predict_proba(X_valid_tf)[:, 1]

    print('KNN')
    print('Training:')
    knn_train_auc, knn_train_accuracy, knn_train_recall, knn_train_precision, knn_train_specificity = print_report(
        y_train, y_train_preds, thresh)
    print('Validation:')
    knn_valid_auc, knn_valid_accuracy, knn_valid_recall, knn_valid_precision, knn_valid_specificity = print_report(
        y_valid, y_valid_preds, thresh)

    lr = LogisticRegression(random_state=100)
    lr.fit(X_train_tf, y_train)
    y_train_preds = lr.predict_proba(X_train_tf)[:, 1]
    y_valid_preds = lr.predict_proba(X_valid_tf)[:, 1]

    print('Logistic Regression')
    print('Training:')
    lr_train_auc, lr_train_accuracy, lr_train_recall, \
        lr_train_precision, lr_train_specificity = print_report(y_train, y_train_preds, thresh)
    print('Validation:')
    lr_valid_auc, lr_valid_accuracy, lr_valid_recall, \
        lr_valid_precision, lr_valid_specificity = print_report(y_valid, y_valid_preds, thresh)

    # sgdc = SGDClassifier(loss='log', alpha=0.1, random_state=42)
    # sgdc.fit(X_train_tf, y_train)
    # y_train_preds = sgdc.predict_proba(X_train_tf)[:, 1]
    # y_valid_preds = sgdc.predict_proba(X_valid_tf)[:, 1]
    #
    # print('Stochastic Gradient Descend')
    # print('Training:')
    # sgdc_train_auc, sgdc_train_accuracy, sgdc_train_recall, sgdc_train_precision, sgdc_train_specificity = print_report(
    #     y_train, y_train_preds, thresh)
    # print('Validation:')
    # sgdc_valid_auc, sgdc_valid_accuracy, sgdc_valid_recall, sgdc_valid_precision, sgdc_valid_specificity = print_report(
    #     y_valid, y_valid_preds, thresh)
    nb = GaussianNB()
    nb.fit(X_train_tf, y_train)
    y_train_preds = nb.predict_proba(X_train_tf)[:, 1]
    y_valid_preds = nb.predict_proba(X_valid_tf)[:, 1]

    print('Naive Bayes')
    print('Training:')
    nb_train_auc, nb_train_accuracy, nb_train_recall, nb_train_precision, nb_train_specificity = print_report(y_train,
                                                                                                              y_train_preds,
                                                                                                              thresh)
    print('Validation:')
    nb_valid_auc, nb_valid_accuracy, nb_valid_recall, nb_valid_precision, nb_valid_specificity = print_report(y_valid,
                                                                                                              y_valid_preds,
                                                                                                              thresh)

    tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    tree.fit(X_train_tf, y_train)

    y_train_preds = tree.predict_proba(X_train_tf)[:, 1]
    y_valid_preds = tree.predict_proba(X_valid_tf)[:, 1]

    print('Decision Tree')
    print('Training:')
    tree_train_auc, tree_train_accuracy, tree_train_recall, tree_train_precision, tree_train_specificity = print_report(
        y_train, y_train_preds, thresh)
    print('Validation:')
    tree_valid_auc, tree_valid_accuracy, tree_valid_recall, tree_valid_precision, tree_valid_specificity = print_report(
        y_valid, y_valid_preds, thresh)
    rf = RandomForestClassifier(max_depth=6, random_state=42)
    rf.fit(X_train_tf, y_train)
    y_train_preds = rf.predict_proba(X_train_tf)[:, 1]
    y_valid_preds = rf.predict_proba(X_valid_tf)[:, 1]

    print('Random Forest')
    print('Training:')
    rf_train_auc, rf_train_accuracy, rf_train_recall, rf_train_precision, rf_train_specificity = print_report(y_train,
                                                                                                              y_train_preds,
                                                                                                              thresh)
    print('Validation:')
    rf_valid_auc, rf_valid_accuracy, rf_valid_recall, rf_valid_precision, rf_valid_specificity = print_report(y_valid,
                                                                                                              y_valid_preds,
                                                                                                              thresh)
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                     max_depth=3, random_state=42)
    gbc.fit(X_train_tf, y_train)
    y_train_preds = gbc.predict_proba(X_train_tf)[:, 1]
    y_valid_preds = gbc.predict_proba(X_valid_tf)[:, 1]

    print('Gradient Boosting Classifier')
    print('Training:')
    gbc_train_auc, gbc_train_accuracy, gbc_train_recall, gbc_train_precision, gbc_train_specificity = print_report(
        y_train, y_train_preds, thresh)
    print('Validation:')
    gbc_valid_auc, gbc_valid_accuracy, gbc_valid_recall, gbc_valid_precision, gbc_valid_specificity = print_report(
        y_valid, y_valid_preds, thresh)
    # df_results = pd.DataFrame(
    #     {'classifier': ['KNN', 'KNN', 'LR', 'LR',  'SGD', 'NB', 'NB', 'DT', 'DT', 'RF', 'RF', 'GB', 'GB'],
    #      'data_set': ['train', 'valid'] * 7,
    #      'auc': [knn_train_auc, knn_valid_auc, lr_train_auc, lr_valid_auc,  nb_train_auc,
    #              nb_valid_auc, tree_train_auc, tree_valid_auc, rf_train_auc, rf_valid_auc, gbc_valid_auc,
    #              gbc_valid_auc, ],
    #      'accuracy': [knn_train_accuracy, knn_valid_accuracy, lr_train_accuracy, lr_valid_accuracy,  nb_train_accuracy, nb_valid_accuracy, tree_train_accuracy,
    #                   tree_valid_accuracy, rf_train_accuracy, rf_valid_accuracy, gbc_valid_accuracy,
    #                   gbc_valid_accuracy, ],
    #      'recall': [knn_train_recall, knn_valid_recall, lr_train_recall, lr_valid_recall,  nb_train_recall, nb_valid_recall, tree_train_recall, tree_valid_recall,
    #                 rf_train_recall, rf_valid_recall, gbc_valid_recall, gbc_valid_recall, ],
    #      'precision': [knn_train_precision, knn_valid_precision, lr_train_precision, lr_valid_precision,
    #                    nb_train_precision, nb_valid_precision,
    #                    tree_train_precision, tree_valid_precision, rf_train_precision, rf_valid_precision,
    #                    gbc_valid_auc, gbc_valid_precision, ],
    #      'specificity': [knn_train_specificity, knn_valid_specificity, lr_train_specificity, lr_valid_specificity,
    #                       nb_train_specificity, nb_valid_specificity,
    #                      tree_train_specificity, tree_valid_specificity, rf_train_specificity, rf_valid_specificity,
    #                      gbc_valid_specificity, gbc_valid_specificity, ]})
    sns.set(style="darkgrid")
    # ax = sns.barplot(x="classifier", y="auc", hue="data_set", data=df_results)
    # ax.set_xlabel('Classifier', fontsize=15)
    # ax.set_ylabel('AUC', fontsize=15)
    # ax.tick_params(labelsize=15)
    #
    # # Put the legend out of the figure
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=15)
    # plt.show()

    print('rf.get_params() ',rf.get_params())
    n_estimators = range(200, 1000, 200)
    # maximum number of features to use at each split
    max_features = ['auto', 'sqrt']
    # maximum depth of the tree
    max_depth = range(1, 10, 1)
    # minimum number of samples to split a node
    min_samples_split = range(2, 10, 2)
    # criterion for evaluating a split
    criterion = ['gini', 'entropy']

    # random grid

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'criterion': criterion}

    print(random_grid)
    auc_scoring = make_scorer(roc_auc_score)
    # create the randomized search cross-validation
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                                   n_iter = 20, cv = 2, scoring=auc_scoring,
                                   verbose = 1, random_state = 42)
    # fit the random search model (this will take a few minutes)
    t1 = time.time()
    rf_random.fit(X_train_tf, y_train)
    t2 = time.time()
    print(t2 - t1)
    rf_random.best_params_
    y_train_preds = rf.predict_proba(X_train_tf)[:, 1]
    y_valid_preds = rf.predict_proba(X_valid_tf)[:, 1]

    print('Baseline Random Forest')
    rf_train_auc_base = roc_auc_score(y_train, y_train_preds)
    rf_valid_auc_base = roc_auc_score(y_valid, y_valid_preds)

    print('Training AUC:%.3f' % (rf_train_auc_base))
    print('Validation AUC:%.3f' % (rf_valid_auc_base))

    print('Optimized Random Forest')
    y_train_preds_random = rf_random.best_estimator_.predict_proba(X_train_tf)[:, 1]
    y_valid_preds_random = rf_random.best_estimator_.predict_proba(X_valid_tf)[:, 1]

    rf_train_auc = roc_auc_score(y_train, y_train_preds_random)
    rf_valid_auc = roc_auc_score(y_valid, y_valid_preds_random)

    print('Training AUC:%.3f' % (rf_train_auc))
    print('Validation AUC:%.3f' % (rf_valid_auc))
    penalty = ['none', 'l2', 'l1']
    max_iter = range(100, 500, 100)
    alpha = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    random_grid_sgdc = {'penalty': penalty,
                        'max_iter': max_iter,
                        'alpha': alpha}
    # create the randomized search cross-validation
    # sgdc_random = RandomizedSearchCV(estimator=sgdc, param_distributions=random_grid_sgdc,
    #                                  n_iter=20, cv=2, scoring=auc_scoring, verbose=0,
    #                                  random_state=42)
    #
    # t1 = time.time()
    # sgdc_random.fit(X_train_tf, y_train)
    # t2 = time.time()
    # print(t2 - t1)
    # number of trees
    n_estimators = range(100, 500, 100)

    # maximum depth of the tree
    max_depth = range(1, 5, 1)

    # learning rate
    learning_rate = [0.001, 0.01, 0.1]

    # random grid

    random_grid_gbc = {'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'learning_rate': learning_rate}

    # create the randomized search cross-validation
    gbc_random = RandomizedSearchCV(estimator=gbc, param_distributions=random_grid_gbc,
                                    n_iter=20, cv=2, scoring=auc_scoring,
                                    verbose=0, random_state=42)

    t1 = time.time()
    gbc_random.fit(X_train_tf, y_train)
    t2 = time.time()
    print(t2 - t1)
    gbc_random.best_params_
    y_train_preds = gbc.predict_proba(X_train_tf)[:, 1]
    y_valid_preds = gbc.predict_proba(X_valid_tf)[:, 1]

    print('Baseline gbc')
    gbc_train_auc_base = roc_auc_score(y_train, y_train_preds)
    gbc_valid_auc_base = roc_auc_score(y_valid, y_valid_preds)

    print('Training AUC:%.3f' % (gbc_train_auc_base))
    print('Validation AUC:%.3f' % (gbc_valid_auc_base))

    print('Optimized gbc')
    y_train_preds_random = gbc_random.best_estimator_.predict_proba(X_train_tf)[:, 1]
    y_valid_preds_random = gbc_random.best_estimator_.predict_proba(X_valid_tf)[:, 1]
    gbc_train_auc = roc_auc_score(y_train, y_train_preds_random)
    gbc_valid_auc = roc_auc_score(y_valid, y_valid_preds_random)

    print('Training AUC:%.3f' % (gbc_train_auc))
    print('Validation AUC:%.3f' % (gbc_valid_auc))
    pickle.dump(gbc_random.best_estimator_, open('best_classifier.pkl', 'wb'),protocol = 4)
    X_test = df_test[col2use].values
    y_test = df_test['OUTPUT_LABEL'].values

    scaler = pickle.load(open('scaler.sav', 'rb'))
    X_test_tf = scaler.transform(X_test)
    best_model = pickle.load(open('best_classifier.pkl', 'rb'))
    y_train_preds = best_model.predict_proba(X_train_tf)[:, 1]
    y_valid_preds = best_model.predict_proba(X_valid_tf)[:, 1]
    y_test_preds = best_model.predict_proba(X_test_tf)[:, 1]
    thresh = 0.5

    print('Training:')
    train_auc, train_accuracy, train_recall, train_precision, train_specificity = print_report(y_train, y_train_preds,
                                                                                               thresh)
    print('Validation:')
    valid_auc, valid_accuracy, valid_recall, valid_precision, valid_specificity = print_report(y_valid, y_valid_preds,
                                                                                               thresh)
    print('Test:')
    test_auc, test_accuracy, test_recall, test_precision, test_specificity = print_report(y_test, y_test_preds, thresh)

def buildTrainTest(df_data):
    # shuffle the samples
    df_data = df_data.sample(n=len(df_data), random_state=42)
    df_data = df_data.reset_index(drop=True)
    # Save 30% of the data as validation and test data
    df_valid_test = df_data.sample(frac=0.30, random_state=42)
    print('Split size: %.3f' % (len(df_valid_test) / len(df_data)))
    df_test = df_valid_test.sample(frac=0.5, random_state=42)
    df_valid = df_valid_test.drop(df_test.index)
    df_train_all = df_data.drop(df_valid_test.index)
    print('Test prevalence(n = %d):%.3f' % (len(df_test), calc_prevalence(df_test.OUTPUT_LABEL.values)))
    print('Valid prevalence(n = %d):%.3f' % (len(df_valid), calc_prevalence(df_valid.OUTPUT_LABEL.values)))
    print('Train all prevalence(n = %d):%.3f' % (len(df_train_all), calc_prevalence(df_train_all.OUTPUT_LABEL.values)))
    print('all samples (n = %d)' % len(df_data))
    assert len(df_data) == (len(df_test) + len(df_valid) + len(df_train_all)), 'math didnt work'
    # split the training data into positive and negative
    rows_pos = df_train_all.OUTPUT_LABEL == 1
    df_train_pos = df_train_all.loc[rows_pos]
    df_train_neg = df_train_all.loc[~rows_pos]

    # merge the balanced data
    df_train = pd.concat([df_train_pos, df_train_neg.sample(n=len(df_train_pos), random_state=42)], axis=0)

    # shuffle the order of training samples
    df_train = df_train.sample(n=len(df_train), random_state=42).reset_index(drop=True)

    print('Train balanced prevalence(n = %d):%.3f' % (len(df_train), calc_prevalence(df_train.OUTPUT_LABEL.values)))
    df_train_all.to_csv('df_train_all.csv', index=False)
    df_train.to_csv('df_train.csv', index=False)
    df_valid.to_csv('df_valid.csv', index=False)
    df_test.to_csv('df_test.csv', index=False)
    X_train = df_train[col2use].values
    X_train_all = df_train_all[col2use].values
    X_valid = df_valid[col2use].values

    y_train = df_train['OUTPUT_LABEL'].values
    y_valid = df_valid['OUTPUT_LABEL'].values

    print('Training All shapes:', X_train_all.shape)
    print('Training shapes:', X_train.shape, y_train.shape)
    print('Validation shapes:', X_valid.shape, y_valid.shape)


    scaler = StandardScaler()
    scaler.fit(X_train_all)
    scalerfile = 'scaler.sav'
    pickle.dump(scaler, open(scalerfile, 'wb'))
    scaler = pickle.load(open(scalerfile, 'rb'))
    X_train_tf = scaler.transform(X_train)
    X_valid_tf = scaler.transform(X_valid)
    buildModels(X_train_tf, y_train, X_valid_tf, y_valid,df_test)

def testUserVal():


    # Given dictionary
    data = {
        "encounter_id": 2278392,
        "patient_nbr": 8222157,
        "race": "Caucasian",
        "gender": "Female",
        "age": "[0-10)",
        "weight": "?",
        "admission_type_id": 6,
        "discharge_disposition_id": 25,
        "admission_source_id": 1,
        "time_in_hospital": 1,
        "payer_code": "?",
        "medical_specialty": "Pediatrics-Endocrinology",
        "num_lab_procedures": 41,
        "num_procedures": 0,
        "num_medications": 1,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "diag_1": "250.83",
        "diag_2": "?",
        "diag_3": "?",
        "number_diagnoses": 1,
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "metformin": "No",
        "repaglinide": "No",
        "nateglinide": "No",
        "chlorpropamide": "No",
        "glimepiride": "No",
        "acetohexamide": "No",
        "glipizide": "No",
        "glyburide": "No",
        "tolbutamide": "No",
        "pioglitazone": "No",
        "rosiglitazone": "No",
        "acarbose": "No",
        "miglitol": "No",
        "troglitazone": "No",
        "tolazamide": "No",
        "examide": "No",
        "citoglipton": "No",
        "insulin": "No",
        "glyburide-metformin": "No",
        "glipizide-metformin": "No",
        "glimepiride-pioglitazone": "No",
        "metformin-rosiglitazone": "No",
        "metformin-pioglitazone": "No",
        "change": "No",
        "diabetesMed": "No",

    }

    # Convert dictionary to DataFrame
    df = pd.DataFrame([data])
    # if(df.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21])):
    #     print("Patient has already dead or hospice")
    # Display the DataFrame
    print(df)
    for c in list(df.columns):

        # get a list of unique values
        n = df[c].unique()

        # if number of unique values is less than 30, print the values. Otherwise print the number of unique values
        if len(n) < 30:
            print(c)
            print(n)
        else:
            print(c + ': ' + str(len(n)) + ' unique values')
    df = df.replace('?', np.nan)
    cols_num = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
    df[cols_num].isnull().sum()
    print(df[cols_num].isnull().sum())
    # Categorical Features
    cols_cat = ['race', 'gender',
                'max_glu_serum', 'A1Cresult',
                'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
                'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
                'tolazamide', 'insulin',
                'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                'metformin-pioglitazone', 'change', 'diabetesMed', 'payer_code']

    df[cols_cat].isnull().sum()
    df['race'] = df['race'].fillna('UNK')
    df['payer_code'] = df['payer_code'].fillna('UNK')
    df['medical_specialty'] = df['medical_specialty'].fillna('UNK')
    top_10 = ['UNK', 'InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice', 'Cardiology', 'Surgery-General',
              'Nephrology', 'Orthopedics',
              'Orthopedics-Reconstructive', 'Radiologist']

    # make a new column with duplicated data
    df['med_spec'] = df['medical_specialty'].copy()

    # replace all specialties not in top 10 with 'Other' category
    df.loc[~df.med_spec.isin(top_10), 'med_spec'] = 'Other';
    df.groupby('med_spec').size()
    cols_cat_num = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']

    df[cols_cat_num] = df[cols_cat_num].astype('str')
    df_cat = pd.get_dummies(df[cols_cat + cols_cat_num + ['med_spec']])
    df_cat.head()
    df = pd.concat([df, df_cat], axis=1)
    print(df)
    cols_all_cat = list(df_cat.columns)
    age_id = {'[0-10)': 0,
              '[10-20)': 10,
              '[20-30)': 20,
              '[30-40)': 30,
              '[40-50)': 40,
              '[50-60)': 50,
              '[60-70)': 60,
              '[70-80)': 70,
              '[80-90)': 80,
              '[90-100)': 90}
    df['age_group'] = df.age.replace(age_id)
    print(df['age_group'])
    df.weight.notnull().sum()
    df['has_weight'] = df.weight.notnull().astype('int')
    cols_extra = ['age_group', 'has_weight']
    print('Total number of features:', len(cols_num + cols_all_cat + cols_extra))
    print('Numerical Features:', len(cols_num))
    print('Categorical Features:', len(cols_all_cat))
    print('Extra features:', len(cols_extra))
    col2use = cols_num + cols_all_cat + cols_extra
    print('col2use ', col2use)
    df_data = df[col2use]
    df_second_array = pd.DataFrame(df_data)
    cols_original =['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'race_Asian', 'race_Caucasian', 'race_Hispanic', 'race_Other', 'race_UNK', 'gender_Male', 'gender_Unknown/Invalid', 'max_glu_serum_>300', 'max_glu_serum_Norm', 'A1Cresult_>8', 'A1Cresult_Norm', 'metformin_No', 'metformin_Steady', 'metformin_Up', 'repaglinide_No', 'repaglinide_Steady', 'repaglinide_Up', 'nateglinide_No', 'nateglinide_Steady', 'nateglinide_Up', 'chlorpropamide_No', 'chlorpropamide_Steady', 'chlorpropamide_Up', 'glimepiride_No', 'glimepiride_Steady', 'glimepiride_Up', 'acetohexamide_Steady', 'glipizide_No', 'glipizide_Steady', 'glipizide_Up', 'glyburide_No', 'glyburide_Steady', 'glyburide_Up', 'tolbutamide_Steady', 'pioglitazone_No', 'pioglitazone_Steady', 'pioglitazone_Up', 'rosiglitazone_No', 'rosiglitazone_Steady', 'rosiglitazone_Up', 'acarbose_No', 'acarbose_Steady', 'acarbose_Up', 'miglitol_No', 'miglitol_Steady', 'miglitol_Up', 'troglitazone_Steady', 'tolazamide_Steady', 'tolazamide_Up', 'insulin_No', 'insulin_Steady', 'insulin_Up', 'glyburide-metformin_No', 'glyburide-metformin_Steady', 'glyburide-metformin_Up', 'glipizide-metformin_Steady', 'glimepiride-pioglitazone_Steady', 'metformin-rosiglitazone_Steady', 'metformin-pioglitazone_Steady', 'change_No', 'diabetesMed_Yes', 'payer_code_CH', 'payer_code_CM', 'payer_code_CP', 'payer_code_DM', 'payer_code_FR', 'payer_code_HM', 'payer_code_MC', 'payer_code_MD', 'payer_code_MP', 'payer_code_OG', 'payer_code_OT', 'payer_code_PO', 'payer_code_SI', 'payer_code_SP', 'payer_code_UN', 'payer_code_UNK', 'payer_code_WC', 'admission_type_id_2', 'admission_type_id_3', 'admission_type_id_4', 'admission_type_id_5', 'admission_type_id_6', 'admission_type_id_7', 'admission_type_id_8', 'discharge_disposition_id_10', 'discharge_disposition_id_12', 'discharge_disposition_id_15', 'discharge_disposition_id_16', 'discharge_disposition_id_17', 'discharge_disposition_id_18', 'discharge_disposition_id_2', 'discharge_disposition_id_22', 'discharge_disposition_id_23', 'discharge_disposition_id_24', 'discharge_disposition_id_25', 'discharge_disposition_id_27', 'discharge_disposition_id_28', 'discharge_disposition_id_3', 'discharge_disposition_id_4', 'discharge_disposition_id_5', 'discharge_disposition_id_6', 'discharge_disposition_id_7', 'discharge_disposition_id_8', 'discharge_disposition_id_9', 'admission_source_id_10', 'admission_source_id_11', 'admission_source_id_13', 'admission_source_id_14', 'admission_source_id_17', 'admission_source_id_2', 'admission_source_id_20', 'admission_source_id_22', 'admission_source_id_25', 'admission_source_id_3', 'admission_source_id_4', 'admission_source_id_5', 'admission_source_id_6', 'admission_source_id_7', 'admission_source_id_8', 'admission_source_id_9', 'med_spec_Emergency/Trauma', 'med_spec_Family/GeneralPractice', 'med_spec_InternalMedicine', 'med_spec_Nephrology', 'med_spec_Orthopedics', 'med_spec_Orthopedics-Reconstructive', 'med_spec_Other', 'med_spec_Radiologist', 'med_spec_Surgery-General', 'med_spec_UNK', 'age_group', 'has_weight']
    df_result = pd.DataFrame(columns=cols_original)

    additional_columns = list(set(cols_original) - set(col2use))

    for col in additional_columns:
        df_second_array[col] = 0
    df_result = df_second_array[cols_original]
    # df_result =df_second_array.reindex(columns=df_result.columns,fill_value=0 )
    print("df_data_user Data_df_result  ", df_result)
    # df_result[df_second_array.columns] = df_second_array
    # Reindex and fill missing values with 0


    print("df_data_user Data ", df_data)
    print("df_data_user Data_df_result  ", df_result)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    testUserVal();

    df = pd.read_csv('diabetic_data.csv')
    print(len(df))
    # df.info()
    # count the number of rows for each type
    print(df.groupby('readmitted').size())
    df.groupby('discharge_disposition_id').size()
    #If we look at the IDs_mapping.csv we can see that 11,13,14,19,20,21 are related to death or hospice. We should remove these samples from the predictive model.
    df = df.loc[~df.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21])];

    # Lets's define an output variable for our binary classification. Here we will try to predict if a patient is likely to be re-admitted within 30 days of discharge.
    df['OUTPUT_LABEL'] = (df.readmitted == '<30').astype('int')
    print("df.head()")
    print(df.head())
    print('Prevalence:%.3f'%calc_prevalence(df['OUTPUT_LABEL'].values))
    df[list(df.columns)[10:20]].head()
    # for each column
    for c in list(df.columns):

        # get a list of unique values
        n = df[c].unique()

        # if number of unique values is less than 30, print the values. Otherwise print the number of unique values
        if len(n) < 30:
            print(c)
            print(n)
        else:
            print(c + ': ' + str(len(n)) + ' unique values')
    df = df.replace('?', np.nan)
    cols_num = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
    df[cols_num].isnull().sum()
    print(df[cols_num].isnull().sum())
    #Categorical Features
    cols_cat = ['race', 'gender',
                'max_glu_serum', 'A1Cresult',
                'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
                'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
                'tolazamide', 'insulin',
                'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                'metformin-pioglitazone', 'change', 'diabetesMed', 'payer_code']
    df[cols_cat].isnull().sum()
    df['race'] = df['race'].fillna('UNK')
    df['payer_code'] = df['payer_code'].fillna('UNK')
    df['medical_specialty'] = df['medical_specialty'].fillna('UNK');
    print('Number medical specialty:', df.medical_specialty.nunique())
    df.groupby('medical_specialty').size().sort_values(ascending=False);
    top_10 = ['UNK', 'InternalMedicine', 'Emergency/Trauma',  'Family/GeneralPractice', 'Cardiology', 'Surgery-General',
              'Nephrology', 'Orthopedics',
              'Orthopedics-Reconstructive', 'Radiologist']

    # make a new column with duplicated data
    df['med_spec'] = df['medical_specialty'].copy()

    # replace all specialties not in top 10 with 'Other' category
    df.loc[~df.med_spec.isin(top_10), 'med_spec'] = 'Other';
    df.groupby('med_spec').size()
    cols_cat_num = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']

    df[cols_cat_num] = df[cols_cat_num].astype('str')
    df_cat = pd.get_dummies(df[cols_cat + cols_cat_num + ['med_spec']], drop_first=True)
    df_cat.head()
    df = pd.concat([df, df_cat], axis=1)
    print(df)
    cols_all_cat = list(df_cat.columns)
    age_id = {'[0-10)': 0,
              '[10-20)': 10,
              '[20-30)': 20,
              '[30-40)': 30,
              '[40-50)': 40,
              '[50-60)': 50,
              '[60-70)': 60,
              '[70-80)': 70,
              '[80-90)': 80,
              '[90-100)': 90}
    df['age_group'] = df.age.replace(age_id)
    print(df['age_group'])
    df.weight.notnull().sum()
    df['has_weight'] = df.weight.notnull().astype('int')
    cols_extra = ['age_group', 'has_weight']
    print('Total number of features:', len(cols_num + cols_all_cat + cols_extra))
    print('Numerical Features:', len(cols_num))
    print('Categorical Features:', len(cols_all_cat))
    print('Extra features:', len(cols_extra))
    col2use = cols_num + cols_all_cat + cols_extra
    print('col2use ',col2use)
    df_data = df[col2use + ['OUTPUT_LABEL']]
    print("df_data ",df_data)
    buildTrainTest(df_data)





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
