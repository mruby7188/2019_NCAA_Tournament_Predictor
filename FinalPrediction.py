import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import fisher_exact
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import NCAATournament2019Model as md
import RFECVFeatureSelect as fs

if len(sys.argv) == 3:
  train = pd.read_csv(str(sys.argv[1]))
  pred = pd.read_csv(str(sys.argv[2]))
else:
  train = pd.read_csv("db1/TeamDataTrain_2019.csv")
  pred = pd.read_csv("db1/TeamDataPred_2019.csv")

def plot_results(pred, y_test):
  # Print model evaluation information.
  fpr, tpr, thresholds = metrics.roc_curve(np.array(y_test, dtype = 'int64'),
                                           np.array(pred))
  roc_auc = metrics.auc(fpr, tpr)

  print(metrics.classification_report(pred, y_test))

  print('accuracy =', metrics.accuracy_score(pred, y_test))

  print('balanced_accuracy =', metrics.balanced_accuracy_score(pred, y_test))

  print('precision =', metrics.precision_score(pred, y_test))

  print('average_precision =', metrics.average_precision_score(pred, y_test))

  print('f1_score =', metrics.f1_score(pred, y_test))

  print('recall =', metrics.recall_score(pred, y_test))

  print('ROC =', roc_auc)

  conf = confusion_matrix(y_test, pred)
  print("\nConfusion Matrix:\n", conf)

  odds, p = fisher_exact(conf)
  print('\nFisher Exact Score:')
  print(p)
  print(odds)

  plt.figure()
  plt.plot(fpr, tpr, label = 'GridSearchCV (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], 'r--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic')
  plt.legend(loc = "lower right")
  plt.show()


def plot_learning_curve(estimator, title, X, y, ylim = None, cv = None,
                        n_jobs = None, train_sizes = np.linspace(.1, 1.0, 5)):
  """
  Generate a simple plot of the test and training learning curve.

  Parameters
  ----------
  estimator : object type that implements the "fit" and "predict" methods
      An object of that type which is cloned for each validation.

  title : string
      Title for the chart.

  X : array-like, shape (n_samples, n_features)
      Training vector, where n_samples is the number of samples and
      n_features is the number of features.

  y : array-like, shape (n_samples) or (n_samples, n_features), optional
      Target relative to X for classification or regression;
      None for unsupervised learning.

  ylim : tuple, shape (ymin, ymax), optional
      Defines minimum and maximum yvalues plotted.

  cv : int, cross-validation generator or an iterable, optional
      Determines the cross-validation splitting strategy.
      Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

      For integer/None inputs, if ``y`` is binary or multiclass,
      :class:`StratifiedKFold` used. If the estimator is not a classifier
      or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

      Refer :ref:`User Guide <cross_validation>` for the various
      cross-validators that can be used here.

  n_jobs : int or None, optional (default=None)
      Number of jobs to run in parallel.
      ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
      ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
      for more details.

  train_sizes : array-like, shape (n_ticks,), dtype float or int
      Relative or absolute numbers of training examples that will be used to
      generate the learning curve. If the dtype is float, it is regarded as a
      fraction of the maximum size of the training set (that is determined
      by the selected validation method), i.e. it has to be within (0, 1].
      Otherwise it is interpreted as absolute sizes of the training sets.
      Note that for classification the number of samples usually have to
      be big enough to contain at least one sample from each class.
      (default: np.linspace(0.1, 1.0, 5))
  """
  plt.figure()
  plt.title(title)
  if ylim is not None:
    plt.ylim(*ylim)
  plt.xlabel("Training examples")
  plt.ylabel("Score")
  train_sizes, train_scores, test_scores = learning_curve(
    estimator, X, y, cv = cv, n_jobs = n_jobs, train_sizes = train_sizes)
  train_scores_mean = np.mean(train_scores, axis = 1)
  train_scores_std = np.std(train_scores, axis = 1)
  test_scores_mean = np.mean(test_scores, axis = 1)
  test_scores_std = np.std(test_scores, axis = 1)
  plt.grid()

  plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                   train_scores_mean + train_scores_std, alpha = 0.1,
                   color = "r")
  plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                   test_scores_mean + test_scores_std, alpha = 0.1, color = "g")
  plt.plot(train_sizes, train_scores_mean, 'o-', color = "r",
           label = "Training score")
  plt.plot(train_sizes, test_scores_mean, 'o-', color = "g",
           label = "Cross-validation score")

  plt.legend(loc = "best")
  return plt

y = train['Result']
X = train.copy()
X_Valid = pred.copy()
X.drop(['Result', 'Season', 'T1_TeamID', 'T2_TeamID', 'T1_Seed', 'T2_Seed',
        'T1_kenpom_Rank', 'T2_kenpom_Rank', 'T1_Avg_Rank', 'T2_Avg_Rank'],
       axis = 1, inplace = True)

GameID = pred['GameID']
X_Valid.drop(['GameID', 'Season', 'T1_TeamID', 'T2_TeamID', 'T1_Seed',
              'T2_Seed', 'T1_kenpom_Rank', 'T2_kenpom_Rank', 'T1_Avg_Rank',
              'T2_Avg_Rank'],
             axis = 1, inplace = True)

# Select Features and standardize Features
rfe = fs.RFECVFeatureSelect()
scaler = StandardScaler()

scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), columns=X.columns)
X_Valid = pd.DataFrame(scaler.transform(X_Valid), columns=X.columns)

X = rfe.fit_transform(X, y)

features = X.columns

X_Valid = X_Valid[features]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=453)

sm = SMOTE(random_state = 536)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
X_res, y_res = sm.fit_resample(X, y)

# Fit Logistic Classifier
clf = md.NCAATournament2019Model(folds = 5)
clf.fit(X_res, y_res)

model1 = clf.model_.best_estimator_
model1.fit(X_train_res, y_train_res)
preds = model1.predict(X_test)

plot_results(preds, y_test)
model1.fit(X_res, y_res)
probs = model1.predict_proba(X_Valid)[:, 1]
out1 = pd.DataFrame({"ID": GameID, 'Pred': probs}).sort_values(by=['ID'])
out1.to_csv('logit.csv', index=False)

# # Fit Random Forest Classifier
# rfc = RandomForestClassifier(oob_score = 'neg_log_loss', bootstrap = True)
#
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf}
#
#
# rff = md.NCAATournament2019Model(rfc, random_grid, folds = 5)
# rff.fit(X_res, y_res)
#
# model2 = rff.model_.best_estimator_
# model2.fit(X_train_res, y_train_res)
# preds2 = model2.predict(X_test)
#
# plot_results(preds2, y_test)
# model2.fit(X_res, y_res)
# probs2 = model2.predict_proba(X_Valid)[:, 1]
# out2 = pd.DataFrame({"ID": GameID, 'Pred': probs2}).sort_values(by=['ID'])
# out2.to_csv('rfc.csv', index=False)
#
# # Support Vector Classifier
# svc_C = np.logspace(start=-1, stop=3, num= 15)
# svc_kernel = ['poly', 'rbf', 'sigmoid']
# svc_degree = np.linspace(1, 4, 1)
# svc_gamma = ['auto', 'scale']
# svc_shrinking = [True, False]
# svc_max_iter = [100, 200, 300, 400, 500, 1000]
# sv = SVC(probability = True,
#          decision_function_shape = 'ovr',
#          random_state = 32)
#
# sv_grid = {'C': svc_C,
#            'kernel': svc_kernel,
#            'gamma': svc_gamma,
#            'degree': svc_degree,
#            'shrinking': svc_shrinking,
#            'max_iter': svc_max_iter}
#
# svc = md.NCAATournament2019Model(sv, sv_grid, folds = 5)
# svc.fit(X_res, y_res)
# model3 = svc.model_.best_estimator_
#
# model3.fit(X_train_res, y_train_res)
# preds3 = model3.predict(X_test)
# plot_results(preds3, y_test)
#
# model3.fit(X_res, y_res)
# probs3 = model3.predict_proba(X_Valid)[:, 1]
# out3 = pd.DataFrame({"ID": GameID, 'Pred': probs3}).sort_values(by=['ID'])
# out3.to_csv('svc.csv', index=False)
#
# # Fit ABABoost
#
# ada = AdaBoostClassifier(base_estimator = model1, random_state=337)
# ada_params = {'learning_rate': np.logspace(-2, 0, 8),
#               'n_estimators': range(8, 81, 8)}
# adac = md.NCAATournament2019Model(ada, ada_params, folds = 5)
#
# adac.fit(X_res, y_res)
#
# model4 = adac.model_.best_estimator_
#
# model4.fit(X_train_res, y_train_res)
# preds4 = model4.predict(X_test)
# plot_results(preds4, y_test)
#
# model4.fit(X_res, y_res)
# probs4 = model4.predict_proba(X_Valid)[:, 1]
# out4 = pd.DataFrame({"ID": GameID, 'Pred': probs4}).sort_values(by=['ID'])
# out4.to_csv('adac.csv', index=False)
#
# # Fit Gradient Boosting
# gb = GradientBoostingClassifier(loss='deviance', random_state=235)
# gb_grid = {'learning_rate': np.logspace(-2, 0, 10),
#            'n_estimators': range(8, 81, 8),
#            'subsample': np.logspace(-0.2, 0, 8),
#            'max_depth': [3, 4, 5, 6]}
#
#
# gbc = md.NCAATournament2019Model(gb, gb_grid, folds = 5)
# gbc.fit(X_res, y_res)
# model5 = gbc.model_.best_estimator_
#
# model5.fit(X_train_res, y_train_res)
# preds5 = model5.predict(X_test)
# plot_results(preds4, y_test)
#
# model5.fit(X_res, y_res)
# probs5 = model5.predict_proba(X_Valid)[:, 1]
# out5 = pd.DataFrame({"ID": GameID, 'Pred': probs5}).sort_values(by=['ID'])
# out5.to_csv('gbc.csv', index=False)
#
#
# plt.figure(figsize = (60, 40))
# plt.subplot(3, 2, 1)
# title = r"Learning Curves (Logistic Regression)"
# # Cross validation with 100 iterations to get smoother mean test and train
# # score curves, each time with 25% data randomly selected as a validation set.
# cv = ShuffleSplit(n_splits = 100, test_size = 0.20, random_state = 0)
#
# plot_learning_curve(model1, title, X, y, ylim = (0, 1.01),
#                     cv = rfe.CV, n_jobs = 4)
# plt.show()
#
# plt.subplot(3, 2, 2)
# title = r"Learning Curves (RandomForestClassifier)"
# plot_learning_curve(model2, title, X, y, (0, 1.01), cv = rfe.CV, n_jobs = 4)
# plt.show()
#
# plt.subplot(3, 2, 3)
# title = r"Learning Curves (SVM)"
# # SVC is more expensive so we do a lower number of CV iterations:
# plot_learning_curve(model3, title, X, y, (0, 1.01), cv = rfe.CV, n_jobs = 4)
# plt.show()
#
# plt.subplot(3, 2, 4)
# title = r"Learning Curves (ADABoost)"
# plot_learning_curve(model4, title, X, y, (0, 1.01), cv = rfe.CV, n_jobs = 4)
# plt.show()
#
# plt.subplot(3, 2, 5)
# title = r"Learning Curves (GradientBoost)"
# plot_learning_curve(model5, title, X, y, (0, 1.01), cv = rfe.CV, n_jobs = 4)
#
# plt.show()
