import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

CV = 7
logit_ = LogisticRegression(max_iter = 500,
                            solver = 'liblinear',
                            penalty = 'l1',
                            fit_intercept = False,
                            multi_class='ovr',
                            random_state=358,
                            warm_start = True)

param_grid = {'C': np.logspace(start=-5, stop=3, num= 10),
              'penalty': ['l1', 'l2'],
              'max_iter': [100, 200, 300, 400, 500, 1000]}

scaler = None

class NCAATournament2019Model:
  """Performs feature selection and fits a logistic regression model on the
  given data."""

  def __init__(self, estimator = logit_, param = param_grid, folds = CV):
    """Creates a model object with response as object the object and the 
    remaining variables as the independent variables, excluding any variable 
    names in exclude."""
    global scaler
    scaler = StandardScaler()
    clf = RandomizedSearchCV(estimator = estimator,
                             param_distributions = param,
                             n_iter = 100,
                             cv = KFold(folds),
                             verbose=2,
                             random_state=425,
                             n_jobs = -1,
                             scoring = 'neg_log_loss',
                             refit = True)
    self.model_ = clf
    self.CV = folds

  def get_CV(self):
    return self.CV

  def get_estimator(self):
    return self.model_

  def fit(self, X, y):
    global scaler
    scaler.fit(X)
    dat = scaler.transform(X)
    self.model_.fit(dat, y)

    return self.model_

  def predict(self, test):
    global scaler
    test = scaler.transform(test.copy())
    return self.model_.predict(test)

  def predict_proba(self, test):
    test = scaler.transform(test.copy())
    return self.model_.predict_proba(test)

