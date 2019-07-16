import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

train = pd.read_csv("db1/TeamDataTrain_2019.csv")
pred = pd.read_csv("db1/TeamDataPred_2019.csv")

class RFECVFeatureSelect:
  CV = None

  def __init__(self, rfecv_folds = None,
               rfecv_step = 1,
               rfecv_scoring = 'neg_log_loss',
               rfecv_min_feat = 4,
               logit_C = 1000,
               logit_iter = 500,
               logit_solver = 'liblinear',
               logit_penalty = 'l1'):
    self.features_ = []
    if rfecv_folds:
      self.CV = rfecv_folds
    else:
      rfecv_folds = 5

    self.rfecv_step = rfecv_step
    self.rfecv_scoring = rfecv_scoring
    self.rfecv_min_feat = rfecv_min_feat
    self.cv_scores_ = {}
    self.logit_ = LogisticRegression(C = logit_C,
                                     max_iter = logit_iter,
                                     solver = logit_solver,
                                     fit_intercept = False,
                                     penalty = logit_penalty,
                                     multi_class='ovr')

    self.clf_ = RFECV(self.logit_,
                      cv = KFold(rfecv_folds),
                      step = self.rfecv_step,
                      scoring = self.rfecv_scoring,
                      min_features_to_select = self.rfecv_min_feat)

  def fit(self, X, y):
    if not self.CV:
      for i in range(2, 11):
        self.cv_scores_[max(abs(cross_val_score(self.logit_,
                                                X,
                                                y,
                                                cv = KFold(i),
                                                scoring = 'neg_log_loss',
                                                n_jobs = -1)))] = i

        self.CV = self.cv_scores_[min(self.cv_scores_.keys())]
    print("Recursive Feature Select initialized with %d folds." % self.CV)
    self.clf_.cv = self.CV
    self.clf_.fit(X, y)

  def transform(self, X):
    self.features_ = X.columns[self.clf_.support_]
    dat = X.copy()
    dat = dat[self.features_]
    print("Recursive Feature Elimination chose "
          "%d features: %s." % (len(self.features_),
                                ',\n'.join(self.features_)))
    return dat

  def fit_transform(self, X, y):
    self.fit(X, y)
    return self.transform(X)
