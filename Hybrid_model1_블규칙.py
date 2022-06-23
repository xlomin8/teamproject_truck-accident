import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score

# file = r"D:/work/python/a-truck-accident/data/Newdata3_truck.xlsx"
# df = pd.read_excel(file, sheet_name="화물더미 (2)")
# 한글 깨짐 방지
import matplotlib
import matplotlib.font_manager as fm
# fm._rebuild()
fm.get_fontconfig_fonts()
font_location = 'C:/Users/ASIA-19/NanumGothic.ttf' # 폰트 파일 이름, 디렉토리 주의
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)

file = r"D:/work/python/a-truck-accident/data/모델변수.xlsx"
df = pd.read_excel(file, sheet_name="Sheet1")

print(df.columns)

# # 기술 속성(descriptive features)
X = df.drop(['사망', '부상자수', '교통사고비용'], axis=1)
# # 대상 속성(target feature)
Y = df['사망']

# train_data, test_data, train_label, test_label \
#     = train_test_split(df, label, test_size=0.3, random_state=0 )
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=10)
# # 1.의사결정나무
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=5, random_state=0) # 기본 지니계수
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

print(df.head())

import lightgbm as lgb
# ▶ Hyper parametre setting

d_train = lgb.Dataset (X_train, label = Y_train)

params = {}
params [ 'learning_rate'] = 0.02
params [ 'boosting_type'] = 'gbdt' # GradientBoostingDecisionTree
params ['objective'] = 'binary'
params [ 'metric' ] = 'binary_logloss' # metric for binary-class
params [ 'max_depth'] = 2
params [ 'num_leaves' ] = 4 # 최대 leaves는 2^(max_depth)
params ['seed'] = 23456
# ▶ 학습
clf = lgb.train (params, d_train, 500) # 1000 epocs에서 모델 훈련

from sklearn.metrics import classification_report

Y_pred_train = clf.predict(X_train)
for i in range(0,len(Y_pred_train)):
    if Y_pred_train[i]>=.3:       # setting threshold to .5
       Y_pred_train[i]=1
    else:
       Y_pred_train[i]=0

Y_pred_test = clf.predict(X_test)
for i in range(0,len(Y_pred_test)):
    if Y_pred_test[i]>=.3:       # setting threshold to .5
       Y_pred_test[i]=1
    else:
       Y_pred_test[i]=0

print(classification_report(Y_train, Y_pred_train))
print(classification_report(Y_test, Y_pred_test))

print(pd.Series(Y_pred_test).value_counts())

# 과적합 문제, Train과 Test set에 성능을 최대한 줄여주는 것이 과적합을 방지
from sklearn.metrics import roc_auc_score

Y_pred_train_proba = clf.predict(X_train)
Y_pred_test_proba = clf.predict(X_test)


roc_score_train = roc_auc_score(Y_train, Y_pred_train_proba)
roc_score_test = roc_auc_score(Y_test, Y_pred_test_proba)

print("roc_score_train :", roc_score_train)
print("roc_score_test :", roc_score_test)

from sklearn.metrics import roc_curve

def roc_curve_plot(Y_test, pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음.
    # FPR : 암환자가 아닌 환자를 암환자라고 잘 못 예측한 비율
    # TPR : Recall
    fprs, tprs, thresholds = roc_curve(Y_test, pred_proba_c1)

    # ROC Curve를 plot 곡선으로 그림.
    plt.plot(fprs, tprs, label='ROC')
    # 가운데 대각선 직선을 그림.
    plt.plot([0, 1], [0, 1], 'k--', label='Random', color='red')

    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('FPR( 1 - SensitivitY )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()

roc_curve_plot(Y_train, Y_pred_train_proba)

roc_curve_plot(Y_test, Y_pred_test_proba)

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
plt.style.use(['dark_background'])

ftr_importances_values = clf.feature_importance() # Randomforest : feature_importance_
ftr_importances = pd.Series(ftr_importances_values, index = X.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.title('Feature Importances')
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()

# model = df.fit(X_train, Y_test)
# print("의사결정나무 train :", model.score(X_train, Y_train))
# print("의사결정나무 test :", model.score(X_test, Y_test))

# # 2.랜덤포레스트
# from sklearn.ensemble import RandomForestClassifier
# # rf = RandomForestClassifier(n_estimators=10, random_state=0)
# rf = RandomForestClassifier(max_depth=5, min_samples_leaf=10, min_samples_split=10,
#                        n_estimators=10, random_state=0)
# model = rf.fit(train_data, train_label)
# print("랜덤포레스트 train :", model.score(train_data, train_label))
# print("랜덤포레스트 test :", model.score(test_data, test_label))

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# # 1. Train and predict with first model
# model_1.fit(X_train_1, Y_train)
# Y_pred_1 = model_1.predict(X_train)
#
# # 2. Train and predict with second model on residuals
# model_2.fit(X_train_2, Y_train - y_pred_1)
# y_pred_2 = model_2.predict(X_train_2)
#
# # 3. Add to get overall predictions
# y_pred = y_pred_1 + y_pred_2



# # 분석 모형 찾기
# train_data, test_data, train_label, test_label \
#     = train_test_split(df, label, test_size=0.3, random_state=0 )
#
# # 1.로지스틱회귀
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()
# model = lr.fit(train_data, train_label)
# print("로지스틱회귀 train :", model.score(train_data, train_label))
# print("로지스틱회귀 test :", model.score(test_data, test_label))
#
# # 2.나이브베이지안
# from sklearn.naive_bayes import GaussianNB
# nb = GaussianNB()
# model = nb.fit(train_data, train_label)
# print("나이브베이지안 train :", model.score(train_data, train_label))
# print("나이브베이지안 test :", model.score(test_data, test_label))
#
# # 3.의사결정나무
# from sklearn.tree import DecisionTreeClassifier
# dt = DecisionTreeClassifier(max_depth=5, random_state=0) # 기본 지니계수
# model = dt.fit(train_data, train_label)
# print("의사결정나무 train :", model.score(train_data, train_label))
# print("의사결정나무 test :", model.score(test_data, test_label))
#
# # 4.랜덤포레스트
# from sklearn.ensemble import RandomForestClassifier
# # rf = RandomForestClassifier(n_estimators=10, random_state=0)
# rf = RandomForestClassifier(max_depth=5, min_samples_leaf=10, min_samples_split=10,
#                        n_estimators=10, random_state=0)
# model = rf.fit(train_data, train_label)
# print("랜덤포레스트 train :", model.score(train_data, train_label))
# print("랜덤포레스트 test :", model.score(test_data, test_label))
#

# from sklearn.metrics import confusion_matrix
# predicts = model.predict(test_data)
# result = confusion_matrix(predicts, test_label)
#
# import seaborn as sns
# fig, ax = plt.subplots(figsize=(5,5))
# sns.heatmap(result, annot=True, linewidths=0.5, linecolor='red', fmt='.0f', ax=ax)
# plt.xlabel("predict")
# plt.xlabel("label")
# plt.show()


# # 랜덤 포레스트
# cancer = load_breast_cancer()
# data = cancer.data
# label = cancer.target
# train_data, test_data, train_label, test_label \
#     = train_test_split( data, label, test_size=0.3, random_state=0 )
# dt = DecisionTreeClassifier(max_depth=5, random_state=0) # 기본 지니계수
#
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(n_estimators=20, random_state=0) # 20개 트리 생성
# model = rf.fit(train_data, train_label)
# print("train :", model.score(train_data, train_label))
# print("test :", model.score(test_data, test_label))
#
# # 가장 큰 영향을 미친 요소
# plt.rcParams['figure.figsize'] = (10, 5)
# features = cancer.data.shape[1]
# plt.xlabel("importance")
# # plt.xlabel("features")
# plt.yticks(np.arange(features), cancer.feature_names)
# plt.barh(range(features), model.feature_importances_) # BAR가로차트
# plt.show()


# #최적화 나무수, 가지치기 결정
# params = {"n_estimators" : [5,10,15,20],
#           "max_depth" : [4,5,6],
#           "min_samples_leaf" : [10,12,14],
#           "min_samples_split" : [10,12,14]}
# from sklearn.model_selection import GridSearchCV
# cv = GridSearchCV(RandomForestClassifier(random_state=0), params, n_jobs = 1)
# models = cv.fit(train_data, train_label)
# model = models.best_estimator_
# print(model)
#
# # import pandas as pd
# # result_df = pd.DataFrame(models.cv_results)
# # print(result_df)
#
# # 반영
# rf = RandomForestClassifier(max_depth=5, min_samples_leaf=10, min_samples_split=10,
#                        n_estimators=10, random_state=0)
# model = rf.fit(train_data, train_label)
# predicts = model.predict(test_data)
#
# from sklearn.metrics import classification_report
# result = classification_report(predicts, test_label)
# print(result)