#!/usr/bin/env python
# coding: utf-8


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


# In[2]:


# 한글 깨짐 방지
import matplotlib
import matplotlib.font_manager as fm
# fm._rebuild()
fm.get_fontconfig_fonts()
font_location = 'C:/Users/ASIA-19/NanumGothic.ttf' # 폰트 파일 이름, 디렉토리 주의
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)


file = r"D:/work/python/a-truck-accident/data/cleandata_final.xlsx"
df = pd.read_excel(file, sheet_name="Sheet1")



print(df.columns)



# # 기술 속성(descriptive features)
X = df.drop(['사망', '부상자수', '교통사고비용'], axis=1)
# # 대상 속성(target feature)
Y = df['사망']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# LightGBM
# 장점)
# - 학습하는데 걸리는 시간이 적다, 빠른 속도
# - 메모리 사용량이 상대적으로 적은편이다
# - categorical feature들의 자동 변환과 최적 분할
# - GPU 학습 지원
# 단점)
# - 작은 dataset을 사용할 경우 과적합 가능성이 크다 (일반적으로 10,000개 이하의 데이터를 적다고 한다)
# 출처: https://mac-user-guide.tistory.com/79 [🌷나의 선인장🌵:티스토리]
# !pip install lightgbm


# #데이터를 표준화 시킴
# from sklearn.preprocessing import StandardScaler


# #훈련용 뿐만 아니라 테스트용도 같이 

# ss = StandardScaler()
# ss.fit(X_train)
# train_scaled = ss.transform(X_train)
# test_scaled = ss.transform(X_test)



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

