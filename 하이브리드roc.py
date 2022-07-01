#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn. preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

# 한글 깨짐 방지
import matplotlib
import matplotlib.font_manager as fm
# fm._rebuild()
fm.get_fontconfig_fonts()
font_location = './NanumGothic.ttf' # 폰트 파일 이름, 디렉토리 주의
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)

from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd

from sklearn.preprocessing import StandardScaler
from collections import Counter
import seaborn as sns



#데이터 로드
file = r"./clearfile_f.xlsx"
df = pd.read_excel(file, sheet_name="ver4")


# 결측치 확인
print(df.isnull().sum())


# 독립변수(특징), 종속변수(라벨) 나누기
X = df.drop(['사망', '부상자수', '교통사고비용','버스평균일교통량','요일_토', '요일_일','법규위반_중앙',
                   '법규위반_차로위반','법규위반_기타', '법규위반_불법유턴', '법규위반_신호위반'], axis=1)
Y = df['사망']


# 훈련셋, 테스트셋 나누기
from sklearn.model_selection import train_test_split

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print('X_train shape:', X_train.shape)  #(2088, 39)
print('X_test shape:', X_test.shape)    #(523, 39)
print('Y_train shape:', Y_train.shape)  #(2088,)
print('Y_test shape:', Y_test.shape)    #(523,)


# 데이터 불균형 확인
print(Y_train.value_counts())
# 0 1919
# 1 171

# 불균형 비율 계산
print(Y_train.value_counts().iloc[0] / Y_train.value_counts().iloc[-1])
# 11.2222222

# 오버샘플링
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN

print('Before OverSampling, counts of label "1": {}'.format(sum(Y_train==1)))
print('Before OverSampling, counts of label "0": {} \n'.format(sum(Y_train==0)))

## SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(k_neighbors=3)
X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train)
X_train_res = pd.DataFrame(X_train_res, columns=X.columns)
Y_train_res = pd.Series(Y_train_res)

print('After OverSampling, X_train shape: {}'.format(X_train_res.shape))
print('After OverSampling, Y_train shape: {} \n'.format(Y_train_res.shape))

print('After OverSampling, counts of label "1": {}'.format(sum(Y_train_res==1)))
print('After OverSampling, counts of label "0": {} \n'.format(sum(Y_train_res==0)))

print('Oversampled dataset shape %s' % Counter(Y_train))
print('Oversampled dataset shape %s' % Counter(Y_train_res))


# 모델링
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

## 개별 분류 모델은 의사결정나무와 KNN 사용
classifiers = {}
classifier1 = KNeighborsClassifier(n_neighbors=8)
classifier2 = DecisionTreeClassifier(max_depth=10)

## 개별 모델을 튜플로 입력, 보팅 방식은 soft로 지정
model = VotingClassifier(estimators=[('KNN', classifier1), ('DT', classifier2)], voting='soft')


# 모델 학습/평가
import sklearn.metrics as mt

model = model.fit(X_train_res, Y_train_res)
print('Model Train Accuracy : ', model.score(X_train_res, Y_train_res), '\n')


## 개별 모델 학습/예측/평가
# estimators = [knn_clf, tree_clf]
#
# for estimator in estimators:
#     estimator.fit(X_train_res, Y_train_res)
#     pred = estimator.predict(X_test)
#     class_name = estimator.__class__.__name__
#     print('{} 분류기 정확도: {}'.format(class_name, accuracy_score(Y_test, pred)))

# https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=winddori2002&logNo=221659080425
# 학습결과 평가


# 교차검증
## 훈련셋을 찐훈련셋(sub_input), 검증셋(val_input)으로 나누기
sub_train, sub_test, val_train, val_test = train_test_split(X_train_res, Y_train_res, test_size=0.2, random_state=42)
print('sub_train shape:', sub_train.shape)  #(3059, 39)
print('sub_test shape:', sub_test.shape)  #(765, 39)
print('val_train shape:', val_train.shape)    #(3059,)
print('val_test shape:', val_test.shape)    #(765,)

print()

## 모델 학습/평가
model.fit(sub_train, val_train)
pred = model.predict(sub_test)
print('Voting Classifier Accuracy : ',accuracy_score(val_test, pred))   #0.8601307189542484
print(model.score(sub_train, val_train))   #0.946060804184374
print(model.score(sub_test, val_test)) #0.8601307189542484

print()

## 교차검증 (훈련셋 섞지 X)
from sklearn.model_selection import cross_validate

scores = cross_validate(model, X_train_res, Y_train_res)
print(scores)
# 'test_score': array([0.89712919, 0.86363636, 0.88995215, 0.89208633, 0.88968825])
print(np.mean(scores['test_score']))    #정확도의 평균 : 0.8864984567370027

print()

## 5폴드 교차검증 (훈련셋 섞음)
from sklearn.model_selection import StratifiedKFold

scores = cross_validate(model, X_train_res, Y_train_res, cv=StratifiedKFold())
print(np.mean(scores['test_score']))    #0.8874576893509115

print()

## 10폴드 교차검증
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(model, X_train_res, Y_train_res, cv=splitter)
print(np.mean(scores['test_score']))    #0.8922532226976336


# 테스트셋 예측
pred = model.predict(X_test)

accuracy = mt.accuracy_score(Y_test, pred)
recall = mt.recall_score(Y_test, pred)
precision = mt.precision_score(Y_test, pred)
f1_score = mt.f1_score(Y_test, pred)
matrix = mt.confusion_matrix(Y_test, pred)

print('Accuracy: ', format(accuracy,'.2f'))
print('Recall: ', format(recall,'.2f'))
print('Precision: ', format(precision,'.2f'))
print('F1_score: ', format(f1_score,'.2f'))
print('Confusion Matrix:','\n', matrix)

scores = cross_validate(model, X_test, Y_test, cv=splitter)
print(np.mean(scores['test_score']))    #0.896734397677794


# 모델 저장
import pickle

with open('./hybridmodel.pkl', 'wb') as f:
    pickle.dump(model, f)
exit()

# 예측
# ▶ 예측은 학습에 사용된 Data와 Test Data 모두 예측하고 평가함(※ 과적합 여부 판별)
pred = vo_clf.predict(X_train_res, Y_train_res)
print(pred)
exit()

Y_pred_train = vo_clf.predict(X_train_res)
Y_pred_test = vo_clf.predict(X_test)

print(classification_report(Y_train_res, Y_pred_train))
print(classification_report(Y_test, Y_pred_test))

pd.Series(Y_pred_train).value_counts()

# ▶ Q. [Test] Precision = 0.60, threshold = ? ↓(base:0.5), Recall = ?
pd.Series(Y_pred_test).value_counts()

from sklearn.preprocessing import Binarizer

# ▶ threshold를 증가시키면, 진짜 정답일 데이터를 예측할 것이므로 precision 값이 증가 (※ 예측하는 개수 감소)
# ▶ threshold를 감소시키면, 조금 이라도 가능성이 있는 정답을 더 많이 예측하므로 recall 값이 증가 (※ 예측하는 개수 증가)
# Input_threshold = 0.405
Input_threshold = 0.100

pred_proba_test = rfc.predict_proba(X_test)[:, 1].reshape(-1, 1)
custom_predict = Binarizer(threshold=Input_threshold).fit_transform(pred_proba_test)

# ▶ 성능평가 확인
print(classification_report(Y_test, custom_predict))

pd.Series(custom_predict.reshape(-1)).value_counts()

# ▶ 과적합 문제, Train과 Test set에 성능을 최대한 줄여주는 것이 과적합을 방지
from sklearn.metrics import roc_auc_score

Y_pred_train_proba = rfc.predict_proba(X_train_res)[:, 1]
Y_pred_test_proba = rfc.predict_proba(X_test)[:, 1]

roc_score_train = roc_auc_score(Y_train_res, Y_pred_train_proba)
roc_score_test = roc_auc_score(Y_test, Y_pred_test_proba)

print("roc_score_train :", roc_score_train)
print("roc_score_test :", roc_score_test)

from sklearn.metrics import roc_curve


def roc_curve_plot(Y_test, pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음.
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
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()


roc_curve_plot(Y_train_res, Y_pred_train_proba)

roc_curve_plot(Y_test, Y_pred_test_proba)


# In[9]:


# ▶ 모델 학습
from sklearn.metrics import classification_report

rfc = DecisionTreeClassifier(max_depth=10)
rfc.fit(X_train_res, Y_train_res)

# ▶ 예측
# ▶ 예측은 학습에 사용된 Data와 Test Data 모두 예측하고 평가함(※ 과적합 여부 판별)
Y_pred_train = rfc.predict(X_train_res)
Y_pred_test = rfc.predict(X_test)

print(classification_report(Y_train_res, Y_pred_train))
print(classification_report(Y_test, Y_pred_test))

pd.Series(Y_pred_train).value_counts()

# ▶ Q. [Test] Precision = 0.60, threshold = ? ↓(base:0.5), Recall = ?
pd.Series(Y_pred_test).value_counts()

from sklearn.preprocessing import Binarizer

# ▶ threshold를 증가시키면, 진짜 정답일 데이터를 예측할 것이므로 precision 값이 증가 (※ 예측하는 개수 감소)
# ▶ threshold를 감소시키면, 조금 이라도 가능성이 있는 정답을 더 많이 예측하므로 recall 값이 증가 (※ 예측하는 개수 증가)
# Input_threshold = 0.405
Input_threshold = 0.100

pred_proba_test = rfc.predict_proba(X_test)[:, 1].reshape(-1, 1)
custom_predict = Binarizer(threshold=Input_threshold).fit_transform(pred_proba_test)

# ▶ 성능평가 확인
print(classification_report(Y_test, custom_predict))

pd.Series(custom_predict.reshape(-1)).value_counts()

# ▶ 과적합 문제, Train과 Test set에 성능을 최대한 줄여주는 것이 과적합을 방지
from sklearn.metrics import roc_auc_score

Y_pred_train_proba = rfc.predict_proba(X_train_res)[:, 1]
Y_pred_test_proba = rfc.predict_proba(X_test)[:, 1]

roc_score_train = roc_auc_score(Y_train_res, Y_pred_train_proba)
roc_score_test = roc_auc_score(Y_test, Y_pred_test_proba)

print("roc_score_train :", roc_score_train)
print("roc_score_test :", roc_score_test)

from sklearn.metrics import roc_curve


def roc_curve_plot(Y_test, pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음.
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
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()


roc_curve_plot(Y_train_res, Y_pred_train_proba)

roc_curve_plot(Y_test, Y_pred_test_proba)

# 학습결과 평가 

print('Train_Accuracy: ', vo_clf.score(X_train, Y_train),'\n')

accuracy = mt.accuracy_score(Y_test, pred)
recall = mt.recall_score(Y_test, pred)
precision = mt.precision_score(Y_test, pred)
f1_score = mt.f1_score(Y_test, pred)
matrix = mt.confusion_matrix(Y_test, pred)

print('Accuracy: ', format(accuracy,'.2f'),'\n')
print('Recall: ', format(recall,'.2f'),'\n')
print('Precision: ', format(precision,'.2f'),'\n')
print('F1_score: ', format(f1_score,'.2f'),'\n')
print('Confusion Matrix:','\n', matrix)


# In[10]:


# ▶ 모델 학습
from sklearn.metrics import classification_report

rfc = KNeighborsClassifier(n_neighbors=8)
rfc.fit(X_train_res, Y_train_res)

# ▶ 예측
# ▶ 예측은 학습에 사용된 Data와 Test Data 모두 예측하고 평가함(※ 과적합 여부 판별)
Y_pred_train = rfc.predict(X_train_res)
Y_pred_test = rfc.predict(X_test)

print(classification_report(Y_train_res, Y_pred_train))
print(classification_report(Y_test, Y_pred_test))

pd.Series(Y_pred_train).value_counts()

# ▶ Q. [Test] Precision = 0.60, threshold = ? ↓(base:0.5), Recall = ?
pd.Series(Y_pred_test).value_counts()

from sklearn.preprocessing import Binarizer

# ▶ threshold를 증가시키면, 진짜 정답일 데이터를 예측할 것이므로 precision 값이 증가 (※ 예측하는 개수 감소)
# ▶ threshold를 감소시키면, 조금 이라도 가능성이 있는 정답을 더 많이 예측하므로 recall 값이 증가 (※ 예측하는 개수 증가)
# Input_threshold = 0.405
Input_threshold = 0.100

pred_proba_test = rfc.predict_proba(X_test)[:, 1].reshape(-1, 1)
custom_predict = Binarizer(threshold=Input_threshold).fit_transform(pred_proba_test)

# ▶ 성능평가 확인
print(classification_report(Y_test, custom_predict))

pd.Series(custom_predict.reshape(-1)).value_counts()

# ▶ 과적합 문제, Train과 Test set에 성능을 최대한 줄여주는 것이 과적합을 방지
from sklearn.metrics import roc_auc_score

Y_pred_train_proba = rfc.predict_proba(X_train_res)[:, 1]
Y_pred_test_proba = rfc.predict_proba(X_test)[:, 1]

roc_score_train = roc_auc_score(Y_train_res, Y_pred_train_proba)
roc_score_test = roc_auc_score(Y_test, Y_pred_test_proba)

print("roc_score_train :", roc_score_train)
print("roc_score_test :", roc_score_test)

from sklearn.metrics import roc_curve


def roc_curve_plot(Y_test, pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음.
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
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()


roc_curve_plot(Y_train_res, Y_pred_train_proba)

roc_curve_plot(Y_test, Y_pred_test_proba)


# In[11]:


import pickle
pred = pickle.load(open('/content/drive/MyDrive/project01/ensemble3_.pickle', 'rb')).predict(testX)
# pred = apply.predict(testX)


# In[ ]:


# VotingClassifier 학습/예측/평가
vo_clf.fit(X_train_res, Y_train_res)
pred = vo_clf.predict(X_test)
print('Voting Classifier Accuracy : ',accuracy_score(Y_test, pred))

# 개별 모델 학습/예측/평가
estimators = [lf_clf, tree_clf]

for estimator in estimators:
    estimator.fit(X_train_res, Y_train_res)
    pred = estimator.predict(X_test)
    class_name = estimator.__class__.__name__
    print('{} 분류기 정확도: {}'.format(class_name, accuracy_score(Y_test, pred)))


# In[ ]:


# import pickle
# with open('D:\work\python\a-truck-accident/ensemble_2model_잘나옴.pickle', 'wb') as f: #모델 저장해두기(피클)
#     pickle.dump(estimators, f)


# In[ ]:


# 한글 깨짐 방지
import matplotlib
import matplotlib.font_manager as fm
# fm._rebuild()
# fm.get_fontconfig_fonts()
# font_location = './content/drive/MyDrive/NanumGothic.ttf' # 폰트 파일 이름, 디렉토리 주의
# font_name = fm.FontProperties(fname=font_location).get_name()
# matplotlib.rc('font', family=font_name)

import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.style.use(['dark_background'])

ftr_importances_values = estimator.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.title('Feature Importances')
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()


# In[ ]:


sns.distplot(df['고속도로 교통량'])


# In[ ]:


# 교통량 + 차종 별 사망자 수 분포
# sns.barplot(x='요일', y='사망자수', hue='가해운전자 차종', data=df)
sns.barplot(x='고속도로 교통량', y='사망', hue='가해운전자 차종', data=df)


# In[ ]:


#데이터 검증
file = r"C:\Users\ASIA-19\Downloads\clearfile_f.xlsx"
test = pd.read_excel(file, sheet_name="2021_원본_2")


# In[ ]:


df.columns


# In[ ]:


test.columns


# In[ ]:


testX = test.drop(['사망', '부상자수', '교통사고비용','연령별_가해진료비'], axis=1) #법규위반_안전거리미확보
testY = test['사망']

pred = estimator.predict(testX)
pred[:5]

from numpy import random
import pandas as pd

pred_ox = pd.DataFrame(pred)
print(pred_ox)

pred_ox.to_excel("D:\work\python\a-truck-accident\예측1.xlsx")


# In[ ]:





# In[ ]:





# In[ ]:




