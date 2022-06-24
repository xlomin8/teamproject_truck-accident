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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score


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


print(df.head())


#결측치 제거
df.isnull().sum()


print(df.columns)

# # 1.의사결정나무
from sklearn.ensemble import RandomForestClassifier

# # 기술 속성(descriptive features)
X = df.drop(['사망', '부상자수', '교통사고비용'], axis=1)
# # 대상 속성(target feature)
Y = df['사망']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20,  stratify=Y)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


#데이터를 표준화 시킴
from sklearn.preprocessing import StandardScaler


#훈련용 뿐만 아니라 테스트용도 같이 

ss = StandardScaler()
ss.fit(X_train)
train_scaled = ss.transform(X_train)
test_scaled = ss.transform(X_test)


# DT 객체 생성 및 훈련
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train,Y_train)


# 예측값 저장
Y_pred = dt_clf.predict(X_test)

# · 모델 학습 및 평가
# 모델을 학습하고 예측을 수행하여 성능을 평가

import sklearn.metrics as mt


# !pip install -U pandas-profiling


# https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=winddori2002&logNo=221659080425
# 학습결과 평가 

print('Train_Accuracy: ', dt_clf.score(X_train, Y_train),'\n')

accuracy = mt.accuracy_score(Y_test, Y_pred)
recall = mt.recall_score(Y_test, Y_pred)
precision = mt.precision_score(Y_test, Y_pred)
f1_score = mt.f1_score(Y_test, Y_pred)
matrix = mt.confusion_matrix(Y_test, Y_pred)

print('Accuracy: ', format(accuracy,'.2f'),'\n')
print('Recall: ', format(recall,'.2f'),'\n')
print('Precision: ', format(precision,'.2f'),'\n')
print('F1_score: ', format(f1_score,'.2f'),'\n')
print('Confusion Matrix:','\n', matrix)


# 학습 데이터에서는 100%의 정확도, 테스트에서는 80%의 정확도

# 교차검증
from sklearn.model_selection import cross_val_score, cross_validate

# 각 폴드의 스코어 
scores = cross_val_score(dt_clf, X, Y, cv = 5)
scores

pd.DataFrame(cross_validate(dt_clf, X, Y, cv =5))
print('교차검증 평균: ', scores.mean())


from sklearn.model_selection import GridSearchCV

# 테스트하고자 하는 파라미터 값들을 사전타입으로 정의

dt_clf = DecisionTreeClassifier(random_state=33)
parameters = {'max_depth': [3, 5, 7],
              'min_samples_split': [3, 5],
              'splitter': ['best', 'random']}

grid_dt = GridSearchCV(dt_clf, # estimator 객체,
                      param_grid = parameters, cv = 5,
                      # n_jobs = -1: 모든 cpu를 사용)
                      )

grid_dt.fit(X_train, Y_train)

result = pd.DataFrame(grid_dt.cv_results_['params'])
result['mean_test_score'] = grid_dt.cv_results_['mean_test_score']
result.sort_values(by='mean_test_score', ascending=False)


# 결과에서 보면 알 수 있듯이 튜닝한 파라미터의 max_depth = 3, min_samples_split = 3, splitter = 'best' 일때 가장 좋은 성능을 보였습니다. mean_test_score는 각 fold 5개의 평균을 의미합니다.
# 튜닝전 79% -> 90%로 성능이 개선된 것을 확인할 수 있습니다. 

# #데이터를 표준화 시킴
# from sklearn.preprocessing import StandardScaler


# #훈련용 뿐만 아니라 테스트용도 같이 

# ss = StandardScaler()
# ss.fit(X_train)
# train_scaled = ss.transform(X_train)
# test_scaled = ss.transform(X_test)


#로지스틱 회귀 훈련 모형을 적용해서 훈련실시 

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_scaled, Y_train)

#훈련용 데이터로 학습한 결과 
print(lr.score(train_scaled, Y_train))

#검증용 데이터 -> 훈련용 모델 -> 결과
print(lr.score(test_scaled, Y_test))


# !pip install pydotplus

#모형의 절편, 기술기

print(lr.coef_, lr.intercept_)


# 모델 생성하기
# 이제 모델을 생성하자. 당연히 학습 데이터를 가지고 모델을 생성한다.

# 방법은 단순선형회귀와 똑같다.


#결정트리를 만들어서 모형을 만든다 : 스무고개 하는 방식

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)

#여기서도 random_statef를 사용했는데 그 이유는? 

dt.fit(train_scaled, Y_train)

print(dt.score(train_scaled, Y_train))
print(dt.score(test_scaled, Y_test))


#분류 과정을 시각화해서 보여준다.
#시간이 좀 걸립니다.

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()


# 가지치기

#max_depth=3 (매개변수를 3로 준다 : 로트노투 하나를 제외하고 3개 더 그려줌]

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, Y_train)

print(dt.score(train_scaled, Y_train))
print(dt.score(test_scaled, Y_test))



dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, Y_train)

print(dt.score(X_train, Y_train))
print(dt.score(X_test, Y_test))


#어떤 요인이 크게 작용했는지 볼 수 있음 
print(dt.feature_importances_)


df


dt = DecisionTreeClassifier(min_impurity_decrease=0.0005, random_state=42)
dt.fit(X_train, Y_train)

print(dt.score(X_train, Y_train))
print(dt.score(X_test, Y_test))


import pickle

with open('D:/work/python/a-truck-accident/decisionTree.pickle', 'wb') as f: #모델 저장해두기(피클)
    pickle.dump(dt, f)

