import os
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter

#데이터 로드
df = pd.read_excel('./cleandata_final.xlsx')
print(df.head())

#결측치 확인
print(df.isnull().sum())

# 데이터 불균형 확인
pd.value_counts(df['사망']).plot.bar()
plt.title('사망사고 발생')
plt.xlabel('사망 여부')
plt.ylabel('frequency')
plt.show()

# 독립변수(특징), 종속변수(라벨) 나누기
X = df.drop(['사망'], axis=1)
Y = df['사망']

# 훈련셋, 검증셋 나누기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print('X_train shape:', X_train.shape)  #(2090, 25)
print('X_test shape:', X_test.shape)    #(523, 25)
print('Y_train shape:', Y_train.shape)  #(2090,)
print('Y_test shape:', Y_test.shape)    #(523,)

# 데이터 불균형 확인
print(Y_train.value_counts())
# 0 1919
# 1 171

# 불균형 비율 계산
print(Y_train.value_counts().iloc[0] / Y_train.value_counts().iloc[-1])
# 11.2222222

# 오버샘플링
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
