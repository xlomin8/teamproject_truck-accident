import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn. preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# 한글 깨짐 방지
import matplotlib
import matplotlib.font_manager as fm
# fm._rebuild()
fm.get_fontconfig_fonts()
font_location = 'D:/work/python/a-truck-accident/NanumBarunGothic.ttf' # 폰트 파일 이름, 디렉토리 주의
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)

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
df = pd.read_excel('./cleandata_final_0624.xlsx')


# 데이터 불균형 확인
pd.value_counts(df['사망']).plot.bar()
plt.title('사망사고 발생')
plt.xlabel('사망 여부')
plt.ylabel('frequency')
plt.show()

# 독립변수(특징), 종속변수(라벨) 나누기
X = df.drop(['사망', '교통사고비용', '부상자수'], axis=1)
Y = df['사망']

# 훈련셋, 검증셋 나누기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,  stratify=Y)
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

# Feature Scaling
sc = StandardScaler()

#Defining the machine learning models
model1 = LogisticRegression()
model2 = DecisionTreeClassifier(max_depth = 3)
model3 = SVC()
model4 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
model5 = GaussianNB()

#Training the machine learning models
model1.fit(X_train_res, Y_train_res)
model2.fit(X_train_res, Y_train_res)
model3.fit(X_train_res, Y_train_res)
model4.fit(X_train_res, Y_train_res)
model5.fit(X_train_res, Y_train_res)

#Making the prediction
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)
y_pred4 = model4.predict(X_test)
y_pred5 = model5.predict(X_test)


#Confusion matrix
cm_LogisticRegression = confusion_matrix(Y_test, y_pred1)
cm_DecisionTree = confusion_matrix(Y_test, y_pred2)
cm_SupportVectorClass = confusion_matrix(Y_test, y_pred3)
cm_KNN = confusion_matrix(Y_test, y_pred4)
cm_NaiveBayes = confusion_matrix(Y_test, y_pred5)

kfold = model_selection.KFold(n_splits=10, random_state = 42)
result1 = model_selection.cross_val_score(model1, X_train, Y_train, cv=kfold)
result2 = model_selection.cross_val_score(model2, X_train, Y_train, cv=kfold)
result3 = model_selection.cross_val_score(model3, X_train, Y_train, cv=kfold)
result4 = model_selection.cross_val_score(model4, X_train, Y_train, cv=kfold)
result5 = model_selection.cross_val_score(model5, X_train, Y_train, cv=kfold)

print('Accuracy of Logistic Regression Model = ',result1.mean())
print('Accuracy of Decision Tree Model = ',result2.mean())
print('Accuracy of Support Vector Machine = ',result3.mean())
print('Accuracy of k-NN Model = ',result4.mean())
print('Accuracy of Naive Bayes Model = ',result5.mean())
#Output:-

