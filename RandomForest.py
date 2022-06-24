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



print(df.head())



#결측치 제거
df.isnull().sum()




print(df.columns)


#
# # 1.RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

# # 기술 속성(descriptive features)
X = df.drop(['사망', '부상자수', '교통사고비용'], axis=1)
# # 대상 속성(target feature)
Y = df['사망']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,  stratify=Y)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=10)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)



# #데이터를 표준화 시킴
# from sklearn.preprocessing import StandardScaler


# #훈련용 뿐만 아니라 테스트용도 같이 

# ss = StandardScaler()
# ss.fit(X_train)
# train_scaled = ss.transform(X_train)
# test_scaled = ss.transform(X_test)



# ▶ stratify='Y' 옵션 결과 확인
pd.Series(Y_train).value_counts()



125/(1271+125)


# ▶ stratify='Y' 옵션 결과 확인
pd.Series(Y_test).value_counts()



42/(424+42)


# · 모델 학습 및 평가
# 모델을 학습하고 예측을 수행하여 성능을 평가



# ▶ 모델 학습
from sklearn.metrics import classification_report

rfc = RandomForestClassifier(random_state=123456)
rfc.fit(X_train, Y_train)

# ▶ 예측
# ▶ 예측은 학습에 사용된 Data와 Test Data 모두 예측하고 평가함(※ 과적합 여부 판별)
Y_pred_train = rfc.predict(X_train)
Y_pred_test = rfc.predict(X_test)

print(classification_report(Y_train, Y_pred_train))
print(classification_report(Y_test, Y_pred_test))



pd.Series(Y_pred_train).value_counts()



# ▶ Q. [Test] Precision = 0.60, threshold = ? ↓(base:0.5), Recall = ?
pd.Series(Y_pred_test).value_counts()



from sklearn.preprocessing import Binarizer

# ▶ threshold를 증가시키면, 진짜 정답일 데이터를 예측할 것이므로 precision 값이 증가 (※ 예측하는 개수 감소)
# ▶ threshold를 감소시키면, 조금 이라도 가능성이 있는 정답을 더 많이 예측하므로 recall 값이 증가 (※ 예측하는 개수 증가)
# Input_threshold = 0.405
Input_threshold = 0.100

pred_proba_test = rfc.predict_proba(X_test)[:,1].reshape(-1,1)
custom_predict = Binarizer(threshold=Input_threshold).fit_transform(pred_proba_test)

# ▶ 성능평가 확인
print(classification_report(Y_test, custom_predict))



pd.Series(custom_predict.reshape(-1)).value_counts()




# ▶ 과적합 문제, Train과 Test set에 성능을 최대한 줄여주는 것이 과적합을 방지
from sklearn.metrics import roc_auc_score

Y_pred_train_proba = rfc.predict_proba(X_train)[:, 1] 
Y_pred_test_proba = rfc.predict_proba(X_test)[:, 1] 


roc_score_train = roc_auc_score(Y_train, Y_pred_train_proba)
roc_score_test = roc_auc_score(Y_test, Y_pred_test_proba)

print("roc_score_train :", roc_score_train)
print("roc_score_test :", roc_score_test)



from sklearn.metrics import roc_curve
def roc_curve_plot(Y_test , pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음. 
    fprs , tprs , thresholds = roc_curve(Y_test ,pred_proba_c1)

    # ROC Curve를 plot 곡선으로 그림. 
    plt.plot(fprs , tprs, label='ROC')
    # 가운데 대각선 직선을 그림. 
    plt.plot([0, 1], [0, 1], 'k--', label='Random', color='red')
  
    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()



roc_curve_plot(Y_train, Y_pred_train_proba)


roc_curve_plot(Y_test, Y_pred_test_proba)


# · 중요 변수 파악
# Feature IMP 분석을 통한 중요변수 파악


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use(['dark_background'])

ftr_importances_values = rfc.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.title('Feature Importances')
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()


# · 모델 Save and Read
# 최종 모델을 저장 및 다시 불러오기


# import pickle
# # 모델 저장
# saved_model = pickle.dumps(rfc)

# # 모델 Read
# clf_from_pickle = pickle.loads(saved_model)

