import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from sklearn.preprocessing import StandardScaler

# 데이터 로드
file = './fim.xlsx'
df = pd.read_excel(file, sheet_name='2021_원본')

# print(df.head())
print(df.columns)

# 스케일링 필요한 컬럼 추출
X = df.drop(['사고발생년도', '사고발생월', '사고발생일', '사고발생시각', '요일', '화물등록_계', '지역', '도로종류', '차대차', '차대사람', '단독차량', '안전운전불이행',
             '가해연령대', '교통사고비용', '피해운전자 차종', '가해운전자 차종', '법규위반', '사고유형'
             ], axis=1)
# print(X.columns)

X_colnames = ['일일 교통량', '일일 교통량 평균', '일일 속도평균', '일일 속도75%', '고속도로 교통량', '고속도로별 로드킬', '화물평균일교통량', '승용평균일교통량', '버스평균일교통량', '위험운전행동',
              '과적', '적불', '주행거리당 사망자수', '부상자수', '사망', '화물_관용', '화물_자가용', '화물_영업용', '시도별_진료비', '연령별_가해진료비', '연령별_피해진료비']
print(X)

print()

# 스케일링
## 표준화(StandardScaler)
# # 원본 데이터들의 평균, 분산 확인
# print('feature들의 평균값')
# print(X.mean())
# print('\nfeature 들의 분산값')
# print(X.var())
#
# ## StandardScaler 객체 생성
# scaler = StandardScaler()
# scaler.fit(X)
# X_scaled = scaler.transform(X)
#
# print("\n=========== 표준화 ==============\n")
#
# X_scaled = pd.DataFrame(data=X_scaled, columns=X_colnames) #transform -> numpy ndarray -> DataFrame
# print('feature 들의 평균값')
# print(X_scaled.mean())
# print('\nfeature 들의 분산값')
# print(X_scaled.var())
#
# # 각 값들의 정규분포 모양 비교
# f, ax = plt.subplots(1,2, figsize=(8,5))
#
# # 표준화 전 정규분포
# x0 = X['일일 교통량'].dropna().values
# sns.distplot(x0, kde=False, rug=False, fit=sp.stats.norm, ax=ax[0])
#
# # 표준화 후 정규분포
# x1 = X_scaled['일일 교통량'].values
# sns.distplot(x1, kde=False, rug=False, fit=sp.stats.norm, ax=ax[1])
# plt.show()
#
# # 이상치 제거
# ## Z score 확인
# df_Zscore = pd.DataFrame()
# outlier_dict = {}
# outlier_idx_list = []
#
# for one_col in X_scaled.columns:
#     print('Check', one_col)
#     df_Zscore[f'{one_col}_Zscore'] = sp.stats.zscore(X_scaled[one_col])
#     outlier_dict[one_col] = df_Zscore[f'{one_col}_Zscore'][(df_Zscore[f'{one_col}_Zscore']>2) | (df_Zscore[f'{one_col}_Zscore']<-2)]
#     outlier_idx_list.append(list(outlier_dict[one_col].index))
#     if len(outlier_dict[one_col]):
#         print(one_col, 'Has outliers\n', outlier_dict[one_col])
#     else:
#         print(one_col, 'Has No outlier')
#     print()
#
# print('Before', X_scaled.shape)
# all_outlier_idx = sum(outlier_idx_list, [])
# X_scaled = X_scaled.drop(all_outlier_idx)
# print('After (drop outlier)', X_scaled.shape)
#
# # 정규화
# from sklearn.preprocessing import MinMaxScaler
#
# ## MinMaxScaler 객체 생성
# scaler = MinMaxScaler()
# scaler.fit(X)
# X_scaled = scaler.transform(X)
#
# X_scaled = pd.DataFrame(data=X_scaled, columns=X_colnames)
# print('feature들의 최솟값')
# print(X_scaled.min())
# print('feature들의 최댓값')
# print(X_scaled.max())
#
# print(X_scaled)
# print(X_scaled.describe())


## minmax scaler
from sklearn.preprocessing import MinMaxScaler
## MinMaxScaler 객체 생성
scaler = MinMaxScaler()
scaler.fit(X)

X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(data=X_scaled, columns=X_colnames)

print(X_scaled)
print(X_scaled.describe())

# 저장
X_scaled.to_excel('./scaled data 21.xlsx')