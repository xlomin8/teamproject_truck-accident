import pandas as pd


df = pd.read_excel("D:/work/python/a-truck-accident/data/Newdata1.xlsx")

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects

df["화물평균일교통량"] = df["화물평균일교통량"].fillna(0.0).astype(int)
# nan값과 미분류를 기타불명으로 대체(단독사고차량)
df['피해운전자 차종'] = df['피해운전자 차종'].fillna(value = '기타불명')
df['피해운전자 차종'].replace(['미분류'],['기타불명'],inplace=True)
# nan값 기타불명으로 대체
df['피해운전자 상해정도'] = df['피해운전자 상해정도'].fillna(value = '기타불명')
#전처리 편리하도록 TYPE변경

df['요일'] = df['요일'].astype('category')
df['지역'] = df['지역'].astype('category')
df['도로종류'] = df['도로종류'].astype('category')
df['사고'] = df['사고'].astype('category')
df['사고유형'] = df['사고유형'].astype('category')
df['법규위반'] = df['법규위반'].astype('category')
df['기상상태'] = df['기상상태'].astype('category')
df['가해운전자 차종'] = df['가해운전자 차종'].astype('category')
df['가해운전자 상해정도'] = df['가해운전자 상해정도'].astype('category')
df['피해운전자 차종'] = df['피해운전자 차종'].astype('category')

# ▶ 더미화 진행
label_columns = ['사고', '요일','사고유형', '법규위반', '기상상태', '가해운전자 차종', '가해연령대', '가해운전자 상해정도']
# 명목척도, 서열척도 연산불가 가변수화, 더미화
label_df = df[label_columns]
label_df = pd.get_dummies(label_df, columns=label_columns)
label_df.info()
label_df.head()

# ▶ 하나의 Row마다 실행되며, col을 하나씩 증가해가면서 isna(Null 값이면 True를 반환), Null값이 아닐시 해당 데이터를 str처리 후 list에 append
# ▶ for 문에 간결화 Version
records = []
for i in range(len(label_df)):
    records.append([str(label_df.values[i,j]) for j in range(len(label_df.columns)) if not pd.isna(label_df.values[i,j])])

label_df = label_df.astype(bool)

# ▶ 구분화 진행

#시간

df['새벽'] = np.logical_and(df['사고발생시각'] > 0, df['사고발생시각'] < 7)
df['아침'] = np.logical_and(df['사고발생시각'] > 6, df['사고발생시각'] < 13)
df['오후'] = np.logical_and(df['사고발생시각'] > 12, df['사고발생시각'] < 19)
df['저녁'] = np.logical_and(df['사고발생시각'] > 18, df['사고발생시각'] < 25)

# ▶ 구분화 진행(2)

# 일일 교통량
df['일일 교통량_q1'] = np.logical_and(df['일일 교통량'] > 0, df['일일 교통량'] < 676704+1)
df['일일 교통량_q2'] = np.logical_and(df['일일 교통량'] > 676704, df['일일 교통량'] < 677280+1)
df['일일 교통량_q3'] = np.logical_and(df['일일 교통량'] > 677280, df['일일 교통량'] < 682656+1)
df['일일 교통량_q4'] = np.logical_and(df['일일 교통량'] > 682656, df['일일 교통량'] < 977184+1)

# 일일 교통량 평균
df['일일 교통량 평균_q1'] = np.logical_and(df['일일 교통량 평균'] > 0, df['일일 교통량 평균'] < 215+1)
df['일일 교통량 평균_q2'] = np.logical_and(df['일일 교통량 평균'] > 215, df['일일 교통량 평균'] < 227+1)
df['일일 교통량 평균_q3'] = np.logical_and(df['일일 교통량 평균'] > 227, df['일일 교통량 평균'] < 249+1)
df['일일 교통량 평균_q4'] = np.logical_and(df['일일 교통량 평균'] > 249, df['일일 교통량 평균'] < 346+1)

# 일일 속도평균
df['일일 속도평균_q1'] = np.logical_and(df['일일 속도평균'] > 0, df['일일 속도평균'] < 85+1)
df['일일 속도평균_q2'] = np.logical_and(df['일일 속도평균'] > 85, df['일일 속도평균'] < 89+1)
df['일일 속도평균_q3'] = np.logical_and(df['일일 속도평균'] > 89, df['일일 속도평균'] < 91+1)
df['일일 속도평균_q4'] = np.logical_and(df['일일 속도평균'] > 91, df['일일 속도평균'] < 100+1)

# 일일 속도75%
df['일일 속도75%_q1'] = np.logical_and(df['일일 속도75%'] > 0, df['일일 속도75%'] < 104+1)
df['일일 속도75%_q2'] = np.logical_and(df['일일 속도75%'] > 104, df['일일 속도75%'] < 105+1)
df['일일 속도75%_q3'] = np.logical_and(df['일일 속도75%'] > 105, df['일일 속도75%'] < 108+1)
df['일일 속도75%_q4'] = np.logical_and(df['일일 속도75%'] > 108, df['일일 속도75%'] < 195+1)

# 고속도로 교통량
df['고속도로 교통량_q1'] = np.logical_and(df['고속도로 교통량'] > 0, df['고속도로 교통량'] < 23904+1)
df['고속도로 교통량_q2'] = np.logical_and(df['고속도로 교통량'] > 23904, df['고속도로 교통량'] < 28800+1)
df['고속도로 교통량_q3'] = np.logical_and(df['고속도로 교통량'] > 28800, df['고속도로 교통량'] < 53376+1)
df['고속도로 교통량_q4'] = np.logical_and(df['고속도로 교통량'] > 53376, df['고속도로 교통량'] < 72960+1)

#고속도로별 로드킬
df['고속도로별 로드킬_q1_x'] = np.logical_and(df['고속도로별 로드킬'] > -1, df['고속도로별 로드킬'] < 1)
df['고속도로별 로드킬_q2'] = np.logical_and(df['고속도로별 로드킬'] > 0, df['고속도로별 로드킬'] < 16+1)
df['고속도로별 로드킬_q3'] = np.logical_and(df['고속도로별 로드킬'] > 16, df['고속도로별 로드킬'] < 59+1)
df['고속도로별 로드킬_q4'] = np.logical_and(df['고속도로별 로드킬'] > 59, df['고속도로별 로드킬'] < 149+1)

# 제한차량_과적
df['제한차량_과적_q1'] = np.logical_and(df['제한차량_과적'] > 0, df['제한차량_과적'] < 1585+1)
df['제한차량_과적_q2'] = np.logical_and(df['제한차량_과적'] > 1585, df['제한차량_과적'] < 2764+1)
df['제한차량_과적_q3'] = np.logical_and(df['제한차량_과적'] > 2764, df['제한차량_과적'] < 6875+1)
df['제한차량_과적_q4'] = np.logical_and(df['제한차량_과적'] > 6875, df['제한차량_과적'] < 25681+1)

# 제한차량_적불
df['제한차량_적불_q1'] = np.logical_and(df['제한차량_적불'] > 0, df['제한차량_적불'] < 4423+1)
df['제한차량_적불_q2'] = np.logical_and(df['제한차량_적불'] > 4423, df['제한차량_적불'] < 7701+1)
df['제한차량_적불_q3'] = np.logical_and(df['제한차량_적불'] > 7701, df['제한차량_적불'] < 15647+1)
df['제한차량_적불_q4'] = np.logical_and(df['제한차량_적불'] > 15647, df['제한차량_적불'] < 37515+1)

# 화물_과속
df['화물_과속_x'] = np.logical_and(df['화물_과속'] > -1, df['화물_과속'] < 1)
df['화물_과속_o'] = np.logical_and(df['화물_과속'] > 0, df['화물_과속'] < 1000000)

# 급가속
df['급가속_x'] = np.logical_and(df['급가속'] > -1, df['급가속'] < 1)
df['급가속_o'] = np.logical_and(df['급가속'] > 0, df['급가속'] < 1000000)

# 급정지
df['급정지_x'] = np.logical_and(df['급정지'] > -1, df['급정지'] < 1)
df['급정지_o'] = np.logical_and(df['급정지'] > 0, df['급정지'] < 1000000)

# 급좌회전
df['급좌회전_x'] = np.logical_and(df['급좌회전'] > -1, df['급좌회전'] < 1)
df['급좌회전_o'] = np.logical_and(df['급좌회전'] > 0, df['급좌회전'] < 1000000)

# 급우회전
df['급우회전_x'] = np.logical_and(df['급우회전'] > -1, df['급우회전'] < 1)
df['급우회전_o'] = np.logical_and(df['급우회전'] > 0, df['급우회전'] < 1000000)

# 급유턴
df['급유턴_x'] = np.logical_and(df['급유턴'] > -1, df['급유턴'] < 1)
df['급유턴_o'] = np.logical_and(df['급유턴'] > 0, df['급유턴'] < 1000000)

# 급앞지르기
df['급앞지르기_x'] = np.logical_and(df['급앞지르기'] > -1, df['급앞지르기'] < 1)
df['급앞지르기_o'] = np.logical_and(df['급앞지르기'] > 0, df['급앞지르기'] < 1000000)

# 급진로
df['급진로_x'] = np.logical_and(df['급진로'] > -1, df['급진로'] < 1)
df['급진로_o'] = np.logical_and(df['급진로'] > 0, df['급진로'] < 1000000)

# 화물_관용
df['화물_관용_q1'] = np.logical_and(df['화물_관용'] > 0, df['화물_관용'] < 2278+1)
df['화물_관용_q2'] = np.logical_and(df['화물_관용'] > 2278, df['화물_관용'] < 2973+1)
df['화물_관용_q3'] = np.logical_and(df['화물_관용'] > 2973, df['화물_관용'] < 5582+1)
df['화물_관용_q4'] = np.logical_and(df['화물_관용'] > 5582, df['화물_관용'] < 5706+1)

# 화물_자가용
df['화물_자가용_q1'] = np.logical_and(df['화물_자가용'] > 0, df['화물_자가용'] < 169050+1)
df['화물_자가용_q2'] = np.logical_and(df['화물_자가용'] > 169050, df['화물_자가용'] < 291305+1)
df['화물_자가용_q3'] = np.logical_and(df['화물_자가용'] > 291305, df['화물_자가용'] < 683295+1)
df['화물_자가용_q4'] = np.logical_and(df['화물_자가용'] > 683295, df['화물_자가용'] < 688478+1)

# 화물_영업용
df['화물_영업용_q1'] = np.logical_and(df['화물_영업용'] > 0, df['화물_영업용'] < 19122+1)
df['화물_영업용_q2'] = np.logical_and(df['화물_영업용'] > 19122, df['화물_영업용'] < 33603+1)
df['화물_영업용_q3'] = np.logical_and(df['화물_영업용'] > 33603, df['화물_영업용'] < 113282+1)
df['화물_영업용_q4'] = np.logical_and(df['화물_영업용'] > 113282, df['화물_영업용'] < 117289+1)

# 주행거리당 사망수
df['주행거리당 사망수_q1'] = np.logical_and(df['주행거리당 사망수'] > 0, df['주행거리당 사망수'] < 7.4+0.1)
df['주행거리당 사망수_q2'] = np.logical_and(df['주행거리당 사망수'] > 7.4, df['주행거리당 사망수'] < 7.8+0.1)
df['주행거리당 사망수_q3'] = np.logical_and(df['주행거리당 사망수'] > 7.8, df['주행거리당 사망수'] < 14.4+0.1)
df['주행거리당 사망수_q4'] = np.logical_and(df['주행거리당 사망수'] > 14.4, df['주행거리당 사망수'] < 19.5+0.1)

# 시도별_진료비
df['시도별_진료비_q1'] = np.logical_and(df['시도별_진료비'] > 0, df['시도별_진료비'] < 103659+1)
df['시도별_진료비_q2'] = np.logical_and(df['시도별_진료비'] > 103659, df['시도별_진료비'] < 111105+1)
df['시도별_진료비_q3'] = np.logical_and(df['시도별_진료비'] > 111105, df['시도별_진료비'] < 123418+1)
df['시도별_진료비_q4'] = np.logical_and(df['시도별_진료비'] > 123418, df['시도별_진료비'] < 196398+1)

# 연령별_가해진료비
df['연령별_가해진료비_q1'] = np.logical_and(df['연령별_가해진료비'] > 0, df['연령별_가해진료비'] < 100816+1)
df['연령별_가해진료비_q2'] = np.logical_and(df['연령별_가해진료비'] > 100816, df['연령별_가해진료비'] < 104964+1)
df['연령별_가해진료비_q3'] = np.logical_and(df['연령별_가해진료비'] > 104964, df['연령별_가해진료비'] < 131781+1)
df['연령별_가해진료비_q4'] = np.logical_and(df['연령별_가해진료비'] > 131781, df['연령별_가해진료비'] < 198854+1)

# 화물평균일교통량
df['화물평균일교통량_q1'] = np.logical_and(df['화물평균일교통량'] > 0, df['화물평균일교통량'] < 19249+1)
df['화물평균일교통량_q2'] = np.logical_and(df['화물평균일교통량'] > 19249, df['화물평균일교통량'] < 26742+1)
df['화물평균일교통량_q3'] = np.logical_and(df['화물평균일교통량'] > 26742, df['화물평균일교통량'] < 45847+1)
df['화물평균일교통량_q4'] = np.logical_and(df['화물평균일교통량'] > 45847, df['화물평균일교통량'] < 123028+1)

# 교통사고비용_천원
df['교통사고비용_천원_q1'] = np.logical_and(df['교통사고비용_천원'] > 0, df['교통사고비용_천원'] < 5245.000+1)
df['교통사고비용_천원_q2'] = np.logical_and(df['교통사고비용_천원'] > 5245.000, df['교통사고비용_천원'] < 10490.00+1)
df['교통사고비용_천원_q3'] = np.logical_and(df['교통사고비용_천원'] > 10490.00, df['교통사고비용_천원'] < 45847+1)
df['교통사고비용_천원_q4'] = np.logical_and(df['교통사고비용_천원'] > 66502.40, df['교통사고비용_천원'] < 1560485+1)

df_merge = pd.concat([df, label_df], axis=1)

df2 = df_merge.drop( ['사고', '사고발생년도', '사고발생월', '사고발생일', '사고발생시각', '일일 교통량', '일일 교통량 평균', '일일 속도평균',  '일일 속도75%',
                '고속도로 교통량', '고속도로별 로드킬', '요일', '지역', '제한차량_과적', '제한차량_적불', '도로종류', '사망자수', '중상자수', '경상자수', '부상신고자수',
                '부상자수', '사고유형', '법규위반', '기상상태', '가해운전자 차종', '가해연령대', '피해연령대', '사고일', '사고시', '가해운전자 상해정도', '피해운전자 차종',
                '피해운전자 상해정도', '교통사고비용_천원', '화물_과속', '급가속', '급출발', '급감속', '급정지', '급좌회전', '급우회전', '급유턴', '급앞지르기', '급진로',
                '승용_관용', '승용_자가용', '승용_영업용', '승용등록_계', '화물_관용', '화물_자가용', '화물_영업용', '화물등록_계', '주행거리당 사망수', '시도별_진료비', '연령별_가해진료비',
                '연령별_피해진료비', '고속도로별 로드킬_q1_x', '화물_과속_x', '급가속_x', '급정지_x', '급좌회전_x', '급우회전_x', '급유턴_x',
                '급앞지르기_x', '급진로_x', '가해운전자 상해정도_사망', '가해운전자 상해정도_기타불명', '사고_경상사고', '사고_부상신고사고', '사고_중상사고', '기상상태_기타', '법규위반_기타', '가해운전자 차종_승용', '화물평균일교통량',
                '교통사고비용_천원_q4', '교통사고비용_천원_q3', '교통사고비용_천원_q2', '교통사고비용_천원_q1', '가해운전자 차종_화물'], axis='columns')
print(df2)

import mlxtend
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
import time
# fpgrowth(df2, min_support=0.6, use_colnames=True)
from mlxtend.frequent_patterns import apriori, association_rules

min_support_per = 0.5
min_trust_per =0.5
result = fpgrowth(df2,min_support=min_support_per, use_colnames=True)
result_chart = association_rules(result, metric="confidence", min_threshold=min_trust_per)
print(result_chart)

rule = result_chart[result_chart['consequents']==frozenset({'사고_사망사고'})]

print(rule)