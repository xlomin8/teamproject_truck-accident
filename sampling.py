from sklearn.datasets import make_classification

df = pd.read_excel("D:/work/python/a-truck-accident/data/Newdata_ver1.xlsx")

X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1,
    n_samples=100, random_state=10
)

df = pd.DataFrame(X)
df['target'] = y


num_0 = len(X[X['target']==0])
num_1 = len(X[X['target']==1])

# undersampled_data = pd.concat([ X[X['target']==0].sample(num_1) , X[X['target']==1] ])
oversampled_data = pd.concat([ X[X['target']==0] , X[X['target']==1].sample(num_0, replace=True) ])

# 출처: https://matamong.tistory.com/entry/불균형-데이터-다루기-Resampling-over-sampling-under-sampling [마타몽의 개발새발제발:티스토리]