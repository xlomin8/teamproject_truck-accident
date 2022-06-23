import mlxtend
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth

# 데이터 로드
df = pd.read_excel('clean_data.xlsx')
# fpgrowth(df, min_support=0.6)
fpgrowth(df, min_support=0.6, use_colnames=True)


