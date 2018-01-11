import matplotlib.pyplot as plt
from matplotlib import  style
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing, cross_validation
import  pandas as pd

style.use('ggplot')

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
# print(df.head())
df.drop(['name','body'], 1,  inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
# print(df.head())

def handle_non_numerical_data(df):
    columns = df.columns.values
    # print(columns)
    for column in columns:
        text_digit_val = {}
        def convert_to_int(val):
            return text_digit_val[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_content = df[column].values.tolist()
            # print(column_content)
            unique_elements = set(column_content)
            # print(unique_elements)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_val:
                    text_digit_val[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))
    return df

df = handle_non_numerical_data(df)
# print(df)

# df.drop(['fare','pclass'],1,inplace= True)
# print(df.head())

x = np.array(df.drop(['survived'],1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])

# x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=0.2)

clf = MeanShift()
clf.fit(x)

labels = clf.labels_
cluster_center = clf.cluster_centers_


original_df['cluster_group'] = np.nan

for i in range(len(x)):
    original_df['cluster_group'].iloc[i] = labels[i]
# print(original_df['cluster_group'].describe())

n_cluster_ = len(np.unique(labels))
# print(n_cluster_)
survival_rate = {}
for i in range(n_cluster_):
    temp_df = original_df[ (original_df['cluster_group']==float(i)) ]
    # print(temp_df)
    survival_cluster = temp_df[ (temp_df['survived']==1) ]
    # print(survival_cluster)
    survival_rates = len(survival_cluster)/len(temp_df)
    survival_rate[i] = survival_rates

print(survival_rate)
print(original_df[ (original_df['cluster_group']==2) ].describe())
