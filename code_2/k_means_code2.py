import matplotlib.pyplot as plt
from matplotlib import  style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, cross_validation
import  pandas as pd

style.use('ggplot')

df = pd.read_excel('titanic.xls')
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

df.drop(['fare','pclass'],1,inplace= True)
print(df.head())

x = np.array(df.drop(['survived'],1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])

# x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=0.2)

clf = KMeans(n_clusters=2)
clf.fit(x)

correct = 0

for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))

    prediction = clf.predict(predict_me)

    if prediction == y[i]:
        correct += 1

print('accuary '+str(correct/len(x)))

# for i in range(len(x_test)):
#     predict_me = np.array(x_test[i].astype(float))
#     predict_me = predict_me.reshape(-1,len(predict_me))
#
#     prediction = clf.predict(predict_me)
#
#     if prediction == y_test[i]:
#         correct += 1
#
# print(correct/len(x_test))
