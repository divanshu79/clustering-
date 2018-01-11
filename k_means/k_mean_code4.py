import matplotlib.pyplot as plt
from matplotlib import  style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import  pandas as pd

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

##plt.scatter(X[:,0], X[:,1], s=150)
##plt.show()

colors = 10*["g","r","c","b","k"]


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]
        # print(self.centroids)

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                # print(distances)
                classification = distances.index(min(distances))
                # print(classification)
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)
            # print(prev_centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                # print(np.sum((current_centroid - original_centroid) / original_centroid))
                if np.sum((current_centroid-original_centroid)/original_centroid) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


df.drop(['fare','pclass','age','parch'],1,inplace= True)
print(df.head())

x = np.array(df.drop(['survived'],1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])
clf = K_Means()
clf.fit(x)

correct = 0
for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))

    prediction = clf.predict(predict_me)

    if prediction == y[i]:
        correct += 1

print('accuary '+str(correct/len(x)))
