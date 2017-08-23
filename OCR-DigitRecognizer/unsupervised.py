import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
import pandas as pd

X = pd.read_csv('machine.csv', sep=',')
df=pd.DataFrame(X)
numpyMatrix = df.as_matrix(columns=df.columns[1:])

kmeans = KMeans(n_clusters=2)

kmeans.fit(numpyMatrix)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print(labels)
a = ["foo", "bar"]
x=map(lambda x: "Extrovert" if x==1 else "Introvert" , labels)
labelList=list(x)
colors = ["g.","r.","c.","y."]

for i,txt in enumerate(numpyMatrix):
    plt.plot(numpyMatrix[i][0],numpyMatrix[i][1], colors[labels[i]+2], markersize = 20,label=labelList[i]+"  "+df["Name"].values[i])
    
    plt.annotate(df["Name"].values[i],( numpyMatrix[i][0],numpyMatrix[i][1]))


plt.legend()

plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

plt.show()		
