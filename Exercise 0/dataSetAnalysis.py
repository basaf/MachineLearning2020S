from arff import load
from pandas import DataFrame
import matplotlib.pyplot as plt 

# load dataset (.arff) into pandas DataFrame
data = load(open('c:\\git\\MachineLearning2020S\\MachineLearning2020S\\Exercise 0\\communities_data\\communities.arff', 'r'))
attributes = list(i[0] for i in data['attributes'])
df = DataFrame(columns=attributes, data=data['data'])
# print(df.describe())

# plot and save histogram
hist = df.hist(figsize=(27, 36))
plt.tight_layout()
plt.savefig('images\\communities_data_histogram.png')

print('Fertig')
