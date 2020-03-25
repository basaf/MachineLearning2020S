from arff import load
from pandas import DataFrame
import matplotlib.pyplot as plt 

# load dataset (.arff) into pandas DataFrame
data = load(open('c:\\git\\MachineLearning2020S\\MachineLearning2020S\\Exercise 0\\communities_data\\communities.arff', 'r'))
all_attributes = list(i[0] for i in data['attributes'])
df = DataFrame(columns=all_attributes, data=data['data'])

# divide attributes in not_predictive, predictive and goal
not_predictive_attributes = [
    'state',
    'county',
    'community',
    'communityname',
    'fold'
    ]
goal_attribute = 'ViolentCrimesPerPop'

predictive_attributes = all_attributes.copy()
predictive_attributes.remove(goal_attribute)
for x in not_predictive_attributes:
    predictive_attributes.remove(x)

# plot and save histograms
hist = df[predictive_attributes].hist(figsize=(27, 36))
plt.tight_layout()
plt.savefig('images\\communities_data_histogram_predictive.png')
plt.close()

hist = df[goal_attribute].hist()
plt.tight_layout()
plt.savefig('images\\communities_data_histogram_goal.png')
plt.close()

print('Ready')
