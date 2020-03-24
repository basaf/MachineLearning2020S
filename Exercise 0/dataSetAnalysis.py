# -*- coding: utf-8 -*-
from arff import load
import pandas as pd
import matplotlib.pyplot as plt 
import configuration as cfg

# load dataset (.arff) into pandas DataFrame
data = load(open(cfg.default.dataset_1_path + '\\communities.arff', 'r'))
all_attributes = list(i[0] for i in data['attributes'])
df = pd.DataFrame(columns=all_attributes, data=data['data'])

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
#%%
for x in not_predictive_attributes:
    predictive_attributes.remove(x)

# plot and save histograms
hist = df[predictive_attributes].hist(figsize=(27, 36))
plt.tight_layout()
#plt.savefig('images\\communities_data_histogram_predictive.png')
#plt.close()

hist = df[goal_attribute].hist()
plt.tight_layout()
#plt.savefig('images\\communities_data_histogram_goal.png')
#plt.close()



#%%
occupancyTrainingData = pd.read_csv(cfg.default.dataset_2_path + '\\datatraining.txt')

occupancyTrainingData=occupancyTrainingData.set_index('date')
occupancyTrainingData['Temperature'].plot()
plt.xticks(rotation=45)
plt.tight_layout()
plt.xlabel("")
plt.ylabel("Temperature [°C]")
#set time stamp as index
#%%

# plot and save histograms
hist = occupancyTrainingData.hist()
plt.tight_layout()
#plt.savefig('images\\occupancy_data_training_histogram.png')
#plt.close()

occupancyData = pd.read_csv(cfg.default.dataset_2_path + '\\datatest.txt')

# plot and save histograms
hist = occupancyData.hist()
plt.tight_layout()
#plt.savefig('images\\occupancy_data_test_histogram.png')
#plt.close()

occupancyData = pd.read_csv(cfg.default.dataset_2_path + '\\datatest2.txt')

# plot and save histograms
hist = occupancyData.hist()
plt.tight_layout()
#plt.savefig('images\\occupancy_data_test2_histogram.png')
#plt.close()

print('Fertig')
