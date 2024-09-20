from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

data = pd.read_csv('goods_dataset.csv')

Feature = data[['good_reviews', 'rebuy_rate']]
Label = data['label']

Feature_train,Feature_test,Label_train,Label_test = train_test_split(Feature,Label,test_size = 0.2, random_state = 42)

model = LogisticRegression()

model.fit(Feature_train,Label_train) # Start training

Label_pred = model.predict(Feature_test)

accuracy = accuracy_score(Label_test,Label_pred)

print(f'模型的准确率: {accuracy * 100:.2f}%')

results = pd.DataFrame({
    'good_reviews' : Feature_test['good_reviews'],
    'rebuy_rate' : Feature_test['rebuy_rate'],
    'actual_label': Label_test,
    'predicted_label': Label_pred
})

results_csv_path = 'prediction results.csv'
results.to_csv(results_csv_path,index=True)
print('results export ok')