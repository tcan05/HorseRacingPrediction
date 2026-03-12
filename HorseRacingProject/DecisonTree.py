import pandas as pd
import os
import xgboost as xgb
import numpy as np
import time
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


script_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(script_dir, "horse_race_details.csv")

dataset = pd.read_csv(filepath)

# Frequency Encoding
def freq_encode(col, dataset):

    freq = dataset.groupby(col).size() / len(dataset)

    dataset.loc[:, "{}_freq".format(col)] = dataset[col].map(freq)
    dataset.drop(columns = [col], axis = 1, inplace = True)

    return dataset["{}_freq".format(col)]


start  = time.time()

# Drop unnecessary columns
dataset = dataset.drop(columns = ["horse_margin", "best_rating", "horse_name", "horse_trainer"], axis = 1)
dataset = dataset.drop(columns = ["race_date", "race_city", "horse_dam", "horse_owner"], axis = 1)
dataset = dataset.drop(columns = ["race_sex_group", "race_race_group", "race_type", "horse_origin", "horse_late_start"], axis = 1)

# Convert data into integer and float
dataset['horse_age'] = dataset['horse_age'].str.extract(r'(\d+)').astype(int)
dataset['race_age_group'] = dataset['race_age_group'].str.extract(r'(\d+)').astype(int)
dataset['horse_sex'] = dataset['horse_sex'].map({'f': 0, 'm': 1})

def time_to_seconds(t):
    try:
        m, s, ms = t.split('.')
        return int(m)*60 + int(s) + int(ms)/100
    except:
        return None
    
dataset['horse_race_degree'] = dataset['horse_race_degree'].apply(time_to_seconds)

le = LabelEncoder()
dataset['race_track_condition'] = le.fit_transform(dataset['race_track_condition'])
dataset["race_track_type"] = le.fit_transform(dataset["race_track_type"])

freq_encode("horse_accessories", dataset)
freq_encode("jockey_name", dataset)
freq_encode("horse_sire", dataset)
freq_encode("hors_broodmare_sire", dataset)


'''
#print(dataset.head())
dataset["win"] = (dataset["result"] == 1).astype(int)
dataset = dataset.drop(columns=["result"])

# Split the dataset
X = dataset.drop(columns = ["win"])
y = dataset["win"]


# Random Forest Classifier %89 accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

model = RandomForestClassifier(n_estimators = 100, max_depth = 20, class_weight = "balanced", random_state = 42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy:.3f}")

#print(classification_report(y_test, y_pred))

#sample = X_test.iloc[0:1]
#prediction = model.predict(sample)


#sample_dict = sample.iloc[0].to_dict()
#print(f"Sample Input: {sample_dict}")
#print(f"Predicted Result: {prediction}")
'''

'''
# XGBoost Regressor %62
X = dataset.drop(columns = ["result"])
y = max(dataset['result']) - dataset['result'] + 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

y_train_inverted = max(y_train) - y_train + 1
y_test_inverted  = max(y_test) - y_test + 1

group_train = X_train.groupby("race_no").size().to_numpy()
group_test = X_test.groupby("race_no").size().to_numpy()

ranker = xgb.XGBRanker(
    objective = "rank:pairwise",
    n_estimators = 300,
    learning_rate = 0.05,
    max_depth = 5,
    subsample = 0.8,
    colsample_bytree = 0.8,
    random_state = 42
)

ranker.fit(
    X_train.drop(columns = ["race_no"]),
    y_train_inverted,
    group = group_train,
    eval_set = [(X_test.drop(columns=["race_no"]), y_test_inverted)],
    eval_group=[group_test],
    verbose = True
)

y_pred = ranker.predict(X_test.drop(columns = ["race_no"]))

predicted_ranks = np.argsort(np.argsort(-y_pred)) + 1

corr, _ = spearmanr(y_test, predicted_ranks)
tau, _ = kendalltau(y_test, predicted_ranks)

end = time.time()

print(f'Spearman Rank Correlation: {corr:.3f}')
print(f'Kendall tau: {tau:.3f}')
print(f"Time taken: {end - start:.2f} seconds")
'''