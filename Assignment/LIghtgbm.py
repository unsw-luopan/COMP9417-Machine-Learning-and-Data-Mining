"""
This is the group work of COMP9417 19T2 Assignment
Group member: Pan Luo:z5192086,
	          Zhidong Luo:z5181142,
              Shuxiang Zou:z5187969,
	          Xinchen Wang:z5197409.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv('train_V2.csv')
df = df.dropna()

df = pd.read_csv('train_V2.csv')
df = df.dropna()
df = df[df['maxPlace'] > 1]

df = df.drop(["matchType"], axis=1)

df = df.groupby(["matchId", "groupId"]).mean()
df = df.reset_index(drop=True)
df.to_csv("test.csv")

df["total_moving"] = df["rideDistance"] + df["swimDistance"] + df["walkDistance"]
Y = df.iloc[:, 24]
df = df.drop("winPlacePerc", axis=1)
X = df.iloc[:, 0:25]
train_data = lgb.Dataset(data=X, label=Y)

params = {'num_leaves': 31, 'num_trees': 100, 'objective': 'regression'}
params['metric'] = 'rmse'
moudle = lgb.train(params,
                   train_data,
                   num_boost_round=20)

df2 = pd.read_csv("test_V2.csv")
test_id = df2["Id"]
df2["total_moving"] = df2["rideDistance"] + df2["swimDistance"] + df2["walkDistance"]
df2 = df2.drop(["rideDistance", "swimDistance", "walkDistance"], axis=1)
df2 = df2.drop(["matchType"], axis=1)
testX = df2.iloc[:, 3:25]
y_pred = moudle.predict(testX)
submit = pd.DataFrame({'Id': test_id, "winPlacePerc": y_pred}, columns=['Id', 'winPlacePerc'])
submit.to_csv("submission.csv", index=False)
