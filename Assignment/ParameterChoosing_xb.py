"""
This is the group work of COMP9417 19T2 Assignment
Group member: Pan Luo:z5192086,
	          Zhidong Luo:z5181142,
              Shuxiang Zou:z5187969,
	          Xinchen Wang:z5197409.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("train_V2.csv")
df = df.dropna()


def reduce_mem_usage(df):
    # iterate through all the columns of a dataframe and modify the data type
    # to reduce memory usage.

    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# This function aims to produce the categorical type without outliers
def merge_match_type(x):
    if x in {'normal-squad-fpp', 'crashfpp', 'crashtpp', 'normal-duo-fpp',
             'flarefpp', 'normal-solo-fpp', 'flaretpp', 'normal-duo',
             'normal-squad', 'normal-solo'}:
        return 'others'
    else:
        return x


df = reduce_mem_usage(df)
df['matchType'] = df.matchType.apply(merge_match_type)
# using one hot encoding to code the maychType attribute
data_dumm = pd.get_dummies(df, columns=['matchType'])
# drop the outlier
data_dumm = data_dumm.drop('matchType_others', axis=1)
data_dumm = data_dumm.groupby("groupId").mean()
# rearrange the header
clean_data = data_dumm.loc[:, ['Id', 'groupId', 'matchId', 'assists', 'boosts', 'damageDealt', 'DBNOs',
                               'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
                               'killStreaks', 'longestKill', 'matchDuration', 'maxPlace', 'numGroups',
                               'rankPoints', 'revives', 'rideDistance', 'roadKills', 'swimDistance',
                               'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired',
                               'winPoints', 'matchType_duo', 'matchType_duo-fpp',
                               'matchType_solo', 'matchType_solo-fpp', 'matchType_squad',
                               'matchType_squad-fpp', 'winPlacePerc']]
# generate attribute
X = clean_data.iloc[:, 3:33]
# generate target regression target
Y = clean_data.iloc[:, 33]

# generate the parameter dict which can be fit in GridSearchCV model
param_test = {'learning_rate': [0.02, 0.03, 0.1], 'min_child_weight': [4, 6, 8], 'max_depth': [8, 10],
              "subsample": [0.6, 0.4], "n_estimators": [300, 500]}
# define the target model and some parameters except these which should be chosen by greedy search
model = XGBRegressor(n_estimators=-1)
grid = GridSearchCV(model, param_test, cv=5, scoring='neg_mean_absolute_error')
# fit the data into greedy search
print("begin to find parameters")
grid.fit(X, Y)
# print the statistic data and choose the best parameters for our target model
print(grid.best_score_, grid.best_estimator_, grid.best_params_)
