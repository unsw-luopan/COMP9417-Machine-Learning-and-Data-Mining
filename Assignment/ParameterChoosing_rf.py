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
from sklearn.ensemble import RandomForestRegressor
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


df = reduce_mem_usage(df)

# drop the outlier which some competition only have very fewer groups
df = df[df['maxPlace'] > 1]
# drop the categorical features which have too much noise
df = df.drop(["matchType"], axis=1)
# group by matchId and groupId based on our feature engineering
df = df.groupby(["matchId", "groupId"]).mean()
df = df.reset_index(drop=True)
# create a new feature which can be helpful
df["total_moving"] = df["rideDistance"] + df["swimDistance"] + df["walkDistance"]
df = df.drop(["rideDistance", "swimDistance", "walkDistance"], axis=1)
# generate our regression target
Y = df.iloc[:, 21]
df = df.drop("winPlacePerc", axis=1)
# generate our features
X = df.iloc[:, 0:22]

# generate the parameter dict which can be fit in GridSearchCV model
param_test = {'n_estimators': range(10, 51, 10), 'max_depth': range(5, 8, 1),
              "min_samples_split": range(50, 101, 10), "min_samples_leaf": range(20, 51, 10)}
# define the target model and some parameters except these which should be chosen by greedy search
model = RandomForestRegressor(random_state=8, max_features='sqrt', n_estimators=-1)
grid = GridSearchCV(model, param_test, cv=5, scoring='neg_mean_absolute_error')
# fit the data into greedy search
print("begin to find parameters")
grid.fit(X, Y)
# print the statistic data and choose the best parameters for our target model
print(grid.best_score_, grid.best_estimator_, grid.best_params_)
