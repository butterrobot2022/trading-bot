from backtesting.test import GOOG, SMA, EURUSD
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


df = pd.read_csv('/Users/motin/Downloads/traffic/traffic/XAUUSD15.csv').drop_duplicates()
df.index = df['Time'].values
del df['Time']
GOOG = df
print(len(GOOG))
GOOG['Tomorrow'] = GOOG['Close'].shift(-1)

GOOG['Target'] = (GOOG['Tomorrow'] > GOOG['Close']).astype(int)

model  = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = GOOG.iloc[:-500]
test = GOOG.iloc[-500:]

predictors = ['Close', 'Open', 'High', 'Low']
model.fit(train[predictors], train['Target'])

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

print(precision_score(test['Target'], preds))

combined = pd.concat([test['Target'], preds], axis=1)

# def predict(train, test, predictors, model):
#     model.fit(train[predictors], train['Target'])
#     preds = model.predict(test[predictors])
#     preds = pd.Series(preds, index=test.index, name='Predictions')
#     combined = pd.concat([test['Target'], preds], axis=1)
#     return combined

def predict(train, test, predictors, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name='Predictions')
    combined = pd.concat([test['Target'], preds], axis=1)
    return combined


def backtest(data, model, predictors, start=125, step=25):
    all_predictions = []
    print(data.shape[0])

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()

        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)


# predictions = backtest(GOOG, model, predictors)
# print(predictions['Predictions'].value_counts())
# print(precision_score(predictions['Target'], predictions['Predictions']))
# print(predictions['Target'].value_counts()/predictions.shape[0])

horizons = [18, 50, 200 ,500, 1000, 1500]
# horizons = [50, 200, 500, 1000]

new_predictors = []

for horizon in horizons:
    rolling_averages = GOOG.rolling(horizon).mean()
    ratio_column = f'Close_Ratio_{horizon}'
    GOOG[ratio_column] = GOOG['Close'] / rolling_averages['Close']
    trend_column = f'Trend_{horizon}'
    GOOG[trend_column] = GOOG.shift(1).rolling(horizon).sum()['Target']
    new_predictors += [ratio_column, trend_column]

GOOG = GOOG.dropna()

model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)

predictions = backtest(GOOG, model, new_predictors)

print(predictions['Predictions'].value_counts())
print(precision_score(predictions['Target'], predictions['Predictions']))
# test_length = int(len(GOOG) * 0.75)
# print(len(GOOG))
# # print(test_length+20)
# train = GOOG.iloc[:test_length]
# test = GOOG.iloc[850:]
# data_test = GOOG.iloc[test_length:]
# # data_test.reshape(-1, 1)
# model.fit(train[new_predictors], train['Target'])
# preds = model.predict_proba(data_test[predictors])[:,1]
# preds[preds >= .6] = 1
# preds[preds < .6] = 0
# preds = pd.Series(preds, index=data_test.index, name='Predictions')
# print(f'Predictions are {preds}')
# print(f'Length of predictions are {len(preds)}')
# # print(data_test)
# print(precision_score(data_test['Target'], preds))
