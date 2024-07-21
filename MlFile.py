import pandas as pd
from numpy import argsort
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.impute import KNNImputer
import pickle

df = pd.read_csv("Bitcoin_dataset_updated.csv")

df = df.rename(columns={'BTC price [USD]': 'BTC_price', 'n-transactions': 'n_transactions', 'fee [USD]': 'fee',
                        'btc search trends': 'BTC_search_trends', 'Gold price[USD]': 'Gold_price',
                        'SP500 close index': 'close_index', 'Oil WTI price[USD]': 'Oil_price',
                        'M2(Not seasonally adjusted)[1e+09 USD]': 'M2'})

reviseDf = df.drop(['Date'], axis=1)

imputer = KNNImputer(n_neighbors=5)

reviseDf.loc[:, :] = imputer.fit_transform(reviseDf)

x = reviseDf.drop(['BTC_price'], axis=1)

y = reviseDf['BTC_price']

x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, test_size=0.50, random_state=1)

x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.33, random_state=1)


def get_models():
    model_list = list()
    model_list.append(('knn', KNeighborsRegressor()))
    model_list.append(('cart', DecisionTreeRegressor()))
    model_list.append(('svm', SVR()))
    model_list.append(('rfr', RandomForestRegressor()))
    return model_list


def evaluate_models(model_list, x_train, x_test, y_train, y_test):
    scores = list()
    for name, model in model_list:
        model.fit(x_train, y_train)
        result = model.predict(x_test)
        mae = mean_absolute_error(y_test, result)
        scores.append(-mae)

    return scores


models = get_models()

scores = evaluate_models(models, x_train, x_val, y_train, y_val)

ranking = 1 + argsort(argsort(scores))

ensemble = VotingRegressor(estimators=models, weights=ranking)

model_to_be_used = ensemble.fit(x_train_full, y_train_full)

file = open('list.txt', 'wb')

pickle.dump(model_to_be_used, file)

file.close()

