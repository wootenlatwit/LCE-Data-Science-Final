import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_csv("DOW_processed.csv")

X = df.drop(['avg', 'volume', 'vwap'], axis=1).drop(df.columns[0], axis=1).to_numpy(dtype='float64')
y = df['avg'].to_numpy(dtype='float64')

from sklearn.model_selection import train_test_split

X1, X2, y1, y2 = train_test_split(X, y, shuffle=False, train_size=0.75)

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3, shuffle=False))

#model = LinearRegression()

model.fit(X1, y1)
y_model = model.predict(X2)

#print('mean_squared error: ', abs(1 - mean_squared_error(y2, y_model)))
#print('r2 error: ', r2_score(y2, y_model))

#print(X2)
#print(y2)

from tqdm import tqdm

#############################################
################rolling_run##################
#############################################

#TODO: Automate adjustment of N and G

n=12500
g=1450

print('N: ', n)
print('G: ', g)

rollingX = pd.DataFrame(X1.copy(), columns=['timeStamp'])
cumulativeY = pd.DataFrame(columns=['timeStamp', 'prediction'])
overall = pd.concat([pd.DataFrame(X1).iloc[-g:], pd.DataFrame(y1).iloc[-g::]], axis=1)
overall.columns=['timeStamp', 'prediction']

t = tqdm(total=n, unit='Rows')

for i in range(n):
    rollingX = rollingX.append(rollingX.copy().iloc[len(rollingX.index)-1], ignore_index=True)
    rollingX['timeStamp'].iloc[len(rollingX.index)-1] += 60000
    rollingX = rollingX.drop(rollingX.index[0])
    rollingX.reset_index(drop=True, inplace=True)

    #print(rollingX)

    temp_Y = model.predict(rollingX)
    #print('temp_Y:', temp_Y)
    tempB = pd.concat([rollingX.copy(), pd.DataFrame(temp_Y.copy(), columns=['prediction'])], axis=1)
    #print('tempB:', tempB)


    cumulativeY = cumulativeY.append(tempB.copy().loc[len(tempB) - 1], ignore_index=True)
    overall = overall.append(tempB.copy().loc[len(tempB) - 1], ignore_index=True)

    overall = overall.drop(overall.index[0])
    overall.reset_index(drop=True, inplace=True)

    model.fit(overall['timeStamp'].to_numpy().reshape(-1,1), overall['prediction'])

    #print(overall)

    t.update(1)

t.close()

#############################################

X3 = pd.DataFrame(X2.copy(), columns=['timeStamp'])
X4 = pd.concat([X3, pd.DataFrame(y_model, columns=['prediction'])], axis=1)

X5 = X4.sort_values('timeStamp')

Y3 = pd.DataFrame(X2.copy(), columns=['timeStamp'])
Y4 = pd.concat([Y3, pd.DataFrame(y2, columns=['actual'])], axis=1)

Y5 = Y4.sort_values('timeStamp')

#print(X5)

#print(cumulativeY)

#plt.plot(X4[::5, 0], y2[::5])
plt.plot(Y5['timeStamp'].iloc[:n:], Y5['actual'].iloc[:n:])
plt.plot(cumulativeY['timeStamp'], cumulativeY['prediction'])
plt.legend(['actual', 'prediction'], ncol=1, loc='upper left')
plt.show()

from joblib import dump, load
dump(model, "mymodel.joblib")