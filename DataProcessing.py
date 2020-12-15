import pandas as pd
import numpy as np
from tqdm import tqdm

ticker = 'TSLA'

print('opening "', ticker, '"...')
df = pd.read_csv((ticker + '.csv'), index_col=False) #doesn't pull index column as it was causing issues for an unknown reason
print('done.')



print('formatting...')

df['avg'] = (df['open'] + df['close']) / 2  #adds an average columns

df = df.drop(['timestamp', 'open', 'high', 'low', 'close'], axis=1) #removes excess columns not needed for out use case
df = df.drop(df.columns[0], axis=1)

print('done')



processed = pd.DataFrame()
#processed.columns = df.columns


print('Filling missing data...')

totalRows = int((df['UTCtimestamps'].iloc[-1] - df['UTCtimestamps'].loc[0]) // 60000) #quickly calculates how many totals rows should be in processed dataframe

t = tqdm(total=totalRows, unit='Rows')  #creates progress bar object

for i in range(len(df.index) - 1):

    processed = processed.append(df.iloc[i]) #includes working line in processed
    t.update(1) #increments number of completed rows in progress bar

    j = 1

    while int(df['UTCtimestamps'].iloc[i] + (60000 * j)) != int(df['UTCtimestamps'].iloc[i+1]): #while there are missing time slots (minutes)

        temp = pd.DataFrame()   #clear temp

        temp = temp.append(df.iloc[i])  #copy last known data
        #temp.columns = df.columns

        temp['UTCtimestamps'] += (60000 * j)    #update time for current fill row
        temp['volume'] = 0  #adjust volume and related values
        temp['vwap'] = 0


        processed = processed.append(temp)  #include finished row in processed data frame
        t.update(1) #increments number of completed rows in progress bar
        j += 1

processed = processed.append(df.iloc[-1]) #adds last row
t.update(1)
t.close()
print('done.')


print('Writing to file...')
processed.to_csv((ticker + '_processed.csv'), mode='w') #outputs processed to csv
print('done.')
#df.to_csv(file, mode=mode)