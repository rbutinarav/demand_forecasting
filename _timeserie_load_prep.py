#%%
def ts_load():

    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    import matplotlib as mpl
    import matplotlib.pyplot as plt   # data visualization

    import os
    
    #import CNH historical demand dataset
    df = pd.read_csv('dfh_thai.csv')

    #create a new dataset with only ITEMNUMBER, DEMANDDATE and DEMANDQUANTITY columns
    df2 = df[['ITEMNUMBER', 'DEMANDDATE', 'DEMANDQUANTITY']]

    #rename the columns ITEMNUMBER to item, DEMANDDATE to date, and DEMANDQUANTITY to demand
    df2.columns = ['item', 'date', 'demand']

    #convert the column date from string to datetime64
    df2['date'] = pd.to_datetime(df2['date'])

    #create a new dataset filtering only the item 84263773 (CNH 4.5T Excavator)
    df3 = df2[df2['item'] == 'MT40007638']

    #drop the column item
    df4 = df3.drop(['item'], axis=1)

    #create a new dataset with date, and the sum of demand for each date
    df4 = df3.groupby('date').sum()
    
    #reset the index
    df4 = df4.reset_index()

    return df4

def ts_load_full():

    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    import matplotlib as mpl
    import matplotlib.pyplot as plt   # data visualization

    import os
    
    #import CNH historical demand dataset
    df = pd.read_csv('dfh_thai.csv')

    #create a new dataset with only ITEMNUMBER, DEMANDDATE and DEMANDQUANTITY columns
    df2 = df[['ITEMNUMBER', 'DEMANDDATE', 'DEMANDQUANTITY']]

    #rename the columns ITEMNUMBER to item, DEMANDDATE to date, and DEMANDQUANTITY to demand
    df2.columns = ['item', 'date', 'demand']

    #convert the column date from string to datetime64
    df2['date'] = pd.to_datetime(df2['date'])

    return df2

def ts_show(df):
    import matplotlib as mpl
    import matplotlib.pyplot as plt   # data visualization
    
    #plot the time series
    df.plot(x='date', y='demand', figsize=(15, 6))

    plt.show()

    return
