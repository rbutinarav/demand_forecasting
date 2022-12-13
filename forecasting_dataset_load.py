#%%
def ts_load(filename):

    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import os
    import streamlit as st

    df = None

    #check if it exists before opening it
    if os.path.exists(filename)==False:
            #ask the user to upload the file
            st.write('Please upload a file with the historical demand, three columns are expected:')
            st.write('ITEMNUMBER, DEMANDDATE, DEMANDQUANTITY')
            uploaded_file = st.file_uploader("Choose a file", type="csv")
                
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
    else:        
        df = pd.read_csv(filename)

    if df is None:
        return

    #create a new dataset with only ITEMNUMBER, DEMANDDATE and DEMANDQUANTITY columns
    df2 = df[['ITEMNUMBER', 'DEMANDDATE', 'DEMANDQUANTITY']]

    #rename the columns ITEMNUMBER to item, DEMANDDATE to date, and DEMANDQUANTITY to demand
    df2.columns = ['item', 'date', 'demand']

    #convert the column date from string to datetime64
    df2['date'] = pd.to_datetime(df2['date'])
      
    return df2


def prepare_dataset(filename, item="All", rows="All"):   #calls the ts_load function
#creates a dataset called historical_demand_monthly, add demand 0 for the entire time range

    import streamlit as st
    import numpy as np

    historical_demand = ts_load(filename) #assumes the file has the following structure: item, date, demand
 
    if historical_demand is None:
        return

    #limit the dataset to the rows = limit
    if rows !="All":
        historical_demand = historical_demand.head(rows)

    if item !="All":
        historical_demand = historical_demand[historical_demand['item'] == item].reset_index(drop=True)

    #get the last data in the dataset
    last_date = historical_demand['date'].max()

    #get the first date in the dataset
    first_date = historical_demand['date'].min()
    
    #for each item add a row with last_date and demand = 0
    historical_demand = historical_demand.append(historical_demand.groupby('item').tail(1).assign(date=last_date, demand=0))

    #for each item add a row with first_date and demand = 0
    historical_demand = historical_demand.append(historical_demand.groupby('item').head(1).assign(date=first_date, demand=0))

    #create a new dataset called historical demand with all the combination of item and date for the time range
    historical_demand = historical_demand.set_index('date').groupby('item').resample('M').sum().fillna(0).reset_index()

    #TO DEBUG, JUST FOR 1 SINGLE ITEM
    if item !='All' and st.session_state.show_debug:
        st.write('Historical Demand Dataset')
        st.write(historical_demand)

    ##ADD YEAR, MONTH, QUARTER

    #add year column defined as integer
    historical_demand['year'] = historical_demand['date'].dt.year.astype(str)

    #add year-month column and define it as integer
    historical_demand['year_month'] = historical_demand['date'].dt.strftime('%Y%m').astype(str)

    #create a new column called year-quarter dividing the month by 3 and rounding up
    historical_demand['year_quarter'] = (historical_demand['date'].dt.month/3).apply(np.ceil).astype(str)


    #CREATE NEW DATASETS GROUPED BY ITEM, MONTH, QUARTER, YEAR

    #create a new dataset totaling the demand for each month and item, reset index, rename columns
    historical_demand_monthly = historical_demand.groupby(['item', 'year_month'])['demand'].sum().reset_index().rename(columns={'demand': 'demand'})
    
    return historical_demand_monthly