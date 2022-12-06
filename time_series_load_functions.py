#%%
def ts_load(filename):

    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    
    #import historical demand dataset
    df = pd.read_csv(filename)

    #create a new dataset with only ITEMNUMBER, DEMANDDATE and DEMANDQUANTITY columns
    df2 = df[['ITEMNUMBER', 'DEMANDDATE', 'DEMANDQUANTITY']]

    #rename the columns ITEMNUMBER to item, DEMANDDATE to date, and DEMANDQUANTITY to demand
    df2.columns = ['item', 'date', 'demand']

    #convert the column date from string to datetime64
    df2['date'] = pd.to_datetime(df2['date'])
    
    return df2


def prepare_dataset(filename, item="All", rows="All"):   #calls the ts_load function

    import streamlit as st
    import numpy as np
 
    historical_demand = ts_load(filename) #assumes the file has the following structure: item, date, demand
    #limit the dataset to the rows = limit
    if rows !="All":
        historical_demand = historical_demand.head(rows)

    if item !="All":
    #reset the index
        historical_demand = historical_demand[historical_demand['item'] == item].reset_index(drop=True)
    
    #filter the dataset 

    #fill all the data from the earliest date to the latest date
    #add a 0 to all the dates that do not have a demand
    #create a new dataset with all the combinations of item and date for the time range
    historical_demand = historical_demand.set_index('date').groupby('item').resample('D').sum().fillna(0).reset_index()

    #filter the dataset to only include rows with date values not null
    historical_demand = historical_demand[historical_demand['date'].notnull()]

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
    #set the index to year_month
    #historical_demand_monthly = historical_demand_monthly.set_index('year_month')

    #create a new dataset totaling the demand for each quarter and item, reset index, rename columns
    historical_demand_quarterly = historical_demand.groupby(['item', 'year_quarter'])['demand'].sum().reset_index().rename(columns={'demand': 'demand'})
    #set the index to year_quarter
    #historical_demand_quarterly = historical_demand_quarterly.set_index('year_quarter')

    #create a new dataset totaling the demand for each year and item, reset index, rename columns
    historical_demand_yearly = historical_demand.groupby(['item', 'year'])['demand'].sum().reset_index().rename(columns={'demand': 'demand'})
    #set the index to year
    #historical_demand_yearly = historical_demand_yearly.set_index('year')

    #PERSIST IN SESSION STATE VARIABLES
    st.session_state.hdm=historical_demand_monthly
    st.session_state.hdq=historical_demand_quarterly
    st.session_state.hdy=historical_demand_yearly

    return