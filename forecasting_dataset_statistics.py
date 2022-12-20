def statistics(dataset, dataset_statistics=None, display=True):

    import pandas as pd
    import streamlit as st

    #calculate first and last year_month
    first_year_month = dataset['year_month'].min()
    last_year_month = dataset['year_month'].max()

    #calculate number of items
    items = dataset['item'].unique()




    if dataset_statistics is None:

        #create a new dataframe called items statistics with the columns: item, total demand, number of months with positive demand, if the item has a positive demand in the last 12 months
        items_statistics = pd.DataFrame(columns=['total demand', 'months with demand', 'last 12 months with positive demand'])

        #calculate the total demand and at to the dataframe
        items_statistics['total demand'] = dataset.groupby('item')['demand'].sum()

        #calculate the number of months with positive demand and at to the dataframe
        items_statistics['months with demand'] = dataset[dataset['demand'] > 0].groupby('item')['demand'].count()

        #calculate the number of items with more than 12 months with positive demand
        items_with_more_than_12_months_with_positive_demand = items_statistics[items_statistics['months with demand'] > 12]

        #create a list of 12 year_months from last_year_month - 12 months to last_year_month
        #convert last_year_month to datetime
        last_year_month_dt = pd.to_datetime(last_year_month, format='%Y%m')
        #create a list of 12 year_months from last_year_month - 12 months to last_year_month
        year_months = [last_year_month_dt - pd.DateOffset(months=x) for x in range(0,12)]
        #convert year_months to string
        year_months = [x.strftime('%Y%m') for x in year_months]
        #show the list
        #st.write(year_months)
        #last 12 months from the dataset
        last_12_months = dataset[dataset['year_month'].isin(year_months)]

        #calculate the number of items with positive demand in the last 12 months
        items_statistics['last 12 months with positive demand'] = last_12_months[last_12_months['demand'] > 0].groupby('item')['demand'].count() > 0

        #order items_statistics by total demand
        items_statistics = items_statistics.sort_values(by=['total demand'], ascending=False)

        #move index to item column
        items_statistics['item'] = items_statistics.index

        #move item as first column
        items_statistics = items_statistics[['item', 'total demand', 'months with demand', 'last 12 months with positive demand']]

        #reset idex
        items_statistics = items_statistics.reset_index(drop=True)
    
    else:
        items_statistics = dataset_statistics


    if display:
        #display number of records
        st.write('Number of records: ', len(dataset))

        #display number of items
        st.write('Number of items: ', len(items))

        #display the first amd last date
        st.write('Range period : ', first_year_month, '- ', last_year_month)

        #show total demand
        st.write('Total demand: ', dataset['demand'].sum())

        #Total demand in the last 12 months
        #sum the demand for items with "last 12 months with positive demand" = True
        total_demand_last_12_months = items_statistics[items_statistics['last 12 months with positive demand'] == True]['total demand'].sum()
        st.write('Total demand in the last 12 months: ', total_demand_last_12_months) 

        #display the number of items with positive demand in the last 12 months, and % on total items
        st.write('Items with demand in the last 12 months: ', len(items_statistics[items_statistics['last 12 months with positive demand'] == True]), ' - ', round(len(items_statistics[items_statistics['last 12 months with positive demand'] == True])/len(items_statistics)*100, 2), '%')
        
        #display number of items with more than 12 months with positive demand, and % on total items
        #st.write('Items with more than 12 months with positive demand: ', len(items_with_more_than_12_months_with_positive_demand), ' - ', round(len(items_with_more_than_12_months_with_positive_demand)/len(items_statistics)*100, 2), '%')
        
        #display items_statistics
        st.write(items_statistics)

    #end the function
    return items_statistics
