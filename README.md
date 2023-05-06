# Supply Chain Demand Forecasting Analysis

This repository provides a framework for performing demand forecasting analysis using ARIMA, ETS, Prophet, and Neural Prophet algorithms. The application is built using Streamlit, providing an easy-to-use environment for testing different models.

## Features

- Load historical demand data from a CSV file
- Perform demand forecasting using ARIMA, ETS, Prophet, and Neural Prophet algorithms
- Automatically select the best model based on evaluation metrics
- Visualize the results at both the individual item and entire dataset levels

## Usage

To use this repository, simply clone the repository and install the required dependencies by running the following command:

pip install -r requirements.txt

Once the dependencies are installed, you can run the application using Streamlit:
streamlit run app.py


From there, you can upload your historical demand data in CSV format and start testing different forecasting algorithms.

## Notes

This repository is designed to be used as a base for exploration, and is not production-ready or fully fault-tolerant. If your series do not fit the minimum requirements for the models used, the application may not function as expected.

## Contributing

Contributions to this repository are welcome and encouraged. If you have any suggestions for improvement or would like to contribute to the project, please submit a pull request.

## License

This repository is licensed under the [MIT License](https://opensource.org/licenses/MIT).

