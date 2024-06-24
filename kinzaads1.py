"""
KinzaADS1.ipynb
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define the functions from the provided code
def error_prop(x, func, parameter, covar):
    var = np.zeros_like(x)  # initialise variance vector
    for i in range(len(parameter)):
        deriv1 = deriv(x, func, parameter, i)
        for j in range(len(parameter)):
            deriv2 = deriv(x, func, parameter, j)
            var = var + deriv1 * deriv2 * covar[i, j]
    sigma = np.sqrt(var)
    return sigma


def deriv(x, func, parameter, ip):
    scale = 1e-6  # scale factor to calculate the derivative
    delta = np.zeros_like(parameter, dtype=float)
    val = scale * np.abs(parameter[ip])
    delta[ip] = val
    diff = 0.5 * (func(x, *parameter + delta) - func(x, *parameter - delta))
    dfdx = diff / val
    return dfdx


def plot_countries(df):
    # Convert the '2020 [YR2010]' column to numeric, coercing errors
    df['2020 [YR2020]'] = pd.to_numeric(df['2020 [YR2020]'], errors='coerce')
    df_filtered = df[df['Series Code'] == 'EN.ATM.CO2E.KD.GD'][['Country Name', '2020 [YR2020]']]

    # Sort the data to get the top 10 countries
    df_top10 = df_filtered.nlargest(10, '2020 [YR2020]')

    # Plot the data
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    sns.set_palette("viridis")

    barplot = sns.barplot(x='2020 [YR2020]', y='Country Name', data=df_top10, edgecolor='w')

    # Add titles and labels
    plt.title('Top 10 Countries for CO2 Emissions per GDP in 2020', fontsize=16, fontweight='bold')
    plt.ylabel('Country', fontsize=14)

    # Add data labels
    for index, value in enumerate(df_top10['2020 [YR2020]']):
        plt.text(value, index, f'{value:.2f}', va='center', ha='left', fontsize=12, color='black')

    # Adjust layout
    plt.tight_layout()
    return plt.show()



def plot_series_for_countries(file_path, series_code, countries):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Filter the dataset for the specified series code and countries
    filtered_data = data[(data['Series Code'] == series_code) & (data['Country Name'].isin(countries))]

    # Set the country names as index
    filtered_data.set_index('Country Name', inplace=True)

    # Drop non-year columns and transpose the dataframe for plotting
    year_columns = filtered_data.columns[4:]
    filtered_data = filtered_data[year_columns].transpose()

    # Convert the data to numeric, setting errors='coerce' to handle non-numeric data
    filtered_data = filtered_data.apply(pd.to_numeric, errors='coerce')

    # Convert the column names to datetime format for better plotting
    filtered_data.index = pd.to_datetime(filtered_data.index.str.extract(r'(\d{4})')[0], format='%Y')

    # Plot the data
    filtered_data.plot(figsize=(12, 8))
    plt.title(f'Chart for series {series_code}')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend(title='Country')
    plt.grid(True)
    return plt.show()

# Function to plot scatter matrix for specified years with simplified year labels and enhanced title
def plot_scatter_matrix(dataframe, start_year, end_year):
    # Extract the year columns
    year_columns = [str(year) for year in range(int(start_year.split()[0]), int(end_year.split()[0]) + 1)]

    # Rename the columns to just the year
    dataframe = dataframe.rename(columns={col: col.split()[0] for col in dataframe.columns if col.split()[0] in year_columns})

    # Select only the numerical columns for the specified years
    numerical_df = dataframe.loc[:, year_columns]

    # Convert the columns to numeric, forcing errors to NaN
    numerical_df = numerical_df.apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values
    numerical_df = numerical_df.dropna()

    # Plot the scatter matrix
    scatter_matrix(numerical_df, alpha=0.2, figsize=(20, 20), diagonal='kde')
    plt.suptitle('Scatter Matrix of CO2 Emissions Data from ' + start_year.split()[0] + ' to ' + end_year.split()[0], fontsize=20, fontweight='bold')
    return plt.show()


def two_lines():
    # Select the data for count
    country_data = df[df['Country Name'] == 'United Kingdom'].iloc[:, 4:]

    # Replace '..' with NaN and convert to numeric
    country_data = country_data.replace('..', np.nan).astype(float)

    # Drop NaN values
    country_data = country_data.dropna(axis=1)

    # Extract years from column names
    years = np.array([int(col.split()[0]) for col in country_data.columns])

    # Convert country_data to a 1D array
    emissions = country_data.values.flatten()

    # Check the shape of the arrays
    print('Shape of years array:', years.shape)
    print('Shape of emissions array:', emissions.shape)

    # Ensure the arrays are aligned
    if len(years) != len(emissions):
        min_length = min(len(years), len(emissions))
        years = years[:min_length]
        emissions = emissions[:min_length]

    # Define the models
    def exponential_growth(x, a, b):
        return a * np.exp(b * x)

    def logistic_function(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def polynomial_model(x, a, b, c):
        return a * x**2 + b * x + c

    # Fit the models
    popt_exp, _ = curve_fit(exponential_growth, years, emissions)
    popt_log, _ = curve_fit(logistic_function, years, emissions, p0=[max(emissions), 0.1, np.mean(years)])
    popt_poly, _ = curve_fit(polynomial_model, years, emissions)

    # Generate future years for prediction
    future_years = np.arange(years.min(), years.max() + 11)

    # Calculate predictions
    pred_exp = exponential_growth(future_years, *popt_exp)
    pred_log = logistic_function(future_years, *popt_log)
    pred_poly = polynomial_model(future_years, *popt_poly)

    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.scatter(years, emissions, label='Actual Data')
    plt.plot(future_years, pred_exp, label='Exponential Growth')
    plt.plot(future_years, pred_log, label='Logistic Function')
    plt.plot(future_years, pred_poly, label='Polynomial Model')
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions (kg per 2015 US$ of GDP)')
    plt.title('CO2 Emissions in Afghanistan: Actual Data and Model Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()

    print('Model parameters:')
    print('Exponential Growth: a = {:.4f}, b = {:.4f}'.format(*popt_exp))
    print('Logistic Function: L = {:.4f}, k = {:.4f}, x0 = {:.4f}'.format(*popt_log))
    print('Polynomial Model: a = {:.4f}, b = {:.4f}, c = {:.4f}'.format(*popt_poly))

def predict_future_co2(country_name, series_name):
    # Filter the data to only include the United Kingdom and the series 'CO2 emissions (kt)'
    # uk_co2_data = df[(df['Country Name'] == 'United Kingdom') & (df['Series Name'] == 'CO2 emissions (kt)')]
    co2_data = df[(df['Country Name'] == country_name) & (df['Series Name'] == series_name)]

    # Extract year and values, converting year from string to integer and values to float
    years = np.array([int(col.split(' ')[0]) for col in co2_data.columns[4:]])
    values = np.array([float(co2_data.iloc[0, i]) for i in range(4, co2_data.shape[1])])

    # Fit a polynomial regression model
    poly_degree = 3
    poly_features = PolynomialFeatures(degree=poly_degree)
    X_poly = poly_features.fit_transform(years.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, values)

    # Predict future years
    future_years = np.array([2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])
    future_years_poly = poly_features.transform(future_years.reshape(-1, 1))
    predictions = model.predict(future_years_poly)

    # Calculate confidence intervals
    residuals = values - model.predict(X_poly)
    std_error = np.std(residuals)
    confidence_interval = 1.96 * std_error

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(years, values, color='blue', label='Actual Data')
    plt.plot(years, model.predict(X_poly), color='red', label='Polynomial Fit Degree ' + str(poly_degree))
    plt.plot(future_years, predictions, color='green', label='Future Predictions')
    plt.fill_between(future_years, (predictions - confidence_interval), (predictions + confidence_interval), color='yellow', alpha=0.5, label='95% Confidence Interval')
    plt.title(f'Future CO2 Emissions Predictions for the {country_name}')
    plt.xlabel('Year')
    plt.ylabel('series_name')
    plt.legend()
    return plt.show()

def plot_series_with_kmeans(df, series_code1, series_code2):

    df.replace('..', np.nan, inplace=True)

    # Filter the data for the given series codes
    df_filter = df[df['Series Code'].isin([series_code1, series_code2])]

    # Extract the year columns
    year_cols = [col for col in df_filter.columns if 'YR' in col]

    # Melt the dataframe to have years and values in separate columns
    df_new = df_filter.melt(id_vars=['Country Name', 'Series Code'], value_vars=year_cols, var_name='Year', value_name='Value')

    # Drop rows with missing values
    df_new.dropna(inplace=True)

    # Convert 'Year' to numerical format and 'Value' to float
    df_new['Year'] = df_new['Year'].str.extract('(\d{4})').astype(int)
    df_new['Value'] = df_new['Value'].astype(float)

    # Prepare data for KMeans clustering
    X = df_new[['Year', 'Value']].values

    # Determine the number of clusters using the elbow method
    wcss = [KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(X).inertia_ for i in range(1, 11)]

    # Find the optimal number of clusters based on the elbow method
    clusters = np.argmax(np.diff(wcss)) + 1

    # Apply KMeans clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    df_new['Cluster'] = kmeans.fit_predict(X)

    # Filter the data for the specific country and series code
    country_series_data = df_new[(df_new['Country Name'] == 'India') & (df_new['Series Code'] == series_code1)]

    # Sort the data by year
    country_series_data.sort_values('Year', inplace=True)

    # Plot the cluster data
    plt.figure(figsize=(14, 7))
    plt.scatter(country_series_data['Year'], country_series_data['Value'], c=country_series_data['Cluster'], cmap='viridis', marker='o')
    plt.title(f'Cluster Data for United Kingdom {series_code1}')
    plt.xlabel('Year')
    plt.ylabel('GNP (Current US$)')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    return plt.show()


if __name__ == "__main__":
    # Load the dataset
    file_path = 'f6373d02-ee13-460d-aa53-ee84421a56b4.csv'
    df = pd.read_csv(file_path, encoding='ascii')

    df.head()

    # Convert the year columns to numeric, coerce errors to NaN
    for col in df.columns[4:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Summary statistics
    df.describe()

    # Extract distinct Series Codes
    unique_series_codes = df['Series Code'].unique()
    print(unique_series_codes)

    # Extract distinct Series Names
    unique_series_names = df['Series Name'].unique()
    print(unique_series_names)

    plot_countries(df)

    # Example usage
    series_code = 'EN.ATM.CO2E.KD.GD'
    countries = ['Ukraine', 'Syrian Arab Republic', 'Mongolia', 'Turkmenistan', 'Iran, Islamic Rep.', 'Kyrgyz Republic', 'South Africa', 'Bosnia and Herzegovina', 'Russian Federation', 'Viet Nam']
    plot_series_for_countries(file_path, series_code, countries)

    # normal_range_line(df, 'EN.ATM.CO2E.KD.GD', 'EN.ATM.CO2E.PC')

    predict_future_co2('United Kingdom', 'CO2 emissions (kt)')

    plot_series_with_kmeans(df, 'EN.ATM.CO2E.KD.GD', 'EN.ATM.CO2E.PC')

    # Call the function to plot scatter matrix for '2010 [YR2010]' to '2020 [YR2020]'
    plot_scatter_matrix(df, '2010 [YR2010]', '2020 [YR2020]')

