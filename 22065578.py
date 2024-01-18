import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


def load_and_process_data(file_path):
    """
    Load and preprocess the dataset.

    Parameters:
    - file_path (str): The path to the dataset CSV file.

    Returns:
    - cleaned_data (pd.DataFrame): Cleaned dataset after handling missing values.
    - original_data (pd.DataFrame): Original dataset before cleaning.
    - transposed_data (pd.DataFrame): Transposed version of the cleaned dataset.
    """

    # Load the data
    original_data = pd.read_csv(file_path)

    # Data preprocessing and handling missing values
    cleaned_data = original_data.dropna()  # Drop rows with any missing values

    # Extract the features for clustering
    features_for_clustering = [
        'Urban population (% of total population) [SP.URB.TOTL.IN.ZS]' ,
        'Tuberculosis case detection rate (%, all forms) [SH.TBS.DTEC.ZS]' ,
        'Trade (% of GDP) [NE.TRD.GNFS.ZS]' ,
        'Total natural resources rents (% of GDP) [NY.GDP.TOTL.RT.ZS]'
    ]

    # Convert selected features to float, excluding non-numeric values
    for feature in features_for_clustering:
        cleaned_data[feature] = pd.to_numeric(cleaned_data[feature] , errors = 'coerce')

    # Drop rows with NaN after conversion
    cleaned_data = cleaned_data.dropna(subset = features_for_clustering)

    # Transpose the cleaned data
    transposed_data = cleaned_data.transpose()

    return cleaned_data , original_data , transposed_data


def fit_model(x , a , b):
    """
    Fit a linear model to the data.

    Parameters:
    - x (array-like): Independent variable data.
    - a (float): Slope of the linear model.
    - b (float): Intercept of the linear model.

    Returns:
    - array-like: Predicted values based on the linear model.
    """
    return a * x + b


def err_ranges(x , params , covariance):
    """
    Calculate the 95% confidence interval for the predicted values of a linear model.

    Parameters:
    - x (array-like): Independent variable data.
    - params (array-like): Parameters (coefficients) of the fitted linear model.
    - covariance (2D array-like): Covariance matrix of the fitted model parameters.

    Returns:
    - tuple: Lower and upper bounds of the 95% confidence interval for the predicted values.
    """
    sigma = np.sqrt(np.diag(covariance))
    lower = fit_model(x , params[0] - 1.96 * sigma[0] , params[1] - 1.96 * sigma[1])
    upper = fit_model(x , params[0] + 1.96 * sigma[0] , params[1] + 1.96 * sigma[1])
    return lower , upper


# Load and process the data
cleaned_data , original_data , transposed_data = \
    load_and_process_data('420ba903-f954-401f-ab90-f5b19238e019_Data.csv')

# Clustering
num_clusters = 3  # adjust as needed
features_for_clustering = ['Urban population (% of total population) [SP.URB.TOTL.IN.ZS]' ,
                           'Tuberculosis case detection rate (%, all forms) [SH.TBS.DTEC.ZS]' ,
                           'Trade (% of GDP) [NE.TRD.GNFS.ZS]' ,
                           'Total natural resources rents (% of GDP) [NY.GDP.TOTL.RT.ZS]']

# Extract the features for clustering
X = cleaned_data[features_for_clustering].values

# Normalize the data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Perform clustering on normalized data
kmeans = KMeans(n_clusters = num_clusters , random_state = 42)
cleaned_data['Cluster'] = kmeans.fit_predict(X_normalized)

# Calculate silhouette score for clustering
silhouette_avg = silhouette_score(X_normalized ,
                                  cleaned_data['Cluster'])
print(f"Silhouette Score: {silhouette_avg}")

# Plot cluster membership and centers
cluster_centers_normalized = kmeans.cluster_centers_

# Back scale the cluster centers to the original scale
cluster_centers_original = scaler.inverse_transform(cluster_centers_normalized)

plt.scatter(cleaned_data['Trade (% of GDP) [NE.TRD.GNFS.ZS]'] ,
            cleaned_data['Total natural resources rents (% of GDP) [NY.GDP.TOTL.RT.ZS]'] ,
            c = cleaned_data['Cluster'] , cmap = 'viridis')
plt.scatter(cluster_centers_original[: , 2] ,
            cluster_centers_original[: , 3] , s = 300 , c = 'red' , marker = 'X')
plt.xlabel('Trade (% of GDP)')
plt.ylabel('Total natural resources rents (% of GDP)')
plt.title('Clustering Results')
plt.show()

# Model fitting
y_data = \
    cleaned_data['Total natural resources rents (% of GDP) [NY.GDP.TOTL.RT.ZS]'].astype(float)
x_data = \
    cleaned_data['Trade (% of GDP) [NE.TRD.GNFS.ZS]'].astype(float)

# Fit the model
params , covariance = curve_fit(fit_model , x_data , y_data)

x_future = np.arange(min(x_data) , max(x_data) + 10 , 1)
y_pred = fit_model(x_future , *params)

lower , upper = err_ranges(x_future , params , covariance)

# Plot the fitting results
plt.scatter(x_data , y_data , label = 'Actual Data')
plt.plot(x_future , y_pred , label = 'Fitted Model' , color = 'red')
plt.fill_between(x_future , lower , upper ,
                 color = 'gray' , alpha = 0.2 , label = 'Confidence Interval')
plt.xlabel('Trade (% of GDP)')
plt.ylabel('Total natural resources rents (% of GDP)')
plt.title('Model Fitting Results')
plt.legend()
plt.show()
