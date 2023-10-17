import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
from math import ceil
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.patheffects as pe
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, FunctionTransformer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from category_encoders import JamesSteinEncoder
from sklearn.experimental import enable_halving_search_cv
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime
import time
from joblib import dump, load
import pytz
import os
import re
import collections
from scipy.stats import skew, kurtosis, iqr
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings


def get_unique(df, n=20, sort='none', list=True, strip=False, count=False, percent=False, plot=False, cont=False):
    """
    Version 0.2
    Obtains unique values of all variables below a threshold number "n", and can display counts or percents
    Parameters:
    - df: dataframe that contains the variables you want to analyze
    - n: int (default is 20). Maximum number of unique values to consider (avoid iterating continuous data)
    - sort: str, optional (default='none'). Determines the sorting of unique values:
        'none' will keep original order,
        'name' will sort alphabetically/numerically,
        'count' will sort by count of unique values (descending)
    - list: boolean, optional (default=True). Shows the list of unique values
    - strip: boolean, optional (default=False). True will remove single quotes in the variable names
    - count: boolean, optional (default=False). True will show counts of each unique value
    - percent: boolean, optional (default=False). True will show percentage of each unique value
    - plot: boolean, optional (default=False). True will show a basic chart for each variable
    - cont: boolean, optional (default=False). True will analyze variables over n as continuous

    Returns: None
    """
    # Calculate # of unique values for each variable in the dataframe
    var_list = df.nunique(axis=0)

    # Iterate through each categorical variable in the list below n
    print(f"\nCATEGORICAL: Variables with unique values equal to or below: {n}")
    for i in range(len(var_list)):
        var_name = var_list.index[i]
        unique_count = var_list[i]

        # If unique value count is less than n, get the list of values, counts, percentages
        if unique_count <= n:
            number = df[var_name].value_counts(dropna=False)
            perc = round(number / df.shape[0] * 100, 2)
            # Copy the index to a column
            orig = number.index
            # Strip out the single quotes
            name = [str(n) for n in number.index]
            name = [n.strip('\'') for n in name]
            # Store everything in dataframe uv for consistent access and sorting
            uv = pd.DataFrame({'orig': orig, 'name': name, 'number': number, 'perc': perc})

            # Sort the unique values by name or count, if specified
            if sort == 'name':
                uv = uv.sort_values(by='name', ascending=True)
            elif sort == 'count':
                uv = uv.sort_values(by='number', ascending=False)
            elif sort == 'percent':
                uv = uv.sort_values(by='perc', ascending=False)

            # Print out the list of unique values for each variable
            if list:
                print(f"\n{var_name} has {unique_count} unique values:\n")
                for w, x, y, z in uv.itertuples(index=False):
                    # Decide on to use stripped name or not
                    if strip:
                        w = x
                    # Put some spacing after the value names for readability
                    w_str = str(w)
                    w_pad_size = uv.name.str.len().max() + 7
                    w_pad = w_str + " " * (w_pad_size - len(w_str))
                    y_str = str(y)
                    y_pad_max = uv.number.max()
                    y_pad_max_str = str(y_pad_max)
                    y_pad_size = len(y_pad_max_str) + 3
                    y_pad = y_str + " " * (y_pad_size - len(y_str))
                    if count and percent:
                        print("\t" + str(w_pad) + str(y_pad) + str(z) + "%")
                    elif count:
                        print("\t" + str(w_pad) + str(y))
                    elif percent:
                        print("\t" + str(w_pad) + str(z) + "%")
                    else:
                        print("\t" + str(w))

            # Plot countplot if plot=True
            if plot:
                print("\n")
                if strip:
                    if sort == 'count':
                        sns.barplot(data=uv, x='name', y='number', order=uv.sort_values('number', ascending=False).name)
                    else:
                        sns.barplot(data=uv, x=uv.loc[0], y='number', order=uv.sort_values('name', ascending=True).name)
                else:
                    if sort == 'count':
                        sns.barplot(data=uv, x='orig', y='number', order=uv.sort_values('number', ascending=False).orig)
                    else:
                        sns.barplot(data=uv, x='orig', y='number', order=uv.sort_values('orig', ascending=True).orig)
                plt.title(var_name)
                plt.xlabel('')
                plt.ylabel('')
                plt.xticks(rotation=45)
                plt.show()

    if cont:
        # Iterate through each categorical variable in the list below n
        print(f"\nCONTINUOUS: Variables with unique values greater than: {n}")
        for i in range(len(var_list)):
            var_name = var_list.index[i]
            unique_count = var_list[i]

            if unique_count > n:
                print(f"\n{var_name} has {unique_count} unique values:\n")
                print(var_name)
                print(df[var_name].describe())

                # Plot countplot if plot=True
                if plot:
                    print("\n")
                    sns.histplot(data=df, x=var_name)
                    # plt.title(var_name)
                    # plt.xlabel('')
                    # plt.ylabel('')
                    # plt.xticks(rotation=45)
                    plt.show()


def plot_charts(df, plot_type='both', n=10, ncols=3, fig_width=20, subplot_height=4, rotation=45, strip=False,
                cat_cols=None, cont_cols=None, dtype_check=True, sample_size=None):
    """
    Version 0.2
    Plot barplots for categorical columns, or histograms for continuous columns, in a grid of subplots.

    Parameters:
    - df: dataframe that contains the variables you want to analyze
    - plot_type: string, optional (default='both'). Type of charts to plot: 'cat' for categorical, 'cont' for
        continuous, 'both' for both
    - n: int (default=20). Threshold of unique values for categorical (equal or below) vs. continuous (above)
    - ncols: int, optional (default=3). The number of columns in the subplot grid.
    - fig_width: int, optional (default=20). The width of the entire plot figure (not the subplot width)
    - subplot_height: int, optional (default=4). The height of each subplot.
    - rotation: int, optional (default=45). The rotation of the x-axis labels.
    - strip: boolean, optional (default=False). Will strip single quotes from ends of column names
    - cat_cols: list, optional (default=None). A list of column names to treat as categorical variables. If not
        provided, inferred based on the unique count.
    - cont_cols: list, optional (default=None). A list of column names to treat as continuous variables. If not
        provided, inferred based on the unique count.
    - dtype_check: boolean, optional (default=True). If True, consider only numeric types (int64, float64) for
        continuous variables.
    - sample_size: float or int, optional (default=None). If provided and less than 1, the fraction of the data to
        sample. If greater than or equal to 1, the number of samples to draw.

    Returns: None
    """

    # Helper function to plot continuous variables
    def plot_continuous(df, cols, ncols, fig_width, subplot_height, strip, sample_size):
        nrows = ceil(len(cols) / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(fig_width, nrows * subplot_height), constrained_layout=True)
        axs = np.array(axs).ravel()  # Ensure axs is always a 1D numpy array

        # Loop through all continuous columns
        for i, col in enumerate(cols):
            if sample_size:
                sample_count = int(len(df[col].dropna()) * sample_size)  # Calculate number of samples
                data = df[col].dropna().sample(sample_count)
            else:
                data = df[col].dropna()

            if strip:
                sns.stripplot(x=data, ax=axs[i])
            else:
                sns.histplot(data, ax=axs[i], kde=False)

            axs[i].set_title(f'{col}', fontsize=20)
            axs[i].tick_params(axis='x', rotation=rotation)
            axs[i].set_xlabel('')

        # Remove empty subplots
        for empty_subplot in axs[len(cols):]:
            empty_subplot.remove()

    # Helper function to plot categorical variables
    def plot_categorical(df, cols, ncols, fig_width, subplot_height, rotation, sample_size):
        nrows = ceil(len(cols) / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(fig_width, nrows * subplot_height), constrained_layout=True)
        axs = np.array(axs).ravel()  # Ensure axs is always a 1D numpy array

        # Loop through all categorical columns
        for i, col in enumerate(cols):
            uv = df[col].value_counts().reset_index().rename(columns={col: 'name', 'count': 'number'})
            uv['perc'] = uv['number'] / uv['number'].sum()

            if sample_size:
                uv = uv.sample(sample_size)

            sns.barplot(data=uv, x='name', y='number', order=uv.sort_values('number', ascending=False).name, ax=axs[i])

            axs[i].set_title(f'{col}', fontsize=20)
            axs[i].tick_params(axis='x', rotation=rotation)
            axs[i].set_ylabel('Count')
            axs[i].set_xlabel('')

        # Remove empty subplots
        for empty_subplot in axs[len(cols):]:
            empty_subplot.remove()

    # Compute unique counts and identify categorical and continuous variables
    unique_count = df.nunique()
    if cat_cols is None:
        cat_cols = unique_count[unique_count <= n].index.tolist()
    if cont_cols is None:
        cont_cols = unique_count[unique_count > n].index.tolist()

    if dtype_check:
        cont_cols = [col for col in cont_cols if df[col].dtype in ['int64', 'float64']]

    if plot_type == 'cat' or plot_type == 'both':
        plot_categorical(df, cat_cols, ncols, fig_width, subplot_height, rotation, sample_size)
    if plot_type == 'cont' or plot_type == 'both':
        plot_continuous(df, cont_cols, ncols, fig_width, subplot_height, strip, sample_size)


def plot_charts_with_hue(df, plot_type='both', n=10, ncols=3, fig_width=20, subplot_height=4, rotation=0,
                         cat_cols=None, cont_cols=None, dtype_check=True, sample_size=None, hue=None, color_discrete_map=None, normalize=False, kde=False, multiple='layer'):
    """
    Version 0.1
    Plot barplots for categorical columns, or histograms for continuous columns, in a grid of subplots.
    Option to pass a 'hue' parameter to dimenions the plots by a variable/column of the dataframe.

    Parameters:
    - df: dataframe that contains the variables you want to analyze
    - plot_type: string, optional (default='both'). Type of charts to plot: 'cat' for categorical, 'cont' for continuous, 'both' for both
    - n: int (default=20). Threshold of unique values for categorical (equal or below) vs. continuous (above)
    - ncols: int, optional (default=3). The number of columns in the subplot grid.
    - fig_width: int, optional (default=20). The width of the entire plot figure (not the subplot width)
    - subplot_height: int, optional (default=4). The height of each subplot.
    - rotation: int, optional (default=45). The rotation of the x-axis labels.
    - cat_cols: list, optional (default=None). A list of column names to treat as categorical variables. If not provided, inferred based on the unique count.
    - cont_cols: list, optional (default=None). A list of column names to treat as continuous variables. If not provided, inferred based on the unique count.
    - dtype_check: boolean, optional (default=True). If True, consider only numeric types (int64, float64) for continuous variables.
    - sample_size: float or int, optional (default=None). If provided and less than 1, the fraction of the data to sample. If greater than or equal to 1, the number of samples to draw.
    - hue: string, optional (default=None). Name of the column to dimension by passing as 'hue' to the Seaborn charts.
    - color_discrete_map: name of array or array, optional (default=None). Pass a color mapping for the values in the 'hue' variable.
    - normalize: boolean, optional (default=False). Set to True to normalize categorical plots and see proportions instead of counts
    - kde: boolean, optional (default=False). Set to show KDE line on continuous countplots
    - multiple: 'layer', 'dodge', 'stack', 'fill', optional (default='layer'). Choose how to handle hue variable when plotted on countplots
    Returns: None
    """
    def plot_categorical(df, cols, ncols, fig_width, subplot_height, rotation, sample_size, hue, color_discrete_map, normalize):
        if sample_size:
            df = df.sample(sample_size)
        nplots = len(cols)
        nrows = nplots//ncols
        if nplots % ncols:
            nrows += 1

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, nrows*subplot_height), constrained_layout=True)
        if isinstance(axs, np.ndarray):
            if len(axs.shape) > 1:
                axs = axs.ravel()
        else:
            axs = [axs]

        for i, col in enumerate(cols):
            if normalize:
                # Normalize the counts
                df_copy = df.copy()
                data = df_copy.groupby(col)[hue].value_counts(normalize=True).rename('proportion').reset_index()
                sns.barplot(data=data, x=col, y='proportion', hue=hue, palette=color_discrete_map, ax=axs[i])
                axs[i].set_ylabel('Proportion', fontsize=12)
            else:
                order = df[col].value_counts().index
                sns.countplot(data=df, x=col, hue=hue, palette=color_discrete_map, ax=axs[i], order=order)
                axs[i].set_ylabel('Count', fontsize=12)
            axs[i].set_xlabel(' ', fontsize=12)
            axs[i].set_title(col, fontsize=16, pad=10)
            axs[i].tick_params(axis='x', rotation=rotation)

        # Remove empty subplots
        for empty_subplot in axs[nplots:]:
            fig.delaxes(empty_subplot)

    def plot_continuous(df, cols, ncols=3, fig_width=15, subplot_height=5, sample_size=None, hue=None, color_discrete_map=None, kde=False, multiple=multiple):
        if sample_size:
            df = df.sample(sample_size)
        nplots = len(cols)
        nrows = nplots//ncols
        if nplots % ncols:
            nrows += 1
            
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, nrows * subplot_height), constrained_layout=True)
        if isinstance(axs, np.ndarray):
            if len(axs.shape) > 1:
                axs = axs.ravel()
        else:
            axs = [axs]
        
        for i, col in enumerate(cols):
            if hue is not None:
                sns.histplot(data=df, x=col, hue=hue, palette=color_discrete_map, ax=axs[i], kde=kde, multiple=multiple)
            else:
                sns.histplot(data=df, x=col, ax=axs[i])
            axs[i].set_title(col, fontsize=16, pad=10)
            axs[i].set_ylabel('Count', fontsize=12)
            axs[i].set_xlabel(' ', fontsize=12)
            axs[i].tick_params(axis='x', rotation=rotation)
            
        # Remove empty subplots
        for empty_subplot in axs[nplots:]:
            fig.delaxes(empty_subplot)
            
    unique_count = df.nunique()
    if cat_cols is None:
        cat_cols = unique_count[unique_count <= n].index.tolist()
        if hue in cat_cols:
            cat_cols.remove(hue)
    if cont_cols is None:
        cont_cols = unique_count[unique_count > n].index.tolist()

    if dtype_check:
        cont_cols = [col for col in cont_cols if df[col].dtype in ['int64', 'float64']]

    if plot_type == 'cat' or plot_type == 'both':
        plot_categorical(df, cat_cols, ncols, fig_width, subplot_height, rotation, sample_size, hue, color_discrete_map, normalize)
    if plot_type == 'cont' or plot_type == 'both':
        plot_continuous(df, cont_cols, ncols, fig_width, subplot_height, sample_size, hue, color_discrete_map, kde, multiple)


def plot_corr(df, column, n, meth='pearson', size=(15, 8), rot=45, pal='RdYlGn', rnd=2):
    """
    Version 0.2
    Create a barplot that shows correlation values for one variable against others.
    Essentially one slice of a heatmap, but the bars show the height of the correlation
    in addition to the color. It will only look at numeric variables.

    Parameters:
    - df: dataframe that contains the variables you want to analyze
    - column: string. Column name that you want to evaluate the correlations against
    - n: int. The number of correlations to show (split evenly between positive and negative correlations)
    - meth: optional (default='pearson'). See df.corr() method options
    - size: tuple of ints, optional (default=(15, 8)). The size of the plot
    - rot: int, optional (default=45). The rotation of the x-axis labels
    - pal: string, optional (default='RdYlGn'). The color map to use
    - rnd: int, optional (default=2). Number of decimal places to round to

    Returns: None
    """
    # Calculate correlations
    corr = round(df.corr(method=meth, numeric_only=True)[column].sort_values(), rnd)

    # Drop column from correlations (correlating with itself)
    corr = corr.drop(column)

    # Get the most negative and most positive correlations, sorted by absolute value
    most_negative = corr.sort_values().head(n // 2)
    most_positive = corr.sort_values().tail(n // 2)

    # Concatenate these two series and sort the final series by correlation value
    corr = pd.concat([most_negative, most_positive]).sort_values()

    # Generate colors based on correlation values using a colormap
    cmap = plt.get_cmap(pal)
    colors = cmap((corr.values + 1) / 2)

    # Plot the chart
    plt.figure(figsize=size)
    plt.axhline(y=0, color='lightgrey', alpha=0.8, linestyle='-')
    bars = plt.bar(corr.index, corr.values, color=colors)

    # Add value labels to the end of each bar
    for bar in bars:
        yval = bar.get_height()
        if yval < 0:
            plt.text(bar.get_x() + bar.get_width() / 3.0, yval - 0.05, yval, va='top')
        else:
            plt.text(bar.get_x() + bar.get_width() / 3.0, yval + 0.05, yval, va='bottom')

    plt.title('Correlation with ' + column, fontsize=20)
    plt.ylabel('Correlation', fontsize=14)
    plt.xlabel('Other Variables', fontsize=14)
    plt.xticks(rotation=rot)
    plt.ylim(-1, 1)
    plt.show()


def split_dataframe(df, n):
    """
    Split a DataFrame into two based on the number of unique values in each column.

    Parameters:
    - df: DataFrame. The DataFrame to split.
    - n: int. The maximum number of unique values for a column to be considered categorical.

    Returns:
    - df_cat: DataFrame. Contains the columns of df with n or fewer unique values.
    - df_num: DataFrame. Contains the columns of df with more than n unique values.
    """
    df_cat = pd.DataFrame()
    df_num = pd.DataFrame()

    for col in df.columns:
        if df[col].nunique() <= n:
            df_cat[col] = df[col]
        else:
            df_num[col] = df[col]

    return df_cat, df_num


def thousands(x, pos):
    """
    Format a number with thousands separators.

    Parameters:
    - x: float. The number to format.
    - pos: int. The position of the number.

    Returns:
    - s: string. The formatted number.
    """
    s = '{:0,d}'.format(int(x))
    return s

def thousand_dollars(x, pos):
    """
    Format a number with thousands separators.

    Parameters:
    - x: float. The number to format.
    - pos: int. The position of the number.

    Returns:
    - s: string. The formatted number.
    """
    s = '${:0,d}'.format(int(x))
    return s

def visualize_kmeans(df, x_var, y_var, centers=3, iterations=100):
    # Select centers at random
    starting_centers = df.sample(centers).reset_index(drop=True)

    # Make a list to hold the center values
    center_values = [starting_centers[[x_var, y_var]].iloc[i].values for i in range(centers)]

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_var, y=y_var, palette='tab10')
    plt.scatter(starting_centers[x_var], starting_centers[y_var], marker='*', s=400, c='red', edgecolor='black')
    plt.title('Starting Centers')
    plt.show()

    # For each iteration
    for i in range(iterations):
        # Determine intercluster variance
        dists = [np.linalg.norm(df[[x_var, y_var]] - center_values[j], axis=1)**2 for j in range(centers)]
        dist_X = pd.DataFrame(np.array(dists).T, columns=['d' + str(j+1) for j in range(centers)])

        # Make cluster assignments
        df['Cluster Label'] = np.argmin(dist_X.values, axis=1)

        # Update centroids
        new_centers = df.groupby('Cluster Label').mean()

        # Update the center values
        center_values = [new_centers[[x_var, y_var]].iloc[j].values for j in range(centers)]

        plt_title = 'Iteration ' + str(i+1)
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x=x_var, y=y_var, hue='Cluster Label', palette='tab10')
        plt.scatter(new_centers[x_var], new_centers[y_var], marker='*', s=400, c='red', edgecolor='black')
        plt.title(plt_title)
        plt.show()

    return df


import seaborn as sns
import plotly.express as px

def plot_3d(df, x, y, z, color=None, color_map=None, scale='linear'):
    """
    Create a 3D scatter plot using Plotly Express.

    Parameters:
    - df: DataFrame. The input dataframe.
    - x: str. The column name to be used for the x-axis.
    - y: str. The column name to be used for the y-axis.
    - z: str. The column name to be used for the z-axis.
    - color: str, optional (default=None). The column name to be used for color coding the points.
    - color_map: list of str, optional (default=None). The color map to be used. If None, the seaborn default color palette will be used.
    - scale: str, optional (default='linear'). The scale type for the axis. Use 'log' for logarithmic scale.

    Returns: None
    """
    if color_map is None:
        color_map = sns.color_palette().as_hex()

    fig = px.scatter_3d(df, 
                    x=x, 
                    y=y, 
                    z=z, 
                    color=color,
                    color_discrete_sequence=color_map,
                    height=600, 
                    width=1000)
    title_text = "{}, {}, {} by {}".format(x, y, z, color)
    fig.update_layout(title={'text': title_text, 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                      showlegend=True,
                      scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                        center=dict(x=0, y=0, z=-0.1),
                                        eye=dict(x=1.5, y=-1.4, z=0.5)),
                      margin=dict(l=0, r=0, b=0, t=0),
                      scene=dict(xaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title=x,
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            type=scale), 
                                 yaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title=y,
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            type=scale), 
                                 zaxis=dict(backgroundcolor='lightgrey',
                                            color='black', 
                                            gridcolor='#f0f0f0',
                                            title=z,
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            type=scale)))
    fig.update_traces(marker=dict(size=3, opacity=1, line=dict(color='black', width=0.1)))
    fig.show()


def plot_map_ca(df, lon='Longitude', lat='Latitude', hue=None, size=None, size_range=(50, 200), title='Geographic Chart', dot_size=None, alpha=0.8, color_map=None, fig_size=(12, 12)):
    """
    Version 0.1
    Plots a geographic map of California with data points overlaid.

    Parameters:
    - df: DataFrame containing the data to be plotted
    - lon: str, optional (default='Longitude'). Column name in `df` representing the longitude coordinates
    - lat: str, optional (default='Latitude'). Column name in `df` representing the latitude coordinates
    - hue: str, optional (default=None). Column name in `df` for color-coding the points
    - size: str, optional (default=None). Column name in `df` to scale the size of points
    - size_range: tuple, optional (default=(50, 200)). Range of sizes if the `size` parameter is used
    - title: str, optional (default='Geographic Chart'). Title of the plot
    - dot_size: int, optional (default=None). Size of all dots if you want them to be uniform
    - alpha: float, optional (default=0.8). Transparency of the points
    - color_map: colormap, optional (default=None). Colormap to be used if `hue` is specified
    - fig_size: tuple, optional (default=(12, 12)). Size of the figure

    Returns: None
    """
    # Define the locations of major cities
    large_ca_cities = {'Name': ['Fresno', 'Los Angeles', 'Sacramento', 'San Diego', 'San Francisco', 'San Jose'],
                       'Latitude': [36.746842, 34.052233, 38.581572, 32.715328, 37.774931, 37.339386],
                       'Longitude': [-119.772586, -118.243686, -121.494400, -117.157256, -122.419417, -121.894956],
                       'County': ['Fresno', 'Los Angeles', 'Sacramento', 'San Diego', 'San Francisco', 'Santa Clara']}
    df_large_cities = pd.DataFrame(large_ca_cities)

    # Create a figure that utilizes Cartopy
    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-125, -114, 32, 42])

    # Add geographic details
    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='lightgrey', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)

    # Add county boundaries
    counties = gpd.read_file('data/cb_2018_us_county_5m.shp')
    counties_ca = counties[counties['STATEFP'] == '06']
    counties_ca = counties_ca.to_crs("EPSG:4326")
    for geometry in counties_ca['geometry']:
        ax.add_geometries([geometry], crs=ccrs.PlateCarree(), edgecolor='grey', alpha=0.3, facecolor='none')

    # Draw the scatterplot of data
    if dot_size:
        ax.scatter(df[lon], df[lat], s=dot_size, cmap=color_map, alpha=alpha, transform=ccrs.PlateCarree())
    else:
        sns.scatterplot(data=df, x=lon, y=lat, hue=hue, size=size, alpha=alpha, ax=ax, palette=color_map, sizes=size_range)

    # Add cities
    ax.scatter(df_large_cities['Longitude'], df_large_cities['Latitude'], transform=ccrs.PlateCarree(), edgecolor='black')
    for x, y, label in zip(df_large_cities['Longitude'], df_large_cities['Latitude'], df_large_cities['Name']):
        text = ax.text(x + 0.05, y + 0.05, label, transform=ccrs.PlateCarree(), fontsize=12, ha='left', fontname='Arial')
        text.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])

    # Finish up the chart
    ax.set_title(title, fontsize=18, pad=15)
    ax.set_xlabel('Longitude', fontsize=14, labelpad=15)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.gridlines(draw_labels=True, color='lightgrey', alpha=0.5)
    plt.show()
    

def get_corr(df, n=5, var=None, show_results=True, return_arrays=False):
    """
    Gets the top n positive and negative correlations in a dataframe. Returns them in two
    arrays. By default, prints a summary of the top positive and negative correlations.

    Parameters
    ----------
    - df : pandas.DataFrame. The dataframe you wish to analyze for correlations
    - n : int, default 5. The number of top positive and negative correlations to list.
    - var : str, (optional) default None. The variable of interest. If provided, the function
        will only show the top n positive and negative correlations for this variable.
    - show_results : boolean, default True. Print the results.
    - return_arrays : boolean, default False. If true, return arrays with column names

    Returns: Tuple (if return_arrays == True)
        - positive_variables: array of variable names involved in top n positive correlations
        - negative_variables: array of variable names involved in top n negative correlations
    """
    pd.set_option('display.expand_frame_repr', False)

    corr = round(df.corr(numeric_only=True), 2)
    
    # Unstack correlation matrix into a DataFrame
    corr_df = corr.unstack().reset_index()
    corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']

    # If a variable is specified, filter to correlations involving that variable
    if var is not None:
        corr_df = corr_df[(corr_df['Variable 1'] == var) | (corr_df['Variable 2'] == var)]

    # Remove self-correlations and duplicates
    corr_df = corr_df[corr_df['Variable 1'] != corr_df['Variable 2']]
    corr_df[['Variable 1', 'Variable 2']] = np.sort(corr_df[['Variable 1', 'Variable 2']], axis=1)
    corr_df = corr_df.drop_duplicates(subset=['Variable 1', 'Variable 2'])

    # Sort by absolute correlation value from highest to lowest
    corr_df['AbsCorrelation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values(by='AbsCorrelation', ascending=False)

    # Drop the absolute value column
    corr_df = corr_df.drop(columns='AbsCorrelation').reset_index(drop=True)

    # Get the first n positive and negative correlations
    positive_corr = corr_df[corr_df['Correlation'] > 0].head(n).reset_index(drop=True)
    negative_corr = corr_df[corr_df['Correlation'] < 0].head(n).reset_index(drop=True)

    # Print the results
    if show_results:
        print("Top", n, "positive correlations:")
        print(positive_corr)
        print("\nTop", n, "negative correlations:")
        print(negative_corr)

    # Return the arrays
    if return_arrays:
        # Remove target variable from the arrays
        positive_variables = positive_corr[['Variable 1', 'Variable 2']].values.flatten()
        positive_variables = positive_variables[positive_variables != var]

        negative_variables = negative_corr[['Variable 1', 'Variable 2']].values.flatten()
        negative_variables = negative_variables[negative_variables != var]

        return positive_variables, negative_variables

    
def sk_vif(exogs, data):
    # Set a high threshold, e.g., 1e10, for very large VIFs
    MAX_VIF = 1e10

    vif_dict = {}

    for exog in exogs:
        not_exog = [i for i in exogs if i !=exog]
        # split the dataset, one independent variable against all others
        X, y = data[not_exog], data[exog]

        # fit the model and obtain R^2
        r_squared = LinearRegression().fit(X,y).score(X,y)

        # compute the VIF, with a check for r_squared close to 1
        if 1 - r_squared < 1e-5:  # or some other small threshold that makes sense for your application
            vif = MAX_VIF
        else:
            vif = 1/(1-r_squared)

        vif_dict[exog] = vif

    return pd.DataFrame({"VIF": vif_dict})


def calc_vif(X):

    # Calculate Variance Inflation Factor (VIF) to find which features have mutlticollinearity
    vif = pd.DataFrame()
    vif['variables'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif.sort_values(by='VIF', ascending=False))


def calc_fpi(model, X, y, n_repeats=10, random_state=42):
    
    # Calculate Feature Permutation Importance to find out which features have the most effect
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)

    return pd.DataFrame({"Variables": X.columns,
                         "Score Mean": r.importances_mean,
                         "Score Std": r.importances_std}).sort_values(by="Score Mean", ascending=False)


from sklearn import linear_model

# List of classes that support the .coef_ attribute
SUPPORTED_COEF_CLASSES = (
    linear_model.LogisticRegression,
    linear_model.LogisticRegressionCV,
    linear_model.PassiveAggressiveClassifier,
    linear_model.Perceptron,
    linear_model.RidgeClassifier,
    linear_model.RidgeClassifierCV,
    linear_model.SGDClassifier,
    linear_model.SGDOneClassSVM,
    linear_model.LinearRegression,
    linear_model.Ridge,
    linear_model.RidgeCV,
    linear_model.SGDRegressor,
    linear_model.ElasticNet,
    linear_model.ElasticNetCV,
    linear_model.Lars,
    linear_model.LarsCV,
    linear_model.Lasso,
    linear_model.LassoCV,
    linear_model.LassoLars,
    linear_model.LassoLarsCV,
    linear_model.LassoLarsIC,
    linear_model.OrthogonalMatchingPursuit,
    linear_model.OrthogonalMatchingPursuitCV,
    linear_model.ARDRegression,
    linear_model.BayesianRidge,
    linear_model.HuberRegressor,
    linear_model.QuantileRegressor,
    linear_model.RANSACRegressor,
    linear_model.TheilSenRegressor
)

def supports_coef(estimator):
    """Check if estimator supports .coef_"""
    return isinstance(estimator, SUPPORTED_COEF_CLASSES)

def extract_features_and_coefficients(grid_or_pipe, X, debug=False):
    # Determine the type of the passed object and set flags
    if hasattr(grid_or_pipe, 'best_estimator_'):
        estimator = grid_or_pipe.best_estimator_
        is_grid = True
        is_pipe = False
        if debug:
            print('Grid: ', is_grid)
    else:
        estimator = grid_or_pipe
        is_pipe = True
        is_grid = False
        if debug:
            print('Pipe: ', is_pipe)

    # Initial setup
    current_features = list(X.columns)
    if debug:
        print('current_features: ', current_features)
    mapping = pd.DataFrame({
        'feature_name': current_features,
        'intermediate_name1': current_features,
        'selected': [True] * len(current_features),
        'coefficients': [None] * len(current_features)
    })

    for step_name, step_transformer in estimator.named_steps.items():
        if debug:
            print(f"Processing step: {step_name} in {step_transformer}")  # Debugging

        n_features_in = len(current_features)  # Number of features at the start of this step

        # If transformer is a ColumnTransformer
        if isinstance(step_transformer, ColumnTransformer):
            new_features = []  # Collect new features from this step
            step_transformer_list = step_transformer.transformers_
            for name, trans, columns in step_transformer_list:
                # OneHotEncoder or similar expanding transformers
                if hasattr(trans, 'get_feature_names_out'):
                    out_features = list(trans.get_feature_names_out(columns))
                    new_features.extend(out_features)
                else:
                    new_features.extend(columns)

            current_features = new_features

            # Update mapping based on current_features
            mapping = pd.DataFrame({
                'feature_name': current_features,
                'intermediate_name1': current_features,
                'selected': [True] * len(current_features),
                'coefficients': [None] * len(current_features)
            })
            if debug:
                print("Mapping: ", mapping)

        # Reduction
        elif hasattr(step_transformer, 'get_support'):
            mask = step_transformer.get_support()
            # Update selected column in mapping
            mapping.loc[mapping['feature_name'].isin(current_features), 'selected'] = mask
            current_features = mapping[mapping['selected']]['feature_name'].tolist()

    # Inside your extract_features_and_coefficients function:

    # If there's a model with coefficients in this step, update coefficients
    if supports_coef(step_transformer):
        coefficients = step_transformer.coef_.ravel()
        selected_rows = mapping[mapping['selected']].index
        if debug:
            print("Coefficients: ", coefficients)
            print(f"Number of coefficients: {len(coefficients)}")  # Debugging
            print(f"Number of selected rows: {len(selected_rows)}")  # Debugging

        if len(coefficients) == len(selected_rows):
            mapping.loc[selected_rows, 'coefficients'] = coefficients.tolist()
        else:
            print(f"Mismatch in coefficients and selected rows for step: {step_name}")

    # For transformers inside ColumnTransformer
    if isinstance(step_transformer, ColumnTransformer):
        if debug:
            print("ColumnTransformer:", step_transformer)
        transformers = step_transformer.transformers_
        if debug:
            print("Transformers: ", transformers)
        new_features = []  # Collect new features from this step
        for name, trans, columns in transformers:
            # OneHotEncoder or similar expanding transformers
            if hasattr(trans, 'get_feature_names_out'):
                out_features = list(trans.get_feature_names_out(columns))
                new_features.extend(out_features)
                if debug:
                    print("Out features: ", out_features)
                    print("New features: ", new_features)
            else:
                new_features.extend(columns)
        
        current_features = new_features

        # Update mapping based on current_features
        mapping = pd.DataFrame({
            'feature_name': current_features,
            'intermediate_name1': current_features,
            'selected': [True] * len(current_features),
            'coefficients': [None] * len(current_features)
        })
        if debug:
            print("Mapping: ", mapping)
    # Filtering the final selected features and their coefficients
    final_data = mapping[mapping['selected']]
    
    return final_data[['feature_name', 'coefficients']]

    
    


# MODEL ITERATION: default_config, create_pipeline, iterate_model

# default_config: Version 0.1
# Default configuration of parameters used by iterate_model and create_pipeline
# New configurations can be passed in by the user when function is called
# 

# create_pipeline: Version 0.1
#
def create_pipeline(transformer_keys=None, scaler_key=None, selector_key=None, model_key=None, config=None, X_cat_columns=None, X_num_columns=None):
    """
    Creates a pipeline for data preprocessing and modeling.

    This function allows for flexibility in defining the preprocessing and 
    modeling steps of the pipeline. You can specify which transformers to apply 
    to the data, whether to scale the data, and which model to use for predictions. 
    If a step is not specified, it will be skipped.

    Parameters:
    - model_key (str): The key corresponding to the model in the config['models'] dictionary.
    - transformer_keys (list of str, str, or None): The keys corresponding to the transformers 
        to apply to the data. This can be a list of string keys or a single string key corresponding 
        to transformers in the config['transformers'] dictionary. If not provided, no transformers will be applied.
    - scaler_key (str or None): The key corresponding to the scaler to use to scale the data. 
        This can be a string key corresponding to a scaler in the config['scalers'] dictionary. 
        If not provided, the data will not be scaled.
    - selector_key (str or None): The key corresponding to the feature selector. 
        This can be a string key corresponding to a scaler in the config['selectors'] dictionary. 
        If not provided, no feature selection will be performed.
    - X_num_columns (list-like, optional): List of numeric columns from the input dataframe. This is used
        in the default_config for the relevant transformers.
    - X_cat_columns (list-like, optional): List of categorical columns from the input dataframe. This is used
        in the default_config for the elevant encoders.

    Returns:
    pipeline (sklearn.pipeline.Pipeline): A scikit-learn pipeline consisting of the specified steps.

    Example:
    >>> pipeline = create_pipeline('linreg', transformer_keys=['ohe', 'poly2'], scaler_key='stand', config=my_config)
    """

    # Check for configuration file parameter, if none, use default in library
    if config is None:
        # If no column lists are provided, raise an error
        if not X_cat_columns and not X_num_columns:
            raise ValueError("If no config is provided, X_cat_columns and X_num_columns must be passed.")
        config = {
            'transformers': {
                'ohe': (OneHotEncoder(drop='if_binary', handle_unknown='ignore'), X_cat_columns),
                'ord': (OrdinalEncoder(), X_cat_columns),
                'js': (JamesSteinEncoder(), X_cat_columns),
                'poly2': (PolynomialFeatures(degree=2, include_bias=False), X_num_columns),
                'poly2_bias': (PolynomialFeatures(degree=2, include_bias=True), X_num_columns),
                'poly3': (PolynomialFeatures(degree=3, include_bias=False), X_num_columns),
                'poly3_bias': (PolynomialFeatures(degree=3, include_bias=True), X_num_columns),
                'log': (FunctionTransformer(np.log1p, validate=True), X_num_columns)
            },
            'scalers': {
                'stand': StandardScaler(),
                'robust': RobustScaler(),
                'minmax': MinMaxScaler()
            },
            'selectors': {
                'sfs': SequentialFeatureSelector(LinearRegression()),
                'sfs_7': SequentialFeatureSelector(LinearRegression(), n_features_to_select=7),
                'sfs_6': SequentialFeatureSelector(LinearRegression(), n_features_to_select=6),
                'sfs_5': SequentialFeatureSelector(LinearRegression(), n_features_to_select=5),
                'sfs_4': SequentialFeatureSelector(LinearRegression(), n_features_to_select=4),
                'sfs_3': SequentialFeatureSelector(LinearRegression(), n_features_to_select=3),
                'sfs_bw': SequentialFeatureSelector(LinearRegression(), direction='backward')
            },
            'models': {
                'linreg': LinearRegression(),
                'ridge': Ridge(),
                'lasso': Lasso(random_state=42),
                'random_forest': RandomForestRegressor(),
                'gradient_boost': GradientBoostingRegressor(),
            }
        }

    # Initialize an empty list for the transformation steps
    steps = []

    # If transformers are provided, add them to the steps
    if transformer_keys is not None:
        transformer_steps = []

        for key in (transformer_keys if isinstance(transformer_keys, list) else [transformer_keys]):
            transformer, cols = config['transformers'][key]

            transformer_steps.append((key, transformer, cols))

        # Create column transformer
        col_trans = ColumnTransformer(transformer_steps, remainder='passthrough')
        transformer_name = 'Transformers: ' + '_'.join(transformer_keys) if isinstance(transformer_keys, list) else 'Transformers: ' + transformer_keys
        steps.append((transformer_name, col_trans))


    # If a scaler is provided, add it to the steps
    if scaler_key is not None:
        scaler_obj = config['scalers'][scaler_key]
        steps.append(('Scaler: ' + scaler_key, scaler_obj))

    # If a selector is provided, add it to the steps
    if selector_key is not None:
        selector_obj = config['selectors'][selector_key]
        steps.append(('Selector: ' + selector_key, selector_obj))

    # If a model is provided, add it to the steps
    if model_key is not None:
        model_obj = config['models'][model_key]
        steps.append(('Model: ' + model_key, model_obj))

    # Create and return pipeline
    return Pipeline(steps)

 
# Define the results_df global variable that will store the results of iterate_model if Save=True:
results_df = None
# This needs to be initialized in notebook with the following lines of code:
# import mytools as my
# my.results_df = pd.DataFrame(columns=['Iteration', 'Train MSE', 'Test MSE', 'Train RMSE', 'Test RMSE', 'Train MAE',
#                'Test MAE', 'Train R^2 Score', 'Test R^2 Score', 'Pipeline', 'Best Grid Params', 'Note', 'Date'])

def iterate_model(Xn_train, Xn_test, yn_train, yn_test, model=None, transformers=None, scaler=None, selector=None, drop=None, iteration='1', note='', save=False, export=False, plot=False, coef=False, perm=False, vif=False, cross=False, cv_folds=5, config=None, debug=False, grid=False, grid_params=None, grid_cv=None, grid_score='r2', grid_verbose=1, decimal=2, lowess=False):
    """
    Creates a pipeline from specified parameters for transformers, scalers, and models. Parameters must be
    defined in configuration dictionary containing 3 dictionaries: transformer_dict, scaler_dict, model_dict.
    See 'default_config' in this library file for reference, customize at will. Then fits the pipeline to the passed
    training data, and evaluates its performance with both test and training data. Options to see plots of residuals
    and actuals vs. predicted, save results to results_df with user-defined note, display coefficients, calculate
    permutation feature importance, variance inflation factor (VIF), and cross-validation.
    
    Parameters:
    - Xn_train, Xn_test: Training and test feature sets.
    - yn_train, yn_test: Training and test target sets.
    - config: Configuration dictionary of parameters for pipeline construction (see default_config)
    - model: Key for the model to be used (ex: 'linreg', 'lasso', 'ridge').
    - transformers: List of transformation keys to apply (ex: ['ohe', 'poly2']).
    - scaler: Key for the scaler to be applied (ex: 'stand')
    - selector: Key for the selector to be applied (ex: 'sfs')
    - drop: List of columns to be dropped from the training and test sets.
    - iteration: A string identifier for the iteration.
    - note: Any note or comment to be added for the iteration.
    - save: Boolean flag to decide if the results should be saved to the global results dataframe (results_df).
    - plot: Flag to plot residual plots and actuals vs. predicted for training and test data.
    - coef: Flag to print and plot model coefficients.
    - perm: Flag to compute and display permutation feature importance.
    - vif: Flag to calculate and display Variance Inflation Factor for features.
    - cross: Flag to perform cross-validation and print results.
    - cv_folds: Number of folds to be used for cross-validation if cross=True.
    - debug: Flag to show debugging information like the details of the pipeline.

    Prerequisites:
    - Dictionaries of parameters for transformers, scalers, and models: transformer_dict, scaler_dict, model_dict.
    - Lists identifying columns for various transformations and encodings, e.g., ohe_columns, ord_columns, etc.

    Outputs:
    - Prints results, performance metrics, and other specified outputs.
    - Updates the global results dataframe if save=True.
    - Displays plots based on flags like plot, coef.
    
    Usage:
    >>> iterate_model(X_train, X_test, y_train, y_test, transformers=['ohe','poly2'], scaler='stand', model='linreg', drop=['col1'], iteration="1", save=True, plot=True)
    """
    # Drop specified columns from Xn_train and Xn_test
    if drop is not None:
        Xn_train = Xn_train.drop(columns=drop)
        Xn_test = Xn_test.drop(columns=drop)
        if debug:
            print('Drop:', drop)
            print('Xn_train.columns', Xn_train.columns)
            print('Xn_test.columns', Xn_test.columns)

    # Check for configuration file parameter, if none, use default in library
    if config is None:
        X_num_columns = Xn_train.select_dtypes(include=[np.number]).columns.tolist()
        X_cat_columns = Xn_train.select_dtypes(exclude=[np.number]).columns.tolist()
        if debug:
            print('Config:', config)
            print('X_num_columns:', X_num_columns)
            print('X_cat_columns:', X_cat_columns)
    else:
        X_num_columns = None
        X_cat_columns = None

    # Create a pipeline from transformer and model parameters
    pipe = create_pipeline(transformer_keys=transformers, scaler_key=scaler, selector_key=selector, model_key=model, config=config, X_cat_columns=X_cat_columns, X_num_columns=X_num_columns)
    if debug:
        print('Pipeline:', pipe)
        print('Pipeline Parameters:', pipe.get_params())

    # Construct format string
    format_str = f',.{decimal}f'
        
    # Print some metadata
    print(f'\nITERATION {iteration} RESULTS\n')
    pipe_steps = " -> ".join(pipe.named_steps.keys())
    print(f'Pipeline: {pipe_steps}')
    if note: print(f'Note: {note}')
    # Get the current date and time
    current_time = datetime.now(pytz.timezone('US/Pacific'))
    timestamp = current_time.strftime('%b %d, %Y %I:%M %p PST')
    print(f'{timestamp}\n')

    if cross or grid:
        print('Cross Validation:\n')
    # Before fitting the pipeline, check if cross-validation is desired:
    if cross:
        # Flatten yn_train for compatibility
        yn_train_flat = yn_train.values.flatten() if isinstance(yn_train, pd.Series) else np.array(yn_train).flatten()
        cv_scores = cross_val_score(pipe, Xn_train, yn_train_flat, cv=cv_folds, scoring='r2')

        print(f'Cross-Validation (R^2) Scores for {cv_folds} Folds:')
        for i, score in enumerate(cv_scores, 1):
            print(f'Fold {i}: {score:{format_str}}')
        print(f'Average: {np.mean(cv_scores):{format_str}}')
        print(f'Standard Deviation: {np.std(cv_scores):{format_str}}\n')

    if grid:
        
        grid = GridSearchCV(pipe, param_grid=config['params'][grid_params], cv=config['cv'][grid_cv], scoring=grid_score, verbose=grid_verbose)
        if debug:
            print('Grid: ', grid)
            print('Grid Parameters: ', grid.get_params())
        # Fit the grid and predict
        grid.fit(Xn_train, yn_train)
        #best_model = grid.best_estimator_
        best_model = grid
        yn_train_pred = grid.predict(Xn_train)
        yn_test_pred = grid.predict(Xn_test)
        if debug:
            print("First 10 actual train values:", yn_train[:10])
            print("First 10 predicted train values:", yn_train_pred[:10])
            print("First 10 actual test values:", yn_test[:10])
            print("First 10 predicted test values:", yn_test_pred[:10])
        best_grid_params = grid.best_params_
        best_grid_score = grid.best_score_
        best_grid_estimator = grid.best_estimator_
        best_grid_index = grid.best_index_
        grid_results = grid.cv_results_
    else:
        best_grid_params = None
        best_grid_score = None
        # Fit the pipeline and predict
        pipe.fit(Xn_train, yn_train)
        best_model = pipe
        yn_train_pred = pipe.predict(Xn_train)
        yn_test_pred = pipe.predict(Xn_test)

    # MSE
    yn_train_mse = mean_squared_error(yn_train, yn_train_pred)
    yn_test_mse = mean_squared_error(yn_test, yn_test_pred)

    # RMSE
    yn_train_rmse = np.sqrt(yn_train_mse)
    yn_test_rmse = np.sqrt(yn_test_mse)

    # MAE
    yn_train_mae = mean_absolute_error(yn_train, yn_train_pred)
    yn_test_mae = mean_absolute_error(yn_test, yn_test_pred)

    # R^2 Score
    if grid:
        if grid_score == 'r2':
            train_score = grid.score(Xn_train, yn_train)
            test_score = grid.score(Xn_test, yn_test)
        else:
            train_score = 0
            test_score = 0
    else:
        train_score = pipe.score(Xn_train, yn_train)
        test_score = pipe.score(Xn_test, yn_test)

    # Print Grid best parameters
    if grid:
        print(f'\nBest Grid mean score ({grid_score}): {best_grid_score:{format_str}}')
        #print(f'Best Grid parameters: {best_grid_params}\n')        
        param_str = ', '.join(f"{key}: {value}" for key, value in best_grid_params.items())
        print(f"Best Grid parameters: {param_str}\n")
        #print(f'Best Grid estimator: {best_grid_estimator}')
        #print(f'Best Grid index: {best_grid_index}')
        #print(f'Grid results: {grid_results}')

    # Print the results
    print('Predictions:')
    print(f'{"":<15} {"Train":>15} {"Test":>15}')
    #print('-'*55)
    print(f'{"MSE:":<15} {yn_train_mse:>15{format_str}} {yn_test_mse:>15{format_str}}')
    print(f'{"RMSE:":<15} {yn_train_rmse:>15{format_str}} {yn_test_rmse:>15{format_str}}')
    print(f'{"MAE:":<15} {yn_train_mae:>15{format_str}} {yn_test_mae:>15{format_str}}')
    print(f'{"R^2 Score:":<15} {train_score:>15{format_str}} {test_score:>15{format_str}}')

    if save:
        # Access to the dataframe for storing results
        global results_df
        # Check if results_df exists in the global scope
        if 'results_df' not in globals():
            # Create results_df if it doesn't exist   
            results_df = pd.DataFrame(columns=['Iteration', 'Train MSE', 'Test MSE', 'Train RMSE', 'Test RMSE',
                'Train MAE', 'Test MAE', 'Train R^2 Score', 'Test R^2 Score', 'Best Grid Mean Score',
                'Best Grid Params', 'Pipeline', 'Note', 'Date'])
            print("\n'results_df' not found in global scope. A new one has been created.")

        # Store results in a dictionary   
        results = {
            'Iteration': iteration,
            'Train MSE': yn_train_mse,
            'Test MSE': yn_test_mse,
            'Train RMSE': yn_train_rmse,
            'Test RMSE': yn_test_rmse,
            'Train MAE': yn_train_mae,
            'Test MAE': yn_test_mae,
            'Train R^2 Score': train_score,
            'Test R^2 Score': test_score,
            'Best Grid Mean Score': best_grid_score,
            'Best Grid Params': best_grid_params,
            'Pipeline': pipe_steps,
            'Note': note,
            'Date': timestamp
        }

        # Convert the dictionary to a dataframe
        df_iteration = pd.DataFrame([results])

        # Append the results dataframe to the existing results dataframe
        results_df = pd.concat([results_df, df_iteration], ignore_index=True)        
        
    # Permutation Feature Importance
    if perm:
        print("\nPermutation Feature Importance:")
        if grid:
            perm_imp_res = calc_fpi(grid, Xn_train, yn_train)
        else:
            perm_imp_res = calc_fpi(pipe, Xn_train, yn_train)

        # Create a Score column
        perm_imp_res['Score'] = perm_imp_res['Score Mean'].apply(lambda x: f"{x:{format_str}}") + " ± " + perm_imp_res['Score Std'].apply(lambda x: f"{x:{format_str}}")

        # Adjust the variable names for better alignment in printout
        perm_imp_res['Variables'] = perm_imp_res['Variables'].str.ljust(25)

        # Create a copy for printing and rename the 'Variables' column header to empty
        print_df = perm_imp_res.copy()
        print_df = print_df.rename(columns={"Variables": ""})

        # Print the DataFrame with only the Variables and the Score column
        print(print_df[['', 'Score']].to_string(index=False))

    if vif:
        all_numeric = not bool(Xn_train.select_dtypes(exclude=[np.number]).shape[1])

        if all_numeric:
            suitable = True
        else:
            # Check if transformers is not empty
            if transformers:
                transformer_list = [transformers] if isinstance(transformers, str) else transformers
                suitable_for_vif = {'ohe', 'ord', 'ohe_drop'}
                if any(t in suitable_for_vif for t in transformer_list):
                    suitable = True
                else:
                    suitable = False
            elif drop:
                suitable = True
            else:
                suitable = False

        if suitable:
            print("\nVariance Inflation Factor:")
            if all_numeric:
                vif_df = Xn_train
            else:
                if transformers is not None:
                    # Create a pipeline with the transformers only
                    #vif_pipe = create_pipeline(transformer_keys=transformers, config=config, X_cat_columns=X_cat_columns, X_num_columns=X_num_columns)
                    if grid:
                        vif_pipe = grid
                        feature_names = grid.best
                    elif pipe:
                        vif_pipe = pipe
                    if debug:
                        print('VIF Pipeline:', vif_pipe)
                        print('VIF Pipeline Parameters:', vif_pipe.get_params())
                    #vif_pipe.fit(Xn_train, yn_train)
                    #feature_names = vif_pipe.get_feature_names_out()
                    #
                    transformed_data = vif_pipe.transform(Xn_train)
                    vif_df = pd.DataFrame(transformed_data, columns=feature_names)
            vif_results = sk_vif(vif_df.columns, vif_df).sort_values(by='VIF', ascending=False)
            vif_results['VIF'] = vif_results['VIF'].apply(lambda x: f'{{:,.{decimal}f}}'.format(x))
            print(vif_results)
        else:
            print("\nVIF calculation skipped. The transformations applied are not suitable for VIF calculation.")

    if plot:
        print('')
        yn_train = yn_train.values.flatten() if isinstance(yn_train, pd.Series) else np.array(yn_train).flatten()
        yn_test = yn_test.values.flatten() if isinstance(yn_test, pd.Series) else np.array(yn_test).flatten()

        yn_train_pred = yn_train_pred.flatten()
        yn_test_pred = yn_test_pred.flatten()

        # Generate residual plots
        plt.figure(figsize=(12, 3))

        plt.subplot(1, 2, 1)
        sns.residplot(x=yn_train, y=yn_train_pred, lowess=lowess, scatter_kws={'s': 30, 'edgecolor': 'white'}, line_kws={'color': 'red', 'lw': '1'})
        plt.title(f'Training Residuals - Iteration {iteration}')

        plt.subplot(1, 2, 2)
        sns.residplot(x=yn_test, y=yn_test_pred, lowess=lowess, scatter_kws={'s': 30, 'edgecolor': 'white'}, line_kws={'color': 'red', 'lw': '1'})
        plt.title(f'Test Residuals - Iteration {iteration}')

        plt.show()

        # Generate predicted vs actual plots
        plt.figure(figsize=(12, 3))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=yn_train, y=yn_train_pred, s=30, edgecolor='white')
        plt.plot([yn_train.min(), yn_train.max()], [yn_train.min(), yn_train.max()], color='red', linewidth=1)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Training Predicted vs. Actual - Iteration {iteration}')

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=yn_test, y=yn_test_pred, s=30, edgecolor='white')
        plt.plot([yn_test.min(), yn_test.max()], [yn_test.min(), yn_test.max()], color='red', linewidth=1)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Test Predicted vs. Actual - Iteration {iteration}')

        plt.show()

            
    #Calculate coefficients if model supports
    if coef:
        # Extract features and coefficients using the function
        coefficients_df = extract_features_and_coefficients(
            grid.best_estimator_ if grid else pipe, Xn_train, debug=debug
        )

        # Check if there are any non-NaN coefficients
        if coefficients_df['coefficients'].notna().any():
            # Ensure the coefficients are shaped as a 2D numpy array
            coefficients = coefficients_df[['coefficients']].values
        else:
            coefficients = None

        # Debugging information
        if debug:
            print("Coefficients: ", coefficients)
            # Print the number of coefficients and selected rows
            print(f"Number of coefficients: {len(coefficients)}")

        if coefficients is not None:
            print("\nCoefficients:")
            with pd.option_context('display.float_format', lambda x: f'{x:,.{decimal}f}'.replace('-0.00', '0.00')):
                coefficients_df.index = coefficients_df.index + 1
                coefficients_df = coefficients_df.rename(columns={'feature_name': 'Feature', 'coefficients': 'Value'})
                print(coefficients_df)

            if plot:
                coefficients = coefficients.ravel()
                plt.figure(figsize=(12, 3))
                x_values = range(1, len(coefficients) + 1)
                plt.bar(x_values, coefficients)
                plt.xticks(x_values)
                plt.xlabel('Feature')
                plt.ylabel('Value')
                plt.title('Coefficients')
                plt.axhline(y = 0, color = 'black', linestyle='dotted', lw=1)
                plt.show()

    if export:
        filestamp = current_time.strftime('%Y%m%d_%H%M%S')
        filename = f'iteration_{iteration}_model_{filestamp}.joblib'
        dump(best_model, filename)

        # Check if file exists and display a message
        if os.path.exists(filename):
            print(f"\nModel saved successfully as {filename}")
        else:
            print(f"\FAILED to save the model as {filename}")
        
    if grid:
        return best_model, grid_results
    else:
        return best_model
       
def split_outliers(df, columns=None, iqr_multiplier=1.5):
    """
    Splits a DataFrame into two: one with outliers and one without.
    
    Uses the IQR method to determine outliers, based on the provided multiplier.
    
    Params:
    - df (pd.DataFrame): The input dataframe.
    - columns (list): List of columns to consider for outlier detection. If None, all columns are considered.
    - iqr_multiplier (float): The multiplier for the IQR range to determine outliers. Default is 1.5.
    
    Returns:
    - df_no_outliers (pd.DataFrame): DataFrame without outliers.
    - df_outliers (pd.DataFrame): DataFrame with only the outliers.
    """
    
    # If columns parameter is not provided, use all columns in the dataframe
    if columns is None:
        columns = df.columns
    
    # Create an initial mask with all False values (meaning no outliers)
    outlier_mask = pd.Series(False, index=df.index)
    
    # For each specified column, update the outlier mask to mark outliers
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Update mask for outliers in current column
        outlier_mask |= (df[col] < (Q1 - iqr_multiplier * IQR)) | (df[col] > (Q3 + iqr_multiplier * IQR))
    
    # Use the mask to split the data
    df_no_outliers = df[~outlier_mask]
    df_outliers = df[outlier_mask]
    
    return df_no_outliers, df_outliers

def log_transform(df, columns=None):
    """
    Apply a log transformation to specified columns in a DataFrame.
    
    Params:
    - df (pd.DataFrame): The input dataframe.
    - columns (list): List of columns to transform. If None, all columns are considered.
    
    Returns:
    - df_log (pd.DataFrame): DataFrame with the log-transformed columns appended.
    """
    
    df_log = df.copy(deep=True)

    if columns is None:
        columns = df.columns

    log_columns=[]
    
    for col in columns:
        if df[col].min() < 0:
            raise ValueError(f"Column '{col}' has negative values and cannot be log-transformed.")
        
        df_log[col + '_log'] = np.log1p(df[col])
        log_columns.append(col + '_log')
    
    return df_log

def format_columns(val, col_type):
    if col_type == "large":
        return '{:,.0f}'.format(val)
    elif col_type == "small":
        return '{:.2f}'.format(val)
    else:
        return val

def format_df(df, large_num_cols=None, small_num_cols=None):
    """
    Returns a formatted DataFrame.
    
    Parameters:
    - df: the DataFrame to format.
    - large_num_cols: list of columns with large numbers to be formatted.
    - small_num_cols: list of columns with small numbers to be formatted.
    
    Returns:
    - formatted_df: DataFrame with specified columns formatted.
    """
    
    formatted_df = df.copy()

    if large_num_cols:
        for col in large_num_cols:
            formatted_df[col] = formatted_df[col].apply(lambda x: format_columns(x, "large"))

    if small_num_cols:
        for col in small_num_cols:
            formatted_df[col] = formatted_df[col].apply(lambda x: format_columns(x, "small"))
        
    return formatted_df

def plot_residuals(results, rotation=45):
    """
    Plot the residuals of an ARIMA model along with their histogram, autocorrelation function (ACF), 
    and partial autocorrelation function (PACF). The residuals are plotted with lines indicating 
    standard deviations from the mean.

    Parameters:
    - results (object): The result object typically obtained after fitting an ARIMA model.
      This object should have a `resid` attribute containing the residuals.

    Outputs:
    A 2x2 grid of plots displayed using matplotlib:
    - Top-left: Residuals with standard deviation lines.
    - Top-right: Histogram of residuals with vertical standard deviation lines.
    - Bottom-left: Autocorrelation function of residuals.
    - Bottom-right: Partial autocorrelation function of residuals.

    Note:
    Ensure that the necessary libraries (like matplotlib, statsmodels) are imported before using this function.
    """
    residuals = results.resid
    std_dev = residuals.std()
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))

    # Plot residuals
    ax[0, 0].axhline(y=0, color='lightgrey', linestyle='-', lw=1)
    ax[0, 0].axhline(y=std_dev, color='red', linestyle='--', lw=1, label=f'1 STD (±{std_dev:.2f})')
    ax[0, 0].axhline(y=2*std_dev, color='red', linestyle=':', lw=1, label=f'2 STD (±{2*std_dev:.2f})')
    ax[0, 0].axhline(y=-std_dev, color='red', linestyle='--', lw=1)
    ax[0, 0].axhline(y=2*-std_dev, color='red', linestyle=':', lw=1)
    ax[0, 0].plot(residuals, label='Residuals')
    ax[0, 0].tick_params(axis='x', rotation=rotation)
    ax[0, 0].set_title('Residuals from ARIMA Model')
    ax[0, 0].legend()

    # Plot histogram of residuals
    ax[0, 1].hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    ax[0, 1].axvline(x=std_dev, color='red', linestyle='--', lw=1, label=f'1 STD (±{std_dev:.2f})')
    ax[0, 1].axvline(x=2*std_dev, color='red', linestyle=':', lw=1, label=f'2 STD (±{2*std_dev:.2f})')
    ax[0, 1].axvline(x=-std_dev, color='red', linestyle='--', lw=1)
    ax[0, 1].axvline(x=2*-std_dev, color='red', linestyle=':', lw=1)
    ax[0, 1].set_title("Histogram of Residuals")
    ax[0, 1].set_xlabel("Residual Value")
    ax[0, 1].set_ylabel("Frequency")
    ax[0, 1].legend()

    # Plot ACF of residuals
    plot_acf(residuals, lags=40, ax=ax[1, 0])
    ax[1, 0].set_title("ACF of Residuals")

    # Plot PACF of residuals
    plot_pacf(residuals, lags=40, ax=ax[1, 1])
    ax[1, 1].set_title("PACF of Residuals")

    plt.tight_layout()
    plt.show()

    
def eval_model(*, y_test, preds, neg_label, pos_label, title, figmulti=1.5, model_name=None,
                            class_type='binary', estimator=None, X_test=None, threshold=0.5,
                            decimal=4, plot=False, figsize=(12,11), class_weight=None, return_metrics=False, output=True):
    """
    Generates a detailed model evaluation report, including a classification report, confusion matrix, ROC curve, and a precision-recall curve.
    Note: This function requires all arguments to be passed as named arguments. It's limited to binary classification for now.

    Parameters:
    - y_test (array-like): True labels for the test set.
    - preds (array-like): Predicted labels for the test set.
    - neg_label (str): Name of the negative class label.
    - pos_label (str): Name of the positive class label.
    - title (str): Title for the classification report and plots.
    - type (str): Type of model, default and only option in this version is 'binclass' for binary classification.
    - estimator (estimator object, optional): The trained model estimator. Required if plotting curves or if using a custom threshold.
    - X_test (array-like, optional): Feature data for the test set. Required if plotting curves or if using a custom threshold.
    - threshold (float, optional): Threshold for classification, default is 0.5. Changes the decision boundary if not set to default.
    - decimal (int, optional): Number of decimal places for metrics in the report, default is 4.
    - plot (bool, optional): Whether to plot the confusion matrix, histogram of predicted probabilities, ROC curve, and precision-recall curve. Default is False.
    - figsize (tuple, optional): Size of the plots, default is (12,11).

    Returns:
    None. Displays the classification report, confusion matrix, and plots if requested.

    Example:
    model_evaluation_report(y_test, y_preds, "Not Churn", "Churn", "Churn Classifier", estimator=my_model, X_test=X_test_data, plot=True)

    Notes:
    - If 'plot' is set to True, 'estimator' and 'X_test' must be provided to generate the ROC and precision-recall curves.
    """    
    if plot or threshold != 0.5:
        if estimator is None or X_test is None:
            raise ValueError("Both estimator and X_test must be provided for custom threshold or plotting curves.")

        if class_type == 'binary':
            probabilities = estimator.predict_proba(X_test)[:, 1]
        elif class_type == 'multi':
            probabilities = estimator.predict_proba(X_test)

        if class_type == 'binary' and threshold != 0.5:
            #preds = np.array([1 if prob >= threshold else 0 for prob in probabilities[:, 1]])
            
            # Print the shape of probabilities for debugging purposes
            #print(f"Shape of probabilities: {probabilities.shape}")

            # Check if probabilities is 2D
            if len(probabilities.shape) == 2 and probabilities.shape[1] > 1:
                preds = np.array([1 if prob >= threshold else 0 for prob in probabilities[:, 1]])
            # Check if probabilities is 1D
            elif len(probabilities.shape) == 1 or probabilities.shape[1] == 1:
                preds = np.array([1 if prob >= threshold else 0 for prob in probabilities])
            else:
                raise ValueError("Unexpected shape for probabilities array")

            # Print the predictions for debugging purposes
            #print(f"Predictions: {preds}")

        
    if class_type == 'multi':
        unique_labels = np.unique(y_test)
        num_classes = len(unique_labels)
        cm = confusion_matrix(y_test, preds)
        roc_auc = roc_auc_score(y_test, estimator.predict_proba(X_test), multi_class='ovr')
        if output:
            print(f"\n{title} Multiple Classification Report\n")
            print(classification_report(y_test, preds, digits=decimal, target_names=[str(label) for label in unique_labels]))
            print("ROC AUC: ", round(roc_auc, decimal))
            print("\nClass Weight: ", class_weight)


    elif class_type == 'binary':
        cm = confusion_matrix(y_test, preds)
        roc_auc = roc_auc_score(y_test, estimator.predict_proba(X_test)[:,1])
        TN, FP, FN, TP = cm.ravel()
        FPR = FP / (FP + TN)
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        FNR = FN / (FN + TP)

        if output:
            print(f"\n{title} Binary Classification Report\n")
            print(classification_report(y_test, preds, digits=decimal, target_names=[str(neg_label), str(pos_label)]))
            
            print("ROC AUC: ", round(roc_auc, decimal), "\n")

            print(f"{'':<15}{'Predicted:':<10}{neg_label:<10}{pos_label:<10}")
            print(f"{'Actual: ' + str(neg_label):<25}{cm[0][0]:<10}{cm[0][1]:<10}")
            print(f"{'Actual: ' + str(pos_label):<25}{cm[1][0]:<10}{cm[1][1]:<10}")

            print("\nTrue Positive Rate / Sensitivity:", round(TPR, decimal))
            print("True Negative Rate / Specificity:", round(TNR, decimal))
            print("False Positive Rate / Fall-out:", round(FPR, decimal))
            print("False Negative Rate / Miss Rate:", round(FNR, decimal))
            print("\nPositive Label:", pos_label)
            print("Class Weight: ", class_weight)

        metrics = {
            "True Positives": TP,
            "False Positives": FP,
            "True Negatives": TN,
            "False Negatives": FN,
            "TPR": round(TPR, decimal),
            "TNR": round(TNR, decimal),
            "FPR": round(FPR, decimal),
            "FNR": round(FNR, decimal),
        }

    if plot and output:
        
        blue = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
        
        if class_type == 'multi':
            
            multiplier = figmulti # adjust this based on your needs
            max_size = 20 # to avoid too large figures
            size = min(num_classes * multiplier, max_size)
            figsize = (size, size)
            
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)

            # Confusion Matrix for Multi-class
            cm_display = ConfusionMatrixDisplay(cm, display_labels=unique_labels)
            cm_display.plot(cmap='Blues', ax=ax1, colorbar=False)
            for text in cm_display.text_:
                for t in text:
                    t.set_fontsize(10)
            ax1.set_title(f'{model_name} Confusion Matrix', fontsize=18, pad=15)
            ax1.set_xlabel('Predicted Label', fontsize=14, labelpad=15)
            ax1.set_ylabel('True Label', fontsize=14, labelpad=10)
            ax1.tick_params(axis='both', which='major', labelsize=10)

            plt.show()

        elif class_type == 'binary':
        
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

            # 1. Confusion Matrix
            cm_matrix = ConfusionMatrixDisplay(cm, display_labels=[neg_label, pos_label])
            cm_matrix.plot(cmap='Blues', ax=ax1, colorbar=False)
            for text in cm_matrix.text_:
                for t in text:
                    t.set_fontsize(14)
            ax1.set_title('Confusion Matrix', fontsize=18, pad=15)
            ax1.set_xlabel('Predicted Label', fontsize=14, labelpad=15)
            ax1.set_ylabel('True Label', fontsize=14, labelpad=10)
            ax1.tick_params(axis='both', which='major', labelsize=11)

            # 2. Histogram of Predicted Probabilities
            ax2.hist(probabilities, color=blue, edgecolor='black', alpha=0.7, label='Probabilities')
            ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=1, label=f'Threshold: {threshold:.{decimal}f}')
            ax2.set_title('Histogram of Predicted Probabilities', fontsize=18, pad=15)
            ax2.set_xlabel('Probability', fontsize=14, labelpad=15)
            ax2.set_ylabel('Frequency', fontsize=14, labelpad=10)
            ax2.set_xticks(np.arange(0, 1.1, 0.1))
            ax2.legend()


            # 3. ROC Curve
            fpr, tpr, thresholds = roc_curve(y_test, probabilities, pos_label=pos_label)

            RocCurveDisplay.from_estimator(estimator, X_test, y_test, pos_label=pos_label, marker='.', ax=ax3, linewidth=2, label=title, color=blue, response_method='predict_proba')
            if threshold is not None:
                # Determine the closest matching fpr and tpr points
                closest_idx = np.argmin(np.abs(thresholds - threshold))
                fpr_point = fpr[closest_idx]
                tpr_point = tpr[closest_idx]
                
                # Scatter plot to highlight the threshold point
                ax3.scatter(fpr_point, tpr_point, color='red', s=80, zorder=5, label=f'Threshold {threshold:.{decimal}f}')

                # Add vertical and horizontal lines to highlight the threshold's TPR and FPR
                ax3.axvline(x=fpr_point, ymax=tpr_point-0.025, color='red', linestyle='--', lw=1, label=f'TPR: {tpr_point:.{decimal}f}, FPR: {fpr_point:.{decimal}f}')
                ax3.axhline(y=tpr_point, xmax=fpr_point+0.05, color='red', linestyle='--', lw=1)

            ax3.set_xticks(np.arange(0, 1.1, 0.1))
            ax3.set_yticks(np.arange(0, 1.1, 0.1))
            ax3.set_ylim(0,1.05)
            ax3.set_xlim(-0.05,1.0)
            ax3.grid(which='both', color='lightgrey', linewidth=0.5)
            ax3.set_title('ROC Curve', fontsize=18, pad=15)
            ax3.set_xlabel('False Positive Rate', fontsize=14, labelpad=15)
            ax3.set_ylabel('True Positive Rate', fontsize=14, labelpad=10)
            ax3.legend(loc='lower right')

            # 4. Precision-Recall Curve
            precision, recall, thresholds = precision_recall_curve(y_test, probabilities, pos_label=pos_label)
            ax4.plot(recall, precision, marker='.', label=title, color=blue)
            if threshold is not None:
                chosen_threshold = threshold
                closest_point = np.argmin(np.abs(thresholds - chosen_threshold))
                ax4.scatter(recall[closest_point], precision[closest_point], color='red', s=80, zorder=5, label=f'Threshold: {chosen_threshold:.{decimal}f}')
                #ax4.axvline(x = recall[closest_point], ymax=precision[closest_point]-0.05, color = 'red', linestyle='--', lw=1, label=f'Precision: {precision[closest_point]:.{decimal}f}, Recall: {recall[closest_point]:.{decimal}f}')
                ax4.axvline(x = recall[closest_point], ymax=precision[closest_point]-0.025, color = 'red', linestyle='--', lw=1, label=f'Precision: {precision[closest_point]:.{decimal}f}, Recall: {recall[closest_point]:.{decimal}f}')
                ax4.axhline(y = precision[closest_point], xmax=recall[closest_point]-0.025, color = 'red', linestyle='--', lw=1)
            ax4.set_xticks(np.arange(0, 1.1, 0.1))
            ax4.set_yticks(np.arange(0, 1.1, 0.1))
            ax4.set_ylim(0,1.05)
            ax4.set_xlim(0,1.05)
            ax4.grid(which='both', color='lightgrey', linewidth=0.5)
            ax4.set_title('Precision-Recall Curve', fontsize=18, pad=15)
            ax4.set_xlabel('Recall', fontsize=14, labelpad=15)
            ax4.set_ylabel('Precision', fontsize=14, labelpad=10)
            ax4.legend(loc='lower left')

            plt.tight_layout()
            plt.show()

    if return_metrics:
        return metrics
        
def compare_models(X, y, test_size=0.25, model_list=None, search_type='grid', grid_params=None, cv_folds=5,
                   ohe_columns=None, drop='if_binary', plot_perf=False, scorer='accuracy', neg_label=None, pos_label=None,
                   random_state=42, decimel=4, verbose=4, title=None, fig_size=(12,6), figmulti=1.5, legend_loc='best',
                   model_eval=False, svm_proba=False, threshold=0.5, class_weight=None, stratify=None,
                   under_sample=None, over_sample=None, notes=None, svm_knn_resample=None, n_jobs=None, output=True):

    if len(np.unique(y)) > 2:
        class_type = 'multi'
        average = 'weighted'
    else:
        class_type = 'binary'
        average = 'binary'

    if (model_list is None) or (grid_params is None):
        raise ValueError("Please specify a model_list and grid_params.")
        
    def get_scorer_and_name(scorer, pos_label=None):
        if scorer == 'accuracy':
            scoring_function = 'accuracy'
            display_name = 'Accuracy'
        elif scorer == 'precision':
            if not pos_label:
                raise ValueError("Please specify a 'pos_label' parameter if you're using precision.")
            scoring_function = make_scorer(precision_score, pos_label=pos_label)
            display_name = 'Precision'
        elif scorer == 'recall':
            if not pos_label:
                raise ValueError("Please specify a 'pos_label' parameter if you're using recall.")
            scoring_function = make_scorer(recall_score, pos_label=pos_label)
            display_name = 'Recall'
        elif scorer == 'f1':
            if not pos_label:
                raise ValueError("Please specify a 'pos_label' parameter if you're using f1 scores.")
            scoring_function = make_scorer(f1_score, pos_label=pos_label)
            display_name = 'F1'
        elif scorer == 'roc_auc':
            scoring_function = 'roc_auc'
            display_name = 'ROC AUC'
        else:
            raise ValueError(f"Unsupported scorer: {scorer}")

        return scoring_function, display_name

    scorer, scorer_name = get_scorer_and_name(scorer=scorer, pos_label=pos_label)
    
    current_time = datetime.now(pytz.timezone('US/Pacific'))
    timestamp = current_time.strftime('%b %d, %Y %I:%M %p PST')

    if output:
        print(f"\n-----------------------------------------------------------------------------------------")
        print(f"Starting Data Processing - {timestamp}")
        print(f"-----------------------------------------------------------------------------------------\n")

        print("Train/Test split, test_size: ", test_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)
    if output:
        print("X_train, X_test, y_train, y_test shapes: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    if over_sample:
        if output:
            print("\nOversampling via SMOTE strategy: ", over_sample)
            print("X_train, y_train shapes before: ", X_train.shape, y_train.shape)
            print("y_train value counts before: ", y_train.value_counts())
            print("Running SMOTE on X_train, y_train...")
        over = SMOTE(sampling_strategy=over_sample, random_state=42)
        X_train, y_train = over.fit_resample(X_train, y_train)
        if output:
            print("X_train, y_train shapes after: ", X_train.shape, y_train.shape)
            print("y_train value counts after: ", y_train.value_counts())

    if under_sample:
        if output:
            print("\nUndersampling via RandomUnderSampler strategy: ", under_sample)
            print("X_train, y_train shapes before: ", X_train.shape, y_train.shape)
            print("y_train value counts before: ", y_train.value_counts())
            print("Running RandomUnderSampler on X_train, y_train...")
        under = RandomUnderSampler(sampling_strategy=under_sample, random_state=42)
        X_train, y_train = under.fit_resample(X_train, y_train)
        if output:
            print("X_train, y_train shapes after: ", X_train.shape, y_train.shape)
            print("y_train value counts after: ", y_train.value_counts())
        
    model_name_list = []
    fit_time_list = []
    fit_count_list = []
    avg_fit_time_list = []
    inference_time_list = []
    train_score_list = []
    test_score_list = []
    overfit_list = []
    overfit_diff_list = []
    best_param_list = []
    best_cv_score_list = []
    best_estimator_list = []
    timestamp_list = []
    
    train_accuracy_list = []
    test_accuracy_list = []

    train_precision_list = []
    test_precision_list = []

    train_recall_list = []
    test_recall_list = []

    train_f1_list = []
    test_f1_list = []

    train_roc_auc_list = []
    test_roc_auc_list = []

    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []
    TPR_list = []
    FPR_list = []
    TNR_list = []
    FNR_list = []
    FR_list = []
    
    resample_list = []
    
    resample_completed = False
    
    def resample_for_knn_svm(X_train, y_train):
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, test_size=1-svm_knn_resample, stratify=y_train, random_state=random_state
            )
            if output:
                print(f"Training data resampled to {svm_knn_resample*100}% of original for KNN and SVM speed improvement")
                print("X_train, y_train shapes after: ", X_train.shape, y_train.shape)
                print("y_train value counts after: ", y_train.value_counts(), "\n")
            
            return X_train, y_train

        
    def run_grid(model_type):
        if search_type == 'grid':
            grid = GridSearchCV(pipe, param_grid=grid_params[model_type], scoring=scorer, verbose=verbose, cv=cv_folds, n_jobs=n_jobs)
        elif search_type == 'random':
            grid = RandomizedSearchCV(pipe, param_distributions=grid_params[model_type], scoring=scorer,
                                  verbose=verbose, cv=cv_folds, random_state=random_state, n_jobs=n_jobs)
        else:
            raise ValueError("search_type should be either 'grid' for GridSearchCV, or 'random' for RandomizedSearchCV")
        
        return grid

    search_string = search_type.capitalize()
        
    for i in range(len(model_list)):
        
        model = model_list[i]
        total = len(model_list)
        
        current_time = datetime.now(pytz.timezone('US/Pacific'))
        timestamp = current_time.strftime('%b %d, %Y %I:%M %p PST')
        timestamp_list.append(timestamp)
    
        if model == LogisticRegression:
            model_name = 'LogisticRegression'
            if output:
                print(f"\n-----------------------------------------------------------------------------------------")
                print(f"{i+1}/{total}: Starting LogisticRegression {search_string} Search - {timestamp}")
                print(f"-----------------------------------------------------------------------------------------\n")
            if ohe_columns:
                transformer = make_column_transformer(
                    (OneHotEncoder(drop=drop), ohe_columns),
                    remainder='passthrough'
                )
                pipe = Pipeline([
                    ('ohe', transformer),
                    ('scaler', StandardScaler(with_mean=False)),
                    ('logreg', LogisticRegression(max_iter=10000, random_state=random_state, class_weight=class_weight))
                ])
            else:
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('logreg', LogisticRegression(max_iter=10000, random_state=random_state, class_weight=class_weight))
                ])            
            grid = run_grid(model_type='logreg')
            
            resample_list.append("None")

        elif model == DecisionTreeClassifier:
            model_name = 'DecisionTreeClassifier'
            if output:
                print(f"\n-----------------------------------------------------------------------------------------")
                print(f"{i+1}/{total}: Starting DecisionTreeClassifier {search_string} Search - {timestamp}")
                print(f"-----------------------------------------------------------------------------------------\n")
            if ohe_columns:
                transformer = make_column_transformer(
                    (OneHotEncoder(drop=drop), ohe_columns),
                    remainder='passthrough'
                )
                pipe = Pipeline([
                    ('ohe', transformer),
                    ('tree', DecisionTreeClassifier(random_state=random_state, class_weight=class_weight))
                ])
            else:
                pipe = Pipeline([
                    ('tree', DecisionTreeClassifier(random_state=random_state, class_weight=class_weight))
                ])
            grid = run_grid(model_type='tree')
            
            resample_list.append("None")

        elif model == KNeighborsClassifier:
            model_name = 'KNeighborsClassifier'
            if output:
                print(f"\n-----------------------------------------------------------------------------------------")
                print(f"{i+1}/{total}: Starting KNeighborsClassifier {search_string} Search - {timestamp}")
                print(f"-----------------------------------------------------------------------------------------\n")
            
            # Check if svm_knn_resample is set and model is SVC
            if svm_knn_resample is not None and resample_completed is False:
                X_train, y_train = resample_for_knn_svm(X_train, y_train)
                resample_completed = True
            if resample_completed:
                resample_list.append(svm_knn_resample)
            else:
                resample_list.append("None")
                
            if ohe_columns:
                transformer = make_column_transformer(
                    (OneHotEncoder(drop=drop), ohe_columns),
                    remainder='passthrough'
                )
                pipe = Pipeline([
                    ('ohe', transformer),
                    ('scaler', StandardScaler(with_mean=False)),
                    ('knn', KNeighborsClassifier())
                ])
            else:
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('knn', KNeighborsClassifier())
                ])
            grid = run_grid(model_type='knn')


        elif model == SVC:
            model_name = 'SVC'
 
            if output:
                print(f"\n-----------------------------------------------------------------------------------------")
                print(f"{i+1}/{total}: Starting SVC {search_string} Search - {timestamp}")
                print(f"-----------------------------------------------------------------------------------------\n")          
            
            # Check if svm_knn_resample is set and model is SVC
            if svm_knn_resample is not None and resample_completed is False:
                X_train, y_train = resample_for_knn_svm(X_train, y_train)
                resample_completed = True
            elif resample_completed:
                resample_list.append(svm_knn_resample)
                if output:
                    print(f"Training data already resampled to {svm_knn_resample*100}% of original for KNN and SVM speed improvement")
                    print("X_train, y_train shapes: ", X_train.shape, y_train.shape)
                    print("y_train value counts: ", y_train.value_counts(), "\n")
            else:
                resample_list.append("None")
                
            if svm_proba:
                if ohe_columns:
                    transformer = make_column_transformer(
                        (OneHotEncoder(drop=drop), ohe_columns),
                        remainder='passthrough'
                    )
                    pipe = Pipeline([
                        ('ohe', transformer),
                        ('scaler', StandardScaler(with_mean=False)),
                        ('svm', SVC(random_state=random_state, probability=True, class_weight=class_weight))
                    ])
                else:
                    pipe = Pipeline([
                        ('scaler', StandardScaler()),
                        ('svm', SVC(random_state=random_state, probability=True, class_weight=class_weight))
                    ])
            else:
                if ohe_columns:
                    transformer = make_column_transformer(
                        (OneHotEncoder(drop=drop), ohe_columns),
                        remainder='passthrough'
                    )
                    pipe = Pipeline([
                        ('ohe', transformer),
                        ('scaler', StandardScaler(with_mean=False)),
                        ('svm', SVC(random_state=random_state, class_weight=class_weight))
                    ])
                else:
                    pipe = Pipeline([
                        ('scaler', StandardScaler()),
                        ('svm', SVC(random_state=random_state, class_weight=class_weight))
                    ])
            grid = run_grid(model_type='svm')
        
        model_name_list.append(model_name)
                
        # Fit the model and measure total time
        start_time = time.time()
        grid.fit(X_train, y_train)
        fit_time = time.time() - start_time
        fit_time_list.append(fit_time)
        if output:
            print(f"\nTotal Time: {fit_time:.{decimel}f} seconds")
        
        # Calculate average fit time
        fit_count = len(grid.cv_results_['params']) * cv_folds
        fit_count_list.append(fit_count)        
        avg_fit_time = fit_time / fit_count
        avg_fit_time_list.append(avg_fit_time)
        if output:
            print(f"Average Fit Time: {avg_fit_time:.{decimel}f} seconds")
        
        # Create train predictions for overfitting comparisons
        y_train_pred = grid.predict(X_train)

        # Measure the inference time and predict the labels
        start_time = time.time()
        y_test_pred = grid.predict(X_test)
        inference_time = time.time() - start_time
        inference_time_list.append(inference_time)
        if output:
            print(f"Inference Time: {inference_time:.{decimel}f}")
 
        # Calculate the train metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average=average)
        train_recall = recall_score(y_train, y_train_pred, average=average)
        train_f1 = f1_score(y_train, y_train_pred, average=average)
        if class_type == 'multi':
            train_roc_auc = roc_auc_score(y_train, grid.predict_proba(X_train), multi_class='ovr')
        else:
            train_roc_auc = roc_auc_score(y_train, grid.predict_proba(X_train)[:,1])
        train_accuracy_list.append(train_accuracy)
        train_precision_list.append(train_precision)
        train_recall_list.append(train_recall)
        train_f1_list.append(train_f1)
        train_roc_auc_list.append(train_roc_auc)

        # Calculate the test metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average=average)
        test_recall = recall_score(y_test, y_test_pred, average=average)
        test_f1 = f1_score(y_test, y_test_pred, average=average)
        if class_type == 'multi':
            test_roc_auc = roc_auc_score(y_test, grid.predict_proba(X_test), multi_class='ovr')
        else:
            test_roc_auc = roc_auc_score(y_test, grid.predict_proba(X_test)[:,1])
        test_accuracy_list.append(test_accuracy)
        test_precision_list.append(test_precision)
        test_recall_list.append(test_recall)
        test_f1_list.append(test_f1)
        test_roc_auc_list.append(test_roc_auc)
        
        best_cv_score = grid.best_score_
        best_cv_score_list.append(best_cv_score)
        if output:
            print(f"Best CV {scorer_name} Score: {best_cv_score:.{decimel}f}")
        
        train_score = grid.score(X_train, y_train)
        train_score_list.append(train_score)
        if output:
            print(f"Train {scorer_name} Score: {train_score:.{decimel}f}")
        
        test_score = grid.score(X_test, y_test)
        test_score_list.append(test_score)
        if output:
            print(f"Test {scorer_name} Score: {test_score:.{decimel}f}")
            
        overfit_diff = train_score - test_score
        overfit_diff_list.append(overfit_diff)
        if train_score > test_score:
            overfit = 'Yes'
        else:
            overfit = 'No'
        overfit_list.append(overfit)
        if output:
            print(f"Overfit: {overfit}")
            print(f"Overfit Difference: {overfit_diff:.{decimel}f}")

        best_estimator = grid.best_estimator_
        best_estimator_list.append(best_estimator)
        best_params = grid.best_params_
        best_param_list.append(best_params)
        if output:
            print(f"Best Parameters: {best_params}")
        
        if model_eval:
            if model_name != 'SVC' or (model_name == 'SVC' and svm_proba == True):
                preds = grid.predict(X_test)
                if class_type == 'binary':
                    binary_metrics = eval_model(y_test=y_test, preds=preds, X_test=X_test, estimator=grid,
                            neg_label=neg_label, pos_label=pos_label, class_type=class_type,
                            title=f"{model_name} {search_string} Search: T={threshold:.{decimel}f}", threshold=threshold,
                            decimal=2, plot=True, figsize=(12,11), class_weight=class_weight, return_metrics=True, output=output)
                elif class_type == 'multi':
                    eval_model(y_test=y_test, preds=preds, X_test=X_test, estimator=grid,
                            neg_label=neg_label, pos_label=pos_label, class_type=class_type,
                            title=f"{model_name} {search_string} Search", output=output,
                            decimal=2, plot=True, figmulti=figmulti, model_name=model_name, class_weight=class_weight)

        if binary_metrics is not None:
            TP = binary_metrics['True Positives']
            FP = binary_metrics['False Positives']
            TN = binary_metrics['True Negatives']
            FN = binary_metrics['False Negatives']
            TPR = binary_metrics['TPR']
            FPR = binary_metrics['FPR']
            TNR = binary_metrics['TNR']
            FNR = binary_metrics['FNR']
            FR = FNR + FPR
            
            TP_list.append(TP)
            FP_list.append(FP)
            TN_list.append(TN)
            FN_list.append(FN)
            TPR_list.append(TPR)
            FPR_list.append(FPR)
            TNR_list.append(TNR)
            FNR_list.append(FNR)
            FR_list.append(FR)
        
        results_df = pd.DataFrame({'Model': model_name_list,
                                   'Test Size': [test_size] * len(model_name_list),
                                   'Over Sample': [over_sample] * len(model_name_list),
                                   'Under Sample': [under_sample] * len(model_name_list),
                                   'Resample': resample_list,
                                   'Total Fit Time': fit_time_list,
                                   'Fit Count': fit_count_list,
                                   'Average Fit Time': avg_fit_time_list,
                                   'Inference Time': inference_time_list,
                                   'Grid Scorer': [scorer_name] * len(model_name_list),
                                   'Best Params': best_param_list,
                                   'Best CV Score': best_cv_score_list,
                                   'Train Score': train_score_list,
                                   'Test Score': test_score_list,
                                   'Overfit': overfit_list,
                                   'Overfit Difference': overfit_diff_list,
                                   'Train Accuracy Score': train_accuracy_list,
                                   'Test Accuracy Score': test_accuracy_list,
                                   'Train Precision Score': train_precision_list,
                                   'Test Precision Score': test_precision_list,
                                   'Train Recall Score': train_recall_list,
                                   'Test Recall Score': test_recall_list,
                                   'Train F1 Score': train_f1_list,
                                   'Test F1 Score': test_f1_list,
                                   'Train ROC AUC Score': train_roc_auc_list,
                                   'Test ROC AUC Score': test_roc_auc_list,
                                   'Threshold': [threshold] * len(model_name_list),
                                   'True Positives': TP_list,
                                   'False Positives': FP_list,
                                   'True Negatives': TN_list,
                                   'False Negatives': FN_list,
                                   'TPR': TPR_list,
                                   'FPR': FPR_list,
                                   'TNR': TNR_list,
                                   'FNR': FNR_list,
                                   'False Rate': FR_list,
                                   'Notes': [notes] * len(model_name_list),
                                   'Timestamp': timestamp_list
                                  })
    
    if plot_perf:
        
        score_df = results_df.melt(id_vars=['Model'], 
                            value_vars=[f'Best CV Score', f'Train Score', f'Test Score'], 
                            var_name='Split', value_name=f'{scorer_name}')

        plt.figure(figsize=fig_size)
        sns.barplot(data=score_df, x='Model', y=f'{scorer_name}', hue='Split')
        plt.title(f'{title} {scorer_name} Scores by Model and Data Split', fontsize=18, pad=15)
        plt.yticks(np.arange(0,1.1,0.1))
        plt.xlabel('Model', fontsize=14, labelpad=10)
        plt.ylabel(f'{scorer_name}', fontsize=14, labelpad=10)
        #plt.legend(loc='upper left', bbox_to_anchor=(1,1))
        plt.legend(loc=legend_loc)
        plt.show()

        plt.figure(figsize=fig_size)
        sns.barplot(data=results_df, x='Model', y='Average Fit Time')
        plt.title(f'{title} Average Fit Time by Model', fontsize=18, pad=15)
        plt.xlabel('Model', fontsize=14, labelpad=10)
        plt.ylabel('Average Fit Time (seconds)', fontsize=14, labelpad=10)
        plt.show()
 
     
    return results_df


def convert_dtypes(df, cols, target_dtype, show_results=False):
    """
    Converts specified columns in a dataframe to the desired data type.
    
    Parameters:
    - df (pd.DataFrame): The dataframe with the columns to be converted.
    - cols (list): List of column names to be converted.
    - target_dtype (type or str): The desired data type. e.g. float, int, 'str', etc.
    - show_results (bool, optional): If True, will print results of each successful conversion. Default is False.

    Returns:
    - None. Modifies the input dataframe in-place.
    """
    for col in cols:
        try:
            current_dtype = df[col].dtype
            df[col] = df[col].astype(target_dtype)
            if show_results:
                print(f"Successfully converted column '{col}' from {current_dtype} to {target_dtype}.")
        except ValueError as e:
            print(f"Error converting column: {col} (Current dtype: {current_dtype}). Error message: {e}")

    return None


def build_tree(X, y, title=None, test_size=0.25, stratify=None, max_depth=5, ohe_columns=None, drop='if_binary', 
               classifier='dt', random_state=42, min_samples_split=2, min_samples_leaf=1, criterion='gini', class_weight=None,
               fig_size=(12,6), watermark=False, iterate=True, list_tree=False, draw_tree=True, plot_perf=True, decimel=4,
               over_sample=None, under_sample=None, fontsize=None, class_names=None, rounded=False, proportion=False):
    """
    Build and visualize a decision tree classifier based on input data and parameters.
    Can also iterate through various tree depths to evaluate performance.
    
    Parameters:
    - X (DataFrame): Features used to train the decision tree.
    - y (Series): Target variable.
    - title (str, optional): Title to be used in the visualization. Defaults to None.
    - test_size (float, optional): Fraction of data to be used as test set. Defaults to 0.25.
    - max_depth (int, optional): Maximum depth of the tree. If set to None, tree grows without restriction. Defaults to 5.
    - ohe_columns (list, optional): Columns to be one-hot encoded. Defaults to None.
    - drop (str, optional): Strategy to use to drop one of the categories per feature during OHE. Defaults to 'if_binary'.
    - random_state (int, optional): Seed for reproducibility. Defaults to 42.
    - fig_size (tuple, optional): Figure size for visualization. Defaults to (12, 6).
    - watermark (bool, optional): Add watermark to the plot. Defaults to False.
    - iterate (bool, optional): Build trees of various depths to visualize performance. Defaults to True.
    - list_tree (bool, optional): Print the tree structure in the console. Defaults to False.
    - draw_tree (bool, optional): Draw the tree structure. Defaults to True.
    - plot_perf (bool, optional): Plot performance of the tree. Defaults to True.
    - decimel (int, optional): Number of decimal places for accuracy. Defaults to 4.
    
    Returns:
    - None: Function mainly used for side effects (visualization, printing, etc.)
    
    Notes:
    - Uses a pipeline to chain multiple operations, especially useful when one-hot encoding is required.
    - Iteratively builds trees of increasing depth (from 1 to max_depth) and captures their performance.
    - Visualizes the performance of these trees to identify potential overfitting.
    """
    is_dataframe = isinstance(X, pd.DataFrame)
  
    print("Train/Test split, test_size: ", test_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)
    print("X_train, X_test, y_train, y_test shapes: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    if over_sample:
        print("\nOversampling via SMOTE strategy: ", over_sample)
        print("X_train, y_train shapes before: ", X_train.shape, y_train.shape)
        print("y_train value counts before: ", y_train.value_counts())
        print("Running SMOTE on X_train, y_train...")
        over = SMOTE(sampling_strategy=over_sample, random_state=42)
        X_train, y_train = over.fit_resample(X_train, y_train)
        print("X_train, y_train shapes after: ", X_train.shape, y_train.shape)
        print("y_train value counts after: ", y_train.value_counts())

    if under_sample:
        print("\nUndersampling via RandomUnderSampler strategy: ", under_sample)
        print("X_train, y_train shapes before: ", X_train.shape, y_train.shape)
        print("y_train value counts before: ", y_train.value_counts())
        print("Running RandomUnderSampler on X_train, y_train...")
        under = RandomUnderSampler(sampling_strategy=under_sample, random_state=42)
        X_train, y_train = under.fit_resample(X_train, y_train)
        print("X_train, y_train shapes after: ", X_train.shape, y_train.shape)
        print("y_train value counts after: ", y_train.value_counts())
       
    actual_max_depth = None
    
    def iterate_model(depth):

        if classifier == 'dt':
            if ohe_columns and is_dataframe:
                transformer = make_column_transformer(
                    (OneHotEncoder(drop=drop), ohe_columns),
                    remainder='passthrough'
                )
                pipe = Pipeline([
                    ('ohe', transformer),
                    ('tree', DecisionTreeClassifier(max_depth=depth, random_state=random_state, class_weight=class_weight,
                                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion))
                ])

            else:
                pipe = Pipeline([
                    ('tree', DecisionTreeClassifier(max_depth=depth, random_state=random_state, class_weight=class_weight,
                                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion))
                ])
        elif classifier == 'rf':
            if ohe_columns and is_dataframe:
                transformer = make_column_transformer(
                    (OneHotEncoder(drop=drop), ohe_columns),
                    remainder='passthrough'
                )
                pipe = Pipeline([
                    ('ohe', transformer),
                    ('tree', RandomForestClassifier(max_depth=depth, random_state=random_state, class_weight=class_weight,
                                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion))
                ])

            else:
                pipe = Pipeline([
                    ('tree', RandomForestClassifier(max_depth=depth, random_state=random_state, class_weight=class_weight,
                                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion))
                ])
            
        
        pipe.fit(X_train, y_train)
        
        train_preds = pipe.predict(X_train)
        test_preds = pipe.predict(X_test)
        
        train_acc = accuracy_score(train_preds, y_train)
        test_acc = accuracy_score(test_preds, y_test)
        
        return pipe, train_acc, test_acc

    
    if iterate:
        train_acc_list = []
        test_acc_list = []
        overfit_depth = None
        overfit_set = False
        if max_depth:
            depth_list = list(range(1, max_depth + 1))
        else:
            # Determine the actual max depth of a tree grown without restriction
            pipe, _, _ = iterate_model(depth=None)
            actual_max_depth = pipe.named_steps.tree.tree_.max_depth

            # Use this max depth to create depth_list
            depth_list = list(range(1, actual_max_depth + 1))
        
        for depth in depth_list:
            pipe, train_acc, test_acc = iterate_model(depth=depth)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            if not overfit_set and train_acc > test_acc:
                overfit_depth = depth
                overfit_set = True
        
        acc_df = pd.DataFrame({'Depth' : depth_list, 'Train Accuracy' : train_acc_list, 'Test Accuracy' : test_acc_list})
        
        if plot_perf:
            fig = plt.figure(figsize=fig_size)
            sns.lineplot(data=acc_df, x='Depth', y='Train Accuracy', label='Train Accuracy')
            sns.lineplot(data=acc_df, x='Depth', y='Test Accuracy', label='Test Accuracy')
            sns.scatterplot(data=acc_df, x='Depth', y='Train Accuracy')
            sns.scatterplot(data=acc_df, x='Depth', y='Test Accuracy')
            if overfit_depth:
                plt.axvline(x=overfit_depth, linestyle='--', color='red', label=f'Overfit depth: {overfit_depth}')
            if actual_max_depth:
                plt.title(f'{title} Decision Tree Performance (Determined Depth={actual_max_depth}, Test Size={test_size})', fontsize=18, pad=15)
            else:
                plt.title(f'{title} Decision Tree Performance (Max Depth={max_depth}, Test Size={test_size})', fontsize=18, pad=15)
            plt.xticks(depth_list)
            plt.xlabel('Depth', fontsize=14, labelpad=10)
            plt.ylabel('Accuracy', fontsize=14, labelpad=10)
            plt.legend(loc='best')
            if watermark:
                fig.text(0.5, 0.5, f'{test_size}', fontsize=100, color='lightgrey', ha='center', va='center', alpha=0.4, zorder=0)
            plt.show()

    else:
        pipe, train_acc, test_acc = iterate_model(depth=max_depth)

    if list_tree:
        if actual_max_depth:
            print(f'\n{title} Decision Tree (Determined Depth={actual_max_depth}, Test Size={test_size}, Train Acc={train_acc:.{decimel}f}, Test Acc={test_acc:.{decimel}f})\n')
        else:
            print(f'\n{title} Decision Tree (Max Depth={max_depth}, Test Size={test_size}, Train Acc={train_acc:.{decimel}f}, Test Acc={test_acc:.{decimel}f})\n')
        if ohe_columns and is_dataframe:
            tree_text = export_text(pipe.named_steps.tree, feature_names = list(pipe.named_steps.ohe.get_feature_names_out()))
        elif is_dataframe:
            tree_text = export_text(pipe.named_steps.tree, feature_names = list(X.columns))
        else:
            tree_text = export_text(pipe.named_steps.tree, feature_names = [f'Feature{i}' for i in range(X.shape[1])])
        print(tree_text)

    if draw_tree:
        plt.figure(figsize=fig_size)
        if ohe_columns and is_dataframe:
            plot_tree(pipe.named_steps.tree, filled=True, rounded=rounded, fontsize=fontsize, class_names=class_names,
                      proportion=proportion, feature_names=list(pipe.named_steps.ohe.get_feature_names_out()))
        elif is_dataframe:
            plot_tree(pipe.named_steps.tree, filled=True, rounded=rounded, fontsize=fontsize, class_names=class_names,
                      proportion=proportion, feature_names=list(X.columns))
        else:
            plot_tree(pipe.named_steps.tree, filled=True, rounded=rounded, fontsize=fontsize, class_names=class_names,
                      proportion=proportion, feature_names = [f'Feature{i}' for i in range(X.shape[1])])
        if actual_max_depth:
            plt.title(f'{title} Decision Tree (Determined Depth={actual_max_depth}, Test Size={test_size}, Train Acc={train_acc:.{decimel}f}, Test Acc={test_acc:.{decimel}f})', fontsize=18, pad=15)         
        else:
            plt.title(f'{title} Decision Tree (Max Depth={max_depth}, Test Size={test_size}, Train Acc={train_acc:.{decimel}f}, Test Acc={test_acc:.{decimel}f})', fontsize=18, pad=15)
        plt.show()
    
    return None


def search_trees(X, y, test_size=0.25, searcher_list=None, grid_params=None, ohe_columns=None, drop='if_binary',
                 plot_perf=False, scorer='accuracy', pos_label=None, random_state=42, decimel=4, verbose=4,
                 title=None, fig_size=(12,6), legend_loc='best'):

    if (searcher_list is None) or (grid_params is None):
        raise ValueError("Please specify a searcher_list and grid_params.")
        
    def get_scorer_and_name(scorer, pos_label=None):
        if scorer == 'accuracy':
            scoring_function = 'accuracy'
            display_name = 'Accuracy'
        elif scorer == 'precision':
            if not pos_label:
                raise ValueError("Please specify a 'pos_label' parameter if you're using precision.")
            scoring_function = make_scorer(precision_score, pos_label=pos_label)
            display_name = 'Precision'
        elif scorer == 'recall':
            if not pos_label:
                raise ValueError("Please specify a 'pos_label' parameter if you're using recall.")
            scoring_function = make_scorer(recall_score, pos_label=pos_label)
            display_name = 'Recall'
        elif scorer == 'f1':
            if not pos_label:
                raise ValueError("Please specify a 'pos_label' parameter if you're using f1 scores.")
            scoring_function = make_scorer(f1_score, pos_label=pos_label)
            display_name = 'F1'
        else:
            raise ValueError(f"Unsupported scorer: {scorer}")

        return scoring_function, display_name

    scorer, scorer_name = get_scorer_and_name(scorer=scorer, pos_label=pos_label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if ohe_columns:
        transformer = make_column_transformer(
            (OneHotEncoder(drop=drop), ohe_columns),
            remainder='passthrough'
        )
        pipe = Pipeline([
            ('ohe', transformer),
            ('tree', DecisionTreeClassifier(random_state=random_state))
        ])

    else:
        pipe = Pipeline([
            ('tree', DecisionTreeClassifier(random_state=random_state))
        ])
    
    searcher_name_list = []
    fit_time_list = []
    train_score_list = []
    test_score_list = []
    best_param_list = []
    best_cv_score_list = []
    best_estimator_list = []
    timestamp_list = []
        
    for i in range(len(searcher_list)):
        
        searcher = searcher_list[i]
        total = len(searcher_list)
        
        current_time = datetime.now(pytz.timezone('US/Pacific'))
        timestamp = current_time.strftime('%b %d, %Y %I:%M %p PST')
        timestamp_list.append(timestamp)
    
        if searcher == GridSearchCV:
            searcher_name = 'GridSearchCV'
            print(f"\n-----------------------------------------------------------------------------------------")
            print(f"{i+1}/{total}: Starting GridSearchCV - {timestamp}")
            print(f"-----------------------------------------------------------------------------------------\n")
            grid = GridSearchCV(pipe, param_grid=params, scoring=scorer, verbose=verbose)

        elif searcher == RandomizedSearchCV:
            searcher_name = 'RandomizedSearchCV'
            print(f"\n-----------------------------------------------------------------------------------------")
            print(f"{i+1}/{total}: Starting RandomizedSearchCV - {timestamp}")
            print(f"-----------------------------------------------------------------------------------------\n")
            print(f'\n{timestamp}\n')
            grid = RandomizedSearchCV(pipe, param_distributions=params, scoring=scorer,
                                      verbose=verbose, random_state=random_state)

        elif searcher == HalvingGridSearchCV:
            searcher_name = 'HalvingGridSearchCV'
            print(f"\n-----------------------------------------------------------------------------------------")
            print(f"{i+1}/{total}: Starting HalvingGridSearchCV - {timestamp}")
            print(f"-----------------------------------------------------------------------------------------\n")
            print(f'\n{timestamp}\n')
            grid = HalvingGridSearchCV(pipe, param_grid=params, scoring=scorer,
                                       verbose=verbose, random_state=random_state)

        elif searcher == HalvingRandomSearchCV:
            searcher_name = 'HalvingRandomSearchCV'
            print(f"\n-----------------------------------------------------------------------------------------")
            print(f"{i+1}/{total}: Starting HalvingRandomSearchCV - {timestamp}")
            print(f"-----------------------------------------------------------------------------------------\n")
            print(f'\n{timestamp}\n')
            grid = HalvingRandomSearchCV(pipe, param_distributions=params, scoring=scorer,
                                         verbose=verbose, random_state=random_state)
        
        searcher_name_list.append(searcher_name)
        start_time = time.time()
        grid.fit(X_train, y_train)
        fit_time = time.time() - start_time
        #fit_time2 = sum(grid.cv_results_['mean_fit_time'])
        fit_time_list.append(fit_time)
        print(f"\nFit Time: {fit_time:.{decimel}f} seconds")
        #print(f"\nFit Time from cv_results_: {fit_time2}")
        
        best_cv_score = grid.best_score_
        best_cv_score_list.append(best_cv_score)
        print(f"Best CV {scorer_name} Score: {best_cv_score:.{decimel}f}")
        
        train_score = grid.score(X_train, y_train)
        train_score_list.append(train_score)
        print(f"Train {scorer_name} Score: {train_score:.{decimel}f}")
        
        test_score = grid.score(X_test, y_test)
        test_score_list.append(test_score)
        print(f"Test {scorer_name} Score: {test_score:.{decimel}f}")

        best_estimator = grid.best_estimator_
        best_estimator_list.append(best_estimator)
        best_params = grid.best_params_
        best_param_list.append(best_params)
        #print(f"Best Estimator: {best_estimator}")
        print(f"Best Parameters: {best_params}")
    
    results_df = pd.DataFrame({'Searcher' : searcher_name_list,
                               'Fit Time' : fit_time_list,
                               f'Best CV {scorer_name} Score' : best_cv_score_list,
                               f'Train {scorer_name} Score' : train_score_list,
                               f'Test {scorer_name} Score' : test_score_list,
                               'Best Params' : best_param_list,
                               #'Best Estimator' : best_estimator_list
                               'Timestamp' : timestamp_list
                              })
    
    if plot_perf:
        
        score_df = results_df.melt(id_vars=['Searcher'], 
                            value_vars=[f'Best CV {scorer_name} Score', f'Train {scorer_name} Score', f'Test {scorer_name} Score'], 
                            var_name='Split', value_name=f'{scorer_name}')

        plt.figure(figsize=fig_size)
        sns.barplot(data=score_df, x='Searcher', y=f'{scorer_name}', hue='Split')
        plt.title(f'{title} Decision Tree {scorer_name} Scores by Searcher and Data Split', fontsize=18, pad=15)
        plt.yticks(np.arange(0,1.1,0.1))
        plt.xlabel('Searcher', fontsize=14, labelpad=10)
        plt.ylabel(f'{scorer_name}', fontsize=14, labelpad=10)
        #plt.legend(loc='upper left', bbox_to_anchor=(1,1))
        plt.legend(loc=legend_loc)
        plt.show()

        plt.figure(figsize=fig_size)
        sns.barplot(data=results_df, x='Searcher', y='Fit Time')
        plt.title(f'{title} Decision Tree Fit Time by Searcher', fontsize=18, pad=15)
        plt.xlabel('Searcher', fontsize=14, labelpad=10)
        plt.ylabel('Fit Time (seconds)', fontsize=14, labelpad=10)
        plt.show()
 
     
    return results_df

    
def convert_data_value(value, target_unit='MB', allow_space_between=True, show_results=False):
    """
    Convert a string value with a unit (e.g., '5 GB') to a specified target unit.
    
    Parameters:
    - value (str): The value with a unit to be converted, e.g., '5 GB', '500 KB'.
    - target_unit (str, optional): The target unit for the conversion. Default is 'MB'.
        Possible values are: 'B', 'KB', 'MB', 'GB', 'TB', 'PB'.
    - allow_space_between (bool, optional): If set to True, allows a space between the number and the unit.
        For instance, both '5GB' and '5 GB' will be valid. Default is False.
    - show_results (bool, optional): If set to True, prints the before-and-after conversion values.

    Returns:
    - float: The converted value in the target unit.
    - NaN: Returns NaN if the input is NaN

    Raises:
    - ValueError: If the unit in the input value is unrecognized.

    Examples:
    >>> convert_data_value('5 GB')
    5000.0
    >>> convert_data_value('5000 KB', target_unit='GB')
    0.005
    >>> convert_data_value('0B')
    0.0
    >>> convert_data_value('0 B', allow_space_between=True)
    0.0
    """
    
    # Conversion factors based on powers of 2
    unit_conversion = {
        'B': 0,
        'KB': 10,
        'MB': 20,
        'GB': 30,
        'TB': 40,
        'PB': 50
    }

    if pd.isna(value):
        return np.nan
    
    if allow_space_between:
        match = re.match(r'(\d+\.?\d*)\s?([A-Za-z]+)', value, re.IGNORECASE)
    else:
        match = re.match(r'(\d+\.?\d*)([A-Za-z]+)', value, re.IGNORECASE)

    if not match:
        raise ValueError(f"Invalid format for value: {value}")

    number, unit = match.groups()
    unit = unit.upper()  # Convert the unit to uppercase
    number = float(number)

    # Convert the number to bytes
    bytes_value = number * (2 ** unit_conversion[unit])
    
    # Convert the bytes value to the target unit
    converted_value = bytes_value / (2 ** unit_conversion[target_unit])

    if show_results:
        print(f"Original: {value} -> Converted: {converted_value:.2f} {target_unit}")

    return converted_value


def convert_time_value(value, target_format='%Y-%m-%d %H:%M:%S', show_results=False, zero_to_nan=False):
    """
    Convert a given time value to a specified target format.

    Parameters:
    - value (str or float): The value representing the time data. This can be:
        1. Excel serial format (e.g., '45161.23458')
        2. String format (e.g., 'YYYY-MM-DD')
        3. UNIX epoch in milliseconds (e.g., '1640304000000.0')

    - target_format (str, optional): The desired datetime format for the conversion. 
        This is a string that defines the desired output format. Uses format codes such as:
        %Y: 4-digit year (e.g., '2023')
        %m: Month as zero-padded decimal (e.g., '02' for February)
        %d: Day of the month as zero-padded decimal (e.g., '09' for the 9th day)
        %H: Hour (24-hour clock) as zero-padded decimal (e.g., '07' for 7 AM, '19' for 7 PM)
        %M: Minute as zero-padded decimal (e.g., '05' for 5 minutes past the hour)
        %S: Second as zero-padded decimal (e.g., '09' for 9 seconds past the minute)
        
        Default format is '%Y-%m-%d %H:%M:%S'. For a complete list of format codes, refer to Python's datetime documentation.

    - show_results (bool, optional): If set to True, prints the before-and-after conversion values, providing a quick visualization of the transformation.

    Returns:
    - str: The converted time value in the desired target format.
    - NaN: Returns NaN if the input is NaN, indicating missing or undefined time data.

    Raises:
    - ValueError: If the input format is unrecognized or incompatible with the known time representations.

    Examples:
    >>> convert_time_value('45161.23458')
    '2023-09-16 05:37:41'
    >>> convert_time_value('2019-02-09', target_format='%d/%m/%Y')
    '09/02/2019'
    >>> convert_time_value('1640304000000.0', target_format='%Y-%m')
    '2022-01'
    """    
    # If value is NaN, return NaN
    if pd.isna(value):
        return np.nan

    # If zero_to_nan is True, replace '0' or '0.0' with NaN
    if zero_to_nan and value in ['0', '0.0', '0.00', 0, 0.0, 0.00]:
        return np.nan

    detected_format = None

    try:
        # Convert Excel Serial to datetime
        if isinstance(value, (float, int)) or (isinstance(value, str) and "." in value):  # Check if it might be a float in string format
            float_value = float(value)
            if 40000 < float_value < 50000:  # Typical range for recent Excel serials
                datetime_val = pd.Timestamp('1900-01-01') + pd.to_timedelta(float_value, 'D')
                detected_format = "Excel Serial"
            else:
                datetime_val = pd.to_datetime(value, unit='ms')
                detected_format = "UNIX Epoch in milliseconds"
        # Assume it's already in a recognizable format (like 'YYYY-MM-DD')
        elif isinstance(value, str):
            # Define expected datetime patterns
            patterns = [
                r"^\d{4}-\d{2}-\d{2}$",               # YYYY-MM-DD
                r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", # YYYY-MM-DD HH:MM:SS
                r"^\d{2}/\d{2}/\d{4}$",               # MM/DD/YYYY
                # Add more patterns as needed
]
            if any(re.match(pattern, value) for pattern in patterns):
                datetime_val = pd.to_datetime(value)
                detected_format = "Standard Datetime String"
            else:
                raise ValueError(f"Unrecognized format for value: {value}")

        # Format conversion
        formatted_datetime = datetime_val.strftime(target_format)

    except Exception as e:
        raise ValueError(f"Error converting value: {value}. Additional info: {e}")

    if show_results:
        print(f"Original: {value} ({detected_format}) -> Converted: {formatted_datetime}")

    return formatted_datetime


def check_for_duplicates(lists_dict):
    for list_name, mylist in lists_dict.items():
        duplicates = [item for item, count in collections.Counter(mylist).items() if count > 1]
        if duplicates:
            print(f"WARNING in {list_name}: The following items are duplicated:\n {', '.join(map(str, duplicates))}")
        else:
            print(f"No duplicates found in {list_name}.")
            

def detect_outliers(df, num_columns, ratio=1.5, exclude_zeros=False, plot=False, width=15, height=2):
    """
    Detect outliers in selected numerical columns of a pandas DataFrame.
    
    Outliers are identified using Tukey's method. The interquartile range (IQR) 
    is calculated, and any points that fall below Q1 - 1.5*IQR or above 
    Q3 + 1.5*IQR are considered outliers.
    
    Parameters:
    df (DataFrame): Input dataframe
    num_columns (list): List of numerical column names to analyze
    ratio (int): Ratio of IQR to evaluate for outliers, adjust as needed
    exclude_zeros (bool): Whether to exclude zero values during analysis 
    plot (bool): Whether to output boxplots of outliers
    width (int): Width of matplotlib subplot figure
    height (int): Height of matplotlib subplot figure  
    
    Returns:
    DataFrame: Summary of outliers detected with columns:
        Column, Total Non-Null, Total Zero, Zero Percent, Outlier Count,
        Outlier Percent, Skewness, Kurtosis
    """
    outlier_data = []

    for col in num_columns:
        non_null_data = df[col].dropna()
        if exclude_zeros:
            non_null_data = non_null_data[non_null_data != 0]
        else:
            non_null_data = non_null_data

        if non_null_data.empty:
            continue
            
        Q1 = np.percentile(non_null_data, 25)
        Q3 = np.percentile(non_null_data, 75)
        IQR = iqr(non_null_data)

        lower_bound = Q1 - ratio * IQR
        upper_bound = Q3 + ratio * IQR

        outliers = non_null_data[(non_null_data < lower_bound) | (non_null_data > upper_bound)]
        outlier_count = outliers.count()
        total_non_null = non_null_data.count()
        total_zero = non_null_data[non_null_data == 0].count()
        zero_percent = round((total_zero / total_non_null) * 100, 2)

        if outlier_count > 0:
            percentage = round((outlier_count / total_non_null) * 100, 2)
            skewness = round(non_null_data.skew(), 2)
            kurtosis = round(non_null_data.kurt(), 2)
            outlier_data.append([col, total_non_null, total_zero, zero_percent, outlier_count, percentage, skewness, kurtosis])

    outlier_df = pd.DataFrame(outlier_data, columns=['Column', 'Total Non-Null', 'Total Zero', 'Zero Percent', 'Outlier Count', 
                                                     'Outlier Percent', 'Skewness', 'Kurtosis'])
    outlier_df = outlier_df.sort_values(by='Outlier Percent', ascending=False)

    if plot:
        plt.figure(figsize=(width, len(outlier_df) * height))
        plot_index = 1

        for index, row in outlier_df.iterrows():
            plt.subplot(len(outlier_df) // 2 + len(outlier_df) % 2, 2, plot_index)
            sns.boxplot(x=df[row['Column']], orient='h')
            plt.title(f"{row['Column']}, Outliers: {row['Outlier Count']} ({row['Outlier Percent']}%)")

            plot_index += 1
            
        plt.tight_layout()
        plt.show()

    return outlier_df


def compare_metrics(df, metric, models=['LogisticRegression', 'DecisionTreeClassifier', 
                                    'KNeighborsClassifier', 'SVC']):
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flatten()

    # Loop through models and plot data
    for i, model in enumerate(models):
        model_df = df[df['Model'] == model]

        # Plot train and test metrics
        axs[i].plot(model_df.index, model_df[f'Train {metric}'], label=f'Train {metric}', marker='x')
        axs[i].plot(model_df.index, model_df[f'Test {metric}'], label=f'Test {metric}', marker='o')

        # Set plot titles and labels
        axs[i].set_title(f'{model} Train/Test {metric}', fontsize=18, pad=15)
        axs[i].set_xlabel('Iteration (Different Test Sizes and Thresholds)', fontsize=14)
        axs[i].set_ylabel(metric, fontsize=14)
        axs[i].set_ylim(0, 1.05)
        axs[i].set_xlim(0, 130)
        axs[i].legend()

    # Adjust layout
    plt.tight_layout(pad=3)
    plt.show()
