from typing import Literal
from typing import Union
import jupyterlab
import ipykernel
import re

import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
import math

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from tabulate import tabulate
from optbinning import OptimalBinning



def print_notebook_versions():
    print(f"JupyterLab: {jupyterlab.__version__}\n")
    print(f"ipykernel: {ipykernel.__version__}")


def round_magnitude(n, order_magnitude:int=1):
    if n <= 0:
        return 0 
    power = 10 ** (len(str(int(n))) - order_magnitude)
    return math.ceil(n / power) * power


def count_zero(array:np.array):
    '''Counts the number of zeros in a numpy array.'''
    
    count = np.size(array) - np.count_nonzero(array)
    return count


def root_n_th(radicand_index, radicand):
    return np.power(radicand, 1/ radicand_index)


def mode_limits(list_values: list, max_mode=True):
    '''Calculates the mode of a list of values, returning the maximum or minimum mode value.'''

    mode_values = mode_list(pd.Series.mode(list_values))
    if max_mode:
        mode_values = np.max(mode_values)
    else:
        mode_values = np.min(mode_values)
    return mode_values


def mode_list(list_values: list):
    freq = mode(list_values, count_mode=True)
    
    if freq == 1:
        return None
    else:
        mode_values = list(pd.Series.mode(list_values))
        return mode_values


def mode(list_values:list, count_mode=False):
    mode_value = stats.mode(list_values, keepdims=False)
    if count_mode:
        mode_value = mode_value[1]
    else:
        mode_value = mode_value[0]
    return mode_value


def mode_freq(list_values:list):
    return mode(list_values, True)


def mean_abs_deviation(list_values: list):
    mean = np.mean(list_values)
    mean_ad = np.mean(abs(list_values - mean))
    return mean_ad


def median_abs_deviation_norm(list_values: list):
    mad = stats.median_abs_deviation(list_values)
    factor = 1 / stats.norm.ppf(0.75)
    mad_norm = mad * factor  # Factor for normal distribution
    return mad_norm


def outlier_values(list_values: list, dtype: Literal['lower', 'upper']):
    q1 = np.percentile(list_values, 25)
    q3 = np.percentile(list_values, 75)
    iqr = q3 - q1
    
    if dtype == 'lower':
        outlier = q1 - 1.5 * iqr
    elif dtype == 'upper':
        outlier = q3 + 1.5 * iqr
    else:
        outlier = None
        
    return outlier


def count_outlier_values(list_values: list, dtype: Literal['lower', 'upper']):
    if dtype == 'lower':
        outlier = outlier_values(list_values, dtype)
        lower_outliers = list_values[list_values < outlier]
        num = len(lower_outliers)
    elif dtype == 'upper':
        outlier = outlier_values(list_values, dtype)
        upper_outliers = list_values[list_values > outlier]
        num = len(upper_outliers)
    else:
        num = None
        
    return num


def describe_full_df(_df: pd.DataFrame, extend_metrics=False):
    '''Generates a full description of a DataFrame, 
    including additional statistics for numeric columns.
    '''

    cols_number = _df.select_dtypes(include='number').columns
    df_describe = _df.describe()
    
    if np.all(df_describe.columns == cols_number):

        count_isnull = lambda x: pd.Series.isnull(x).sum()
        count_negative = lambda x: sum(1 for number in x if number < 0)
        count_positive = lambda x: sum(1 for number in x if number > 0)
        perc_count_isnull = lambda x: pd.Series.isnull(x).mean() * 100
        percentile_25 = lambda x: np.percentile(x, 25)
        percentile_75 = lambda x: np.percentile(x, 75)
        amplitude = lambda x: np.max(x) - np.min(x)
        lower_outlier = lambda x: outlier_values(x, 'lower')
        upper_outlier = lambda x: outlier_values(x, 'upper')
        count_lower_outlier = lambda x: count_outlier_values(x, 'lower')
        count_upper_outlier = lambda x: count_outlier_values(x, 'upper')
        coefficient_variation = lambda x: pd.Series.std(x) / pd.Series.mean(x)
        int_mean_5p = lambda x: stats.trim_mean(x, proportiontocut=0.05)
        int_mean_25p = lambda x: stats.trim_mean(x, proportiontocut=0.25)
        ampl_over_avg = lambda x: amplitude(x) / pd.Series.mean(x)

        dict_functions = {
            'count': pd.Series.count,
            'count_isnull': count_isnull,
            '%_count_isnull': perc_count_isnull,
            'count_zero': count_zero,
            'count_nonzero': np.count_nonzero,
            'count_positive': count_positive,
            'count_negative': count_negative,
            'count_unique': pd.Series.nunique,
            'mean': pd.Series.mean,
            'geo_mean': stats.gmean,
            'harm_mean': stats.hmean,
            'int_mean_5%': int_mean_5p,
            'median': pd.Series.median,
            'mode': mode,
            'mode_list': mode_list,
            'mode_freq': mode_freq,
            'min': np.min,
            '25%': percentile_25,
            '50%': pd.Series.median,
            '75%': percentile_75,
            'max': np.max,
            'amplitude': amplitude,
            'iqr': stats.iqr,
            'lower_outlier': lower_outlier,
            'count_lower_outlier': count_lower_outlier,
            'upper_outlier': upper_outlier,
            'count_upper_outlier': count_upper_outlier,
            'int_mean_25%': int_mean_25p,
            'mean_abs_dev': mean_abs_deviation,
            'std_dev': pd.Series.std,
            'pop_std_dev': np.std,
            'median_abs_dev': stats.median_abs_deviation,
            'median_abs_dev_norm': median_abs_deviation_norm,
            'coefficient_var': coefficient_variation, 
            'ampl_over_avg': ampl_over_avg,
            'variance': np.var,
        }

        if extend_metrics:
            shapiro_stat = lambda x: stats.shapiro(x)[0]
            shapiro_pvalue = lambda x: stats.shapiro(x)[1]

            dict_functions.update({
                'kurt': pd.Series.kurt,
                'kurtosis': stats.kurtosis,
                'skew': pd.Series.skew,
                'shapiro_stat': shapiro_stat,
                'shapiro_pvalue': shapiro_pvalue,
                'autocorr': pd.Series.autocorr,
                'circvar': stats.circvar,
                'circmean': stats.circmean,
                'circstd': stats.circstd,
                'entropy': stats.entropy,
                'kstat': stats.kstat,
                'kstatvar': stats.kstatvar,
            })

        
        dict_data = {}
        for col_n in cols_number:
            
            list_metric = []
            for col, function in dict_functions.items():
                try:
                    list_metric.append(function(_df[col_n]))
                except:
                    list_metric.append(None)

            dict_data.update({col_n: list_metric})

        df_temp = pd.DataFrame(dict_data)
        df_temp.index = dict_functions.keys()

        return pd.concat([df_temp])
    

def describe_full_df_segmented(df:pd.DataFrame, column_numeric:str, column_category:str, category_values:list=None, extend_metrics=False):
    if category_values is None:
        category_values = df[column_category].unique()

    df_desc = []
    for cv in category_values:
        _ = describe_full_df(df.query(f"{column_category} == @cv"),extend_metrics=extend_metrics)[column_numeric]
        df_desc.append(_)

    df_desc = pd.concat(df_desc, axis=1)
    df_desc.columns = category_values
    df_desc.index.name = column_numeric

    return df_desc
    

def save_profile_report(df: pd.DataFrame, filename: str, title: str = "Pandas Profiling Report"):
    '''Saves a profile report of a DataFrame to an HTML file.'''
    
    from ydata_profiling import ProfileReport

    filename=f'{filename}.html'
    ProfileReport(df, title=title, explorative=True, progress_bar=False).to_file(filename)
    #profile = ProfileReport(df, title=title, explorative=True)
    #profile.to_file(filename)

    print(f"Profile report saved to {filename}")


def describe_df_category(df, col_category, col_metric):
    list_category = df[col_category].unique()
    list_describe_category = [df[col_metric].describe()]
    
    for cat in list_category:
        series_category = df[df[col_category] == cat][col_metric]
        series_category.name = cat
        list_describe_category.append(series_category.describe())
        
    df_describe = pd.concat(list_describe_category, axis=1)
    return df_describe


def convert_binary_to_descriptive(df: pd.DataFrame, cols_binary: list, values_descriptive: list = ['Não', 'Sim']):
    for c in cols_binary:
        df[c] = df[c].map({0: values_descriptive[0], 1: values_descriptive[1]})
    return df


def apply_percent_category(list_values: list):
    count_list = len(list_values)
    sum_list = sum(list_values) 
    return sum_list / count_list *100    


def convert_types(df, list_columns: list, list_types: list):
    '''Converts the specified columns of a DataFrame to a given data type.'''
    
    for col, dtype in zip(list_columns, list_types):
        if col in df.columns:
            df[col] = df[col].astype(dtype)
        else:
            print(f"Column {col} not found in DataFrame.")
    
    return df


def get_chi_square(df:pd.DataFrame, reference_category: str, target_category: list, significance_limit=0.05):
    '''Calculates the chi-square statistic for two categorical variables in a DataFrame.'''
    
    data_stat = []
    for c in target_category:
        contingency_table = pd.crosstab(df[c], df[reference_category])
        # chi2: Chi-square statistic
        chi2, p, dof, _ = stats.chi2_contingency(contingency_table)
        # p: p-value of the test 
        if p < significance_limit:
            significance = 1
        else:
            significance = 0
        values = dict(variable = c, chi_square = chi2, p_value = p, dof = dof, significance = significance)

        data_stat.append(values)

        df_stat = pd.DataFrame(data_stat)
        df_stat.chi_square = df_stat.chi_square.round(4)
        df_stat.p_value = df_stat.p_value.round(6)
        df_stat.dof = df_stat.dof.round(4)
        df_stat.sort_values(by='variable', ascending=True, inplace=True)

    return df_stat


def get_ttest_ind(df:pd.DataFrame, reference_group: str, numerical_variables: list):
    '''Calculates the ttest_ind statistic for two group of numerical variables in a DataFrame.'''
    
    data_stat = []
    for nv in numerical_variables:
        
        group_1 = df[df[reference_group] == False][nv]
        group_2 = df[df[reference_group] == True][nv]

        # t_stat: ttest_ind statistic
        t_stat, p = stats.ttest_ind(group_1, group_2)
        # p: p-value of the test 
        if p < 0.05:
            significance = 1
        else:
            significance = 0
        values = dict(variable = nv, t_stat = t_stat, p_value = p, significance = significance)

        data_stat.append(values)

        df_stat = pd.DataFrame(data_stat)
        df_stat.t_stat = df_stat.t_stat.round(4)
        df_stat.p_value = df_stat.p_value.round(6)
        df_stat.sort_values(by='variable', ascending=True, inplace=True)

    return df_stat


def get_anova_levene_test(df, tyte_test: Literal['anova', 'levene'], values_columns:list, categorical_columns:list):

    data_stat = []

    for vc in values_columns:
    
        for category in categorical_columns:
            domains = df[category].unique()
            groups = []
            for d in domains:
                group = df[df[category] == d][vc]
                groups.append(list(group))
    
            # stat: type test statistic
            if tyte_test == 'anova':
                stat, p = stats.f_oneway(*groups)
            elif tyte_test == 'levene':
                stat, p = stats.levene(*groups)
    
            # p: p-value of the test 
            if p < 0.05:
                significance = 1
            else:
                significance = 0
            values = dict(variable_numerical = vc, categorical = category, stat = stat, p_value = p, significance = significance)
    
            data_stat.append(values)

    df_stat = pd.DataFrame(data_stat)

    numeric_cols = df_stat.select_dtypes(include='number')
    df_stat[numeric_cols.columns] = numeric_cols.round(6)
    df_stat.reset_index(drop=True, inplace=True)
    
    return df_stat


def get_welch_anova_test(df:pd.DataFrame, binary_column:str, variables_multi_category:list):

    list_df = []
    
    for v in variables_multi_category:
        anova_welch = pg.welch_anova(dv=binary_column, between=v, data=df)
        anova_welch.insert(0, 'dv', binary_column)
        list_df.append(anova_welch)
    
    df_stat = pd.concat(list_df, axis=0)

    numeric_cols = df_stat.select_dtypes(include='number')
    df_stat[numeric_cols.columns] = numeric_cols.round(6)
    df_stat.reset_index(drop=True, inplace=True)
    
    return df_stat


def get_gameshowell_test(df, binary_column:str, variables_multi_category:list):

    list_df = []
    
    for v in variables_multi_category:
        gameshowell = pg.pairwise_gameshowell(dv=binary_column, between=v, data=df)
        gameshowell.insert(0, 'dv', binary_column)
        gameshowell.insert(1, 'between', v)
        list_df.append(gameshowell)

    df_stat = pd.concat(list_df, axis=0)

    numeric_cols = df_stat.select_dtypes(include='number')
    df_stat[numeric_cols.columns] = numeric_cols.round(6)
    df_stat.reset_index(drop=True, inplace=True)
    
    return df_stat


def get_variance_inflation_factor(df, variables_analyzed, print_vif = True):
    X = df[variables_analyzed]
    X = add_constant(X)
    
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    # Check the VIF for each column of X
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["tolerance"] = 1 / vif["VIF"]
    if print_vif:
        print_tabulate_df(vif)
    else:
        return vif


def get_columns_categorical(df, limit_):
    columns = []
    values = []
    for i, v in zip(df.nunique().index, df.nunique().values):
        if v <= limit_:
            unique_values = df[i].unique()
            columns.append(i)
            values.append(sorted(unique_values))
    return columns, values


def get_columns_numeric(df, limit_categorical:int=5):
    numerics = df.select_dtypes(include=['number']).columns.tolist()
    categorical, _ = get_columns_categorical(df, limit_categorical)
    numerics = list(set(numerics).difference(set(categorical)))
    return numerics


def get_bins_rule_freedman_diaconis(list_values):
    iqr = np.percentile(list_values, 75) - np.percentile(list_values, 25)
    n = len(list_values)
    bins = (2 * iqr) / np.power(n, 1/3)
    return bins


def get_bins_rule_sturges(list_values):
    ampl = np.max(list_values) - np.min(list_values)
    n = len(list_values)
    bins = ampl / (1 + 3.3 * np.log(n))
    return bins


def get_bins_rule_scott(list_values):
    std = np.std(list_values)
    n = len(list_values)
    bins = (3.49 * std) / np.power(n, 1/3)
    return bins


def create_standardized_normal_table():
    """Creates a standardized normal distribution table with Z-scores and their cumulative probabilities."""
    standardized_normal_table = pd.DataFrame(
        [], 
        index=["{0:0.2f}".format(i / 100) for i in range(0, 400, 10)],
        columns = ["{0:0.2f}".format(i / 100) for i in range(0, 10)])
    
    for index in standardized_normal_table.index:
        for column in standardized_normal_table.columns:
            Z = np.round(float(index) + float(column), 2)
            standardized_normal_table.loc[index, column] = f"{stats.norm.cdf(Z):0.8f}"
    
    standardized_normal_table.rename_axis('Z', axis = 'columns', inplace = True)

    return standardized_normal_table


def get_standardized_normal(value_z, df_norm=None, is_print=False):
    """Returns the cumulative probability for a given Z-score from a standardized normal distribution table."""
    if df_norm is None:
        df_norm = create_standardized_normal_table()
    
    value_z = round(value_z, 2)
    row = f"{np.math.floor(value_z * 10) /10:.2f}"  # integer part and first decimal place
    column = f"{value_z * 10 % 1 /10:.2f}"  # second decimal place
    if is_print:
        print(f'Row {row} e column {column}')
    
    # quering normalized table
    return float(df_norm.loc[row, column])


def get_categorical_column(df: pd.DataFrame, threshold=5, print_info=True):
    """Identifies categorical columns in a DataFrame based on a threshold for unique values."""
    categorical_column = []
    for c in df.columns:
        if df[c].nunique() <= threshold:
            categorical_column.append(c)
            
            if print_info:
                print(f'\nColumn: {c}. Number of Domains: {df[c].nunique()}. Domain Values: {np.unique(df[c])}')

    return categorical_column


def identify_columns_binary_values(df: pd.DataFrame):
    """Identifies columns in a DataFrame that contain only binary values (0 and 1)."""
    columns = []
    for c in df.columns:
        if len(set(df[c].unique()).difference(set([0, 1]))) == 0:
            columns.append(c)
            
    return columns


def get_columns_binary(df:pd.DataFrame):
    return identify_columns_binary_values(df)


def print_tabulate_df(df:pd.DataFrame, limit_rows:int=100, **kwargs):
    '''
    Print up to the first 100 rows of the dataframe
    '''
    limit_rows = min(limit_rows, 100)
    
    allowed_parameters = {
        'tablefmt', 
        'floatfmt', 
        'stralign', 
        'intfmt', 
        'numalign', 
        'missingval',
        'showindex',
        'disable_numparse',
        'colalign',
        'maxcolwidths',
        'rowalign',
        'maxheadercolwidths',
    }

    # Check for invalid keys
    invalid_keys = set(kwargs.keys()) - allowed_parameters
    if invalid_keys:
        raise ValueError(f"Allowed Parameters: {', '.join(invalid_keys)}")

    # Set default values
    tablefmt = kwargs.get('tablefmt', 'orgtbl')
    floatfmt = kwargs.get('floatfmt', '.2f')
    stralign = kwargs.get('stralign', 'left')
    intfmt = kwargs.get('intfmt', '')
    numalign = kwargs.get('numalign', 'default')
    missingval = kwargs.get('missingval', '')
    showindex = kwargs.get('showindex', 'default')
    disable_numparse = kwargs.get('disable_numparse', False)
    colalign = kwargs.get('colalign', None)
    maxcolwidths = kwargs.get('maxcolwidths', None)
    rowalign = kwargs.get('rowalign', None)
    maxheadercolwidths = kwargs.get('maxheadercolwidths', None)

    table = tabulate(
        df.head(limit_rows).values.tolist(), 
        headers=df.columns, 
        tablefmt=tablefmt, 
        floatfmt=floatfmt, 
        stralign=stralign,
        intfmt=intfmt,
        numalign=numalign,
        missingval=missingval,
        showindex=showindex,
        disable_numparse=disable_numparse,
        colalign=colalign,
        maxcolwidths=maxcolwidths,
        rowalign=rowalign,
        maxheadercolwidths=maxheadercolwidths,
    )
    
    print(table)


def adjust_column_names(df):
    df = df.copy()
    df.columns = [
        re.sub(r"[^\w]+", "_", col.strip())  # replaces non-alphanumeric characters and spaces with "_"
        for col in df.columns
    ]
    return df


def optimal_bins(df, feature, target):
    optb = OptimalBinning(name=feature, dtype="numerical", solver="cp")
    optb.fit(df[feature], df[target])
    return optb.transform(df[feature], metric="bins")


def filter_startswith(list_value, startswith_value:str):
    """
    Returns all list elements that begin with 'startswith_value' prefix.

    Args:
    list_value (list): List of strings.
    startswith_value (str): The prefix to filter.

    Returns:
    list: List containing only elements that begin with 'startswith_value' prefix.
    """
    return [item for item in list_value if isinstance(item, str) and item.startswith(startswith_value)]