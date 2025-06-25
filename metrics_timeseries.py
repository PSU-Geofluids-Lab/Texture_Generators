import antropy as ant
import pandas as pd
import tsfel

def generate_entropy_measures(x):
    """
    Generate entropy measures and fractal dimension from a given time series.
    Parameters
    ----------
    x : array_like
        Time series data.

    Combines :
    -------
    dict_entropy : dict
        Dictionary of entropy measures.
    dict_fractal : dict
        Dictionary of fractal dimension measures.
        
    Return :
    -------
    df : DataFrame
        DataFrame of entropy measures and fractal dimension measures.
    Notes
    -----
    Permutation entropy is a measure of the complexity of the time series,
    which is computed by finding the permutation of the values in the time
    series that has the highest probability of occurring.

    Spectral entropy is a measure of the complexity of the time series in the
    frequency domain, which is computed by finding the probability distribution
    of the values in the time series in the frequency domain and computing the
    entropy of the distribution.

    Singular value decomposition entropy is a measure of the complexity of the
    time series, which is computed by finding the singular values of the time
    series and computing the entropy of the distribution of the singular values.

    Approximate entropy is a measure of the complexity of the time series, which
    is computed by finding the probability distribution of the values in the
    time series and computing the entropy of the distribution.

    Sample entropy is a measure of the complexity of the time series, which is
    computed by finding the probability distribution of the values in the time
    series and computing the entropy of the distribution.

    Hjorth mobility and complexity are measures of the complexity of the time
    series, which are computed by finding the probability distribution of the
    values in the time series and computing the entropy of the distribution.

    Number of zero-crossings is a measure of the complexity of the time series,
    which is computed by finding the number of zero-crossings in the time series.

    Lempel-Ziv complexity is a measure of the complexity of the time series,
    which is computed by finding the number of distinct substrings of the time
    series.

    Petrosian fractal dimension is a measure of the fractal dimension of the time
    series, which is computed by finding the box-counting dimension of the time
    series.

    Katz fractal dimension is a measure of the fractal dimension of the time
    series, which is computed by finding the box-counting dimension of the time
    series.

    Higuchi fractal dimension is a measure of the fractal dimension of the time
    series, which is computed by finding the box-counting dimension of the time
    series.

    Detrended fluctuation analysis is a measure of the fractal dimension of the
    time series, which is computed by finding the self-similarity of the time
    series.
    """
    dict_entropy = {}
    # Permutation entropy
    dict_entropy['permutation_entropy'] = ant.perm_entropy(x, normalize=True)
    # Spectral entropy
    dict_entropy['spectral_entropy'] = ant.spectral_entropy(x, sf=100, method='welch', normalize=True)
    # Singular value decomposition entropy
    dict_entropy['svd_entropy'] = ant.svd_entropy(x, normalize=True)
    # Approximate entropy
    dict_entropy['approximate_entropy'] = ant.app_entropy(x)
    # Sample entropy
    dict_entropy['sample_entropy'] = ant.sample_entropy(x)
    # Hjorth mobility and complexity
    dict_entropy['hjorth_mobility'], dict_entropy['hjorth_complexity'] = ant.hjorth_params(x)
    # Number of zero-crossings
    dict_entropy['num_zerocross'] = ant.num_zerocross(x)
    # Lempel-Ziv complexity
    dict_entropy['lziv_complexity'] = ant.lziv_complexity('01111000011001', normalize=True)

    dict_fractal = {}
    # Petrosian fractal dimension
    dict_fractal['petrosian_fd'] = ant.petrosian_fd(x)
    # Katz fractal dimension
    dict_fractal['katz_fd'] = ant.katz_fd(x)
    # Higuchi fractal dimension
    dict_fractal['higuchi_fd'] = ant.higuchi_fd(x)
    # Detrended fluctuation analysis
    dict_fractal['detrended_fluctuation'] = ant.detrended_fluctuation(x)
    # Combine the two dictionaries into a single DataFrame
    df = pd.concat([pd.DataFrame(dict_entropy, index=[0]), 
                    pd.DataFrame(dict_fractal, index=[0])], axis=1)
    return df

def generate_stat_features(x,use_windows=False,window_size=2000,overlap=0.5):
    """
    Extracts and combines statistical features from a time series.

    This function extracts time series features using the TSFEL library
    and entropy measures, and combines them into a single DataFrame.

    Parameters:
    x : array-like
        The time series data from which features are to be extracted.
    window_size : int, optional
        The size of the moving window for feature extraction. Default is 2000.
    overlap : float, optional
        The overlap ratio between consecutive windows. Default is 0.5.

    Returns:
    DataFrame
        A pandas DataFrame containing the combined features extracted from
        the time series.
    """
    if use_windows == True:
        features_set1 = tsfel.time_series_features_extractor(cfg_file_metrics_timeseries, x,resample_rate=1,
                                                        window_size=window_size,overlap=overlap)
    else : 
        features_set1 = tsfel.time_series_features_extractor(cfg_file_metrics_timeseries, x,resample_rate=1)
    features_antr = generate_entropy_measures(x)
    return features_antr,features_set1



cfg_file_metrics_timeseries = {'temporal': {'Area under the curve': {'complexity': 'log',
   'description': 'Computes the area under the curve of the signal computed with trapezoid rule.',
   'function': 'tsfel.auc',
   'parameters': {'fs': 1},
   'n_features': 1,
   'use': 'yes'},
  'Centroid': {'complexity': 'constant',
   'description': 'Computes the centroid along the time axis.',
   'function': 'tsfel.calc_centroid',
   'parameters': {'fs': 1},
   'n_features': 1,
   'use': 'yes'},
  'Lempel-Ziv complexity': {'complexity': 'linear',
   'description': "Computes the Lempel-Ziv's (LZ) complexity index, normalized by the signal's length.",
   'function': 'tsfel.lempel_ziv',
   'parameters': {'threshold': None},
   'n_features': 1,
   'use': 'yes'},
  'Mean absolute diff': {'complexity': 'constant',
   'description': 'Computes mean absolute differences of the signal.',
   'function': 'tsfel.mean_abs_diff',
   'parameters': '',
   'n_features': 1,
   'use': 'yes'},
  'Signal distance': {'complexity': 'constant',
   'description': 'Computes signal traveled distance.',
   'function': 'tsfel.distance',
   'parameters': '',
   'n_features': 1,
   'use': 'yes'},
  'Sum absolute diff': {'complexity': 'constant',
   'description': 'Computes sum of absolute differences of the signal.',
   'function': 'tsfel.sum_abs_diff',
   'parameters': '',
   'n_features': 1,
   'use': 'yes'}},
'statistical': {'Average power': {'complexity': 'constant',
   'description': 'Computes the average power of the signal.',
   'function': 'tsfel.average_power',
   'parameters': {'fs': 1},
   'n_features': 1,
   'use': 'yes',
   'tag': 'audio'},
  'ECDF Percentile': {'complexity': 'log',
   'description': 'Determines the percentile value of the ECDF.',
   'function': 'tsfel.ecdf_percentile',
   'parameters': {'percentile': '[0.2, 0.8]'},
   'n_features': 'percentile',
   'use': 'yes'},
  'Kurtosis': {'complexity': 'constant',
   'description': 'Computes kurtosis of the signal.',
   'function': 'tsfel.kurtosis',
   'parameters': '',
   'n_features': 1,
   'use': 'yes'},
  'Median absolute deviation': {'complexity': 'constant',
   'description': 'Computes median absolute deviation of the signal.',
   'function': 'tsfel.median_abs_deviation',
   'parameters': '',
   'n_features': 1,
   'use': 'yes'},
  'Skewness': {'complexity': 'constant',
   'description': 'Computes skewness of the signal.',
   'function': 'tsfel.skewness',
   'parameters': '',
   'n_features': 1,
   'use': 'yes'}},
   'fractal': {'Detrended fluctuation analysis': {'complexity': 'nlog',
   'description': 'Computes the Detrended Fluctuation Analysis (DFA) of the signal.',
   'function': 'tsfel.dfa',
   'parameters': '',
   'n_features': 1,
   'use': 'yes'},
  'Higuchi fractal dimension': {'complexity': 'squared',
   'description': "Computes the fractal dimension of a signal using Higuchi's method (HFD).",
   'function': 'tsfel.higuchi_fractal_dimension',
   'parameters': '',
   'n_features': 1,
   'use': 'yes'},
  'Hurst exponent': {'complexity': 'squared',
   'description': 'Computes the Hurst exponent of the signal through the Rescaled range (R/S) analysis.',
   'function': 'tsfel.hurst_exponent',
   'parameters': '',
   'n_features': 1,
   'use': 'yes'},
  'Maximum fractal length': {'complexity': 'squared',
   'description': "Computes the Maximum Fractal Length (MFL) of the signal, which is the average length at the smallest scale, measured from the logarithmic plot determining FD. The Higuchi's method is used.",
   'function': 'tsfel.maximum_fractal_length',
   'parameters': '',
   'n_features': 1,
   'use': 'yes'},
  'Petrosian fractal dimension': {'complexity': 'log',
   'description': 'Computes the Petrosian Fractal Dimension of a signal.',
   'function': 'tsfel.petrosian_fractal_dimension',
   'parameters': '',
   'n_features': 1,
   'use': 'yes'},
  'Multiscale entropy': {'complexity': 'linear',
   'description': 'Computes the Multiscale entropy (MSE) of the signal, that performs the entropy analysis over multiple time scales.',
   'function': 'tsfel.mse',
   'parameters': {'m': 3, 'maxscale': None, 'tolerance': None},
   'n_features': 1,
   'use': 'yes'}}}

