from scipy import stats  # Import the 'stats' module from SciPy.

def print_test_results(statistic, p_value):
    """
    Print test results including the test statistic and p-value.

    :param statistic: Test statistic.
    :param p_value: P-value.
    """
    print(f"\t\t\tTest statistic: {statistic}")
    print(f"\t\t\tP-value: {p_value}")

def check_significance(p_value, alphas=[0.01, 0.05, 0.1]):
    """
    Check the significance of a test based on the given p-value.

    :param p_value: P-value.
    :param alphas: List of significance levels to check against (default is [0.01, 0.05, 0.1]).
    :return: The significance level at which the test is significant or None if not significant.
    """
    for alpha in alphas:
        significant = p_value < alpha
        if significant:
            break
    print(f"\t\t\t--> The test is{' NOT' if not significant else ''} significant at {alpha} level")
    return alpha if significant else None

def shapiro_test(sample, alphas=[0.01, 0.05, 0.1]):
    """
    Perform the Shapiro-Wilk test for normality on a sample.

    :param sample: Sample data to be tested for normality.
    :param alphas: List of significance levels to check against (default is [0.01, 0.05, 0.1]).
    :return: The significance level at which the test is significant or None if not significant.
    """
    statistic, p_value = stats.shapiro(sample)
    print(f"\t\tShapiro-Wilk test for normality:")
    print_test_results(statistic, p_value)
    significance = check_significance(p_value, alphas)
    print()
    return significance

def t_test(sample1, sample2=None, pop_mean=0, alternative="greater", alphas=[0.01, 0.05, 0.1]):
    """
    Calculate a T-test for the mean of one group of scores or two related samples.

    :param sample1: First sample data.
    :param sample2: Second sample data (for two related samples).
    :param pop_mean: Population mean (for one group of scores).
    :param alternative: Alternative hypothesis for the test ('greater', 'less', or 'two-sided').
    :param alphas: List of significance levels to check against (default is [0.01, 0.05, 0.1]).
    :return: The significance level at which the test is significant or None if not significant.
    """
    if sample2 is None:
        print(f"\t\tT-test for the mean of ONE group of scores with population mean {pop_mean} and {alternative} alternative:")
        statistic, p_value = stats.ttest_1samp(sample1, pop_mean=pop_mean, alternative=alternative)
    else:
        print(f"\t\tT-test for the mean of TWO RELATED samples of scores with {alternative} alternative:")
        statistic, p_value = stats.ttest_rel(sample1, sample2, alternative=alternative)
    print_test_results(statistic, p_value)
    significance = check_significance(p_value, alphas)
    print()
    return significance

def wilcoxon_test(sample1, sample2=None, alternative="greater", alphas=[0.01, 0.05, 0.1], zero_method=["wilcox", "pratt", "zsplit"]):
    """
    Calculate the Wilcoxon signed-rank test.

    :param sample1: First sample data.
    :param sample2: Second sample data (for two related samples).
    :param alternative: Alternative hypothesis for the test ('greater', 'less', or 'two-sided').
    :param alphas: List of significance levels to check against (default is [0.01, 0.05, 0.1]).
    :param zero_method: Method to handle zero differences ('wilcox', 'pratt', or 'zsplit').
    :return: List of significance levels at which the test is significant or None if not significant for each zero_method.
    """
    significance = []
    if isinstance(zero_method, str):
        zero_method = [zero_method]
    for zero_method in zero_method:
        print(f"\t\tWilcoxon signed-rank test with {zero_method} method and {alternative} alternative:")
        statistic, p_value = stats.wilcoxon(sample1, sample2, zero_method=zero_method, correction=False, alternative=alternative)
        print_test_results(statistic, p_value)
        significance.append(check_significance(p_value, alphas))
    print()
    return significance
