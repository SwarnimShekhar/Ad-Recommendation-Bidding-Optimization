import numpy as np
from scipy import stats

def simulate_campaign(strategy, n_rounds=1000, base_conversion=0.05):
    """
    Simulate campaign conversions.
    :param strategy: string, either 'ctr_based' or 'rule_based'
    :param n_rounds: number of ad impressions.
    :param base_conversion: base conversion rate.
    :return: array of conversion outcomes (1 if converted, 0 otherwise)
    """
    if strategy == "ctr_based":
        # Simulate slightly higher conversion rate
        conversion_rate = base_conversion + 0.02
    elif strategy == "rule_based":
        conversion_rate = base_conversion
    else:
        conversion_rate = base_conversion
    return np.random.binomial(1, conversion_rate, n_rounds)

def perform_ab_test(n_rounds=1000):
    group_A = simulate_campaign("ctr_based", n_rounds)
    group_B = simulate_campaign("rule_based", n_rounds)
    
    # Compute conversion rates
    rate_A = np.mean(group_A)
    rate_B = np.mean(group_B)
    
    # Perform a t-test (for demonstration)
    t_stat, p_value = stats.ttest_ind(group_A, group_B)
    
    print(f"CTR-based Conversion Rate: {rate_A:.4f}")
    print(f"Rule-based Conversion Rate: {rate_B:.4f}")
    print(f"T-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    
if __name__ == "__main__":
    perform_ab_test()