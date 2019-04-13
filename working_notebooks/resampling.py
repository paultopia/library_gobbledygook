import numpy as np
import pandas as pd
import statsmodels.formula.api as sm

np.random.seed(52242)

def beta(pairs):
    x = [point[0] for point in pairs]
    mean_x = np.mean(x)
    y = [point[1] for point in pairs]
    mean_y = np.mean(y)
    numerator = 0
    for point in pairs:
        numerator += (point[0] - mean_x) * (point[1] - mean_y)
    denominator = np.sum([np.square(item - mean_x) for item in x])
    return numerator / denominator

heights = np.random.normal(loc=72.0, scale=5.0, size=1000)

shoe_sizes = [(height / 10) + np.random.normal(scale=2.0) for height in heights]

pairs = np.array(list(zip(heights, shoe_sizes)))

coefficient = beta(pairs)
print(coefficient)

df = pd.DataFrame({"x": heights, "y": shoe_sizes})
print(df.head())

reg = sm.ols(formula='y ~ x', data=df).fit()
print(reg.summary())

def resample(pairs):
    size = len(pairs)
    indices = np.random.choice(a=size, size=size, replace=True)
    sample = pairs[indices]
    return beta(sample)

bootstrap_samples = np.array([resample(pairs) for x in range(10000)])
print(np.percentile(bootstrap_samples, 5))
print(np.percentile(bootstrap_samples, 95))
