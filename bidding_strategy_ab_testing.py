# BIDDING STRATEGY - Analyse and Present A/B Test Results

'''
XYZ company recently introduced a new bidding type, “average bidding”, as an alternative to its exisiting bidding type, called “maximum bidding”.
One of our clients, abc.com, has decided to test this new feature and wants to conduct an A/B test to understand if average bidding brings more
conversions than maximum bidding.
In this A/B test, abc.com randomly splits its audience into two equally sized groups, e.g. the test and the control group. XYZ company ad campaign
with “maximum bidding” is served to “control group” and another campaign with “average bidding” is served to the “test group”.
The A/B test has run for 1 month and abc.com now expects you to analyze and present the results of this A/B test.

You should answer the following questions in your presentation:

    - How would you define the hypothesis of this A/B test?
    - Can we conclude statistically significant results?
    - Which statistical test did you use, and why?
    - Based on your answer to Question 2, what would be your recommendation to client?

Hints:

    - The ultimate success metric for abc.com is Number of Purchases. Therefore, you should focus on Purchase metrics for statistical testing.
    - Explain the concept of statistical testing for a non-technical audience.
    - The customer journey for this campaign is:
        User sees an ad (Impression)
        User clicks on the website link on the ad (Website Click)
        User makes a search on the website (Search)
        User views details of a product (View Content)
        User adds the product to the cart (Add to Cart)
        User purchases the product (Purchase)
    - Use visualizations to compare test and control group metrics, such as Website Click Through Rate, Cost per Action, and Conversion Rates in addition to Purchase numbers.
    - If you see trends, anomalies or other patterns, discuss these in your presentation.
    - You can make assumptions if needed.

Data Source: “ab_testing_data.xlsx” --> The control and test group campaign results are in Control Group and Test Group tabs, respectively.

'''

'''
Steps to follow:
    - Define hypothesises and interpretation
    - Check data, outliers, assumptions (Normality and Variance Homogenity)
    - Implement appropriateness test (parametric/nonparametric, dependent/independent) according to scenario and assumptions
    - Check p-value and interpret the results
'''

# Import dependencies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro, levene, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# Load the datasets
# Maximum Bidding (Control Group)
df_A = pd.read_excel(r'C:\Users\yakup\PycharmProjects\dsmlbc\datasets\ab_testing_data.xlsx', 'Control Group')
# Average Bidding (Test Group)
df_B = pd.read_excel(r'C:\Users\yakup\PycharmProjects\dsmlbc\datasets\ab_testing_data.xlsx', 'Test Group')

# Show first rows of data
df_A.head()
df_B.head()


# Exploratory Data Analysis
df_A.describe()
df_B.describe()

# Histogram for df_A
df_A.hist()
plt.show()

# Histogram for df_B
df_B.hist()
plt.show()


# Missing Values Analysis
df_A.isnull().sum() # 0
df_B.isnull().sum() # 0
# There are no missing values in our observations.


# Outlier Analysis


# Hypothesis Test Implementation -- Independent Two Samples Test (A/B Testing)

'''
Steps to follow:
    - Define hypothesises and interpretation
    - Check data, outliers, assumptions (Normality and Variance Homogenity) 
    - Implement appropriateness test (parametric/nonparametric, dependent/independent) according to scenario and assumptions
    - Check p-value and interpret the results
'''


# Define a function to show the result for a Hypothesis Test
def hypothesis_test_result(test_result):
    test_statistics, p_value = test_result
    if p_value < 0.05:
        print('Test Statistics = %.4f, p-value = %.4f, so that H0 can be rejected!' % (test_statistics, p_value))
    else:
        print('Test Statistics = %.4f, p-value = %.4f, so that H0 can NOT be rejected!' % (test_statistics, p_value))


# Before, we need rearrange the dataset.

# Rearranging the datasets

# In our test, we are going to use only 'Purchase' column.
# df_A = df_A[['Purchase']]
# df_B = df_B[['Purchase']]

# A and Group A
GROUP_A = np.arange(len(df_A[['Purchase']]))
GROUP_A = pd.DataFrame(GROUP_A)
GROUP_A[:] = "A"
A_ = pd.concat([df_A[['Purchase']], GROUP_A], axis=1)

# B and Group B
GROUP_B = np.arange(len(df_B[['Purchase']]))
GROUP_B = pd.DataFrame(GROUP_B)
GROUP_B[:] = "B"
B_ = pd.concat([df_B[['Purchase']], GROUP_B], axis=1)

# All the data
AB = pd.concat([A_, B_])
AB.columns = ["Purchase", "GROUP"]
print(AB.head())
print(AB.tail())

# Some groupby operations.
AB["GROUP"].value_counts()
AB.groupby('GROUP').agg({'Purchase': ['count', np.mean, np.median, np.std]})

# Visualization
sns.boxplot(x="GROUP", y="Purchase", data=AB)
plt.show()
# As it can be seen from the box-plot, the average purchase of group A is higher than that of group B.
# But we don't know if this is random. So we will test to find out whether it is statistically significant or not.

## 1. Hypothesis Definitions

H0 = 'There is no significant difference between the mean of Purchase of the two groups'
H1 = 'There is significant difference between the mean of Purchase of the two groups'

A = AB.loc[AB["GROUP"] == 'A', 'Purchase']
B = AB.loc[AB["GROUP"] == 'B', 'Purchase']

## 2. Assumption Control

# 1.Normality Assumption
# 2.Variance Homogenity

# Normality Assumption (Shapiro Test)
'''
Shapiro-Wilk Test : Test of normality. Shapiro Wilk test uses only the right-tailed test.

H0 : The sample is drawn from a normally distributed population.
H1 : The sample is drawn from a population that is NOT normally distributed. 

alpha = 0.05 (significance level)

p < alpha : Reject the H0.(**significant**) (We have enough evidence to say that the population is not normally distributed.)
p > alpha : Fail to reject H0.(**non-significant**)(We don'thave enough evidence to say that the population is not normally distributed.)

NOTE: Hypotheses testing; more about evidence, it doens't more about True or False

NOTE : To meet the normality assumption we need to fail to reject the null hypothesis. (p > 0.05)'''

H0 = 'The sample is drawn from a normally distributed population.'
H1 = 'The sample is drawn from a population that is NOT normally distributed.'

hypothesis_test_result(shapiro(A)) # Test Statistics = 0.9773, p-value = 0.5891, so that H0 can NOT be rejected! --> Normality OK.
hypothesis_test_result(shapiro(B)) # Test Statistics = 0.9589, p-value = 0.1541, so that H0 can NOT be rejected! --> Normality OK.

# Variance Homogenity Assumption (Levene Test)
'''
Levene Test: Test wheter the variances of two samples are aproximately equal. (F-TEST likE ANOVA)

H0 : There is no difference between the variance of the first group and variance of the second group.

H1 : There is a difference between the variance of the first group and variance of the second group.

alpha = 0.05 (significance level)

p < alpha : Reject the H0.(significant)

p > alpha : Fail to reject H0.(non-significant)

NOTE : We want to Levene test to be not-significan because we dont want the variances to be diffrent.

NOTE : To satisfy the homogeneity to variance assumption we need to fail to reject the null hypothesis. (p > 0.05)'''

H0 = 'There is no difference between the variance of the first group and variance of the second group.'
H1 = 'There is a significant difference between the variance of the first group and variance of the second group.'

hypothesis_test_result(stats.levene(A, B)) # Test Statistics = 2.6393, p-value = 0.1083, so that H0 can NOT be rejected! --> Variance Homogenity OK.

## 3. Implement hypothesis test (parametric/nonparametric, dependent/independent) according to scenario and assumptions, Check p-value and interpret the results

# We saw, that both normality and variance homogenity assumptions are met. So, we can implement ttest_ind. If were not, then mannwhitneyu.

H0 = 'There is no significant difference between the mean of Purchase of the two groups'
H1 = 'There is significant difference between the mean of Purchase of the two groups'

test_statistics, pvalue = stats.ttest_ind(A, B, equal_var=True) # Test Statistics = -0.9416, p-value = 0.3493
hypothesis_test_result(stats.ttest_ind(A, B, equal_var=True))
# Test Statistics = -0.9416, p-value = 0.3493, so that H0 can NOT be rejected!

A.mean() # 550.8940587702316
B.mean() # 582.1060966484677

# Visualization
sns.boxplot(x="GROUP", y="Purchase", data=AB)
plt.show()

# --> Statistically, THERE IS NO DIFFERENCE between two groups! --> H0: µ1 = µ2
# So, it is advisable to continue with the existing bidding system.

# -> Explain non-technical audience:
#       Normally, we see the mean for each group with simple calculation. We can see the difference between mean values.
#       However, let's say, that we measured the values for one month and we want prove, that this difference is not by chance.
#       For that reason, we implemented statistical test.


'''
If there were a siginificant difference between them, we would say:

Based on the result of the hypothesis test, we can conlude and recommend our customer to opt for 'Maximum Bidding' strategy.
Becasue with that option the company can increase the revenue.
Test result shows, that 'Purchase' of that option is not by chance. There is statistically significant difference between two groups for the benefit of 'Maximum Bidding' strategy.
'''

