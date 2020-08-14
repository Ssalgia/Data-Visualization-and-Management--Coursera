# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:17:36 2019

@author: Voltas
"""

import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt
nesarc = pandas.read_csv ('nesarc_pds.csv' , low_memory=False)     # load NESARC dataset

#Set PANDAS to show all columns in DataFrame
pandas.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pandas.set_option('display.max_rows', None)

nesarc.columns = map(str.upper , nesarc.columns)

pandas.set_option('display.float_format' , lambda x:'%f'%x)


# Change my variables to numeric
nesarc['AGE'] = nesarc['AGE'].convert_objects(convert_numeric=True)
nesarc['S3BQ4'] = nesarc['S3BQ4'].convert_objects(convert_numeric=True)
nesarc['S3BQ1A5'] = nesarc['S3BQ1A5'].convert_objects(convert_numeric=True)
nesarc['S3BD5Q2B'] = nesarc['S3BD5Q2B'].convert_objects(convert_numeric=True)
nesarc['S3BD5Q2E'] = nesarc['S3BD5Q2E'].convert_objects(convert_numeric=True)
nesarc['MAJORDEP12'] = nesarc['MAJORDEP12'].convert_objects(convert_numeric=True)
nesarc['GENAXDX12'] = nesarc['GENAXDX12'].convert_objects(convert_numeric=True)


subset1 = nesarc[(nesarc['AGE']>=18) & (nesarc['AGE']<=30) & (nesarc['S3BD5Q2B']==3) & (nesarc['S3BQ1A5']==1)]    # Cannabis users both last 12 months and prior, ages 18-30
subsetc1 = subset1.copy()

subset2 = nesarc[(nesarc['AGE']>=18) & (nesarc['AGE']<=30) & (nesarc['S3BQ1A5']==2)]      # Non-users, ages 18-30  
subsetc2 = subset2.copy()

subset3 = nesarc[(nesarc['AGE']>=18) & (nesarc['AGE']<=30)]      # Ages 18-30
subsetc3 = subset3.copy()

subset5 = nesarc[(nesarc['AGE']>=18) & (nesarc['AGE']<=30) & (nesarc['S3BQ1A5']==1)]    # Cannabis users, ages 18-30
subsetc5 = subset5.copy()


# Frequency distributions of variables (groupby function) of the entire sample

print("Counts for cannabis use, variable S3BQ1A5 - Section 3B")
cu1 = nesarc.groupby('S3BQ1A5').size()   # Cannabis use counts
print(cu1)
print("Percentages for cannabis use, variable S3BQ1A5 - Section 3B")
cu2 = nesarc.groupby('S3BQ1A5').size() * 100 / len(nesarc)   # Cannabis use percentages
print(cu2)

print("Counts for used cannabis in the last 12 months/prior to last 12 months/both time periods, variable S3BD5Q2B - Section 3B")
uc1 = nesarc.groupby('S3BD5Q2B').size()    # Used cannabis time periods counts
print(uc1)
print("Percentages for used cannabis in the last 12 months/prior to last 12 months/both time periods, S3BD5Q2B - Section 3B")
uc2 = nesarc.groupby('S3BD5Q2B').size() * 100 / len(nesarc)    # Used cannabis time periods percentages
print(uc2)

nesarc.loc[(nesarc['S3BQ1A5']!=1) & (nesarc['S3BD5Q2E'].isnull()), 'S3BD5Q2E'] = 11

print("Counts for frequency of used cannabis when using the most, variable S3BD5Q2E - Section 3B")
fuc1 = nesarc.groupby('S3BD5Q2E').size()    # Frequency of used cannabis counts
print(fuc1)
print("Percentages for frequency of used cannabis when using the most, variable S3BD5Q2E - Section 3B")
fuc2 = nesarc.groupby('S3BD5Q2E').size() * 100 / len(nesarc)    # Frequency of used cannabis percentages
print(fuc2)

print("Counts for non-hierarchical major depression diagnoses in last 12 months, variable MAJORDEP12 - Section 14")
md1 = nesarc.groupby('MAJORDEP12').size()     # Major depression diagnoses counts
print(md1)
print("Percentages for non-hierarchical major depression diagnoses in last 12 months, variable MAJORDEP12 - Section 14")
md2 = nesarc.groupby('MAJORDEP12').size() * 100 / len(nesarc)     # Major depression diagnoses percentages
print(md2)

print("Counts for non-hierarchical generalized anxiety diagnoses in last 12 months, variable GENAXDX12 - Section 14")
ga1 = nesarc.groupby('GENAXDX12').size()     # Generalized anxiety diagnoses counts
print(ga1)
print("Percentages for non-hierarchical generalized anxiety diagnoses in last 12 months, variable GENAXDX12 - Section 14")
ga2 = nesarc.groupby('GENAXDX12').size() * 100 / len(nesarc)     # Generalized anxiety diagnoses percentages
print(ga2)

# Frequency distributions of major depression and general anxiety diagnoses variables for both last 12 months and prior cannabis users, ages 18-30 (subset1)

print("Counts for non-hierarchical major depression diagnoses in last 12 months (both last 12 months and prior cannabis users, ages 18-30), variable MAJORDEP12 - Section 14")
mdu1 = subsetc1.groupby('MAJORDEP12').size()     # Major depression diagnoses counts (subset1)
print(mdu1)
print("Percentages for non-hierarchical major depression diagnoses in last 12 months (both last 12 months and prior cannabis users, ages 18-30), variable MAJORDEP12 - Section 14")
mdu2 = subsetc1.groupby('MAJORDEP12').size() * 100 / len(subsetc1)     # Major depression diagnoses percentages (subset1)
print(mdu2)

print("Counts for non-hierarchical generalized anxiety diagnoses in last 12 months (both last 12 months and prior cannabis users, ages 18-30), variable GENAXDX12 - Section 14")
gau1 = subsetc1.groupby('GENAXDX12').size()     # Generalized anxiety diagnoses counts (subset1)
print(gau1)
print("Percentages for non-hierarchical generalized anxiety diagnoses in last 12 months (both last 12 months and prior cannabis users, ages 18-30), variable GENAXDX12 - Section 14")
gau2 = subsetc1.groupby('GENAXDX12').size() * 100 / len(subsetc1)     # Generalized anxiety diagnoses percentages (subset1)
print(gau2)

# Frequency distributions of major depression and general anxiety diagnoses variables of non-users, ages 18-30 (subset2)

print("Counts for non-hierarchical major depression diagnoses in last 12 months (non-users, ages 18-30), variable MAJORDEP12 - Section 14")
mdn1 = subsetc2.groupby('MAJORDEP12').size()     # Major depression diagnoses counts (subset2)
print(mdn1)
print("Percentages for non-hierarchical major depression diagnoses in last 12 months (non-users, ages 18-30), variable MAJORDEP12 - Section 14")
mdn2 = subsetc2.groupby('MAJORDEP12').size() * 100 / len(subsetc2)     # Major depression diagnoses percentages (subset2)
print(mdn2)

print("Counts for non-hierarchical generalized anxiety diagnoses in last 12 months (non-users, ages 18-30), variable GENAXDX12 - Section 14")
gan1 = subsetc2.groupby('GENAXDX12').size()     # Generalized anxiety diagnoses counts (subset2)
print(gan1)
print("Percentages for non-hierarchical generalized anxiety diagnoses in last 12 months (non-users, ages 18-30), variable GENAXDX12 - Section 14")
gan2 = subsetc2.groupby('GENAXDX12').size() * 100 / len(subsetc2)     # Generalized anxiety diagnoses percentages (subset2)
print(gan2)


###################################################################################################################################

# Quartile age split, cut function, 4 groups (18-21, 21-24, 24-27, 27-30)

subsetc3['AGE4GROUPS'] = pandas.qcut(subsetc3.AGE, 4, labels=["1=17-21","2=21-24","3=24-27","4=27-30"]) 

print("Counts for age splitted in 4 groups: 18-21, 21-24, 24-27, 27-30")
age4g1 = subsetc3.groupby('AGE4GROUPS').size()
print(age4g1)

print("Percentages for age splitted in 4 groups: 18-21, 21-24, 24-27, 27-30")
age4g2 = subsetc3.groupby('AGE4GROUPS').size() * 100 / len(subsetc3)
print(age4g2)

print("Counts of observations within each of the age group four categories")
subsetc3['AGE4GROUPS'] = pandas.cut(subsetc3.AGE, [17, 21, 24, 27, 30])
print (pandas.crosstab(subsetc3['AGE4GROUPS'], subsetc3['AGE']))

# Frequency distribution of cannabis use variable for ages 18-30 (subset3) with 9 set to NaN, number of missing data

subsetc5['S3BD5Q2E'] = subsetc5['S3BD5Q2E'].replace(99, numpy.nan)


recode = {1: 1, 2: 2, 9: "NaN"}
subsetc3['CUMD'] = subsetc3['S3BQ1A5'].map(recode)

print("Counts for cannabis use ages 18-30 with missing data set to NaN, variable S3BQ1A5 - Section 3B")
cuy1 = subsetc3.groupby('CUMD').size()   # Cannabis use counts (subset3)
print(cuy1)
print("Percentages for cannabis use age 18-30 with missing data set to NaN, variable S3BQ1A5 - Section 3B")
cuy2 = subsetc3.groupby('CUMD').size() * 100 / len(subsetc3)   # Cannabis use percentages (subset3)
print(cuy2)

# Frequency distribution of monthly average cannabis used (when using the most) variable for ages 18-30 (subset5)

recode1 = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}       # Dictionary with details of frequency variable reverse-recode
subsetc5['CUFREQ'] = subsetc5['S3BD5Q2E'].map(recode1)     # Change variable name from S3BD5Q2E to CUFREQ
recode2 = {1: 30, 2: 25, 3: 14, 4: 6, 5: 3, 6: 1, 7: 0.8, 8: 0.5, 9: 0.3, 10: 0.1}    # Monthly average cannabis used
subsetc5['CUFREQMO'] = subsetc5['S3BD5Q2E'].map(recode2)      # Change variable name from S3BD5Q2E to CUFREQMO

print("Counts for average cannabis used per month when using the most, variable CUFREQMO")
fucy1 = subsetc5.groupby('CUFREQMO').size()    # Frequency of used cannabis counts (subset5)
print(fucy1)
print("Percentages for average cannabis used per month when using the most, variable CUFREQMO")
fucy2 = subsetc5.groupby('CUFREQMO').size() * 100 / len(subsetc5)    # Frequency of used cannabis percentages (subset5)
print(fucy2)

# Secondary variable creation, NUMJOPMOTH_EST, number of joints per month

subsetc5['NUMJOPMOTH_EST'] = subsetc5['CUFREQMO'] * subsetc5['S3BQ4']
subsetc4 = subsetc5[['IDNUM' , 'S3BQ4' , 'CUFREQMO' , 'NUMJOPMOTH_EST']]
head30 = subsetc4.head(30)
print("Number of cannabis joints smoked per month when using the most, first 30 observations, new variable NUMJOPMOTH_EST")
print(head30)

# Frequency distribution for both major depression and general anxiety diagnoses, new variable cration NUMMDGENANX

subsetc1['NUMMDGENANX'] = subsetc1['MAJORDEP12'] + subsetc1['GENAXDX12']

print("Counts for major depression and general anxiety diagnoses in cannabis users, ages 18-30, variable NUMMDGENANX")
ndg1 = subsetc1.groupby('NUMMDGENANX').size()
print(ndg1)
print("Percentages for major depression and general anxiety diagnoses in cannabis users, ages 18-30, variable NUMMDGENANX")
ndg2 = subsetc1.groupby('NUMMDGENANX').size() * 100 / len(subsetc1)
print(ndg2)

# Newly managed depression and anxiety variables, in cannabis users ages 18-30, both last 12 months and prior (subset1), define function

def DEPRESSIONANXIETY (row):
    if row['NUMMDGENANX'] == 0 :
        return 0
    if row['NUMMDGENANX'] > 1 :
        return 1
    if row['MAJORDEP12'] == 1 :
        return 2
    if row['GENAXDX12'] == 1 :
        return 3
   
subsetc1['DEPRESSIONANXIETY'] = subsetc1.apply (lambda row: DEPRESSIONANXIETY (row), axis=1)
subsetc6 = subsetc1[['IDNUM' , 'MAJORDEP12' , 'GENAXDX12' , 'NUMMDGENANX' , 'DEPRESSIONANXIETY']].copy()
first30 = subsetc6.head(30)
print("Depression and anxiety diagnoses counts for cannabis users in last 12 months and prior, ages 18-30, new variable DEPRESSIONANXIETY")
print(first30)

####################################################################################################################################


print("Counts for average number of joints smoked per month, ages 18-30, variable NUMJOPMOTH_EST")
njpm1 = subsetc5.groupby('NUMJOPMOTH_EST').size()
print(njpm1)
print("Percentages for average number of joints smoked per month, ages 18-30, variable NUMJOPMOTH_EST")
njpm2 = subsetc5.groupby('NUMJOPMOTH_EST').size() * 100 / len(subsetc5)
print(njpm2)

# Change format of S3BQ1A5, S3BD5Q2E, MAJORDEP12, GENAXDX12 to categorical

subsetc3['S3BQ1A5'] = subsetc3['S3BQ1A5'].astype('category')
subsetc5['S3BD5Q2E'] = subsetc5['S3BD5Q2E'].astype('category')
subsetc1['MAJORDEP12'] = subsetc1['MAJORDEP12'].astype('category')
subsetc1['GENAXDX12'] = subsetc1['GENAXDX12'].astype('category')

# Change the numbers with strings, rename x-axis categories

subsetc1['MAJORDEP12'] = subsetc1['MAJORDEP12'].cat.rename_categories(["No","Yes"])
subsetc1['GENAXDX12'] = subsetc1['GENAXDX12'].cat.rename_categories(["No","Yes"])
subsetc3['S3BQ1A5'] = subsetc3['S3BQ1A5'].cat.rename_categories(["Yes","No","Unknown"])
subsetc5['S3BD5Q2E'] = subsetc5['S3BD5Q2E'].cat.rename_categories(["Every day","Nearly every day","3-4 times/week","1-2 times/week","2-3 times/month","Once a month","7-11 times/year","3-6 times/year","2 times/year","Once a year"])

# Univariate bar chart for categorical variables, S3BQ1A5, S3BD5Q2E, MAJORDEP12, GENAXDX12, which stand for cannabis use, frequency of use, major depression and general anxiety

seaborn.countplot(x='S3BQ1A5', data=subsetc3)
plt.xlabel('Cannabis use in ages 18-30')
plt.title('Cannabis use, ages 18-30')
plt.show()
print ('Describe cannabis use variable')
djpm2 = subsetc3['S3BQ1A5'].describe()
print (djpm2)

plt.figure(figsize=(12,4))      # Change plot size
ax5 = seaborn.countplot(x='S3BD5Q2E', data=subsetc5)
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=40, ha="right")    # X-axis labels rotation
plt.xlabel('Frequency of cannabis use in ages 18-30')
plt.title('Frequency of cannabis use, ages 18-30')
plt.show()
print ('Describe frequency of cannabis use variable')
djpm5 = subsetc5['S3BD5Q2E'].describe()
print (djpm5)

seaborn.countplot(x='MAJORDEP12', data=subsetc1)
plt.xlabel('Major depression diagnoses in last 12 months')
plt.title('Major depression diagnosed in cannabis users (both last 12 months and prior), ages 18-30')
plt.show()
print ('Describe major depression variable')
djpm3 = subsetc1['MAJORDEP12'].describe()
print (djpm3)
seaborn.countplot(x='GENAXDX12', data=subsetc1)
plt.xlabel('General anxiety diagnoses in last 12 months')
plt.title('General anxiety diagnosed in cannabis users (both last 12 months and prior), ages 18-30')
plt.show()
print ('Describe general anxiety variable')
djpm4 = subsetc1['GENAXDX12'].describe()
print (djpm4)

# Univariate bar chart for quantitative variable, NUMJOPMOTH_EST, that stands for the number of joints smoked per month when using the most, ages 18-30

plt.figure(figsize=(10,4))     # Change plot size
seaborn.distplot(subsetc5["NUMJOPMOTH_EST"].dropna(), kde=False)
plt.xlabel('Number of joints smoked per month')
plt.title('Estimated number of joints smoked per month by cannabis users, ages 18-30')
plt.show()

# Center and spread measurements

print ('Spread')
std1 = subsetc5['NUMJOPMOTH_EST'].std()
print (std1)

print ('Mode')
mode1 = subsetc5['NUMJOPMOTH_EST'].mode()
print (mode1)

print ('Mean')
mean1 = subsetc5['NUMJOPMOTH_EST'].mean()
print (mean1)

print ('Median')
median1 = subsetc5['NUMJOPMOTH_EST'].median()
print (median1)

# Plot for groups of total number of joints smoked per month, variable NUMJOPMOTH_EST

subsetc5['NUMJOPMOTH_EST'] = pandas.cut(subsetc5.NUMJOPMOTH_EST, [0, 1, 10, 20, 30, 50, 70, 90, 110, 130, 150, 200, 250, 300, 2970])   # Split the number into groups
subsetc5['NUMJOPMOTH_EST'] = subsetc5['NUMJOPMOTH_EST'].astype('category')
# Rename x-axis categories of the plot
subsetc5['NUMJOPMOTH_EST'] = subsetc5['NUMJOPMOTH_EST'].cat.rename_categories(["<1","1-10","11-20","21-30","31-50","51-70","71-90","90-110","111-130","131-150","151-200","201-250","251-300",">300"])
plt.figure(figsize=(10,4))      # Change plot size
ax = seaborn.countplot(x='NUMJOPMOTH_EST', data=subsetc5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")    # X-axis labels rotation
plt.xlabel('Number of joints smoked per month')
plt.title('Estimated number of joints smoked per month by cannabis users, ages 18-30')
plt.show()

print ('Describe number of joints smoked per month')
djpm1 = subsetc5['NUMJOPMOTH_EST'].describe()
print (djpm1)

# Bivariate bar graph C->Q, major depression as response variable and number of joints smoked per month as explanatory variable 

nesarc['MAJORDEP12'] = nesarc['MAJORDEP12'].convert_objects(convert_numeric=True)

plt.figure(figsize=(10,4))      # Change plot size
ax1 = seaborn.factorplot(x="NUMJOPMOTH_EST", y="MAJORDEP12", data=subsetc5, kind="bar", ci=None)
ax1.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")    # X-axis labels rotation
plt.xlabel('Joints smoked per Month')
plt.ylabel('Proportion of major depression')
plt.show()

# Bivariate bar graph C->Q, general anxiety as response variable and number of joints smoked per month as explanatory variable

nesarc['GENAXDX12'] = nesarc['GENAXDX12'].convert_objects(convert_numeric=True)

plt.figure(figsize=(10,4))      # Change plot size
ax2 = seaborn.factorplot(x="NUMJOPMOTH_EST", y="GENAXDX12", data=subsetc5, kind="bar", ci=None)
ax2.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")    # X-axis labels rotation
plt.xlabel('Joints smoked per Month')
plt.ylabel('Proportion of general anxiety')
plt.show()

nesarc['S3BQ1A5'] = nesarc['S3BQ1A5'].convert_objects(convert_numeric=True)
subsetc3['S3BQ1A5']=subsetc3['S3BQ1A5'].replace(9, numpy.nan)

# Bivariate bar graph C->C, major depression as response variable and cannabis use in ages 18 to 30 as explanatory variable

plt.figure(figsize=(10,4))      # Change plot size
seaborn.factorplot(x="S3BQ1A5", y="MAJORDEP12", data=subsetc3, kind="bar", ci=None)
plt.xlabel('Cannabis use ages 18-30')
plt.ylabel('Proportion of major depression')
plt.show()

# Bivariate bar graph C->C, general anxiety as response variable and cannabis use in ages 18 to 30 as explanatory variable

plt.figure(figsize=(10,4))      # Change plot size
seaborn.factorplot(x="S3BQ1A5", y="GENAXDX12", data=subsetc3, kind="bar", ci=None)
plt.xlabel('Cannabis use ages 18-30')
plt.ylabel('Proportion of general anxiety')
plt.show()

# Change frequency variable type to categorical

subsetc5['S3BD5Q2E'] = subsetc5['S3BD5Q2E'].astype('category')

# Bivariate bar graph C->C, major depression as response variable and frequency of cannabis use in ages 18 to 30 as explanatory variable

plt.figure(figsize=(12,4))      # Change plot size
ax3 = seaborn.factorplot(x="S3BD5Q2E", y="MAJORDEP12", data=subsetc5, kind="bar", ci=None)
ax3.set_xticklabels(rotation=40, ha="right")    # X-axis labels rotation
plt.xlabel('Frequency of cannabis use ages 18-30')
plt.ylabel('Proportion of major depression')
plt.show()

# Bivariate bar graph C->C, general anxiety as response variable and frequency of cannabis use in ages 18 to 30 as explanatory variable

plt.figure(figsize=(12,4))      # Change plot size
ax4 = seaborn.factorplot(x="S3BD5Q2E", y="GENAXDX12", data=subsetc5, kind="bar", ci=None)
ax4.set_xticklabels(rotation=40, ha="right")    # X-axis labels rotation
plt.xlabel('Frequency of cannabis use ages 18-30')
plt.ylabel('Proportion of general anxiety')
plt.show()








