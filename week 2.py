# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:17:36 2019

@author: Voltas
"""

import pandas
import numpy
nesarc = pandas.read_csv ("nesarc_pds.csv" , low_memory=False)     # load NESARC dataset
nesarc.columns = map(str.upper , nesarc.columns)
pandas.set_option('display.float_format' , lambda x:'%f'%x)
print (len(nesarc))    # Number of observations
print (len(nesarc.columns))    # Number of variables

# Change my variables to numeric

nesarc['AGE'] = nesarc['AGE'].convert_objects(convert_numeric=True)
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

# Frequency distribution of cannabis use variable for ages 18-30 (subset3)
print("Counts for cannabis use ages 18-30, variable S3BQ1A5 - Section 3B")
cuy1 = subsetc3.groupby('S3BQ1A5').size()   # Cannabis use counts (subset3)
print(cuy1)
print("Percentages for cannabis use age 18-30, variable S3BQ1A5 - Section 3B")
cuy2 = subsetc3.groupby('S3BQ1A5').size() * 100 / len(subsetc3)   # Cannabis use percentages (subset3)
print(cuy2)

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












