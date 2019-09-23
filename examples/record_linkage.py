import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jellyfish import levenshtein_distance
from nameparser import HumanName
import os
import re

# 3 examples (Ex1-3) of Matching
# Ex 1: Data integration - Data from multiple sources

# Ex 2: enrichment, enhance dataset

# Ex 3: Record linkage (e.g. medical and financial research)

A = np.arange(1, 10).reshape((3, 3))
B = A * np.random.randint(1, 3, A.shape)  # print(A, B)
dfA = pd.DataFrame(data=A, columns=["A", "B", "C"])
dfB = pd.DataFrame(data=B, columns=["A", "B", "C"])
print(dfA, dfB)

# 1. Pairwise Matching
# no duplicates, all possible pairs are A x B
print(np.cross(A, B))

M, U = list(), list()
for rowa, rowb in zip(A, B):
    for a, b in zip(rowa, rowb):
        if a == b:
            M.append((a, b))
        else:
            U.append((a, b))
print(M, U)
AxB = set(M).union(U)
print("all pairs: ", AxB)

# or we could build a cluster from that, let us have a look at the data:
plt.figure()
plt.grid(True)
plt.plot(A.ravel(), B.ravel(), "bo")
plt.axis("equal")
plt.xlim([0, 15])
plt.ylim([0, 15])
plt.savefig("cluster_for_pairs.png")

# let us load the example dataset
with open("data.dir") as fid: data_dir = fid.readline()
wageFname = "WageandHourComplianceActions.csv"
inspectionFname = "FloridaRestaurantInspections.csv"

df_wage = pd.read_csv(os.path.join(data_dir, wageFname))
df_inspection = pd.read_csv(os.path.join(data_dir, inspectionFname))

print(df_wage.head())
print(df_inspection.head())

# Correlation between underpaid emplyees and hygiene violations

# profiling
# - accurate?
# - complete ?
# - consistent ?
# - recent ?
df_wage.info(verbose=True, null_counts=True)

# Filter (restaurant data and hygiene data):
fields = ["dba", "location_address", "location_city", "location_zip_code",
          "district", "county_number"]
df_inspection_a = df_inspection[fields].drop_duplicates()
df_inspection_a.info()

fields_wage = ["trade_nm", "legal_name", "street_addr_1_txt", "cty_nm",
               "st_cd", "zip_cd", "naic_cd", "naics_code_description"]
df_wage_a = df_wage[fields_wage].drop_duplicates()
df_wage_a.info()

# Transformations: apply and groupby
# different zip code formats:
t = {ord(x): y for (x, y) in zip("0123456789", "999999999")}
print(t)

def zip_trans(z): return str(z).translate(t) if pd.notnull(z) else z

df_inspection_a["zip_code_pattern"] = df_inspection_a.location_zip_code.apply(zip_trans)
print(df_inspection_a.groupby("zip_code_pattern").size())

df_wage_a["zip_code_pattern"] = df_wage_a.zip_cd.apply(zip_trans)
print(df_wage_a.groupby("zip_code_pattern").size())

# apply with regular expressions:
def leading_patt(s):
    p = (("^[A-Z][a-z]", "Aa"), ("^[A-Z][A-Z]", "AA"), ("^[0-9][0-9]", "99"))
    for rx, patt in p:
        if re.search(rx, s):
            return patt
    return "OTHER"

df_wage_a["leading_pattern"] = df_wage_a.trade_nm.apply(leading_patt)
print(df_wage_a.groupby("leading_pattern").size())

# can be used to check whether dates or emails are in a particular format

# Filter with lambda => group by having in SQL
# cases where company address and city are the same, looking for duplicates
df = df_inspection_a.groupby(["dba", "location_zip_code"])\
    .filter(lambda x: len(x["dba"]) > 2)\
    .sort_values(by="dba")[["dba", "location_address", "location_zip_code",
                            "district", "county_number"]].head(10)
print(df.head(5))

# How about ambigious legal names?
print(df_wage_a.legal_name.value_counts(dropna=True).head())

# But trade names quite business names:
print(df_wage_a.trade_nm.value_counts(dropna=True).head(5))

# Histogram of matching
plt.figure()
plt.hist(df_inspection_a.dba.apply(lambda x: len(x) if pd.notnull(x) else 0),
         bins=50, color="red", alpha=0.3)
plt.hist(df_wage_a.trade_nm.apply(lambda x: len(x) if pd.notnull(x) else 0),
         bins=50, color="blue", alpha=0.3)
plt.savefig("test_word_length_distributions.png")

# Comparing Records and Fields
# we would like Ismatch: A x B -> {true, false}
# we compare against attributes of rA oder rB to create comparison vector:
# sim(rA, rB) = Sim(a1, a2, ..., an; b1, b2, ..., bn) = [s1(a1, b1), ... sn(an, bn)]
# then create a decision model which classifies comparison as match or non-match

# this requires pre-processing and transformation (e.g. lowercase, &-->and, abbreviation):
name_prep = lambda x: x.lower().translate({None: "'.,"}) if pd.notnull(x) else ''
df_wage_a["nname"] = df_wage_a.trade_nm.apply(name_prep)
df_inspection_a["nname"] = df_inspection_a.dba.apply(name_prep)

# convert zip codes to 5 digits
df_inspection_a["nzip"] = df_inspection_a.location_zip_code.apply(lambda x:x[:5] if pd.notnull(x) else x)
df_wage_a["nzip"] = df_wage_a.zip_cd.apply(lambda x: str(int(x)) if pd.notnull(x) else str(0))

# split out street number into separate field
df_inspection_a["snum"] = df_inspection_a.location_address.str.split(" ", 1).apply(lambda x:x[0])
print(df_wage_a.columns)
# print(df_wage_a.street_addr_1_txt.str.split(" ", 1).apply(lambda x: x[0]))
# df_wage_a["snum"] = df_wage_a.street_addr_1_txt.str.split(" ", 1)[0]
# .apply(lambda x:str(x[0]))

print(df_inspection_a[["nname", "nzip", "snum"]])

print(levenshtein_distance("pydata", "padata"))  # 1-character-change
print(levenshtein_distance("pydata", "pandata"))  # insert character
print(levenshtein_distance("pydata", "pandasa"))  # 1-character-change
print(levenshtein_distance("pydata", "pandas"))  # character-deletion

# define jaccard:
def jaccard(s1, s2):
    a = set(s1)
    b = set(s2)
    if len(a) == 0 and len(b) == 0:
        return 0
    return len(a & b) / len(a | b)

print(jaccard("pydata", "pandas"))
print(jaccard(["bag", "of", "words"], ["words", "bag"]))

# now let us look at the n-grams
def ngrams(seq, n):
    return [seq[i:i+n] for i in range(1 + len(seq) -n)]

print(ngrams("pydata", 2), ngrams("pandas", 2))
print(ngrams("pydata", 3), ngrams("pandas", 3))
print(jaccard(ngrams("pydata", 2), ngrams("pandas", 2)))
print(jaccard(ngrams("pydata", 3), ngrams("pandas", 3)))

# Human names tricky
same_name = "RICHARD ALLAN BREGGER", "RICHARD A BREGGER", "BREGGER RICHARD A"
# parse them, compare to firstname databases, dictionary of found matches
print(HumanName("Dr. Richard Allan 'Rick' Bregger, CFP"))
