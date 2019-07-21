import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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

# TODO: apply and groupby
