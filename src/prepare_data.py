# %% load modules

import numpy as np
import pandas as pd
import spacy
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pd.set_option(
    "display.max_rows",
    8,
    "display.max_columns",
    None,
    "display.width",
    None,
    "display.expand_frame_repr",
    True,
    "display.max_colwidth",
    None,
)

np.set_printoptions(
    edgeitems=5,
    linewidth=233,
    precision=4,
    sign=" ",
    suppress=True,
    threshold=50,
    formatter=None,
)

#%%

xls = pd.ExcelFile("../data/raw/Online Retail.xlsx")
df = pd.read_excel(xls, "Online Retail")

#%%

df.dtypes

# no missing values
df[df["InvoiceNo"].isna()]
df[df["InvoiceNo"].isnull()]

# no missing values
df[df["StockCode"].isna()]
df[df["StockCode"].isnull()]

# replace missing values with 'MISSING'
df[df["Description"].isna()]
df[df["Description"].isnull()]
df["Description"] = df["Description"].fillna("MISSING")

# sale quantity cannot be negative or zero, so filtering those out
df[df["Quantity"].isna()]
df[df["Quantity"].isnull()]
df = df[df["Quantity"] > 0]

# negative or zero price suggests items that were not bought, so filtering those out
df[df["UnitPrice"].isna()]
df[df["UnitPrice"].isnull()]
df = df[df["UnitPrice"] > 0]

# replace missing values with 'MISSINGCUST'
df[df["CustomerID"].isna()]
df[df["CustomerID"].isnull()]
df["CustomerID"] = df["CustomerID"].fillna("MISSINGCUST")

# no missing values
df[df["Country"].isna()]
df[df["Country"].isnull()]

# split the date column
df[["Year", "Month", "Date"]] = df["InvoiceDate"].apply(
    lambda x: pd.Series(str(x).split("-"))
)

# there is only partial data for december 2011 so filtering it out
df = df.loc[(df["Year"] != "2011") | (df["Month"] != "12")]
df["TimePeriod"] = df["Month"].astype(int) + 1
df["TimePeriod"] = df["TimePeriod"].replace(13, 1)


df.to_csv("../data/clean/clean_data.csv", index=False)

# new data frame with single row per time period per customer per stockcode

# InvoiceNo is not needed as it is most probably randomly generated and is not meaningful
df = df.drop(columns=["InvoiceNo"])

# InvoiceDate, Year, Month, and Date and now captured by the timeperiod
df = df.drop(columns=["InvoiceDate", "Year", "Month", "Date"])

df.dtypes

# create a new column called revenue
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# create stockcode desc mapping
df_stockdesc = df[["StockCode", "Description"]]
df_stockdesc = df_stockdesc.drop_duplicates()
# set desc as multiple if stock code is still repeated after dropping duplicates
df_stockdesc.loc[
    df_stockdesc.StockCode.duplicated(keep=False), "Description"
] = "MULTIPLE"
# drop duplicates again
df_stockdesc = df_stockdesc.drop_duplicates()

df_stockdesc.StockCode.unique().size  # same as number of rows in the df
df_stockdesc.to_csv("../data/clean/df_stockdesc.csv", index=False)


# create customer country mapping
df_custcountry = df[["CustomerID", "Country"]]
df_custcountry = df_custcountry.drop_duplicates()
# set country as multiple if customer ID is still repeated after dropping duplicates
df_custcountry.loc[
    df_custcountry.CustomerID.duplicated(keep=False), "Country"
] = "MULTIPLE"
# drop duplicates again
df_custcountry = df_custcountry.drop_duplicates()

df_custcountry.CustomerID.unique().size  # same as number of rows in the df
df_custcountry.to_csv("../data/clean/df_custcountry.csv", index=False)

#%%

# collapsing to have [stockcode, customer, timeperiod] to be the primary key / unique in each row
df2 = df.groupby(["CustomerID", "TimePeriod"]).mean()
df2 = df2.reset_index()
df2 = df2.rename(
    columns={
        "Quantity": "AvgQuantity",
        "UnitPrice": "AvgUnitPrice",
        "Revenue": "AvgRevenue",
    }
)
df3 = df.groupby(["CustomerID", "TimePeriod"]).sum()
df3 = df3.reset_index()
df3 = df3.rename(columns={"Quantity": "TotalQuantity", "Revenue": "TotalRevenue"})
df3 = df3.drop(columns=["UnitPrice"])
df2 = pd.merge(df2, df3, on=["CustomerID", "TimePeriod"], how="left")
df2 = pd.merge(df2, df_custcountry, on=["CustomerID"], how="left")

df2.to_csv("../data/clean/customer_data.csv", index=False)

df4 = df.groupby(["StockCode", "TimePeriod"]).mean()
df4 = df4.reset_index()
df4 = df4.rename(
    columns={
        "Quantity": "AvgQuantity",
        "UnitPrice": "AvgUnitPrice",
        "Revenue": "AvgRevenue",
    }
)
df5 = df.groupby(["StockCode", "TimePeriod"]).sum()
df5 = df5.reset_index()
df5 = df5.rename(columns={"Quantity": "TotalQuantity", "Revenue": "TotalRevenue"})
df5 = df5.drop(columns=["UnitPrice"])
df4 = pd.merge(df4, df5, on=["StockCode", "TimePeriod"], how="left")
df4 = pd.merge(df4, df_stockdesc, on=["StockCode"], how="left")

df4.to_csv("../data/clean/stock_data.csv", index=False)

#%%
