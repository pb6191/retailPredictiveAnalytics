# %% load modules

from pathlib import Path
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import string
from sklearn.preprocessing import FunctionTransformer


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

nlp = spacy.load("en_core_web_lg")

#%%

df1 = pd.read_csv("../data/clean/customer_data.csv")

# %%

df1.dtypes

# %%

cols = [
    "CustomerID",
    "TimePeriod",
    "Country",
]
X = df1[cols]
X
y = df1[["AvgQuantity", "AvgUnitPrice", "AvgRevenue", "TotalQuantity", "TotalRevenue"]]
y

# %%

imp_constant = SimpleImputer(strategy="constant", fill_value="other")
ohe = OneHotEncoder()
imp_ohe = make_pipeline(imp_constant, ohe)

scaler = StandardScaler()
imp_median_indicator = SimpleImputer(strategy="median", add_indicator=True)
imp_median_scal = make_pipeline(imp_median_indicator, scaler)

ct = make_column_transformer(
    (
        imp_ohe,
        [
            "Country",
            "CustomerID"
        ],
    ),
    (
        imp_median_scal,
        [
            "TimePeriod"
        ],
    ),
    remainder="passthrough",
)

# %%----------------------------------------------------------------
ct.fit_transform(X)


# %%----------------------------------------------------------------


# %%
linreg = LinearRegression()
pipe = make_pipeline(ct, linreg)
pipe.fit(X, y)
pipe.score(X, y)

# %%

df2 = pd.read_csv("../data/clean/df_custcountry.csv")
df2["TimePeriod"] = 13

df3 = pd.DataFrame(pipe.predict(df2))
df3 = df3.rename(columns={0: "AvgQuantity", 1: "AvgUnitPrice", 2: "AvgRevenue", 3: "TotalQuantity", 4: "TotalRevenue"})

df4 = pd.concat([df2, df3], axis=1)

df_withPred = pd.concat([df1, df4])




df2 = pd.read_csv("../data/clean/df_custcountry.csv")
df2["TimePeriod"] = 14

df3 = pd.DataFrame(pipe.predict(df2))
df3 = df3.rename(columns={0: "AvgQuantity", 1: "AvgUnitPrice", 2: "AvgRevenue", 3: "TotalQuantity", 4: "TotalRevenue"})

df4 = pd.concat([df2, df3], axis=1)

df_withPred = pd.concat([df_withPred, df4])



df2 = pd.read_csv("../data/clean/df_custcountry.csv")
df2["TimePeriod"] = 15

df3 = pd.DataFrame(pipe.predict(df2))
df3 = df3.rename(columns={0: "AvgQuantity", 1: "AvgUnitPrice", 2: "AvgRevenue", 3: "TotalQuantity", 4: "TotalRevenue"})

df4 = pd.concat([df2, df3], axis=1)

df_withPred = pd.concat([df_withPred, df4])

# %%

df_withPred.to_csv("../data/clean/customer_data_with_predictions.csv", index=False)

# %%
