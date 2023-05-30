import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

download_root = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
HOUSING_URL = download_root + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = housing_path
    return pd.read_csv(csv_path)


housing_df = load_housing_data()

print(housing_df.describe())


# housing_df.hist(bins=50, figsize=(20, 15))
# plt.show()


#### Creating a test set
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(housing_df, 0.2)
print("Train set:", len(train_set), "Test Set:", len(test_set))

#### Creating a test set using a hash
from zlib import crc32


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


### unfortunately the housing dataset does not have an identifier column. The simplest solution is to use
### the row index as the ID
housing_with_id = housing_df.reset_index()  # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

### A district's latitude and longitude are guaranteed to be stable for a few millon years, so we could combine them into an ID like so:
housing_with_id["id"] = housing_df["longitude"] * 1000 + housing_df["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

# Sckit-Learn a simplest function

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing_df, test_size=0.2, random_state=42)

# Lets use the function pd.cut to create an income category attribute with five categories
housing_df["income_cat"] = pd.cut(housing_df["median_income"],
                                  bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                  labels=[1, 2, 3, 4, 5])

# housing_df["income_cat"].hist()
# plt.show()

### Now we can do stratified sampling based on the income category, using Scikit-Learn's StratifiedShuffleSplit class
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_df, housing_df["income_cat"]):
    strat_train_set = housing_df.loc[train_index]
    strat_test_set = housing_df.loc[test_index]

### Let's see if it worked as expected
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

### Let's remove `income_cat` attribute so the data is back to its original state:

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

### Discover and visualize the data gain insights
housing = strat_train_set.copy()

### Visualizing geographical data
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()

### Visualizing the housing prices
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"] / 100, label="population", figsize=(10, 7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
# plt.show()

### Looking for correlations
corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)

### Scatter Matrix using Pandas
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()


### Zoom in the median income correlation scatterplot
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
# plt.show()

### Experimenting with attribute combinations
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

## Let's see the crrelation matrix again
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

### Prepare the data for Machine Learning Algorithms

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

## Data Cleaning
# Missing values in total_bedrooms

# Option 1: Get rid of the corresponding districts
# housing.dropna(subset=["total_bedrooms"])

# Option 2: Get rid of the whole attribute
# housing.drop("total_bedrooms, axis=1")

# Option 3: Set the values to some value (zero, the mean, the median, etc)
# median = housing["total_bedrooms"].median()
# housing["total_rooms"].fillna(median, inplace=True)

# Scikit-Learn to take care of missing values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

# the imputer can only be computed on numerical attributes, so lets create a copy without ocean_proximity
housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)

print(imputer.statistics_)
print(housing_num.median().values)

X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

print(housing_tr.head())

### HANDLING TEXT AND CATEGORICAL ATTRIBUTES
housing_cat = housing[["ocean_proximity"]]
print("10 categories", housing_cat.head(10))

## Convert text to numbers
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])

print(ordinal_encoder.categories_)

## ONE HOT ENCODER
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()

housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print("ONE-HOT ENCODER array", housing_cat_1hot.toarray())

print("Categories", cat_encoder.categories_)

# CUSTOME TRANSFORMERS FOR TJE COMBINE ATTRIBUTES MENTIONED EARLIER
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedroom_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # Nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedroom_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

## TRANSFORMATION PIPELINES

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

## COLUMN TRANSFORMER
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)

print(housing_prepared)

### SELECT AND TRAIN A MODEL

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Let's try it out on a few instances from the begining

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))

print("Labels:", list(some_labels))

## LET'S MEASURE THIS REGRESSION MODEL'S RMSE ON THE WHOLE TRAINING SET USING SCIKIT-LEARN'S MEAN_SQUARED_ERROR()

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("RMSE:", lin_rmse)

# Let's use a powerful model - DECISION TREE MODEL

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

## Let's evaluate it
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("DECISION TREE RMSE:", tree_rmse)

## Let's randomly splits the training set into 10 distinct subset called 'folds'
## Then it trains and evaluates the Decision Tree model 10 times,
# picking a different fold for evaluation every time and training on the other 9 folds. The result is an array containing the 10 evaluation socres:

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


display_scores(tree_rmse_scores)

## lETS COMPUTE THE SAME SCORES FOR THE LINEAR REGRESSION MODEL JUST TO BE SURE
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
