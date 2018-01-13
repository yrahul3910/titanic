import numpy as np
import pandas as pd
import os

def read_data():
    # Set the path of the raw data
    raw_data_path = os.path.join(os.path.pardir, "data", "raw")
    train_data_path = os.path.join(raw_data_path, "train.csv")
    test_data_path = os.path.join(raw_data_path, "test.csv")
    
    # Read the data with all default parameters
    train_df = pd.read_csv(train_data_path, index_col="PassengerId")
    test_df = pd.read_csv(test_data_path, index_col="PassengerId")

    test_df["Survived"] = -888
    df = pd.concat((train_df, test_df))
    return df


def process_data(df):
    # Use method chaining concept
    return (df
            # Create title attribute
            .assign(Title = lambda x: x.Name.map(extract_title))
            # Working missing values
            .pipe(fill_missing_values)
            # Create fare_bin feature
            .assign(Fare_Bin = lambda x: pd.qcut(x.Fare, 4, labels=["Very_Low", "Low", "High", "Very_High"]))
            # Create age state
            .assign(AgeState = lambda x: np.where(x.Age >= 18, "Adult", "Child"))
            .assign(FamilySize = lambda x: x.Parch + x.SibSp + 1)
            .assign(IsMother = lambda x: np.where((x.Sex == "female") & (x.Parch > 0) & (x.Age > 18) & (x.Title != "Miss"), 1, 0))
            # Create deck feature
            .assign(Cabin = lambda x: np.where(x.Cabin == "T", np.NaN, x.Cabin))
            .assign(Deck = lambda x: x.Cabin.map(get_deck))
            # Feature encoding
            .assign(IsMale = lambda x: np.where(x.Sex == "male", 1, 0))
            .pipe(pd.get_dummies, columns=["Deck", "Pclass", "Title", "Fare_Bin", "Embarked", "AgeState"])
            # Drop unnecessary columns
            .drop(["Cabin", "Name", "Ticket", "Parch", "SibSp", "Sex"], axis=1)
            .pipe(reorder_columns)
           )
    
    
def extract_title(name):
    title_groups = {
        "mr": "Mr",
        "mrs": "Mrs",
        "miss": "Miss",
        "master": "Master",
        "don": "Sir",
        "rev": "Sir",
        "dr": "Officer",
        "mme": "Mrs",
        "ms": "Mrs",
        "major": "Officer",
        "lady": "Lady",
        "sir": "Sir",
        "mlle": "Miss",
        "col": "Officer",
        "capt": "Officer",
        "the countess": "Lady",
        "jonkheer": "Sir",
        "dona": "Lady"
    }
    return title_groups[name.split(", ")[1].split(". ")[0].lower()]
    

def fill_missing_values(df):
    # embarked
    df.Embarked.fillna("C", inplace=True)
    
    # fare
    median_fare = df[(df.Pclass == 3) & (df.Embarked == "S")]["Fare"].median()
    df.Fare.fillna(median_fare, inplace=True)
    
    # age
    title_age_median = df.groupby("Title").Age.transform("median")
    df.Age.fillna(title_age_median, inplace=True)
    
    return df
    
    
def get_deck(cabin):
    return np.where(pd.notnull(cabin), str(cabin)[0].upper(), "Z")  # Z for NaN
    
    
def reorder_columns(df):
    columns = [column for column in df.columns if column != "Survived"]
    columns += ["Survived"]
    df = df[columns]
    return df


def write_data(df):
    processed_data_path = os.path.join(os.path.pardir, "data", "processed")
    train_path = os.path.join(processed_data_path, "train.csv")
    test_path = os.path.join(processed_data_path, "test.csv")

    # Training data
    df.loc[df.Survived != -888].to_csv(train_path)
    # Test data
    columns = [column for column in df.columns if column != "Survived"]
    df.loc[df.Survived == -888, columns].to_csv(test_path)

    
if __name__ == "__main__":
    df = read_data()
    df = process_data(df)
    write_data(df)