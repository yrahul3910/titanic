from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import json
import os

app = Flask(__name__)

# Load model and scaler
model_path = os.path.join(os.path.pardir, os.path.pardir, "models")
model_file = os.path.join(model_path, "lr_model.pkl")
scaler_file = os.path.join(model_path, "lr_scaler.pkl")

scaler = pickle.load(open(scaler_file, "rb"))
model = pickle.load(open(model_file, "rb"))

columns = ["Age", "Fare", "FamilySize",
       "IsMother", "IsMale", "Deck_A", "Deck_B", "Deck_C", "Deck_D", 
       "Deck_E", "Deck_F", "Deck_G", "Deck_Z", "Pclass_1", "Pclass_2", 
       "Pclass_3", "Title_Lady", "Title_Master", "Title_Miss", "Title_Mr", 
       "Title_Mrs", "Title_Officer", "Title_Sir", "Fare_Bin_Very_Low", 
       "Fare_Bin_Low", "Fare_Bin_High", "Fare_Bin_Very_High", "Embarked_C", 
       "Embarked_Q", "Embarked_S", "AgeState_Adult", "AgeState_Child"]

@app.route("/api", methods=["POST"])
def make_prediction():
    # Read JSON from the request, convert to JSON string
    data = json.dumps(request.get_json(force=True))
    
    # Create Pandas DataFrame
    df = pd.read_json(data)
    
    # Extract PassengerId
    pids = df["PassengerId"].ravel()
    
    X = df[columns].as_matrix().astype("float")
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    df_response = pd.DataFrame({"PassengerId": pids, "Predicted": predictions})
    
    return df_response.to_json()

if __name__ == "__main__":
    app.run(port=9001, debug=True)