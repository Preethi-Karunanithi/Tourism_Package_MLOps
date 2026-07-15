
import pandas as pd
import sklearn
import os

#from google.colab import userdata
from huggingface_hub import HfApi

from huggingface_hub.utils import RepositoryNotFoundError
import os

token = os.getenv("HF_TOKEN")



# for data preprocessing and pipeline creation
#from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
#from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
#from huggingface_hub import login, HfApi

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# Define constants for the dataset and output paths


token = os.getenv("HF_TOKEN")

if token is None:
    raise ValueError("HF_TOKEN environment variable is not set.")

api = HfApi(token=token)

# create  dataset as you creating space
#DATASET_PATH = "/content/tourism_project/data/tourism-package-prediction.csv"

DATASET_PATH = "tourism_project/data/tourism-package-prediction.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop('CustomerID', axis=1, inplace=True)

# Drop the unique identifier
if 'CustomerID' in df.columns:
    df.drop('CustomerID', axis=1, inplace=True)

# Create Age_Group
bins = [0, 18, 30, 45, 60, 100]
labels = ['Child', 'Young Adult', 'Adult', 'Middle Aged', 'Senior']

df['Age_Group'] = pd.cut(
    df['Age'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# Drop the original Age column
df.drop('Age', axis=1, inplace=True)

# Dummy encode Age_Group
df = pd.get_dummies(df, columns=['Age_Group'], drop_first=True)

# Correct inconsistent values
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')

# Convert categorical columns (only those that actually exist)
categorical_columns = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'ProductPitched'
]

for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Dummy encode categorical columns
dummy_columns = [
    'Type',
    'Gender',
    'MaritalStatus',
    'TypeofContact',
    'Occupation',
    'ProductPitched'
]

dummy_columns = [col for col in dummy_columns if col in df.columns]

df = pd.get_dummies(df, columns=dummy_columns, drop_first=True)

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]



# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} was not created.")

    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id="PreethiKarunanithi/Preethi-first-project-space",
        repo_type="dataset",
    )

print("All files uploaded successfully.")
