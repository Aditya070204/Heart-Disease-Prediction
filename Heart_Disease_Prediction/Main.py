import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix


class HeartDiseasePredictionModel:

    def __init__(self,datset):
        # Loading Dataset
        self.disease_df = pd.read_csv(datset)
        # print(disease_df.columns)
        # print(disease_df.shape)
    
    def prepare_dataset(self):
        # Removing Unnecessary data feature education from disease_df
        self.disease_df.drop(['education'],inplace=True,axis = 1)
        # Renaming feature male to sex_male
        self.disease_df.rename(columns = {'male':'Sex_male'}, inplace=True)
        # Handling missing values
        self.disease_df.dropna(axis=0,inplace=True)
    
    def preprocess_dataset(self):
        # Splitting the dataset into training and testing datasets
        # X = disease_df.drop(columns='TenYearCHD',axis=1)
        X = np.asarray(self.disease_df[['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']])
        y = self.disease_df['TenYearCHD']
        # Preprocessing data
        self.scaler = preprocessing.StandardScaler()
        X = self.scaler.fit_transform(X)
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,y,test_size=0.3,random_state=22261)
        
    def fit_model(self):
        # Fitting Logistic Regression Model
        self.logreg = LogisticRegression()
        self.logreg.fit(self.X_train,self.y_train)
        y_pred = self.logreg.predict(self.X_test)
        score = accuracy_score(y_pred,self.y_test)

    def predict_chd(self, input_values):
        # Scale the input values using the same scaler used for training
        scaled_input_values = self.scaler.transform([input_values])
        # Make prediction
        prediction = self.logreg.predict_proba(scaled_input_values)
        return prediction[0][1]  # return the probability of CHD
    
# Usage example
if __name__ == "__main__":
    model = HeartDiseasePredictionModel("Heart_Disease_Prediction/framingham.csv")
    model.prepare_dataset()
    model.preprocess_dataset()
    model.fit_model()
    # Example input values
    input_values = [50, 1, 40, 240, 180, 126]  # Example values for age, sex, cigsPerDay, totChol, sysBP, glucose
    chd_probability = model.predict_chd(input_values)
    print("Probability of CHD:", chd_probability)











