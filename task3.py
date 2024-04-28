#Hello welcome this is Jhansi Madugula
#This is a task in ML internship provided by CODSOFT,this is task -3 customer churn prediction (April-2024)!!
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("C://Users//yaswa//OneDrive//Desktop//vscode//Codsoft//Churn_Modelling.csv")
X = data.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
y = data['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                               ('cat', categorical_transformer, categorical_features)])

logistic_reg_model = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', LogisticRegression())])

random_forest_model = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', RandomForestClassifier())])

gradient_boosting_model = Pipeline(steps=[('preprocessor', preprocessor),
                                          ('classifier', GradientBoostingClassifier())])

logistic_reg_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)
gradient_boosting_model.fit(X_train, y_train)

lr_predictions = logistic_reg_model.predict(X_test)
rf_predictions = random_forest_model.predict(X_test)
gb_predictions = gradient_boosting_model.predict(X_test)

def evaluate_model(model, predictions):
    accuracy = accuracy_score(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(report)
    print("\n")

evaluate_model(logistic_reg_model, lr_predictions)
evaluate_model(random_forest_model, rf_predictions)
evaluate_model(gradient_boosting_model, gb_predictions)
