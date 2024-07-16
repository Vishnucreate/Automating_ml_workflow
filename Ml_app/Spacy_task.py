import spacy
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

nlp = spacy.load("en_core_web_sm")

# Function to clean a text column using SpaCy
def clean_text_column(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].apply(lambda x: clean_text(x))
    return dataframe

def clean_text(text):
    doc = nlp(text)
    cleaned_text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return cleaned_text

# Function to detect and remove outliers using the IQR method
def remove_outliers(dataframe, column_name):
    Q1 = dataframe[column_name].quantile(0.25)
    Q3 = dataframe[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    cleaned_df = dataframe[(dataframe[column_name] >= lower_bound) & (dataframe[column_name] <= upper_bound)]
    return cleaned_df

# Function to train classification model using ensemble methods
def train_classification_model(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using RandomForestClassifier
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Classification Accuracy: {rf_accuracy}")
    
    # Using GradientBoostingClassifier
    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    gb_accuracy = accuracy_score(y_test, y_pred_gb)
    print(f"Gradient Boosting Classification Accuracy: {gb_accuracy}")
    
    # Select the best model
    if rf_accuracy > gb_accuracy:
        print("Random Forest selected as the best model.")
        return rf_model
    else:
        print("Gradient Boosting selected as the best model.")
        return gb_model

# Function to train regression model using ensemble methods
def train_regression_model(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using RandomForestRegressor
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, y_pred_rf)
    print(f"Random Forest Regression Mean Squared Error: {rf_mse}")
    
    # Using GradientBoostingRegressor
    gb_model = GradientBoostingRegressor()
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    gb_mse = mean_squared_error(y_test, y_pred_gb)
    print(f"Gradient Boosting Regression Mean Squared Error: {gb_mse}")
    
    # Select the best model
    if rf_mse < gb_mse:
        print("Random Forest selected as the best model.")
        return rf_model
    else:
        print("Gradient Boosting selected as the best model.")
        return gb_model

# Main function to handle the complete process
def main():
    # Load your dataset
    file_path = input("Enter the path to your dataset: ").strip()
    
    # Remove any surrounding quotes
    if file_path.startswith('"') and file_path.endswith('"'):
        file_path = file_path[1:-1]
    elif file_path.startswith("'") and file_path.endswith("'"):
        file_path = file_path[1:-1]
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"File loaded successfully with {encoding} encoding.")
                break
            except UnicodeDecodeError:
                print(f"Failed to load file with {encoding} encoding. Trying next encoding...")
        else:
            print("Failed to load the file with all tried encodings.")
            return
    except OSError as e:
        print(f"Error loading the file: {e}")
        return
    
    # Check if the user wants to clean the text data
    clean_data = input("Do you need to clean the text data? (yes/no): ").strip().lower()
    if clean_data == 'yes':
        text_column = input("Enter the text column name to clean: ")
        df = clean_text_column(df, text_column)
        print("Text data cleaned successfully.")
    else:
        print("Skipping text data cleaning.")
    
    # Handle missing values in numerical columns
    numerical_columns = ['numeric_column1', 'numeric_column2']  # specify your numerical columns
    imputer = SimpleImputer(strategy='mean')
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])
    print("Missing values handled successfully.")
    
    # Remove outliers
    for column in numerical_columns:
        df = remove_outliers(df, column)
    print("Outliers removed successfully.")
    
    # Get the target column and task type
    target_column = input("Enter the target column name: ")
    task_type = input("Enter the task type (classification/regression): ")
    
    # Train the appropriate model
    if task_type == 'classification':
        model = train_classification_model(df, target_column)
    elif task_type == 'regression':
        model = train_regression_model(df, target_column)
    else:
        print("Invalid task type. Please enter 'classification' or 'regression'.")
        return
    
    print("Model trained successfully.")

if __name__ == "__main__":
    main()
