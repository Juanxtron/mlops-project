import click
import pandas as pd
import numpy as np

# Funciones de limpieza y feature engineering
def impute_values(train_df, test_df):
    train_df['Credit Score'] = train_df['Credit Score'].fillna(train_df['Credit Score'].mean())
    train_df['Number of Dependents'] = train_df['Number of Dependents'].fillna(train_df['Number of Dependents'].mode()[0])
    train_df['Customer Feedback'] = train_df['Customer Feedback'].fillna(train_df['Customer Feedback'].mode()[0])
    train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
    train_df['Health Score'] = train_df['Health Score'].fillna(train_df['Health Score'].median())
    train_df['Annual Income'] = train_df['Annual Income'].fillna(train_df['Annual Income'].mean())
    train_df['Marital Status'] = train_df['Marital Status'].fillna(train_df['Marital Status'].mode()[0])
    train_df['Vehicle Age'] = train_df['Vehicle Age'].fillna(train_df['Vehicle Age'].mean())
    train_df['Insurance Duration'] = train_df['Insurance Duration'].fillna(train_df['Insurance Duration'].median())
    
    test_df['Credit Score'] = test_df['Credit Score'].fillna(test_df['Credit Score'].mean())
    test_df['Number of Dependents'] = test_df['Number of Dependents'].fillna(test_df['Number of Dependents'].mode()[0])
    test_df['Customer Feedback'] = test_df['Customer Feedback'].fillna(test_df['Customer Feedback'].mode()[0])
    test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
    test_df['Health Score'] = test_df['Health Score'].fillna(test_df['Health Score'].median())
    test_df['Annual Income'] = test_df['Annual Income'].fillna(test_df['Annual Income'].mean())
    test_df['Marital Status'] = test_df['Marital Status'].fillna(test_df['Marital Status'].mode()[0])
    test_df['Vehicle Age'] = test_df['Vehicle Age'].fillna(test_df['Vehicle Age'].mean())
    test_df['Insurance Duration'] = test_df['Insurance Duration'].fillna(test_df['Insurance Duration'].median())
    
    train_df = train_df.drop_duplicates()
    test_df = test_df.drop_duplicates()
    return train_df, test_df

def feature_eng(df, numeric_features):
    if 'Policy Start Date' in df.columns:
        df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'], errors='coerce')
        df['Policy Start Year'] = df['Policy Start Date'].dt.year.astype(str)
        df.drop(columns=['Policy Start Date'], inplace=True)
    
    for col in numeric_features:
        if col in df.columns:
            skewness = df[col].skew()
            if skewness > 1 or skewness < -1:
                df[col] = np.log1p(df[col])
    return df

@click.command()
@click.option('--train-input', default='data/train.csv', help='Path to input train CSV file.')
@click.option('--test-input', default='data/test.csv', help='Path to input test CSV file.')
@click.option('--train-output', default='data/train_clean.csv', help='Path to output cleaned train CSV file.')
@click.option('--test-output', default='data/test_clean.csv', help='Path to output cleaned test CSV file.')
def clean_data(train_input, test_input, train_output, test_output):
    """Clean and Feature Engineer the Insurance Premium Dataset."""
    train_df = pd.read_csv(train_input)
    test_df = pd.read_csv(test_input)
    
    train_df, test_df = impute_values(train_df, test_df)
    
    numeric_features = train_df.select_dtypes(['float64', 'int64']).columns.tolist()
    if 'Premium Amount' in numeric_features:
        numeric_features.remove('Premium Amount')
    
    train_df = feature_eng(train_df, numeric_features)
    test_df = feature_eng(test_df, numeric_features)
    
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    click.echo(f"Data cleaning completed successfully.")
    click.echo(f"Cleaned Train Data: {train_output}")
    click.echo(f"Cleaned Test Data: {test_output}")

if __name__ == '__main__':
    clean_data()
