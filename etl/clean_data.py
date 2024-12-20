import click
import pandas as pd
import numpy as np
import os

def impute_values(df):
    # Imputaciones para todas las columnas
    df['Credit Score'] = df['Credit Score'].fillna(df['Credit Score'].mean())
    df['Number of Dependents'] = df['Number of Dependents'].fillna(df['Number of Dependents'].mode()[0])
    df['Customer Feedback'] = df['Customer Feedback'].fillna(df['Customer Feedback'].mode()[0])
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Health Score'] = df['Health Score'].fillna(df['Health Score'].median())
    df['Annual Income'] = df['Annual Income'].fillna(df['Annual Income'].mean())
    df['Marital Status'] = df['Marital Status'].fillna(df['Marital Status'].mode()[0])
    df['Vehicle Age'] = df['Vehicle Age'].fillna(df['Vehicle Age'].mean())
    df['Insurance Duration'] = df['Insurance Duration'].fillna(df['Insurance Duration'].median())
    
    # Eliminar duplicados
    df = df.drop_duplicates()
    
    return df

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
@click.option('--train-input', default='data/raw/train.csv', help='Path to input train CSV file.')
@click.option('--test-input', default='data/raw/test.csv', help='Path to input test CSV file.')
@click.option('--train-output', default='data/clean/train_clean.csv', help='Path to output cleaned train CSV file.')
@click.option('--test-output', default='data/clean/test_clean.csv', help='Path to output cleaned test CSV file.')
def clean_data(train_input, test_input, train_output, test_output):
    """Clean and Feature Engineer the Insurance Premium Dataset."""
    # Cargar datos crudos
    train_df = pd.read_csv(train_input)
    test_df = pd.read_csv(test_input)
    
    # Imputar valores faltantes y eliminar duplicados
    train_df = impute_values(train_df)
    test_df = impute_values(test_df)
    
    # Determinar columnas numéricas y categóricas
    numeric_features = train_df.select_dtypes(['float64', 'int64']).columns.tolist()
    if 'Premium Amount' in numeric_features:
        numeric_features.remove('Premium Amount')
    
    # Aplicar feature engineering
    train_df = feature_eng(train_df, numeric_features)
    test_df = feature_eng(test_df, numeric_features)
    
    # Cargar datos limpios existentes si existen
    if os.path.exists(train_output):
        existing_train = pd.read_csv(train_output)
        # Identificar nuevas filas que no están en existing_train
        train_df = train_df[~train_df.apply(tuple,1).isin(existing_train.apply(tuple,1))]
        # Concatenar nuevas filas
        train_df = pd.concat([existing_train, train_df], ignore_index=True).drop_duplicates()
    
    if os.path.exists(test_output):
        existing_test = pd.read_csv(test_output)
        # Identificar nuevas filas que no están en existing_test
        test_df = test_df[~test_df.apply(tuple,1).isin(existing_test.apply(tuple,1))]
        # Concatenar nuevas filas
        test_df = pd.concat([existing_test, test_df], ignore_index=True).drop_duplicates()
    
    # Guardar datos limpios actualizados
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    click.echo(f"Data cleaning and feature engineering completed successfully.")
    click.echo(f"Cleaned Train Data: {train_output}")
    click.echo(f"Cleaned Test Data: {test_output}")

if __name__ == '__main__':
    clean_data()


