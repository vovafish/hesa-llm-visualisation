import pandas as pd
import numpy as np

def transform_chart_data(csv_path):
    """
    Transform the CSV data into the required format for charts.
    Expected columns after transformation: Academic Year, Level of study, Number
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Skip metadata rows (first 6 rows)
    data_start = 6
    df = pd.read_csv(csv_path, skiprows=range(data_start))
    
    # Rename columns if needed
    if 'Year' in df.columns:
        df = df.rename(columns={'Year': 'Academic Year'})
    
    # Ensure required columns exist
    required_columns = ['Academic Year', 'Level of study', 'Number']
    
    # If the data is in wide format (years as columns), melt it to long format
    if not all(col in df.columns for col in required_columns):
        # Identify the year columns (assuming they follow a pattern like YYYY/YY)
        year_columns = [col for col in df.columns if '/' in str(col)]
        
        if year_columns:
            # Melt the dataframe to convert years to rows
            df = pd.melt(
                df,
                id_vars=[col for col in df.columns if col not in year_columns],
                value_vars=year_columns,
                var_name='Academic Year',
                value_name='Number'
            )
    
    # Clean up the data
    df = df.dropna()
    df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
    
    # Verify required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    return df

def prepare_chart_data(df, chart_type):
    """
    Prepare the data for different chart types
    """
    if chart_type == 'line':
        # Group by Academic Year and Level of study
        data = df.pivot(index='Academic Year', columns='Level of study', values='Number')
        return {
            'labels': data.index.tolist(),
            'datasets': [
                {
                    'label': level,
                    'data': data[level].tolist(),
                    'fill': False,
                    'borderColor': f'hsl({(i * 360/len(data.columns))}, 70%, 50%)'
                }
                for i, level in enumerate(data.columns)
            ]
        }
    
    elif chart_type == 'bar':
        # Group by Level of study and sum the numbers
        data = df.groupby('Level of study')['Number'].sum().sort_values(ascending=False)
        return {
            'labels': data.index.tolist(),
            'datasets': [{
                'label': 'Total Students',
                'data': data.values.tolist(),
                'backgroundColor': [
                    f'hsl({(i * 360/len(data))}, 70%, 50%)'
                    for i in range(len(data))
                ]
            }]
        }
    
    elif chart_type == 'pie':
        # Group by Level of study and sum the numbers
        data = df.groupby('Level of study')['Number'].sum()
        return {
            'labels': data.index.tolist(),
            'datasets': [{
                'data': data.values.tolist(),
                'backgroundColor': [
                    f'hsl({(i * 360/len(data))}, 70%, 50%)'
                    for i in range(len(data))
                ]
            }]
        }
    
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}") 