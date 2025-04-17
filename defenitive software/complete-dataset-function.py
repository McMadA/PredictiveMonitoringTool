import pandas as pd
import glob
import os

#user input variables
input_directory = '../data/raw/*.csv'  # Pad naar de CSV-bestanden
output_directory = '../data/processed'  # Pad naar de verwerkte bestanden
datasheet_name = 'full_datasheet.csv'

# Functie om bestanden correct te sorteren
def sort_key(filename):
    base_name = os.path.basename(filename).replace('.csv', '')
    if base_name == 'exportToCSV':
        return (0, 0)  # Zorg dat 'exportToCSV.csv' eerst komt
    elif '(' in base_name:
        number = int(base_name.split('(')[-1].split(')')[0])
        return (1, number)  # Sorteer op het nummer in de haakjes
    else:
        return (2, 0)  # Catch-all voor andere gevallen

# Vind en sorteer CSV-bestanden
csv_files = glob.glob(input_directory)
csv_files.sort(key=sort_key)

# Verwerk elk bestand
for file in csv_files:
    # Laad dataset
    df = pd.read_csv(file, skiprows=2)

    # Gemeten tijden onder elkaar zetten met transactie als kolomnaam
    df.columns = df.iloc[0]  # Stel eerste rij in als header
    df = df[1:].reset_index(drop=True)  # Verwijder eerste rij uit data
    df = df.T
    df.columns = df.iloc[0]  # Stel eerste rij in als header
    df = df[1:].reset_index(drop=True)  # Verwijder eerste rij uit data

    # Tijden onder elkaar zetten
    times = pd.read_csv(file, skiprows=2)
    times.iloc[0] = times.columns
    first_column = times.iloc[0, :]
    first_column = first_column.loc[~first_column.str.contains('Unnamed')]
    first_column = first_column.dropna().reset_index(drop=True)

    # Tijden toevoegen aan eerste dataframe
    df.insert(0, 'time', first_column)
    df.set_index('time', inplace=True)

    df = df.dropna(how="all")

    # Opslaan naar processed directory
    output_file = os.path.join(output_directory, os.path.basename(file))
    df.to_csv(output_file)
    
    print(f"Processed {file} and saved to {output_file}")

# Combineer de verwerkte CSV-bestanden in één bestand
def combine_processed_csvs(input_directory, output_file):
    csv_files = glob.glob(input_directory + '/*.csv')
    
    # Sorteer opnieuw om consistentie te garanderen
    csv_files.sort(key=sort_key)
    
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    
    combined_df.to_csv(output_file, index=False)
    
    print(f"Combined CSV saved to {output_file}")

combine_processed_csvs(output_directory, datasheet_name)
