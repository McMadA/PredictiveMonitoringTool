import pandas as pd
import glob
import os

csv_files = glob.glob('../data/raw/*.csv')

for file in csv_files:
    #load dataset
    df = pd.read_csv(file, skiprows=2)

    #gemeten tijden onder elkaar zetten met transactie als 
    df.columns = df.iloc[0]  # Set first row as header
    df = df[1:].reset_index(drop=True)  # Remove the first row from data
    df = df.T
    df.columns = df.iloc[0]  # Set first row as header
    df = df[1:].reset_index(drop=True)  # Remove the first row from data


    #tijden onder elkaar zetten
    times = pd.read_csv(file, skiprows=2)
    times.iloc[0] = times.columns
    times.iloc[0] = times.columns
    first_column = times.iloc[0, :] 
    first_column = first_column.loc[~first_column.str.contains('Unnamed')]
    # first_column = first_column.melt(var_name='time slot', value_name='time')
    first_column = first_column.dropna().reset_index(drop=True)


    #tijden toevoegen aan eerste dataframe
    df.insert(0, 'time', first_column)
    df.set_index('time', inplace=True)

    df = df.dropna(how="all")
  
    #delete all excisting csv files in processed folder

    
    output_file = os.path.join('../data/processed', os.path.basename(file))
    df.to_csv(output_file)
    
    print(f"Processed {file} and saved to {output_file}")
    
def combine_processed_csvs(input_directory, output_file):
    csv_files = glob.glob(input_directory + '/*.csv')
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        df_list.append(df)
    
    df_list.reverse()
    
    combined_df = pd.concat(df_list, ignore_index=True)
    
    combined_df.to_csv(output_file, index=False)
    
    print(f"Combined CSV saved to {output_file}")
    
combine_processed_csvs('../data/processed', '../data/datasheet.csv')    
    