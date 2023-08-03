import pandas as pd

# Load the Excel file into a pandas DataFrame
df = pd.read_excel('skipsliste.xlsx')
df["test"] = 1
output_file = "output_file.xlsx"

# Save the modified DataFrame to a new Excel file
df.to_excel(output_file, index=False)

def read(file_name):
    df = pd.read_excel(file_name)
    return df["imo"]

