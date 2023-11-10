import csv
import random

with open("MOSES.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    smiles_data = [row[0] for row in reader]  

random.shuffle(smiles_data)

train_split = int(0.7 * len(smiles_data))
valid_split = int(0.85 * len(smiles_data))

train_data = smiles_data[:train_split]
valid_data = smiles_data[train_split:valid_split]
test_data = smiles_data[valid_split:]

# Write the data to separate .smi files
with open("train.smi", "w") as file:
    for smi in train_data:
        file.write(smi + "\n")

with open("valid.smi", "w") as file:
    for smi in valid_data:
        file.write(smi + "\n")

with open("test.smi", "w") as file:
    for smi in test_data:
        file.write(smi + "\n")