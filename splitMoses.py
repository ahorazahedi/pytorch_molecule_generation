import pandas as pd
from tools import *
# Step 1: Read the CSV File
data = pd.read_csv('MOSES.csv', header=None, names=['SMILES', 'SPLIT'])

# Step 2: Separate into train, test, and validation sets
train_data = data[data['SPLIT'] == 'train']['SMILES']
test_data = data[data['SPLIT'] == 'test_scaffolds']['SMILES']
validation_data = data[data['SPLIT'] == 'test']['SMILES']

# Step 3: Shuffle the training data
train_data = train_data.sample(frac=1).reset_index(drop=True)

# Step 4: Split the training data into 5 chunks
train_chunks = []
chunk_size = len(train_data) // 5
for i in range(4):
    train_chunks.append(train_data[i*chunk_size:(i+1)*chunk_size])
train_chunks.append(train_data[4*chunk_size:])  # Last chunk

# Step 5: Write each chunk to a new SMI file and print their lengths
for i, chunk in enumerate(train_chunks):
    chunk.to_csv(f'./data/train_chunk_{i+1}.smi', header=False, index=False)
    print(f'Length of train_chunk_{i+1}.smi: {len(chunk)}')

# Write test and validation sets and print their lengths
test_data.to_csv('./data/test.smi', header=False, index=False)
print(f'Length of test.smi: {len(test_data)}')

validation_data.to_csv('./data/validation.smi', header=False, index=False)
print(f'Length of validation.smi: {len(validation_data)}')

all_smiles = data['SMILES']
# Step 3: Write all SMILES to a single file
all_smiles.to_csv('./data/all.smi', header=False, index=False)


max_number_of_atoms = get_max_n_atoms('./data/all.smi')
print("Maximum Number of Atoms in a Molecule:" , max_number_of_atoms)

atom_types = get_atom_types('./data/all.smi')
print("Atom Types:" , atom_types)

formal_charge = get_formal_charges('./data/all.smi')
print("Formal Charges:" , formal_charge)