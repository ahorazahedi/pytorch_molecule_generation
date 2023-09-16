# Molecular Graph Generator for Drug-like Molecules

## Overview 

This repository contains a powerful tool for creating and visualizing molecular graphs for drug-like molecules. The core functionality is based on action prediction distribution. Users can generate and analyze molecule structures to aid their research or development projects in pharmaceuticals, drug discovery, and molecular biology.

## Features

1. **Molecular Graph Generation:** Create sophisticated and detailed molecular graphs for drug-like molecules. 
2. **Action Prediction Distribution:** Utilize advanced algorithms to predict the action distribution of molecules.
3. **Customizability:** Modify parameters and control the flow of the program as per your requirements.
   
## Getting Started

### Prerequisites

Make sure you have installed all of the following prerequisites on your development machine:

- Python 3.6 or higher
- Other dependencies (specify all the necessary libraries and their versions).

Clone the repository:

```sh
git clone https://github.com/ahorazahedi/pytorch_molecule_generation.git
```

### Installation & Usage

Navigate to the project directory:


Install the necessary packages:

```sh
pip install torch rdkit tensorboard pprint tqdm 
```

To start the program, run the following command in the project directory:

```sh
python flow.py
```

## Customizing the Generator

You can customize the program to suit your needs by modifying parameters and the control flow of the program:

1. **Parameters:** You can modify the parameters in the `parameters.py` file. The parameters include, but are not limited to, molecule size, bond length, and atom types. Each parameter is thoroughly commented for your understanding.

2. **Control Flow:** The `flow.py` file contains the main control flow of the program. Here, you can customize the sequence of operations and their behavior. The control flow is thoroughly commented for easier understanding and modification.

## Contributing

We appreciate all contributions. If you're interested in contributing, please see the `CONTRIBUTING.md` for details.

## License

This project is licensed under the MIT License 

## Contact

If you have any questions, feel free to contact us at `ahora.zhd@gmail.com` or open an issue.

## Acknowledgements

We thank all the contributors who have helped to develop this project. Your help is much appreciated!
