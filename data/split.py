import argparse
import random

def split_smi_file(file_path, count=10):
    print(file_path, count)

    # Read the contents of the original file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Shuffle the lines
    random.shuffle(lines)

    # Calculate the number of lines in each chunk
    chunk_size = len(lines) // count

    # Splitting the file into chunks
    for i in range(count):
        start = i * chunk_size
        # Ensure the last chunk gets any extra lines
        end = None if i == count - 1 else start + chunk_size

        # Write each chunk to a new file
        with open(f"{file_path}_chunk_{i+1}.smi", 'w', encoding='utf-8') as chunk_file:
            chunk_file.writelines(lines[start:end])

def main():
    parser = argparse.ArgumentParser(description='Split an SMI file into chunks after shuffling.')
    parser.add_argument('--file', help='Path to the SMI file to split')
    parser.add_argument('--count', type=int, default=10, help='Number of chunks to split into')

    args = parser.parse_args()

    split_smi_file(args.file, args.count)

if __name__ == '__main__':
    main()
