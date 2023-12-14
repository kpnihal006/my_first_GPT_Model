#This file is to extract the text from the zip file of the dataset openwebtext
#interact with operating system
import os

#support for reading and writing LZMA compressed files
import lzma

#creates progress bars for loops to track progress
from tqdm import tqdm

print("Running file extraction program")
#list of files
def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

#folder path, output files for train and validation. Vocabulary file with list of characters
folder_path = "/Users/moxieevil/Documents/dev/my_first_gptModel/openwebtext"
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"

print("extracting files from path ", folder_path)

files = xz_files_in_dir(folder_path)
total_files = len(files)

# Calculate the split indices
split_index = int(total_files * 0.9) # 90% for training
files_train = files[:split_index]
files_val = files[split_index:]

#define vocab file as set
vocab = set()

# Process the training files
#writing to output file for training
with open(output_file_train, "w", encoding="utf-8") as outfile:

    #using tqdm to extract files with .tz compressions
    for filename in tqdm(files_train, total=len(files_train)):
        file_path = os.path.join(folder_path, filename)

        #read write format
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:

            #reading the file and extracting text
            text = infile.read()

            #writing to output file
            outfile.write(text)

            #updating vocabulary set
            characters = set(text)
            vocab.update(characters)
            print('training vocab \n',vocab)

# Process the validation files
with open(output_file_val, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_val, total=len(files_val)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)
            print('validation vocab \n',vocab)

# Write the vocabulary to vocab.txt
with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in vocab:
        vfile.write(char + '\n')
