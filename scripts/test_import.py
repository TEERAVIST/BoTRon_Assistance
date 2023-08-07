import nltk
from nltk import sent_tokenize
from nltk import word_tokenize

# Set the custom data path
custom_data_path = "libary"
nltk.data.path.append(custom_data_path)

# Download the 'punkt' resource if not already downloaded
nltk.download('punkt', download_dir=custom_data_path)

# Read the content of the file
with open("../Text/Natural_language_Processing_Text.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Tokenize the text into sentences
sentences = word_tokenize(text)

print(len(sentences))
print(sentences)

