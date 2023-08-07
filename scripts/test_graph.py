import nltk
from nltk import word_tokenize, FreqDist
import matplotlib.pyplot as plt

# Set the custom data path
custom_data_path = "libary"
nltk.data.path.append(custom_data_path)

# Download the 'punkt' resource if not already downloaded
nltk.download('punkt', download_dir=custom_data_path)

# Read the content of the file
with open("../Text/Natural_language_Processing_Text.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Tokenize the text into words
words = word_tokenize(text)

# Calculate word frequencies
freq_dist = FreqDist(words)

# Plot the word distribution
plt.figure(figsize=(10, 6))
freq_dist.plot(30, cumulative=False)
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Word Distribution in the Text')
plt.show()
