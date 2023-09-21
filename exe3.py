import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
with open('moby_dick.txt', 'r', encoding='utf-8') as file:
    moby_dick_text = file.read()

tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(moby_dick_text.lower())  


stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]


pos_tags = nltk.pos_tag(filtered_tokens)


pos_counts = FreqDist(tag for (word, tag) in pos_tags)
top_pos = pos_counts.most_common(5)


lemmatizer = WordNetLemmatizer()
top_lemmas = [lemmatizer.lemmatize(word, pos=tag) for word, tag in pos_tags[:20]]


import matplotlib.pyplot as plt

x, y = zip(*pos_counts.items())
plt.bar(x, y)
plt.xlabel('Parts of Speech')
plt.ylabel('Frequency')
plt.title('POS Frequency Distribution')
plt.show()


print(f"Top 5 Parts of Speech:")
for tag, count in top_pos:
    print(f"{tag}: {count}")

print("\nTop 20 Lemmas:")
for lemma in top_lemmas:
    print(lemma)
