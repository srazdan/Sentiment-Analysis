import pandas as pd
import csv
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
from wordcloud import WordCloud

data = []
filename = input('Enter the file address: ')
df = pd.read_csv(filename)
data = df['text']
print(len(data))

positive = []
with open('data/lexicons/lexicon.finance.positive.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        positive.append(row[0].strip())
        
with open('data/lexicons/lexicon.generic.positive.HuLiu.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        positive.append(row[0].strip())
        
with open('data/lexicons/lexicon.finance.positive.LoughranMcDonald.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        positive.append(row[0].strip())
        
positive = list(set(positive))

for item in positive:
    item.replace(' ', '')
        
negative = []
with open('data/lexicons/lexicon.finance.negative.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        negative.append(row[0].strip())
        
with open('data/lexicons/lexicon.generic.negative.HuLiu.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        negative.append(row[0].strip())
        
with open('data/lexicons/lexicon.finance.negative.LoughranMcDonald.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        negative.append(row[0].strip())
        
negative = list(set(negative))

for item in negative:
    item.replace(' ', '')

scores = []
x = []
skip = 0
for line in data:
    if skip != 0:
        words = line.split()
        positive_count = 0
        negative_count = 0
        for item in words:
            if item in positive:
                positive_count += 1
            elif item in negative:
                negative_count += 1
        scores.append(positive_count - negative_count)
        x.append(skip)
        skip += 1
    else:
        skip = 1
        
print(len(scores))
b = len(x)
plt.hist(scores, bins = b)
plt.title("Sentiment Scores")
plt.xlabel("scores")
plt.show()

stopwords = []
file = open('data/stopwords/stopwords.currencies.txt', 'r')
for line in file:
    test = line.split()
    for word in test:
        if(word != '|'):
            stopwords.append( word )
            
file = open('data/stopwords/stopwords.dates.numbers.txt', 'r')
for line in file:
    test = line.split()
    for word in test:
        stopwords.append( word )
        
file = open('data/stopwords/stopwords.finance.txt', 'r')
for line in file:
    test = line.split()
    for word in test:
        stopwords.append( word )
        
file = open('data/stopwords/stopwords.generic.txt', 'r')
for line in file:
    test = line.split()
    for word in test:
        stopwords.append( word )
        
file = open('data/stopwords/stopwords.geographic.txt', 'r')
for line in file:
    test = line.split()
    for word in test:
        stopwords.append( word )
        
file = open('data/stopwords/stopwords.names.txt', 'r')
for line in file:
    test = line.split()
    for word in test:
        stopwords.append( word )
        

whole_text = ''
skip = 0
for line in data:
    if skip != 0:
        words = line.split(" ")
        for item in words:
            whole_text = whole_text + ' ' + item
    else:
        skip = 1
        
unicode_text = nltk.word_tokenize(whole_text)
stemmer = SnowballStemmer("english")
stemmed_text = ''

print("reading words from unicode")
for word in unicode_text:
    stemmed_text += stemmer.stem(word) + ' '
    
wc = WordCloud(background_color="white", max_words=2000, stopwords=stopwords)
wc.generate(stemmed_text)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()