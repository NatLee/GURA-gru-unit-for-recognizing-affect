from pathlib import Path
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

def check_data_path(file_path:str) -> bool:
    if Path(file_path).exists():
        print(f'[Path][OK] {file_path}')
        return True
    print(f'[Path][FAILED] {file_path}')
    return False

sentences = []

# =====================
# Anime Reviews
# =====================
dataset = '/data/animeReviewsSkipThoughtSummarization.pkl'
if check_data_path(dataset):
    with open(dataset, 'rb') as p:
        X, Y = pickle.load(p)
        sentences.extend(X)
        sentences.extend(Y)


# =====================
# MPQA
# =====================
dataset = '/data/MPQA.pkl'
if check_data_path(dataset):
    with open(dataset, 'rb') as p:
        mpqa = pickle.load(p)
        sentences.extend(list(mpqa.sentence))


# =====================
# IMDB
# =====================
dataset = '/data/aclImdb.pkl'
if check_data_path(dataset):
    with open(dataset, 'rb') as p:
        x_test, y_test, x_train, y_train = pickle.load(p)
        sentences.extend(x_train)
        sentences.extend(y_train)

# =====================
# MR
# =====================
dataset = '/data/MR.pkl'
if check_data_path(dataset):
    with open(dataset, 'rb') as p:
        mr = pickle.load(p)
        sentences.extend(list(mr.sentence))

# =====================
# SST2
# =====================
dataset = '/data/SST2.pkl'
if check_data_path(dataset):
    with open(dataset, 'rb') as p:
        sst2 = pickle.load(p)
        sentences.extend(list(sst2.sentence))

# =====================
# SUBJ
# =====================
dataset = '/data/SUBJ.pkl'
if check_data_path(dataset):
    with open(dataset, 'rb') as p:
        subj = pickle.load(p)
        sentences.extend(list(subj.sentence))

sentences = map(str, sentences)

#Tokenize the sentences
myTokenizer = Tokenizer(
    num_words = 100,
    oov_token="{OOV}"
)
myTokenizer.fit_on_texts(sentences)
print(myTokenizer.word_index)

with open('/model/big-tokenizer.pkl', 'wb') as p:
    pickle.dump(myTokenizer, p)
