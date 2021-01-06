import gensim
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def load_yelp_orig_data():
    PATH_TO_YELP_REVIEWS = 'review1000th.json'

    with open(PATH_TO_YELP_REVIEWS, 'r') as f:
        data = f.readlines()

    data = map(lambda x: x.rstrip(), data)

    data_json_str = "[" + ','.join(data) + "]"
    data_df = pd.read_json(data_json_str)
    data_df.head(50).to_csv('output_reviews_top.csv')

def map_sentiment(stars_received):
    if stars_received <= 2:
        return -1
    elif stars_received == 3:
        return 0
    else:
        return 1

def wordProcess():
    load_yelp_orig_data()
    top_data_df = pd.read_csv('output_reviews_top.csv')
    top_data_df['sentiment'] = [map_sentiment(x) for x in top_data_df['stars']]

    top_data_df['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in top_data_df['text']]
    porter_stemmer = PorterStemmer()
    # Get the stemmed_tokens
    top_data_df['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in
                                           top_data_df['tokenized_text']]

    return top_data_df

def split_train_test(top_data_df, test_size=0.3, shuffle_state=True):
    X_train, X_test, Y_train, Y_test = train_test_split(top_data_df[['business_id', 'cool', 'date', 'funny', 'review_id', 'stars', 'text', 'useful', 'user_id', 'stemmed_tokens']],
        top_data_df['sentiment'], shuffle=shuffle_state, test_size=test_size, random_state=15)
    # print("Value counts for Train sentiments")
    # print(Y_train.value_counts())
    # print("Value counts for Test sentiments")
    # print(Y_test.value_counts())
    # print(type(X_train))
    # print(type(Y_train))
    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    Y_train = Y_train.to_frame()
    Y_train = Y_train.reset_index()
    Y_test = Y_test.to_frame()
    Y_test = Y_test.reset_index()
    # print(X_train.head())
    return X_train, X_test, Y_train, Y_test

    # Call the train_test_split
    X_train, X_test, Y_train, Y_test = split_train_test(top_data_df)

def make_word2vec_model(top_data_df_small, padding=True, sg=1, min_count=1, size=10, workers=3, window=3):
    if  padding:
        # print(len(top_data_df_small))
        temp_df = pd.Series(top_data_df_small['stemmed_tokens']).values
        temp_df = list(temp_df)
        temp_df.append(['pad'])
        word2vec_file = 'models/' + 'word2vec_' + str(size) + '_PAD.model'
    else:
        temp_df = top_data_df_small['stemmed_tokens']
        word2vec_file = 'models/' + 'word2vec_' + str(size) + '.model'
    w2v_model = Word2Vec(temp_df, min_count = min_count, size = size, workers = workers, window = window, sg = sg)

    w2v_model.save(word2vec_file)
    return w2v_model, word2vec_file

def make_word2vec_vector_cnn(sentence, w2vmodel ,max_sen_len, padding_idx):

    padded_X = [padding_idx for i in range(max_sen_len)]
    i = 0
    for word in sentence:
        if word not in w2vmodel.wv.vocab:
            padded_X[i] = 0
            # print(word)
        else:
            padded_X[i] = w2vmodel.wv.vocab[word].index
        i += 1
    return torch.tensor(padded_X, dtype=torch.long, device='cpu').view(1, -1)

def make_target(label):
    device = "cpu"

    if label == -1:
        return torch.tensor([0], dtype=torch.long, device=device)
    elif label == 0:
        return torch.tensor([1], dtype=torch.long, device=device)
    else:
        return torch.tensor([2], dtype=torch.long, device=device)


class CnnTextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, window_sizes=(1,2,3,5)):
        super(CnnTextClassifier, self).__init__()
        w2vmodel = gensim.models.KeyedVectors.load('./models/' + 'word2vec_10_PAD.model')
        weights = w2vmodel.wv
        # With pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors), padding_idx=w2vmodel.wv.vocab['pad'].index)
        # Without pretrained embeddings
        # self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE)

        NUM_FILTERS = 9
        EMBEDDING_SIZE = 10
        self.convs = nn.ModuleList([
                                   nn.Conv2d(1, NUM_FILTERS, [window_size, EMBEDDING_SIZE], padding=(window_size - 1, 0))
                                   for window_size in window_sizes
        ])

        self.fc = nn.Linear(NUM_FILTERS * len(window_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x) # [B, T, E]
        # Apply a convolution + max_pool layer for each window size
        x = torch.unsqueeze(x, 1)
        xs = []
        for conv in self.convs:
            # =========================
            x2 = torch.tanh(conv(x))
            x2 = torch.squeeze(x2, -1)
            x2 = F.max_pool1d(x2, x2.size(2))
            xs.append(x2)
        x = torch.cat(xs, 2)

        # FC
        x = x.view(x.size(0), -1)
        logits = self.fc(x)

        probs = F.softmax(logits, dim = 1)

        return probs


