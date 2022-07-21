import torch
import numpy as np
from torch.utils.data import Dataset
from konlpy.tag import Mecab
from collections import Counter
from itertools import chain
from functools import partial
from tqdm import tqdm
import pickle

from preprocessing import basic_preprocess, preprocess
#def tokenize(tokenizer):

class Vocab():
    PAD_TOKEN = '<PAD>'
    PAD_TOKEN_ID = 0
    UNK_TOKEN = '<UNK>'
    UNK_TOKEN_ID = 1
    SOS_TOKEN = '<SOS>'
    SOS_TOKEN_ID = 2
    EOS_TOKEN = '<EOS>'
    EOS_TOKEN_ID = 3

    def __init__(self, tokenizer=Mecab, do_preprocess=True) -> None:
        self.tokenizer = tokenizer
        
        data_path = "./data/ratings_train.txt" 
        train_f = open(data_path, 'r', encoding="utf-8")
        lines = train_f.readlines()
        
        self.train_corpus, self.train_labels = [], []
        for i in tqdm(range(len(lines)), desc="get_train_data"):
            line = lines[i]
            if i==0:
                continue
            _, sentence, label = line.split(sep='\t')
            
            # preprocessing
            label = label[0]
            sentence = self._preprocess(sentence, do_preprocess)

            self.train_corpus.append(sentence)
            self.train_labels.append(int(label))

        self.test_corpus, self.test_labels = [], []
        test_data_path = "./data/ratings_test.txt" 
        test_f = open(test_data_path, 'r', encoding="utf-8")
        lines = test_f.readlines()
        for i in tqdm(range(len(lines)), desc="get_test_data"):
            line = lines[i]
            if i==0:
                continue
            _, sentence, label = line.split(sep='\t')
            # preprocessing
            label = label[0]
            sentence = self._preprocess(sentence, do_preprocess)
            self.test_corpus.append(sentence)
            self.test_labels.append(int(label))
        
        # test corpus도 포함해서 Vocab을 구성하는 게 맞겠죠?
        self.train_corpus = list(map(self._tokenize_sent,self.train_corpus))
        self.test_corpus = list(map(self._tokenize_sent,self.test_corpus))

        self.id2token, self.token2id = self._build_vocab(self.train_corpus + self.test_corpus, min_freq=2)
        self.vocab_len = len(self.token2id)
        
        # save
        self.obj_to_save = [self.train_corpus, self.train_labels, self.test_corpus, self.test_labels, self.token2id, self.id2token]
        with open('./data/vocab.pkl', 'wb') as f:
            for obj in self.obj_to_save:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
    def _preprocess(self, text, do_preprocess):
        if do_preprocess:
            new_text = preprocess(text)
        else:
            new_text = basic_preprocess(text)
        return new_text

    def _tokenize_sent(self, sentence):
        tokens = self.tokenizer.morphs(sentence)
        return tokens

    def _build_vocab(self, tokens, min_freq=1):
        SPECIAL_TOKENS= [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        id2token = SPECIAL_TOKENS + [word for word, count in Counter(chain(*tokens)).items() if count >= min_freq]
        token2id = {word: idx for idx, word in enumerate(id2token)}

        assert id2token[self.UNK_TOKEN_ID] == self.UNK_TOKEN and token2id[self.UNK_TOKEN] == self.UNK_TOKEN_ID, \
            "[UNK] 토큰을 적절히 삽입하세요"
        assert len(id2token) == len(token2id), \
            "id2word과 word2id의 크기는 같아야 합니다"
        return id2token, token2id

class NSMCDataset(Dataset):
    def __init__(self, corpus, labels, token2id, id2token, tokenizer=Mecab()):
        super(NSMCDataset, self).__init__()
        print(len(token2id))
        self.corpus = corpus
        self.labels = labels
        self.token2id = token2id
        self.id2token = id2token
        self.tokenizer = tokenizer
        

        # test corpus도 포함해서 Vocab을 구성하는 게 맞겠죠?
        self.vocab_len = len(self.token2id)
        self._encoding_corpus()
        self.corpus = self.input_ids
        #self.corpus = list(map(self._one_hot_encoding, self.input_ids))


    def __getitem__(self, index):
        return self.corpus[index], self.labels[index]

    def __len__(self):
        return len(self.corpus)

    def _encoding_corpus(self):
        self.input_ids = list(map(partial(self._encode_sent, token2id=self.token2id), self.corpus))

    def _encode_sent(self, sentence, token2id):
        UNK_TOKEN_ID = 1
        token_ids = torch.tensor([token2id.get(token, UNK_TOKEN_ID) for token in sentence])
        return token_ids

    def one_hot_encoding(self, tok_sent):
        #print(len(tok_sent))
        one_hot = to_categorical(tok_sent)
        return one_hot
        










