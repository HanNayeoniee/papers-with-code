import os
import pickle
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from konlpy.tag import Mecab
from torch.nn.utils.rnn import pad_sequence

from dataset import Vocab, NSMCDataset
from models.rnn import RNNClassification
from models.lstm import LSTMClassification

'''Config'''
model_name = "RNN"
batch_size = 256
learning_rate = 1e-5
emb_size = 5
hidden_size = 128
num_layers = 4
num_epochs = 200

'''Data'''
print("Preparing Data ...")
# vocab
if os.path.isfile('./data/vocab.pkl'):
    vocab_objs = []
    with open('./data/vocab.pkl', 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
            except EOFError:
                break
            vocab_objs.append(data)
    train_corpus, train_labels, test_corpus, test_labels, token2id, id2token = vocab_objs
else:   
    vocab = Vocab(tokenizer=Mecab(), do_preprocess=True)
    train_corpus, train_labels, test_corpus, test_labels, token2id, id2token = vocab.train_corpus, vocab.train_labels, vocab.test_corpus, vocab.test_labels, vocab.token2id, vocab.id2token

# dataset
train_dataset = NSMCDataset(train_corpus, train_labels, token2id, id2token, tokenizer=Mecab())
test_dataset = NSMCDataset(test_corpus, test_labels, token2id, id2token, tokenizer=Mecab())
print(len(train_dataset))


# dataloader
def collate_fn(batched_samples):
    PAD_TOKEN_ID=0
    corpus, labels = zip(*batched_samples)
    
    input_lengths = [len(sent) for sent in corpus]

    src_sentences = pad_sequence([
        torch.Tensor(sentence).to(torch.long) for sentence in corpus
    ], batch_first=True, padding_value=PAD_TOKEN_ID)

    labels = torch.tensor(labels).unsqueeze(-1)
    return src_sentences, labels, input_lengths

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


'''Model'''
print("Set Model, Loss, Optimizer")
#sample_sent, sample_label = next(iter(train_loader))
#print(sample_sent,sample_label)
vocab_size = train_dataset.vocab_len

if model_name =="RNN":
    model = RNNClassification(vocab_size, emb_size, hidden_size, num_layers)
    # model = RNN()
elif model_name =="LSTM":
    model = LSTMClassification(vocab_size, emb_size, hidden_size, num_layers)

'''Loss , Optimizer'''
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

print("Training...")
"""Training Loop"""
for epoch in range(num_epochs):
    model.train()
    for i, (sentences, labels, lens) in enumerate(train_loader):
        # forward
        #print(sentences.unsqueeze(-1).size())
        outputs = model(sentences, lens)

        # loss
        #print(outputs, labels)
        loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, iter {i}/{len(train_loader)} : train_loss = {loss.item()}")

    model.eval()
    for i, (sentences, labels, lens) in enumerate(test_loader):
        # forward
        outputs = model(sentences, lens)

        # loss
        loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))

        # acc

        print(f"Epoch {epoch}/{num_epochs}, iter {i}/{len(test_loader)} : eval_loss = {loss.item()}")

    

