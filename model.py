import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from transformers import BertTokenizer, BertModel
from transformers import AdamW, BertForSequenceClassification,XLMRobertaForSequenceClassification

class CNN(nn.Module):
    def __init__(self,max_len=30,word_dim=300,class_size=2,size='normal'):
        super(CNN, self).__init__()

        self.MAX_SENT_LEN = max_len
        self.WORD_DIM = word_dim
        self.CLASS_SIZE = class_size
        print("size=",size)
        if size=='normal':
            print("Init Normal")
            self.FILTERS = [2,3,4]
            self.FILTER_NUM = [100, 100, 100]
            self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        elif size=='tiny':
            print("Tiny Size")
            self.FILTERS = [3]
            self.FILTER_NUM = [20]
            self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        self.DROPOUT_PROB = 0.5
        self.IN_CHANNEL = 1

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, f'conv_{i}', conv)


    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def forward(self, inp):
        # [B 1 C]
        x = inp.view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        # print(x.size())
        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1)
                .view(-1, self.FILTER_NUM[i])
            for i in range(len(self.FILTERS))]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)
        # x = torch.softmax(x,1)
        return x

# Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
class BLSTMATT(nn.Module):
    def __init__(self, max_len=30,word_dim=300,class_size=2):
        super(BLSTMATT,self).__init__()
        self.hidden_dim = 50
        self.emb_dim = word_dims
        self.dropout = 0.3
        self.encoder = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=2, bidirectional=True, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_dim, class_size)
        self.dropout = nn.Dropout(self.dropout)
        #self.hidden = nn.Parameters(self.batch_size, self.hidden_dim)
    
    def attnetwork(self, encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)
        #M = torch.tanh(encoder_out)
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(encoder_out.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        #print (wt.shape, new_hidden.shape)
        #new_hidden = torch.tanh(new_hidden)
        #print ('UP:', new_hidden, new_hidden.shape)
        
        return new_hidden
    
    def forward(self, sequence):
        # emb_input = self.embedding(sequence)    
        inputx = self.dropout(sequence)
        output, (hn, cn) = self.encoder(inputx)
        fbout = output[:, :, :self.hidden_dim]+ output[:, :, self.hidden_dim:] #sum bidir outputs F+B
        fbout = fbout.permute(1,0,2)
        fbhn = (hn[-2,:,:]+hn[-1,:,:]).unsqueeze(0)
        #print (fbhn.shape, fbout.shape)
        attn_out = self.attnetwork(fbout, fbhn)
        #attn1_out = self.attnetwork1(output, hn)
        logits = self.fc(attn_out)
        return logits
