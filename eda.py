import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numexpr as en
sns.set()
import random
import json
import time
import nltk
import torch
import math
from nlp_aug import eda_4
from random import shuffle
import kenlm

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.tmt.v20180321 import tmt_client, models

random.seed(1)

# stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
              'ours', 'ourselves', 'you', 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his',
              'himself', 'she', 'her', 'hers', 'herself',
              'it', 'its', 'itself', 'they', 'them', 'their',
              'theirs', 'themselves', 'what', 'which', 'who',
              'whom', 'this', 'that', 'these', 'those', 'am',
              'is', 'are', 'was', 'were', 'be', 'been', 'being',
              'have', 'has', 'had', 'having', 'do', 'does', 'did',
              'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
              'because', 'as', 'until', 'while', 'of', 'at',
              'by', 'for', 'with', 'about', 'against', 'between',
              'into', 'through', 'during', 'before', 'after',
              'above', 'below', 'to', 'from', 'up', 'down', 'in',
              'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no',
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
              'very', 's', 't', 'can', 'will', 'just', 'don',
              'should', 'now', '']

# cleaning up text
import re


def get_only_chars(line):
    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm$- ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

# for the first time you use wordnet
# import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet
import nltk
from pywsd import simple_lesk

def synonym_replacement(words, n, adjusted):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words and word != '$t$']))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        if not adjusted:
            synonyms = get_synonyms(random_word)
        else:
            synonyms = get_synonyms_adjusted(words, random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            temp = []
            replaced = False
            for word in new_words:
                if word == random_word and not replaced:
                    temp.append(synonym)
                    replaced = True
                else:
                    temp.append(word)
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
            new_words = temp
        if num_replaced >= n:  # only replace up to n words
            break

    # this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def get_synonyms_adjusted(words, random_word):
    pos_tags = nltk.pos_tag(words)
    for word, func in pos_tags:
        if word == random_word and not get_wordnet_pos(func):
            return []
        elif word == random_word:
            meaning = simple_lesk(' '.join(words), random_word, pos=get_wordnet_pos(func))
    synonyms = []
    if meaning:
        for syn in meaning.lemma_names():
            synonym = syn.lower()
            synonyms.append(synonym)
        if random_word in synonyms:
            synonyms.remove(random_word)
    return synonyms


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):
    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p or '$t$' in word:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n, adjusted=False):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(new_words, adjusted=False):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word) if not adjusted else get_synonyms_adjusted(new_words, random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr=1, alpha_ri=1, alpha_rs=1, p_rd=0, percentage=.2, adjusted=True,num_aug=1):
    # sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    words = [word for word in words if word is not '']
    num_words = len(words)

    augmented_sentences = []
    n_sr = max(1, int(percentage * num_words))
    n_ri = max(1, int(percentage * num_words))
    n_rs = max(1, int(percentage * num_words))

    # sr
    for _ in range(alpha_sr):
        a_words = synonym_replacement(words, n_sr, adjusted)
        augmented_sentences.append(' '.join(a_words))

    # ri
    for _ in range(alpha_ri):
        a_words = random_insertion(words, n_ri, adjusted)
        augmented_sentences.append(' '.join(a_words))
    # rs
    if not adjusted:
        for _ in range(alpha_rs):
            a_words = random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words))
    # rd
    for _ in range(p_rd):
        a_words = random_deletion(words, percentage)
        augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [sentence for sentence in augmented_sentences]
    shuffle(augmented_sentences)
    augmented_sentences = augmented_sentences[:num_aug]
    augmented_sentences.append(sentence)
    return augmented_sentences

########################################################################
# the proposed score of REM and CEM
########################################################################
def JointH(a,b):
    s = 0.0
    for i in range(a.size()[0]):
        for j in range(b.size()[0]):
            _a = a[i]
            _b = b[j]
            p = _a * _b
            if p<=1e-10:
                continue
            s += p*torch.log2(p)
    return s*-1.0
def mutal_info(a,b):
    return H(a)+H(b)-JointH(a,b)
EPS = 1e-10
def MI(z,zt):
    C = z.size()[1]
    # actually they are not independent
    P = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
    P = ((P + P.t())/2) / P.sum()
    P[(P<EPS).data] = EPS
    Pi = P.sum(dim=1).view(C,1).expand(C,C)
    Pj = P.sum(dim=0).view(1,C).expand(C,C)
    # revise by 1.0
    return 1.0-(P * (-torch.log(Pi)-torch.log(Pj)+torch.log(P))).sum()
def CEM(z,zt):
    return MI(z,zt)-H(z)
def H(P):
    P[(P<EPS).data] = EPS
    return -(P*torch.log(P)).sum()
def REM(z,zt):
    EPS = 1e-10
    zt[(zt<EPS).data] = EPS
    return -torch.sum(z*torch.log(zt))
def gradmutualgain(output,label,one_hot,softmaxed,softmaxed_y,loss_fn=None):
    up = REM(softmaxed.unsqueeze(0),one_hot.unsqueeze(0))
    # make all the less than zero > 0
    down = 1.0+CEM(softmaxed.unsqueeze(0),softmaxed_y.unsqueeze(0))
    return up,down
# PPL_model = kenlm.Model('lms/trec.arpa')
def PPLSim(sentence,out_x,out_y):
    up = PPL_model.perplexity(sentence)
    down = torch.cosine_similarity(out_x.unsqueeze(0),out_y.unsqueeze(0))[0]
    return up,down
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
              'ours', 'ourselves', 'you', 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his',
              'himself', 'she', 'her', 'hers', 'herself',
              'it', 'its', 'itself', 'they', 'them', 'their',
              'theirs', 'themselves', 'what', 'which', 'who',
              'whom', 'this', 'that', 'these', 'those', 'am',
              'is', 'are', 'was', 'were', 'be', 'been', 'being',
              'have', 'has', 'had', 'having', 'do', 'does', 'did',
              'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
              'because', 'as', 'until', 'while', 'of', 'at',
              'by', 'for', 'with', 'about', 'against', 'between',
              'into', 'through', 'during', 'before', 'after',
              'above', 'below', 'to', 'from', 'up', 'down', 'in',
              'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no',
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
              'very', 's', 't', 'can', 'will', 'just', 'don',
              'should', 'now', '']

def tackle(input_text):
    input_text = input_text.lower()
    input_text = ' '.join(nltk.word_tokenize(input_text))
    punctuations = '''!()-[]{}|;:'"\,<>./?@#$%^&*_~`'''

    # To take input from the user
    # my_str = input("Enter a string: ")

    # remove punctuation from the string
    no_punct = ""
    for char in input_text:
        if char not in punctuations:
            no_punct = no_punct + char
    no_punct = no_punct.split()
    ans = []
    for word in no_punct:
        if word not in stop_words:
            ans.append(word)
    return ' '.join(ans)

def epda(txt,label,num_aug=1,get_embed_fn=None,loss_fn=None,model=None,translator=None,engine='EDA',alpha=0.1,mix_up=False,nlp_auger=None,alpha_epda=0.5):
    NEED_SAV = False
    model.eval()
    # print("txt=",txt)
    # print("PLZ input sth")
    # txt = input()
    if engine =='EEDA':
        txts = eda(txt,num_aug=2*num_aug)
    elif engine=='EDA':
        txts = eda_4(txt,alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=3*num_aug)
    else:
        txts = nlp_auger.augment(txt,n=num_aug*5)
    try:
        if translator is not None and random.random()<=0.0:
            txts += [tackle(translator(txt,num_aug=1)[0])]
    except:
        print("Error! Don't Back Translate!")
    txts = list(set(txts))
    # print("??",len(txts))
        # print("txts:=",txts)
    txts = [txt] + txts
    oldtxts = txts
    if get_embed_fn is not None:
        newtxts = []
        for txt in txts:
            out = get_embed_fn(txt)
            newtxts.append(out)
        txts = newtxts
    inputs = torch.stack(newtxts).cuda()
    outputs = model(inputs).detach().cpu()
    ups,downs,scores = [],[],[]
    # alpha_epda = 0.4
    for i in range(0,len(newtxts)):
        # a = torch.softmax(outputs[0],0)
        b = torch.softmax(outputs[i],0)
        c = torch.softmax(outputs[0],0)
        C = b.size(0)
        a = torch.zeros(C)
        a[label] = 1.0
        # print('a',a,'b',b)
        # _up,_down = PPLSim(oldtxts[i],a,b)
        _up,_down = gradmutualgain(outputs[i],label,a,b,c,loss_fn)
        ups.append(_up)
        downs.append(_down)
    ups = np.array(ups)
    downs = np.array(downs)
    ups = (ups-np.min(ups))/(np.max(ups)-np.min(ups))
    downs = (downs-np.min(downs))/(np.max(downs)-np.min(downs))
    for i in range(downs.shape[0]):
        _up,_down=ups[i],downs[i]
        score = alpha_epda * _up + (1.0-alpha_epda)*_down
        # print("s = ",oldtxts[i],"gm = ",_up,"mim=",_down)
        # print("??",type(score),score)
        if score == np.nan or math.isnan(score):
            # print("Oh no!")
            score = 1.0
        # print("up=",_up,'down=',_down)
        scores.append(score)
        # print(oldtxts[i+1],'score:',_up,_down,score)
    scores = np.array(scores)
    sortargs = np.argsort(-scores).tolist()
    # print(scores,sortargs)
    model.train()
    newtxts = []
    newscores = []
    if not mix_up or i+num_aug>=len(sortargs):
        if NEED_SAV:
            f = open("data/epda.txt",'a')
            f.write(str(label)+'\t'+oldtxts[0]+'\n')
            f.close()
        for idx in sortargs[:num_aug]:
            if NEED_SAV:
                f = open("data/epda.txt",'a')
                f.write(str(label)+'\t'+oldtxts[idx+1]+'\n')
                f.close()
            newtxts.append(txts[idx])
            newscores.append(scores[idx])
    else:
        for i in range(num_aug):
            # print("LEN",len(sortargs),i+num_aug)
            idx_1, idx_2 = sortargs[i], sortargs[i+num_aug]
            newtxts.append(txts[idx_1])
            newscores.append(scores[idx_1])
    # newtxts.append(txts[0])
    # newscores.append(2.0)
    # print("Final",newscores)
    return newtxts,newscores
def move_to_device(batch, rank = None):
    ans = {}
    if (rank is None):
        device = 'cuda'
    else:
        device = 'cuda:{}'.format(rank)
    for key in batch:
        try:
            ans[key] = batch[key].to(device = device)
        except Exception as e:
            # print(str(e))
            ans[key] = batch[key]
    return ans
def epda_bert(txt,label,num_aug=1,get_embed_fn=None,loss_fn=None,model=None,translator=None,engine='EDA',alpha=0.1,mix_up=False,nlp_auger=None,alpha_epda=0.5):
    NEED_SAV = False
    model.eval()
    # print("txt=",txt)
    # print("PLZ input sth")
    # txt = input()
    # label = 1
    if engine =='EEDA':
        txts = eda(txt,num_aug=2*num_aug)
    elif engine=='EDA':
        txts = eda_4(txt,alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=50*num_aug)
    else:
        txts = nlp_auger.augment(txt,n=num_aug*5)
    try:
        if translator is not None and random.random()<=0.0:
            txts += [tackle(translator(txt,num_aug=1)[0])]
    except:
        print("Error! Don't Back Translate!")
    txts = list(set(txts))
    txts = [txt] + txts
    oldtxts = txts
    encoding = get_embed_fn(txts, return_tensors = 'pt', padding = True, truncation = True)
    labels = []
    for i in range(len(txts)):
        labels.append(label)
    labels = torch.tensor(labels).long()
    encoding['labels'] = labels
    encoding = move_to_device(encoding)

    try:
        outputs = model(**encoding)
        outputs = outputs.logits.detach().cpu()
    except:
        outputs = model(encoding).detach().cpu()
    ups,downs,scores = [],[],[]
    for i in range(0,len(oldtxts)):
        softmaxed = torch.softmax(outputs[i],0)
        c = torch.softmax(outputs[0],0)
        C = softmaxed.size(0)
        one_hot = torch.zeros(C)
        one_hot[label] = 1.0
        # print(outputs[i],label,a,b)
        _up,_down = gradmutualgain(outputs[i],label,one_hot,softmaxed,c,loss_fn)
        ups.append(_up)
        downs.append(_down)
    ups = np.array(ups)
    downs = np.array(downs)
    ups = (ups-np.min(ups))/(np.max(ups)-np.min(ups))
    downs = (downs-np.min(downs))/(np.max(downs)-np.min(downs))
    for i in range(downs.shape[0]):
        _up,_down=ups[i],downs[i]
        score = alpha_epda*_up + (1.0-alpha_epda)*_down
        if score == np.nan or math.isnan(score):
            score = 1.0
        scores.append(score)
    scores = np.array(scores)
    sortargs = np.argsort(-scores).tolist()
    model.train()
    newtxts = []
    newscores = []
    if not mix_up or i+num_aug>=len(sortargs):
        if NEED_SAV:
            f = open("data/epda.txt",'a')
            f.write(str(label)+'\t'+oldtxts[0]+'\n')
            f.close()
        for idx in sortargs[:num_aug]:
            if NEED_SAV:
                f = open("data/epda.txt",'a')
                f.write(str(label)+'\t'+oldtxts[idx+1]+'\n')
                f.close()
            newtxts.append(oldtxts[idx])
            newscores.append(scores[idx])
    else:
        for i in range(num_aug):
            # print("LEN",len(sortargs),i+num_aug)
            idx_1, idx_2 = sortargs[i], sortargs[i+num_aug]
            # MIX UP
            if random.random()<=0.2:
                score_1 ,score_2= scores[idx_1],scores[idx_2]
                lamda = score_1/(score_1+score_2)
                if lamda == np.nan or math.isnan(lamda):
                    lamda = 0.5
                state = oldtxts[idx_1+1]*lamda+(1.0-lamda)*oldtxts[idx_2+1]
                newtxts.append(state)
                newscores.append(2.0)
            else:
                newtxts.append(oldtxts[idx_1+1])
                newscores.append(scores[idx_1])
    newtxts.append(oldtxts[0])
    newscores.append(2.0)
    return newtxts,newscores
