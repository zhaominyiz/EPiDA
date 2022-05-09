# EPDA Easy Plug-in Data Augumentation
# The Code following SUB^2
# For various tasks, various parameters should be set
# Here, we provide the basic set for CWE
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
import numpy as np
import re
import random
import transformers

from torch.utils.data import Dataset, DataLoader
# import model
from model import CNN
from transformers import BertTokenizer, BertModel, XLNetTokenizer, XLMRobertaTokenizer
from transformers import AdamW, BertForSequenceClassification,XLNetForSequenceClassification,XLMRobertaForSequenceClassification
from eda import eda,epda,Translator,epda_bert
from nlp_aug import eda_4
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score,f1_score,classification_report
from utils import SoftCrossEntropy,FocalLoss
# NLP AUG
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
# hyperparameters
# input dir
train_dataset = 'data/sst'
# split name
data_split = '10'

BATCH_SIZE = 32
MODEL_NAME = 'XLM-R'
# need online aug or not
ONLINE = False
# need aug or not
NEED_AUG = True
# n
NUM_AUG = 3
# unused parameter!
MIX_UP = False
AUG_METHOD = 'EPDA'# EDA or CWE or EPDA
# the DA for EPDA
EPDA_ENGINE = 'EDA'
# for EDA
ALPHA = 0.1
train_file_name = 'train_'+str(data_split)+'.txt'
train_dir = train_dataset+'/'+train_file_name
test_dir = train_dataset+'/'+'test.txt'
dev_dir = train_dataset+'/'+'dev.txt'
devtest_dir = train_dataset+'/'+'devtest.txt'
num_classes = 0
LR = 2e-5
# LR = 0.001
min_lr = 1e-5
if 'sst' in train_dir:
    num_classes = 5
    BASIC_EPOCH = 10
    BATCH_SIZE = 64
    # LR = 0.001
    
device = torch.device('cuda')
import torch.distributed as dist

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
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
def reduce_loss_dict(loss_dict):
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim = 0)
        dist.reduce(all_losses, dst = 0)
        if dist.get_rank() == 0:
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses
class Collect_FN():
    def __init__(self, with_label_hard, with_GT_labels = False):
        super(Collect_FN, self).__init__()
        if MODEL_NAME=='Bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',local_files_only=True)
        elif MODEL_NAME=='XLNet':
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased',local_files_only=True)
        elif MODEL_NAME == 'XLM-R':
            self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base',local_files_only=True)
        self.with_label = with_label_hard
        self.with_GT_labels = with_GT_labels

    def __call__(self, batchs):
        # print(batchs)
        if (self.with_label and self.with_GT_labels == False):
            sentences, labels = map(list, zip(*batchs))
        elif (self.with_label == True and self.with_GT_labels == True):
            sentences, labels, GT_labels = map(list, zip(*batchs))
        else:
            sentences = batchs
        encoding = self.tokenizer(sentences, return_tensors = 'pt', padding = True, truncation = True)
        # input_ids = encoding['input_ids']
        # attention_mask = encoding['attention_mask']
        # ans = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if (self.with_label):
            labels = torch.tensor(labels).long()
            encoding['labels'] = labels
        if (self.with_GT_labels):
            GT_labels = torch.tensor(GT_labels).long()
            encoding['GT_labels'] = GT_labels
        encoding['sentences'] = sentences
        return encoding

# the dataset
class EPDADataSet(Dataset):
    def __init__(self,input_dir,max_len=30,num_classes=2):
        self.max_len = max_len
        self.num_classes = num_classes
        self.dir = input_dir
        print("Start to read: ",input_dir, flush = True)
        # pre-load
        lines = open(input_dir,'r').readlines()
        Xs,Ys=[],[]
        count = [0] * num_classes
        for line in lines:
            y,x = line.split('\t')
            y = int(y)

            x = x[:-1]
            # if 'train' in input_dir:
            #     if y==0 or y==2:
            #         continue
            # if count[y] >= int(434*int(data_split)/10*2) and 'train' in input_dir:
            #     continue
            count[y] += 1
            if len(x)<=2:
                continue
            x = self.get_only_chars(x)
            Xs.append(x)
            Ys.append(y)
        weight_per_class = [0.] * num_classes
        N = float(sum(count))                                                   
        for i in range(num_classes):
            weight_per_class[i] = N/float(count[i])                     
        weight = [0] * len(Ys)
        for idx, val in enumerate(Ys):
            weight[idx] = weight_per_class[val]
        self.weights = weight_per_class
        print(weight_per_class,count)
        # os._exit(233)
        # if not 'test' in input_dir:
        #     Xs,Ys = self.upsample_balance(Xs,Ys)
        #     print("Balance dataset Over.")
        self.Xs = Xs
        self.Ys = Ys
        
        self.O_Xs = self.Xs
        self.O_Ys = self.Ys
        print("Load Over, Find: ",len(self.Xs)," datas.", flush = True)
    def get_only_chars(self,line):

        clean_line = ""

        line = line.lower()
        line = line.replace(" 's", " is") 
        line = line.replace("-", " ") #replace hyphens with spaces
        line = line.replace("\t", " ")
        line = line.replace("\n", " ")
        line = line.replace("'", "")

        for char in line:
            if char in 'qwertyuiopasdfghjklzxcvbnm ':
                clean_line += char
            else:
                clean_line += ' '

        clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
        if clean_line[0] == ' ':
            clean_line = clean_line[1:]
        return clean_line

    def __getitem__(self, idx):
        assert idx < len(self.Xs)      
        return self.Xs[idx],self.Ys[idx]
    def __len__(self):
        return len(self.Xs)

    def update(self,Xs,Ys):
        print("Start Update Dataset, Find ",len(self.Xs),'datas.', flush = True)
        # if not 'test' in self.dir:
        #     Xs,Ys = self.upsample_balance(Xs,Ys)
        #     print("Balance dataset Over.")
        self.Xs = Xs
        self.Ys = Ys
        print("Update Dataset Finish, Find ",len(self.Xs),'datas.', flush = True)
        return

    def reset(self):
        if self.O_Xs is not None:
            self.Xs = self.O_Xs
            self.Ys = self.O_Ys

    def upsample_balance(self, sentences, labels):
        sample_number_per_class = [0]*self.num_classes
        for y in labels:
            sample_number_per_class[y] +=1
        sample_number_per_class = np.array(sample_number_per_class)
        max_number = np.max(sample_number_per_class)
        fill_number_each_class = max_number - sample_number_per_class
        # print("??",sample_number_per_class,fill_number_each_class)
        sentence_each_class = [[] for i in range(self.num_classes)]
        for s, l in zip(sentences, labels):
            sentence_each_class[l].append(s)
        for class_index, (sentences_cur_class, fill_num_cur_class) in enumerate(
                zip(sentence_each_class, fill_number_each_class)):
            append_cur_class = []

            for i in range(fill_num_cur_class):
                append_cur_class.append(sentences_cur_class[i % len(sentences_cur_class)])
            sentence_each_class[class_index] = sentences_cur_class + append_cur_class
        ans_sentences = []
        ans_labels = []
        for class_index in range(self.num_classes):
            for s in sentence_each_class[class_index]:
                ans_sentences.append(s)
                ans_labels.append(class_index)
        return ans_sentences, ans_labels

def do_aug(inputs,labels,aug_method,get_embed_fn,model=None,num_aug=1):
    if aug_method == 'EDA':
        aug_fn = eda_4
    elif aug_method == 'EEDA':
        aug_fn = eda
    elif aug_method == 'EPDA':
        aug_fn = epda_bert
    # print(len(inputs),'vs',len(labels))
    Xs,Ys= [],[]
    for i in range(len(inputs)):
        if aug_method == 'EPDA':
            translator = Translator()
            if EPDA_ENGINE == 'CWE':
                # insert, substitute
                nlp_auger = naw.ContextualWordEmbsAug(action='insert',device='cuda')
            elif EPDA_ENGINE == 'BT':
                nlp_auger = naw.BackTranslationAug (device='cuda')
            else:
                nlp_auger = None
            augedtxts,_ = aug_fn(txt=inputs[i],label=labels[i],num_aug=NUM_AUG,model=model,translator=translator,
                engine=EPDA_ENGINE,alpha=ALPHA,mix_up=MIX_UP,get_embed_fn=get_embed_fn,loss_fn=nn.CrossEntropyLoss(),
                nlp_auger = nlp_auger,alpha_epda=0.5)
            Xs+=augedtxts
            # 再填一堆同样的
            for j in range(len(augedtxts)):
                Ys.append(labels[i])
        else:
            txts = aug_fn(inputs[i],num_aug=NUM_AUG)
            for txt in txts:
                embed = get_embed_fn(txt)
                # print("Size",embed.size())
                Xs.append(embed)
                label_tensor = torch.zeros(1)
                label_tensor[0] = labels[i]
                label_tensor = label_tensor.long()
                Ys.append(label_tensor)

    return Xs,Ys

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def get_model():
    if MODEL_NAME == 'Bert':
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
            num_labels = num_classes,
            gradient_checkpointing = True,
            ).cuda()
    elif MODEL_NAME == 'XLNet':
        model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased',
        num_labels = num_classes,
            ).cuda()
    elif MODEL_NAME == 'XLM-R':
        model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base',
        num_labels = num_classes,
            ).cuda()
    return model
def train(train_data_loader,test_data_loader,dev_data_loader,devtest_data_loader,model):
    EPOCHES = BASIC_EPOCH
    if NEED_AUG:
        EPOCHES *= 2
    max_acc = 0.0
    max_dev, max_devtest = 0.0,0.0
    final_score = 0.0
    trained_iter = 0
    UPDATED = False
    if NEED_AUG == False:
        UPDATED = True

    # T_max = EPOCHES * (len(train_dataset)//BATCH_SIZE+1)
    # 定义优化器和调度器
    # optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8, weight_decay=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # scheduler = transformers.get_linear_schedule_with_warmup(optimizer,EPOCHES,T_max)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'max', factor=0.2, min_lr=min_lr
    # )
    # reset the dataset.
    train_data_loader.dataset.reset()
    # print("WEIGHTS,",train_data_loader.dataset.weights)
    # ,alpha=torch.Tensor(train_data_loader.dataset.weights)
    # loss_fn = FocalLoss(class_num=num_classes)
    # loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(train_data_loader.dataset.weights).cuda())
    loss_fn = nn.CrossEntropyLoss()
    print("Pretraining To Generate Samples")
    for epoch in range(EPOCHES):
        if epoch == int(EPOCHES//4*3):
            for param_group in optimizer.param_groups:
                param_group['lr'] = min_lr/2
        model.train()
        if (NEED_AUG and ONLINE == False and epoch == BASIC_EPOCH) or (NEED_AUG and ONLINE and epoch % 5==0 and epoch>=BASIC_EPOCH):
            print("Start to update Dataset")
            # model = torch.load("./release/"+train_dataset.split('/')[-1]+".pth").cuda()
            input_dir = train_data_loader.dataset.dir
            lines = open(input_dir,'r').readlines()
            Xs,Ys=[],[]
            count = [0]*num_classes
            for line in lines:
                y,x = line.split('\t')
                y = int(y)
                # if count[y] >= int(434*int(data_split)*2):
                #     continue
                count[y] += 1

                x = x[:-1]
                x = train_data_loader.dataset.get_only_chars(x)
                Xs.append(x)
                Ys.append(y)
            inputs,label = do_aug(Xs,Ys,AUG_METHOD,get_embed_fn=XLMRobertaTokenizer.from_pretrained('xlm-roberta-base'),
                model=model,num_aug=NUM_AUG)
            # print("??",inputs,"vs",label)
            # os._exit(233)
            print('Before',len(train_data_loader))
            train_data_loader.dataset.update(inputs,label)
            print("< Update Done.")
            print('After',len(train_data_loader))
            UPDATED = True
            for param_group in optimizer.param_groups:
                param_group['lr'] = min_lr
            model.train()
            # finish the rest
            # if not ONLINE:
            #     max_acc = 0.0
            #     model = get_model()
            #     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #         optimizer, 'max', factor=0.2, min_lr=min_lr
            #     )
                
        for i,batch in enumerate(train_data_loader):
            # print(batch)
            batch = move_to_device(batch)
            optimizer.zero_grad()
            # print(batch)
            # input_ids: [16,128]   label_id:[16]
            # print(batch['sentences'])
            # os._exit(233)
            del batch['sentences']
            # print(batch)
            # os._exit(233)
            output = model(**batch)
            # loss = output.loss
            loss = loss_fn(output.logits,batch['labels'])
            # loss_dict_reduced = reduce_loss_dict({'loss_all': loss})
            # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            # meters.update(loss = losses_reduced, **loss_dict_reduced)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            trained_iter += 1
            if trained_iter % (len(test_data_loader)//5) == 0:
                print('Start Dev.')
                # start eval
                model.eval()
                pred_y,gt_y=[],[]
                for i,batch in enumerate(devtest_data_loader):
                    label = batch['labels']
                    del batch['sentences']
                    batch = move_to_device(batch)
                    outputs = model(**batch).logits
                    b,_ = outputs.size()
                    outputs = torch.softmax(outputs,1)
                    # confidence_mat = torch.ones(outputs.size())
                    
                    outputs = torch.argmax(outputs,1).detach().cpu()
                    for j in range(b):
                        pred_y.append(outputs[j])
                        gt_y.append(label[j])
                score_devtest = accuracy_score(gt_y, pred_y)
                pred_y,gt_y=[],[]
                for i,batch in enumerate(dev_data_loader):
                    label = batch['labels']
                    del batch['sentences']
                    batch = move_to_device(batch)
                    outputs = model(**batch).logits
                    b,_ = outputs.size()
                    outputs = torch.softmax(outputs,1)
                    # confidence_mat = torch.ones(outputs.size())
                    
                    outputs = torch.argmax(outputs,1).detach().cpu()
                    for j in range(b):
                        pred_y.append(outputs[j])
                        gt_y.append(label[j])
                score_dev = accuracy_score(gt_y, pred_y)
                score = score_devtest
                # scheduler.step(score)
                print(optimizer.state_dict()['param_groups'][0]['lr'])
                # if UPDATED:
                print("--Current Dev",score_dev,"Devtest",score_devtest,'Max Dev',max_dev,'Max DevTest',max_devtest,'Epoch',epoch, flush = True)
                    # print("Report",classification_report(gt_y,pred_y))
                pred_y,gt_y=[],[]
                for i,batch in enumerate(test_data_loader):

                    label = batch['labels']
                    del batch['sentences']
                    batch = move_to_device(batch)
                    outputs = model(**batch).logits
                    b,_ = outputs.size()
                    outputs = torch.softmax(outputs,1)
                    
                    outputs = torch.argmax(outputs,1).detach().cpu()
                    for j in range(b):
                        pred_y.append(outputs[j])
                        gt_y.append(label[j])
                        
                test_score = accuracy_score(gt_y, pred_y)
                print("Test Acc,",test_score)
                if (score > max_acc) or (abs(score-max_acc)<0.001 and score_dev>max_dev):
                    max_acc = score
                    max_dev = score_dev
                    max_devtest = score_devtest
                    final_score = test_score
                    torch.save(model,"./release/"+train_dataset.split('/')[-1]+".pth")
    # model = torch.load("./release/"+train_dataset.split('/')[-1]+".pth").cuda()
    # model.eval()
        # os._exit(233)
    return final_score
def compute_model(train_dir,test_dir):
    acc_scores = []
    train_dataset = EPDADataSet(train_dir,num_classes=num_classes)
    test_dataset = EPDADataSet(test_dir,num_classes=num_classes)
    dev_dataset = EPDADataSet(dev_dir,num_classes=num_classes)
    devtest_dataset = EPDADataSet(devtest_dir,num_classes=num_classes)
    collate_fn = Collect_FN(True)
    # train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_dataset.weights, BATCH_SIZE)
    # print(max(train_dataset.weights),min(train_dataset.weights))
    train_data_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,collate_fn=collate_fn,shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=False,collate_fn=collate_fn)
    dev_data_loader = DataLoader(dataset=dev_dataset,batch_size=64,shuffle=False,collate_fn=collate_fn)
    devtest_data_loader = DataLoader(dataset=devtest_dataset,batch_size=64,shuffle=False,collate_fn=collate_fn)
    # Test them for 5 times
    for i in range(1):
        setup_seed(i)
        model = get_model()
        acc = train(train_data_loader,test_data_loader,dev_data_loader,devtest_data_loader,model)
        acc_scores.append(acc)
        print("[IMPORTANT] i=",i,"Current Acc",acc,"Average Acc Score: ",sum(acc_scores)/len(acc_scores), flush = True)
    print("> Done.", flush = True)
if __name__ == "__main__":
    compute_model(train_dir,test_dir)
    
