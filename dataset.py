import torch 
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import pandas as pd 
import spacy 
from torch.nn.utils.rnn import pad_sequence
import os 
import torchvision.transforms as transforms
import pickle 

spacy_eng = spacy.load('en_core_web_sm')

class Vocabulary:
    def __init__(self, freqThresold):
        self.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.freqThresold = freqThresold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self,sentenceList):
        freq = {}
        idx = 4

        for sent in sentenceList:
            for word in sent:
                if word not in freq:
                    freq[word] = 1
                else:
                    freq[word] += 1

                if freq[word] == self.freqThresold:
                    self.itos[idx] = word
                    self.stoi[word] = idx 
                    idx += 1

    def encode(self,text):
        tokenizedText = self.tokenizer(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in tokenizedText
        ]

    def storeVocab(self):
        print("INSIDE STOREING")
        with open('G:/AIP/Image-Captioning/vocab/itos' + '.pkl', 'wb') as f:
            pickle.dump(self.itos, f, pickle.HIGHEST_PROTOCOL)

        with open('G:/AIP/Image-Captioning/vocab/stoi' + '.pkl', 'wb') as f:
            pickle.dump(self.stoi, f, pickle.HIGHEST_PROTOCOL)


class FlickerDataset(Dataset):
    def __init__(self, imagePath, captionFile, transform = None, freqThresold = 5):
        self.imagePath = imagePath
        self.df = pd.read_csv(captionFile, sep = '\t')
        #print(self.df.columns)
        self.transform = transform 

        self.imgs = self.df['image']
        self.captions = self.df['caption']

        self.vocab = Vocabulary(freqThresold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(self.imagePath + img_id).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        encoded_caption = [self.vocab.stoi["<SOS>"]] 
        encoded_caption += self.vocab.encode(caption)
        encoded_caption.append(self.vocab.stoi["<EOS>"])

        #print("CAPTION::",caption)
        #print("ENCODED::",encoded_caption)

        return img, torch.tensor(encoded_caption)


class MyCollate:
    def __init__(self,padIdx):
        self.padIdx = padIdx

    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]       
        imgs = torch.cat(imgs,dim = 0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value= self.padIdx)

        return imgs,targets 

class FlickerDataset(Dataset):
    def __init__(self, imagePath, captionFile, transform = None, freqThresold = 5):
        self.imagePath = imagePath
        self.df = pd.read_csv(captionFile, sep = '\t')
        #print(self.df.columns)
        self.transform = transform 

        self.imgs = self.df['image']
        self.captions = self.df['caption']

        self.vocab = Vocabulary(freqThresold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(self.imagePath + img_id).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        encoded_caption = [self.vocab.stoi["<SOS>"]] 
        encoded_caption += self.vocab.encode(caption)
        encoded_caption.append(self.vocab.stoi["<EOS>"])

        #print("CAPTION::",caption)
        #print("ENCODED::",encoded_caption)

        return img, torch.tensor(encoded_caption)


class MyCollate:
    def __init__(self,padIdx):
        self.padIdx = padIdx

    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]       
        imgs = torch.cat(imgs,dim = 0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value= self.padIdx)

        return imgs,targets 


# def get_loader(imagePath, captionPath, transform, batch_size = 32,shuffle = True):
#     dataset = FlickerDataset(imagePath,captionPath, transform = transform)
#     dataset.vocab.storeVocab()
#     padIdx = dataset.vocab.stoi['<PAD>']
#     loader = DataLoader(
#         dataset = dataset,
#         batch_size= batch_size,
#         shuffle = shuffle, 
#         collate_fn = MyCollate(padIdx = padIdx )
#     )

#     return loader,dataset 


# transforms = transforms.Compose(
#     [
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#     ]
# )
# dataloader = get_loader(
#     imagePath = "C:/Users/SKS/Desktop/AAIC/Image_Captioning/Flicker8k_Dataset/", 
#     captionPath = 'C:/Users/SKS/Desktop/AAIC/Image_Captioning/image_train_dataset.tsv',
#     transform = transforms 
#     )

# for idx, (imgs, captions) in enumerate(dataloader):
#     print(imgs.shape)
#     print(captions.shape)
#     if idx == 1:
#         break



