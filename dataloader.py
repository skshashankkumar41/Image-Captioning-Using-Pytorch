from dataset import FlickerDataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader



class MyCollate:
    def __init__(self,padIdx):
        self.padIdx = padIdx

    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]       
        imgs = torch.cat(imgs,dim = 0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value= self.padIdx)

        return imgs,targets 


def get_loader(imagePath, captionPath, transform, batch_size = 16,shuffle = True):
    dataset = FlickerDataset(imagePath,captionPath, transform = transform)
    dataset.vocab.storeVocab()
    padIdx = dataset.vocab.stoi['<PAD>']
    loader = DataLoader(
        dataset = dataset,
        batch_size= batch_size,
        shuffle = shuffle, 
        collate_fn = MyCollate(padIdx = padIdx )
    )

    return loader,dataset


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
