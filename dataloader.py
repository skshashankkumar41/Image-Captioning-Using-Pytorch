from dataset import FlickerDataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class MyCollate:
    def __init__(self,padIdx):
        self.padIdx = padIdx

    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]       
        imgs = torch.cat(imgs,dim = 0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value= self.padIdx)

        return imgs,targets 


def get_loader(imagePath, root_path, transform, batch_size = 16,shuffle = True, num_workers = 2,pin_memory = True):
    trainDataset = FlickerDataset(imagePath, root_path + 'train_df.csv',transform=transform)
    valDataset = FlickerDataset(imagePath, root_path + 'validate_df.csv',transform=transform)
    testDataset = FlickerDataset(imagePath, root_path + 'test_df.csv', transform=transform)
    
    trainDataset.vocab.storeVocab('cap')
    
    padIdx = trainDataset.vocab.stoi['<PAD>']
    
    trainLoader = DataLoader(
        dataset =trainDataset,
        batch_size= batch_size,
        shuffle = shuffle, 
        pin_memory = pin_memory,
        collate_fn = MyCollate(padIdx = padIdx )
    )

    valLoader = DataLoader(
        dataset = valDataset,
        batch_size= batch_size,
        shuffle = shuffle, 
        pin_memory = pin_memory,
        collate_fn = MyCollate(padIdx = padIdx )
    )

    testLoader = DataLoader(
        dataset = testDataset,
        batch_size= batch_size,
        shuffle = shuffle, 
        pin_memory = pin_memory,
        collate_fn = MyCollate(padIdx = padIdx )
    )

    return trainLoader, valLoader, testLoader, trainDataset


# transforms = transforms.Compose(
#     [
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#     ]
# )
# trainLoader, valLoader, testLoader, trainDataset = get_loader(
#     imagePath = "C:/Users/SKS/Desktop/AAIC/Image_Captioning/Flicker8k_Dataset/", 
#     root_path = 'input/',
#     transform = transforms 
#     )

# for idx, (imgs, captions) in enumerate(trainLoader):
#     print("IDX",idx)
#     print(imgs.shape)
#     print(captions.shape)
#     if idx == 1:
#         break
