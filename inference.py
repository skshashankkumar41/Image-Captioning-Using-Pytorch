import torch 
import torchvision.transforms as transforms
from dataset import FlickerDataset
from model import EncoderCNN,DecoderRNN,CNNtoRNN
import pickle
from PIL import Image


def inference(imagePath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('G:/AIP/Image-Captioning-Using-Pytorch/output/cap_itos' +'.pkl', 'rb') as f:
        itos = pickle.load(f)
        
    embedSize = 256
    hiddenSize = 256
    vocabSize = len(itos)
    numLayers = 1

    model = CNNtoRNN(embedSize = embedSize, hiddenSize = hiddenSize, vocabSize = vocabSize, numLayers =  numLayers, inference= True).to(device)
    checkpoint = torch.jit.load("G:/AIP/Image-Captioning-Using-Pytorch/output/my_checkpoint.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])
    
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    
    )
    image = transform(Image.open(imagePath).convert("RGB")).unsqueeze(0)

    output = model.caption_image(image.to(device), itos)

    print(output)
    return output


inference("test.jpg")  
    
    
     

    