import torch 
import torch.nn as nn
import torch.optim as optim 
import torchvision.transforms as transforms 
from dataloader import get_loader
from model import CNNtoRNN
from utils import save_checkpoint,load_checkpoint

def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356,356)),
            transforms.RandomCrop((299,299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    trainLoader, dataset = get_loader(
        imagePath = "C:/Users/SKS/Desktop/AAIC/Image_Captioning/Flicker8k_Dataset/", 
        captionPath = 'C:/Users/SKS/Desktop/AAIC/Image_Captioning/image_train_dataset.tsv',
        transform = transform 
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = False
    train_CNN = False

    embedSize = 256
    hiddenSize = 256
    vocabSize = len(dataset.vocab)
    numLayers = 1
    learning_rate = 3e-4
    num_epochs = 100
    step = 0

    model = CNNtoRNN(embedSize, hiddenSize, vocabSize, numLayers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index= dataset.vocab.stoi['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()
    for epoch in range(num_epochs):
        print("EPOCH::",epoch)
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs,captions) in enumerate(trainLoader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs,captions[:-1])

            loss = criterion(outputs.reshape(-1,outputs.shape[2]),captions.reshape(-1))

            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

if __name__ == "__main__":
    train()