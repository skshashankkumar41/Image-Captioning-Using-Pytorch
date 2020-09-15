import torch 
import torch.nn as nn
import torch.optim as optim 
import torchvision.transforms as transforms 
from dataloader import get_loader
from model import CNNtoRNN
from utils import save_checkpoint,load_checkpoint,print_examples

def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356,356)),
            transforms.RandomCrop((299,299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    trainLoader, valLoader, testLoader, trainDataset = get_loader(
        imagePath = "C:/Users/SKS/Desktop/AAIC/Image_Captioning/Flicker8k_Dataset/", 
        root_path = 'C:/Users/SKS/Desktop/AAIC/Image_Captioning/image_train_dataset.tsv',
        transform = transform 
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = False
    train_CNN = False

    embedSize = 256
    hiddenSize = 256
    vocabSize = len(trainDataset.vocab)
    numLayers = 1
    learning_rate = 3e-4
    num_epochs = 100
    step = 0

    model = CNNtoRNN(embedSize, hiddenSize, vocabSize, numLayers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index= trainDataset.vocab.stoi['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        print("---------------------------------")
        print("EPOCH::",epoch)
        print("---------------------------------")
        print_examples(model, device, dataset)
        #print_examples(model, device, dataset)
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs,captions) in tqdm(enumerate(trainLoader), total = len(trainLoader), leave = False):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs,captions[:-1])

            loss = criterion(outputs.reshape(-1,outputs.shape[2]),captions.reshape(-1))
            if iter % 500 == 0:
                print("---------------------------------")
                print("LOSS::",loss)
                print("---------------------------------")
                #print_examples(model, device, dataset)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
            iter += 1

if __name__ == "__main__":
    train()