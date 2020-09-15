import torch 
import torch.nn as nn
import torchvision.models as models 

class EncoderCNN(nn.Module):
    def __init__(self, embedSize, trainCNN = False):
        super(EncoderCNN,self).__init__()
        self.trainCNN = trainCNN 
        self.inception = models.inception_v3(pretrained=True,aux_logits = False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features,embedSize)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self,images):
        features = self.inception(images)
        return self.dropout(self.relu(features))

class DecoderRNN(nn.Module):
    def __init__(self,embedSize, hiddenSize, vocabSize, numLayers):
        super(DecoderRNN,self).__init__()
        self.embed = nn.Embedding(vocabSize, embedSize)
        self.lstm = nn.LSTM(embedSize,hiddenSize,numLayers)
        self.linear = nn.Linear(hiddenSize, vocabSize)
        self.dropout = nn.Dropout(0.5)

    def forward(self,features,captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0),embeddings),dim = 0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs 

class CNNtoRNN(nn.Module):
    def __init__(self,embedSize, hiddenSize, vocabSize, numLayers):
        super(CNNtoRNN,self).__init__()
        self.encoderCNN = EncoderCNN(embedSize)
        self.decoderRNN = DecoderRNN(embedSize, hiddenSize, vocabSize, numLayers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)

        return outputs 

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
        print("MODELCAP::",result_caption)

        return [vocabulary.itos[idx] for idx in result_caption]