import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# data lol
train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = DataLoader(train, batch_size=1, shuffle=True)
testset = DataLoader(test, batch_size=1, shuffle=True)

# The network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # linear is full connected straight thing
        self.fc1 = nn.Linear((28*28), 64) # image is 28 x 28
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10) # 10 is the ten neurons because we only have 10 digits

    def forward(self, x):
        # data flows through the layers and we return the output of the last layer (fc4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1) # dim means dimention

def train():
    net = Net() # create network
    print(net, end="\n\n")

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    EPOCHS = 69 # hehe train 69 time lmao i know im so funny
    for epoch in range(EPOCHS):
        for data in trainset:
            X, y = data
            net.zero_grad()
            output = net(X.view(-1, 28*28))
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print(epoch+1, loss)

    # save model 
    torch.save(net, "./model.pt")

def test():
    total = 0
    correct = 0

    model = torch.load("./model.pt")
    model.eval()

    for data in testset:
        X, y = data

        prediction = torch.argmax(model(X[0].view(-1, 28*28))[0]).tolist()
        answer = y[0].tolist()
        
        if prediction == answer:
            correct +=1 

        print("Prediction:", prediction, "Actually:", answer)
        
        total +=  1
        return
    
    print(f"\nAmount correct: {round((correct/total)*100, 2)}% ({correct}/{total})")


def test_real_image(image_name, answer):
    img = Image.open(image_name).convert('RGB')
    resize = transforms.Resize([28, 28])
    img = resize(img)
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(img)
    tensor = tensor.unsqueeze(0)


    model = torch.load("./model.pt")
    model.eval()

    prediction = torch.argmax(model(tensor.view(-1, 28*28))[0]).tolist()

    print("Prediction:", prediction, "Actually:", answer)

# train() # Trains the model
# test() # Tests the model with the dataset 
test_real_image("seven.jpeg", answer=7) # tests it with an actual image