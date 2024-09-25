import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import numpy as np 
from tqdm import tqdm
import pickle
import pandas as pd

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    # transforms.ColorJitter(hue=.01, saturation=.01, contrast=.01),
                                    transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR),
                                    transforms.GaussianBlur(3, sigma=(0.1, 0.5)),  # Smaller kernel for blur
                                    # normalize
                                ])

def map_and_clamp_intensity(img, factor = 1, percentage = 0.01):
    # imadjust refer to 
    # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html (opencv)
    # https://www.sciencedirect.com/science/article/pii/S0893608012000524#f000015 (mcdnn)
    # & GPT
    C, H, W = img.shape
    reshaped_img = img.permute(1, 2, 0).view(-1, C)
    reshaped_img = factor * reshaped_img
    lower_percentile, upper_percentile = torch.quantile(reshaped_img, q = percentage, dim = 0), torch.quantile(reshaped_img, q = 1-percentage, dim = 0) 
    clamped_img = torch.clamp(reshaped_img, lower_percentile, upper_percentile)
    rescaled_img = (clamped_img - lower_percentile) /(upper_percentile - lower_percentile)

    transformed_img = rescaled_img.view(H, W, C).permute(2, 0, 1)
    return transformed_img
    

class MyDataset(Dataset):

    def __init__(self, X_path="X.pt", y_path="y.pt", transform = train_transforms):

        self.X = torch.load(X_path).squeeze(1)
        self.y = torch.load(y_path).squeeze(1)
        self.transform = transform

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if self.transform:
            # self.X[idx] = self.transform(self.X[idx])
            self.X[idx] = map_and_clamp_intensity(img = self.X[idx], factor = 1.2)
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx]
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, kernel_size=4)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(150)
        self.conv2_drop = nn.Dropout2d(0.2)
        
        self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(250)
        self.conv3_drop = nn.Dropout2d(0.2)
        
        self.fc1 = nn.Linear(250*4, 200)
        self.fc2 = nn.Linear(200, nclasses, )

    def forward(self, x):
        x = F.gelu(F.max_pool2d(F.gelu(self.bn1(self.conv1(x))), 2))
        x = F.gelu(F.max_pool2d(self.conv2_drop(F.gelu(self.bn2(self.conv2(x)))), 2))
        x = F.gelu(F.max_pool2d(self.conv3_drop(F.gelu(self.bn3(self.conv3(x)))), 2))
        
        x = x.view(-1, 250*4)
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(comb_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(comb_loader.dataset),
                100. * batch_idx / len(comb_loader), loss.item()))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        validation_loss += F.nll_loss(output, target, reduction="sum").item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    acc = correct/len(val_loader.dataset)
    return validation_loss, acc


def eval_on_kaggle(model, outfile = '.csv'):
    output_file = open(outfile, "w")
    dataframe_dict = {"Filename" : [], "ClassId": []}
    
    test_data = torch.load('testing/test.pt')
    file_ids = pickle.load(open('testing/file_ids.pkl', 'rb'))
    model.eval() # Don't forget to put your model on eval mode !
    
    for i, data in enumerate(test_data):
        data = data.unsqueeze(0)
        data = data.to(device)
    
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1].item()
        file_id = file_ids[i][0:5]
        dataframe_dict['Filename'].append(file_id)
        dataframe_dict['ClassId'].append(pred)
    
    df = pd.DataFrame(data=dataframe_dict)
    df.to_csv(outfile, index=False)
    print("Written to csv file {}".format(outfile))

if __name__ == '__main__':
    batch_size = 32
    momentum = 0.9
    lr = 0.01
    epochs = 10
    log_interval = 100
    nclasses = 43 # GTSRB has 43 classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_dataset = MyDataset(X_path="train/X.pt", y_path="train/y.pt", transform = None)
    train_dataset = torch.utils.data.Subset(train_dataset, indices= np.random.permutation(len(train_dataset))[:len(train_dataset)//5])
    
    aug_dataset = MyDataset(X_path="train/X.pt", y_path="train/y.pt", transform = train_transforms)
    # aug_dataset = torch.utils.data.Subset(aug_dataset, indices= np.random.permutation(len(train_dataset))[:len(train_dataset)//5])
    
    comb_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_dataset])
    val_dataset = MyDataset(X_path="validation/X.pt", y_path="validation/y.pt")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    comb_loader = torch.utils.data.DataLoader(
        comb_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    aug_loader = torch.utils.data.DataLoader(
        aug_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print('loaded datasets...')
    
    model = Net()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', patience=3)
    
    pbar = tqdm(range(1, epochs + 1))
    best_model = {'acc': 0, 'epoch': 1}
    prefix = 'mcdnn_acc_433_200_'
    for epoch in pbar:
        train(epoch)
        validation_loss, val_acc = validation()
        lr_scheduler.step(val_acc)
        model_file = prefix + 'model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('\nSaved model to ' + model_file + '.')
        if val_acc > best_model.get('acc'):
            best_model['acc'], best_model['epoch'] = val_acc, epoch

    best_cp = prefix + f'model_{best_model['epoch']}.pth'
    with open('best_cp.p', 'a') as f:
        f.write(f'\nmodel: {best_cp}\tepoch: {best_model['epoch']}\tacc: {best_model['acc']}\n')

    with open('best_cp.p', 'r') as f:
        lines = f.readlines()

    best_cp = lines[-1].split('\t')[0].split(':')[-1].strip()
    model.load_state_dict(torch.load(best_cp))
    model.to(device)
    eval_on_kaggle(model, outfile = best_cp + '.csv')    
