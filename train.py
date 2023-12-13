import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from data_loader import eyes_dataset
from model import Net
import torch.optim as optim

# 코드 작성 전 학습, 데이터 시각화 위한 라이브러리들과 이전에 작성한 model, data처리 클래서 import

x_train = np.load('./dataset/x_train.npy').astype(np.float32)  # (2586, 26, 34, 1)
y_train = np.load('./dataset/y_train.npy').astype(np.float32)  # (2586, 1)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
])

train_dataset = eyes_dataset(x_train, y_train, transform=train_transform)

#data loader
plt.style.use ('dark_background')
fig = plt. figure()

for i in range(len(train_dataset)):
  x, y = train_dataset[i]

    plt.subplot(2, 1, 1)
    plt.title(str(y_train[i]))
    plt.imshow(x_train[i].reshape((26, 34)), cmap='gray')

    plt.show()
#matpltlib로 데이터 제대로 불러왔는지 확인(라벨링 확인)
#눈을 뜸- 1 , 눈을 감음 -0

PATH = 'weights/trained.pth'
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

model = Net()
model.to('cuda')

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 50
#trained parameter를 저장할 path 를 설정
#train, val data loader 를 작성

for epoch in range(epochs):
    running_loss = 0.0
    running_acc = 0.0

    model.train()

    for i, data in enumerate(train_dataloader, 0):
        input_1, labels = data[0].to('cuda'), data[1].to('cuda')

        input = input_1.transpose(1, 3).transpose(2, 3)

        optimizer.zero_grad()

        outputs = model(input)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)

        if i % 80 == 79:
            print('epoch: [%d/%d] train_loss: %.5f train_acc: %.5f' % (
                epoch + 1, epochs, running_loss / 80, running_acc / 80))
            running_loss = 0.0

print("learning finish")
torch.save(model.state_dict(), PATH)
#train 전 moodel.train()으로 학습 모드를 선언해주기
#index 0 은 이미지이고 1은 라벨 --> 이를 받을 때 to('cuda')로 받아야 gpu 사용 가능
#transpose를 통해 형상 변화시킨 후 model에 넣기



def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc
#accuracy 함수
#sigmoid로 확률화 시키기, round로 0또는 1로 변환시키기 -0.5이하는 0, 초과는 1






