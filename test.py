PATH = './weights/trained.pth'

x_test = np.load('./dataset/x_val.npy').astype(np.float32)  # (288, 26, 34, 1)
y_test = np.load('./dataset/y_val.npy').astype(np.float32)  # (288, 1)

test_transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = eyes_dataset(x_test, y_test, transform=test_transform)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

model = Net()
model.to('cuda')
model.load_state_dict(torch.load(PATH))
model.eval()

count = 0

with torch.no_grad():
    total_acc = 0.0
    acc = 0.0
    for i, test_data in enumerate(test_dataloader, 0):
        data, labels = test_data[0].to('cuda'), test_data[1].to('cuda')

        data = data.transpose(1, 3).transpose(2, 3)

        outputs = model(data)

        acc = accuracy(outputs, labels)
        total_acc += acc

        count = i

    print('avarage acc: %.5f' % (total_acc/count),'%')

print('test finish!')
