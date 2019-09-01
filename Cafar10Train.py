import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import os
from Cafar10Net import Net
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

SAVE_PATH = 'model'
SAVE_MODEL = 'model/vae_cafar10_singel.pkl'
EPOCH = 50
BATCHSIZE = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
net = Net().to(device)
if os.path.exists(SAVE_MODEL):
    net.load_state_dict(torch.load(SAVE_MODEL))
    print('加载模型ing')

lossfn = nn.MSELoss(reduction='mean')
opt = torch.optim.Adam(net.parameters(), lr=0.01)
# scheduler = lr_scheduler.StepLR(opt, 20, 0.1)
s = SummaryWriter()

dataset = torchvision.datasets.CIFAR10(
    root='cafadata',
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(1),#将三通道的图片转换为1个通道的，一般都是先转换之后在进行转张量，不然会报错
        torchvision.transforms.ToTensor()
    ])

)
train_data = data.DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True)

for epoch in range(EPOCH):
    for i, (x, _) in enumerate(train_data):
        # print(x.size())
        out, kl_loss = net(x.to(device))
        mse_loss = lossfn(out, x.to(device))
        loss = kl_loss + mse_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        # scheduler.step()
        if i % 10 == 0:
            print('epoch: {} | i/len : {}/{} | kl_loss: {} | mse_loss: {}'.format(epoch, i, len(train_data),
                                                                                  kl_loss.item(), mse_loss.item()))
            # out = out.cpu().permute(0, 2, 3, 1)[0].detach().numpy()
            # print(out.shape)
            # img = Image.fromarray(out, 'RGB')
            # img.show()
            # print(x.size())
            # imgs = torchvision.transforms.ToPILImage()(x[0])
            # imgs.show()
            # img = torchvision.transforms.ToPILImage()(out.cpu()[0])
            # img.show()
            # print("x.shape", x.shape)
            # x = x.permute([0, 2, 3, 1])[0].detach() * 255
            # print('x', x.shape)
            # x = np.array(x, dtype=np.uint8)
            # # print(x)
            # x = Image.fromarray(x, 'RGB', )
            # # print(x)
            # x.show()
            #
            # print("out；", out.shape)
            # out = out.cpu().permute([0, 2, 3, 1])[0].detach() * 255
            # out = np.array(out, dtype=np.uint8)
            # print('1', out.shape)
            # out = Image.fromarray(out, 'RGB', )
            # print(out)
            # out.show()


            save_image(out.cpu(), 'pic/out_cafar10.jpg', nrow=10, normalize=True,
                       scale_each=True)  # 每一行多少个，是否对每一张图片做归一化，
            save_image(x, 'pic/input_cafar10.jpg', nrow=10, normalize=True, scale_each=True)

    torch.save(net.state_dict(), SAVE_MODEL)
    s.add_histogram('w', net.encoder.layer[0].weight.data, global_step=epoch)
    s.add_scalar('kl_loss', kl_loss, global_step=epoch)
    s.add_scalar('mse_loss', mse_loss, global_step=epoch)
    for j in range(1, 11):
        img = torchvision.transforms.ToPILImage()(out.cpu()[int('{}'.format(j))])
        plt.subplot(2, 10, int('{}'.format(j)))
        plt.imshow(img)
        plt.title('out')
        imgs = torchvision.transforms.ToPILImage()(x[int('{}'.format(j))])
        plt.subplot(2, 10, int('{}'.format(j + 10)))
        plt.imshow(imgs)
        plt.title('input')
        plt.savefig('pic/compare_cafar10.jpg')
