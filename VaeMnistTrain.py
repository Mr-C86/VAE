import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import os
from MnistNets import Net
from torchvision.utils import save_image
import matplotlib.pyplot as plt

SAVE_PATH = 'model'
SAVE_MODEL = 'model/vae_mnist.pkl'
EPOCH = 50
BATCHSIZE = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
net = Net().to(device)
if os.path.exists(SAVE_MODEL):
    net.load_state_dict(torch.load(SAVE_MODEL))
    print('加载模型ing')
lossfn = nn.MSELoss(reduction='sum')
opt = torch.optim.Adam(net.parameters(), )
# scheduler = lr_scheduler.StepLR(opt, 20, 0.1)
s = SummaryWriter()

dataset = torchvision.datasets.MNIST(
    root='datasets',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
train_data = data.DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True)

for epoch in range(EPOCH):
    for i, (x, _) in enumerate(train_data):
        c = torch.randn(128)
        out, kl_loss = net(x.to(device), c.to(device))
        mse_loss = lossfn(out, x.to(device))
        loss = kl_loss + mse_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        # scheduler.step()
        if i % 10 == 0:
            print('epoch: {} | i/len : {}/{} | kl_loss: {} | mse_loss: {}'.format(epoch, i, len(train_data),
                                                                                  kl_loss.item(), mse_loss.item()))
            save_image(out.cpu(), 'pic/out.png', nrow=10, normalize=True,
                       scale_each=True)  # 每一行多少个，是否对每一张图片做归一化，
            save_image(x, 'pic/input.png', nrow=10, normalize=True, scale_each=True)
    torch.save(net.state_dict(), SAVE_MODEL)
    s.add_histogram('w', net.encoder.layer[0].weight.data, global_step=epoch)
    s.add_scalar('kl_loss', kl_loss, global_step=epoch)
    s.add_scalar('mse_loss', mse_loss, global_step=epoch)
    for j in range(1, 11):
        img = torchvision.transforms.ToPILImage()(out.cpu().reshape(-1, 1, 28, 28)[int('{}'.format(j))])
        plt.subplot(2, 10, int('{}'.format(j)))
        plt.imshow(img)
        plt.title('output')
        imgs = torchvision.transforms.ToPILImage()(x[int('{}'.format(j))])
        plt.subplot(2, 10, int('{}'.format(j + 10)))
        plt.imshow(imgs)
        plt.title('input')
        plt.savefig('pic/compare.jpg')
