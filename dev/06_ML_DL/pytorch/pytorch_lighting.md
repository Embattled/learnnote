# Pytorch Lighting

PyTorch Lightning is the deep learning framework with “batteries included” for professional AI researchers and machine learning engineers who need maximal flexibility while super-charging performance at scale.

是一种用于快速实现 idea 的网络框架, 旨在通过最少的代码实现网络的各种调参  

样例代码
```py
import lightning as L

# define the LightningModule, 定义 lighting 格式的网络
class LitAutoEncoder(L.LightningModule):
    # 实现几个内置函数
    def __init__(self, encoder, decoder):
        super().__init__()
        # 定义网络层

    # 定义训练过程
    def training_step(self, batch, batch_idx):

    # 配置优化器 
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# 实例化
autoencoder = LitAutoEncoder(encoder, decoder)

# 定义 Pytorch DataLoader
...

# 训练模型
# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)


```