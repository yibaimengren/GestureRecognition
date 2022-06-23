import time
import torch
import math
from torch.utils.tensorboard import SummaryWriter

#获取当前时间与给定参数时间的时间差
def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{:02d}m {:02d}s'.format(int(m), int(s))

#获取准确率
def get_accuracy(model, x, y_ref):
    """Get the accuracy of the pytorch model on a batch"""
    acc = 0.
    model.eval()
    with torch.no_grad():
        predicted = model(x)
        _, predicted = predicted.max(dim=1)
        acc = 1.0 * (predicted == y_ref).sum().item() / y_ref.shape[0]
    return acc

def train(model, criterion, optimizer, dataloader,
          x_train, y_train, x_test, y_test,
          force_cpu=False, num_epochs=5):
    # 使用GPU进行训练
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # 使用tensorboard对结果进行可视化
    writer = SummaryWriter()

    # 开始训练时间
    start = time.time()

    print('[INFO] 开始训练模型.')
    print('在设备 {}上训练模型.'.format('GPU' if device == torch.device('cuda') else 'CPU'))

    for ep in range(num_epochs):
        model.train()
        current_loss = 0.0
        best_accu = 0.0
        for idx_batch, data in enumerate(dataloader):
            x, y = data
            x, y = x.to(device), y.to(device)
            # 清除梯度累计
            optimizer.zero_grad()
            # 前向传播
            y_pred = model(x)
            # 计算损失
            loss = criterion(y_pred, y)
            # 反向传播计算梯度
            loss.backward()
            # 优化
            optimizer.step()
            # 累积损失
            current_loss += loss.item()

        train_acc = get_accuracy(model, x_train, y_train)
        test_acc = get_accuracy(model, x_test, y_test)

        if best_accu < train_acc:
            best_accu = train_acc
            torch.save(model.state_dict(), 'best_train_model.pt')

        writer.add_scalar('data/accuracy_train', train_acc, ep)
        writer.add_scalar('data/accuracy_test', test_acc, ep)
        print(
            'Epoch #{:03d} | Time elapsed : {} | Loss : {:.4e} | Accuracy_train : {:.2f}% | Accuracy_test : {:.2f}% '.format(
                ep + 1, time_since(start), current_loss, 100 * train_acc, 100 * test_acc))

    print('[INFO] 完成模型训练. 耗时 : {}.'.format(time_since(start)))
