from pandas.io import pickle
from torch.utils.data import Dataset

class GestureDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class DHG_Dataset(Dataset):
    def __init__(self, filepath):
        self.x_train, self.x_test, self.y_train_14, self.y_train_28, self.y_test_14, self.y_test_28 = self.load_data(filepath)

    def __getitem__(self, index):
        return self.x_train[index], self.x_test[index], self.y_train_14[index], self.y_train_28[index], self.y_test_14[index], self.y_test_28[index]

    def __len__(self):
        return len(self.x_train)

    def load_data(self, filepath=''):
        file = open(filepath, 'rb')
        data = pickle.load(file, encoding='latin1')  #change to 'latin1' to 'utf8' if the data does not load
        file.close()
        return data['x_train'], data['x_test'], data['y_train_14'], data['y_train_28'], data['y_test_14'], data[
            'y_test_28']