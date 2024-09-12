import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import time

def mlp(dim, hidden_dim, output_dim, layers, variance):
    """Create a mlp from the configurations."""

    seq = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    seq += [nn.Linear(hidden_dim, output_dim)]
    # if variance ==  True:
    #     seq += [nn.ReLU()]
    return nn.Sequential(*seq)

class TradeoffLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, y):
        if self.alpha >= 1:
            return torch.mean(torch.pow(torch.abs(x-y), self.alpha))
        if self.alpha < 1:
            return torch.mean(torch.clamp((1/self.alpha)*torch.abs(x-y), 1))

class L2loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        return torch.mean(torch.pow((x-y), 2))

class Retriever(object):
    def __init__(self, alpha = 2, regular = 0.01, variance = False):
        self.model = None
        self.variance = variance
        self.loss = self._get_loss(alpha)
        self.optimizer = None
        self.X_SC = None
        self.y_SC = None
        self.regular = regular

    def _build_model(self, n_observations, n_layer, hidden_dim, variance):
        return mlp(n_observations, hidden_dim, 1, n_layer, variance = variance).cuda()
    
    def _get_loss(self, alpha):
        if alpha == 2:
            return L2loss()
        else:
            return TradeoffLoss(alpha)

    def _get_optimizer(self, lr):
        return torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = self.regular)
        # return torch.optim.SGD(self.model.parameters(), lr = lr, weight_decay = 0.01)
    
    def train(self, X, y, epochs, batch_size, n_observations, n_layer, hidden_dim, learning_rate = 5e-4,
              ifvalidation = False, validation_size = 0.1):
        """
        X: [N_samples, N_features]
        y: [N_samples, 1]
        """
        y = y.reshape(-1, 1)
        self.model = self._build_model(n_observations, n_layer, hidden_dim, self.variance)
        self.optimizer = self._get_optimizer(learning_rate)
        
        self.X_SC = StandardScaler().fit(X)
        self.y_SC = StandardScaler().fit(y)
        X_norm = self.X_SC.transform(X)
        y_norm = self.y_SC.transform(y)
        X_norm_train, X_norm_val, y_norm_train, y_norm_val = train_test_split(X_norm,y_norm, test_size = validation_size, shuffle = True)
        X_norm_train = torch.from_numpy(X_norm_train).cuda()
        y_norm_train = torch.from_numpy(y_norm_train).cuda()

        for j in range(epochs): 
            for i in range(np.int_(X_norm_train.shape[0]/batch_size)):
                batch_x_norm = X_norm_train[i*batch_size:(i+1)*batch_size, :]
                batch_y_norm = y_norm_train[i*batch_size:(i+1)*batch_size, :]
                y_pred = self.model(batch_x_norm)
                loss = self.loss(y_pred, batch_y_norm)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if j % 5 == 0 and i == np.int_(X_norm_train.shape[0]/batch_size) - 1: 
                    print("Epoch = ", j , 'Loss = ', loss)
            if ifvalidation:
                pass
    
    def test(self, X, y):
        if self.model == None:
            raise Exception('No loaded model.')
        if self.X_SC == None:
            raise Exception('No standard scalar.')
        X_norm = torch.from_numpy(self.X_SC.transform(X)).cuda()
        y_output = self.model(X_norm)
        y_pred = self.y_SC.inverse_transform(y_output.detach().cpu().numpy())
        return np.sqrt(mean_squared_error(y_pred, y)), y_pred.flatten()
    
    def predict(self, X):
        if self.model == None:
            raise Exception('No loaded model.')
        if self.X_SC == None:
            raise Exception('No standard scalar.')
        X_norm = torch.from_numpy(self.X_SC.transform(X)).cuda()
        y_output = self.model(X_norm)
        y_pred = self.y_SC.inverse_transform(y_output.detach().cpu().numpy())
        return y_pred.flatten()
    
    def save_model(self, path, sc_path):
        import shelve
        torch.save(self.model, path)
        ds = shelve.open(sc_path)
        ds['X_SC']= self.X_SC
        ds['y_SC'] = self.y_SC
        ds.close()
        
    def load_model(self, path, sc_path):
        import shelve
        self.model = torch.load(path)
        self.X_SC= shelve.open(sc_path)["X_SC"]
        self.y_SC = shelve.open(sc_path)["y_SC"]


if __name__ == "__main__":
    pass

