# https://github.com/mgroncki/DataScienceNotebooks/blob/master/DeepHedging/DeepHedging_Part1.ipynb

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import datetime as dt

class RnnModel(nn.Module):
    def __init__(self, time_steps, batch_size, features, nodes=[62, 46, 46, 1], name='model'):
        super(RnnModel, self).__init__()
        self.batch_size = batch_size
        self.name = name
        self.time_steps = time_steps

        # LSTM layers
        self.lstm = nn.LSTM(features, nodes[0], batch_first=True)
        self.lstm_layers = nn.ModuleList([nn.LSTM(nodes[i], nodes[i+1], batch_first=True) for i in range(len(nodes)-1)])

        # Optimizer
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, S_t_input, K):
        S_T = S_t_input[-1, :, 0]
        dS = S_t_input[1:, :, 0] - S_t_input[:-1, :, 0]
        
        S_t = S_t_input[:-1, :, :]

        out, _ = self.lstm(S_t)
        for lstm_layer in self.lstm_layers:
            out, _ = lstm_layer(out)

        strategy = out[:, :, 0]  # Assuming the strategy is the output of the last LSTM layer

        option = torch.maximum(S_T - K, torch.tensor(0.0))
        Hedging_PnL = -option + torch.sum(dS * strategy, dim=0)
        Hedging_PnL_Paths = -option + dS * strategy

        return Hedging_PnL, Hedging_PnL_Paths, strategy

    def _execute_graph_batchwise(self, paths, strikes, riskaversion, epochs=1, train_flag=False):
        sample_size = paths.shape[1]
        batch_size = self.batch_size
        idx = np.arange(sample_size)
        start = dt.datetime.now()

        for epoch in range(epochs):
            pnls = []
            strategies = []
            if train_flag:
                np.random.shuffle(idx)
            for i in range(int(sample_size / batch_size)):
                indices = idx[i*batch_size : (i+1)*batch_size]
                batch = paths[:, indices, :]
                batch = torch.tensor(batch, dtype=torch.float32)
                K = torch.tensor(strikes[indices], dtype=torch.float32)

                if train_flag:
                    self.optimizer.zero_grad()
                    Hedging_PnL, _, strategy = self.forward(batch, K)
                    CVaR, _ = torch.topk(-Hedging_PnL, int((1-riskaversion) * batch_size))
                    CVaR = torch.mean(CVaR)
                    CVaR.backward()
                    self.optimizer.step()
                else:
                    Hedging_PnL, _, strategy = self.forward(batch, K)

                pnls.append(Hedging_PnL.detach().numpy())
                strategies.append(strategy.detach().numpy())

            CVaR = np.mean(-np.sort(np.concatenate(pnls))[:int((1-riskaversion) * sample_size)])
            if train_flag and epoch % 10 == 0:
                print('Time elapsed:', dt.datetime.now() - start)
                print('Epoch', epoch, 'CVaR', CVaR)
                torch.save(self.state_dict(), f"/Users/matthiasgroncki/models/{self.name}/model.ckpt")
        torch.save(self.state_dict(), f"/Users/matthiasgroncki/models/{self.name}/model.ckpt")
        return CVaR, np.concatenate(pnls), np.concatenate(strategies, axis=1)

    def training(self, paths, strikes, riskaversion, epochs):
        self.train()
        self._execute_graph_batchwise(paths, strikes, riskaversion, epochs, train_flag=True)

    def predict(self, paths, strikes, riskaversion):
        self.eval()
        with torch.no_grad():
            return self._execute_graph_batchwise(paths, strikes, riskaversion, epochs=1, train_flag=False)

    def restore(self, checkpoint):
        self.load_state_dict(torch.load(checkpoint))
