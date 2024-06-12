
# https://github.com/mgroncki/DataScienceNotebooks/blob/master/DeepHedging/DeepHedging_Part1.ipynb

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import datetime as dt

class RnnModel:
    def __init__(self, time_steps, batch_size, features, nodes=[62, 46, 46, 1], name='model'):
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.features = features
        self.nodes = nodes
        self.name = name

        self._build_model()

    def _build_model(self):
        S_t_input = Input(shape=(self.time_steps, self.features), batch_size=self.batch_size, name='S_t_input')
        
        lstm_layers = S_t_input
        for node in self.nodes[:-1]:
            lstm_layers = LSTM(node, return_sequences=True, stateful=True)(lstm_layers)
        
        lstm_layers = LSTM(self.nodes[-1], stateful=True)(lstm_layers)

        strategy = Dense(self.time_steps - 1)(lstm_layers)
        self.model = Model(inputs=S_t_input, outputs=strategy)
        self.model.compile(optimizer=Adam(), loss=self._cvar_loss)

    def _cvar_loss(self, y_true, y_pred):
        S_T = y_true[:, -1, 0]
        dS = y_true[:, 1:, 0] - y_true[:, :-1, 0]
        option = tf.maximum(S_T - y_pred[:, 0], 0)
        Hedging_PnL = -option + tf.reduce_sum(dS * y_pred[:, 1:], axis=1)
        alpha = 0.95  # set your alpha here
        CVaR, _ = tf.nn.top_k(-Hedging_PnL, k=tf.cast((1 - alpha) * tf.shape(Hedging_PnL)[0], tf.int32))
        CVaR = tf.reduce_mean(CVaR)
        return CVaR

    def train(self, paths, strikes, riskaversion, epochs):
        sample_size = paths.shape[1]
        idx = np.arange(sample_size)
        start = dt.datetime.now()

        for epoch in range(epochs):
            np.random.shuffle(idx)
            for i in range(int(sample_size / self.batch_size)):
                indices = idx[i * self.batch_size: (i + 1) * self.batch_size]
                batch = paths[:, indices, :]
                self.model.fit(batch, batch, epochs=1, verbose=0)
            if epoch % 10 == 0:
                print('Time elapsed:', dt.datetime.now() - start)
                print('Epoch', epoch)
                self.model.save(f"./models/{self.name}/model_epoch_{epoch}.h5")
        self.model.save(f"./models/{self.name}/model_final.h5")

    def predict(self, paths, strikes, riskaversion):
        sample_size = paths.shape[1]
        idx = np.arange(sample_size)
        start = dt.datetime.now()
        
        predictions = []
        for i in range(int(sample_size / self.batch_size)):
            indices = idx[i * self.batch_size: (i + 1) * self.batch_size]
            batch = paths[:, indices, :]
            predictions.append(self.model.predict(batch))
        
        return np.concatenate(predictions, axis=0)

    def restore(self, checkpoint):
        self.model.load_weights(checkpoint)
