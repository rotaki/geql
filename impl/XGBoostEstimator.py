from RLInterfaces import IQEstimator

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import zlib
from PIL import Image

class XGBoostEstimator(IQEstimator):
    def __init__(self, actions, discount, learning_rate, policy=None):
        self.path = []
        self.target = []
        self.discount = discount
        self.learning_rate = learning_rate
        self.compression = 6
        self.downsampling = 8
        self.xg_reg = xgb.XGBRegressor(objective ='reg:linear',
                                       colsample_bytree = 0.3,
                                       learning_rate = 0.1,
                                       max_depth = 5,
                                       alpha = 10,
                                       n_estimators = 10)


    def summary(self):
        temporal_mode = 'XGBoost'
        return 'TabQ [{}, $\\alpha={}$, $\\gamma={}$, $|A| = {}$]'.format(temporal_mode,
                                                                          self.learning_rate,
                                                                          self.discount,
                                                                          len(self.actions))

    def encode_state(self, state):
        i = Image.fromarray(state)
        i = i.crop((0, 40, 256, 240))
        i = i.convert(mode='L')
        width, height = i.size
        i = i.resize((round(width / self.downsampling), round(height / self.downsampling)),
                     resample=Image.NEAREST)
        return zlib.compress(i.tobytes(), self.compression)


    def estimate(self, state, action):
        pred = xg_reg.predict([state, action])
        return pred

    def batch_estimate(self, state, actions):
        return map(lambda a: (a, self.estimate(state, a), actions))

    def reward(self, state, action, reward, state2, action2):
        old_estimate = self.estimate(state, action)
        result_state_value = self.estimate(state2, action2)

        #TODO: adapt to state shape when path.append
        self.path.append([self.encode_state(state), action, reward + self.discount * result_state_value])

        if episode_finished():
            # Train with XGBoost
            data = pd.DataFrame(self.path)
            X, y = data.iloc[:,:-1], data.iloc[:, -1]
            data_dmatrix = xgb.DMatrix(data=X,label=y)
            xg_reg.fit(X, y)

            self.path = []
            self.target = []
            
    def episode_finished(self):
        # TODO return true when finished with the last step

        pass

        


