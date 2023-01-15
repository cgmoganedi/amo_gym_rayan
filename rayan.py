
from datetime import datetime
from stable_baselines3 import SAC
import os
import time

models_dir = f'models/SAC-{int(time.time())}'
logdir = f'logs/SAC-{int(time.time())}'

def learnSAC(env, total_ts=1000) -> str:
    model_path = ''
    n_divisions = 10
    save_step = int(total_ts/n_divisions)
    for i in range(1, n_divisions):
        # 1 - create the reinforcement learning model and 2 - train the model for total_ts episodes
        modelOptimalPolicy = SAC('MlpPolicy', env, verbose=1, tensorboard_log=logdir).learn(total_timesteps=save_step, reset_num_timesteps=False, tb_log_name=f'SAC-{int(time.time())}')
        # Save the trained agent to disk
        model_name = str(save_step * i) + '_ts_at_' + str(datetime.now()).split(' ')[0]
        model_name = model_name.replace(" ", "_").replace(":", "_").replace(".", "_")
        model_path = f'{models_dir}/{model_name}'
        modelOptimalPolicy.save(path=model_path)
    return model_path

class Rayan:
    def __init__(self, name=None) -> None:
        self.model_name = name
        self.learned = True if name else False
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        if not os.path.exists(logdir):
            os.makedirs(logdir)
    
    def learn(self, env, total_ts=1000):
        if self.model_name:
            return self.model_name
        self.model_name = learnSAC(env, total_ts)
        self.learned = True
        return self.model_name
    
    def policy_learned(self):
        return SAC.load(self.model_name)
    