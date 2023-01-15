
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import os
import time

models_dir = f'models/SAC-{int(time.time())}'
logdir = f'logs/SAC-{int(time.time())}'


def learnSAC(env, total_ts=1000) -> str:
    model_path = ''
    n_divisions = 10
    save_step = int(total_ts/n_divisions)
    
    vec_env = Monitor(env, logdir, allow_early_resets=True)
    
    # create the reinforcement learning model
    modelOptimalPolicy = SAC('MlpPolicy', vec_env, verbose=1, tensorboard_log=logdir)
    for i in range(1, n_divisions):
        # train the model for total_ts episodes
        modelOptimalPolicy.learn(total_timesteps=save_step, reset_num_timesteps=False, tb_log_name='SAC')
        
        # define path to save at
        model_name = str(save_step * i) + '_ts_at_' + str(datetime.now()).split(' ')[0]
        model_path = f'{models_dir}/{model_name}'
        
        # Save the trained agent to disk
        modelOptimalPolicy.save(path=model_path)
    
    vec_env.close()
    return model_path

class Rayan:
    def __init__(self, name=None) -> None:
        self.model_path = name
        self.learned = True if name else False
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        if not os.path.exists(logdir):
            os.makedirs(logdir)
    
    def learn(self, env, total_ts=1000):
        if self.model_path:
            return self.model_path
        self.model_path = learnSAC(env, total_ts)
        self.learned = True
        return self.model_path
    
    def policy_learned(self):
        return SAC.load(self.model_path)
    
