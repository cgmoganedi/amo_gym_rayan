
from datetime import datetime
from stable_baselines3 import SAC


def learnSAC(env, total_ts=1000) -> str:
    # 1 - create the reinforcement learning model and 2 - train the model for 2880 episaodes
    modelOptimalPolicy = SAC('MlpPolicy', env, verbose=1).learn(total_timesteps=total_ts)
    # Save the trained agent to disk
    model_name = 'sac_' + str(total_ts) + '_ts_at_' + str(datetime.now())
    model_name = model_name.replace(" ", "_").replace(":", "_").replace(".", "_")
    model_path = 'models/sac/' + model_name
    modelOptimalPolicy.save(path=model_path)
    return model_path

class Rayan:
    def __init__(self, name=None) -> None:
        self.model_name = name
        self.learned = True if name else False
    
    def learn(self, env, total_ts=1000):
        if self.model_name:
            return self.model_name
        self.model_name = learnSAC(env, total_ts)
        self.learned = True
        return self.model_name
    
    def policy_learned(self):
        return SAC.load(self.model_name)
    