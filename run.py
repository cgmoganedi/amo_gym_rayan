# import the necessary modules
from matplotlib import pyplot as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from amo_gym_env import AmoGymEnv
from rayan import Rayan

# Agents
# ---------------------------------
# 1. Ryda: Group B, Loop A, TD3
# 2. Rayrei: Group B, Loop B, TD3
# 3. Rayan: Group A, Loop A, SAC
# ---------------------------------
# 4. Revel: Group C, Loop A, SAC
# ---------------------------------


# Technical indicator groups
# Group A: [Price, Ballinger Bands & CCi]
# Group B: [Price, Ichimoku & MACD]
# Group C: [Price, 18EMA, 50EMA & 200SMA]

# Stepping loop choices
# Step loop A: 
# Step loop B:

# Algorithm choice, continous action space 
# TD3:
# SAC:

symbols = ['CADJPY', 'CHFJPY', 'EURJPY', 'AUDJPY', 'NZDJPY', 'GBPJPY']

# initialize the ForexStockTradingEnv environment
env = AmoGymEnv(symbols, tech_indicator_strategy_group='GROUP_A')
# env_valid = check_env(env)
# if not env_valid:
#      raise ValueError('Environment not valid')

def main():
    # Load the trained agent from disk
    print('Rayan starting ...')
    rayan = Rayan()
    if not rayan.learned: 
        rayan.learn(env, 500)

    rayan_policy = rayan.policy_learned()
    
    # evaluate the models over 50 episodes
    mean_reward_rayan, std_reward_rayan = evaluate_policy(rayan_policy, env, n_eval_episodes=100)

    print(f"Rayan mean reward: {mean_reward_rayan:.2f} +/- {std_reward_rayan:.2f}")

    # Go live
    observation = env.reset()

    while True:
        # Act the model's trade recommendation for the current observation and advance to the next obervation after acting
        actions = rayan_policy.predict(observation)
        observation, reward, done, info = env.step(actions)

        if done:
            print('Rayan done: ', info)
            print('Total reward: ', reward)
            break
        
    # Plot the results
    plt.figure(figsize=(15,6))
    plt.cla()
    env.render_all()
    plt.show()
    print('Amo Rayan agent done, bye ...!')

main()