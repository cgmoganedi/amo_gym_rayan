import gym
import time
import MetaTrader5 as _mt5
import numpy as np
import pandas as pd

from gym import spaces
from ta.volatility import BollingerBands
from ta.trend import CCIIndicator
from datetime import datetime


def strategy_group_A(df)-> pd.DataFrame:
    # Initialize Bollinger Bands and CCI Indicators
    indicator_bb = BollingerBands(close=df['close'], window=26, window_dev=2, fillna=True)
    indicator_cci = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=26, constant=.015, fillna=True)
    
    new_df = pd.DataFrame()
    
    # Add compulsory, uniform features
    new_df['time'] = df['time']
    new_df['close'] = df['close']
    new_df['adj_close'] = (df['high'] + df['low'] + 2 * df['close'])/4
    
    # Add Bollinger Bands features
    new_df['bb_bbm'] = indicator_bb.bollinger_mavg()
    new_df['bb_bbh'] = indicator_bb.bollinger_hband()
    new_df['bb_bbl'] = indicator_bb.bollinger_lband()

    # Add Bollinger Band high and low indicators
    new_df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()
    new_df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

    # Add commodity channels indicator value and normalize it
    new_df['cci'] = indicator_cci.cci()
    new_df['cci'] = new_df['cci'] / new_df['cci'].abs().max()
    
    # normalize the price
    new_df['close'] = new_df['close'] / new_df['close'].abs().max()
    new_df['adj_close'] = new_df['adj_close'] / new_df['adj_close'].abs().max()
    
    return new_df

class AmoGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, symbols, tech_indicator_strategy_group, window_size=6, n_candles=126):
        if not _mt5.initialize():
            raise ValueError('MT5 initialization failed: ',  _mt5.last_error())
            
        # system workspace defination
        self.time_frame = _mt5.TIMEFRAME_M1
        self.n_candles = n_candles
        self.symbols = symbols
        self.window_size = window_size
        
        # get data and setup features
        self.num_sf = 0
        self.tech_indicator_strategy_group = tech_indicator_strategy_group
        self._process_data()
        
        # observable features
        self._signal_features = np.zeros((self.n_candles, len(symbols), self.num_sf))
        
        # spaces
        self.observation_shape = (window_size, len(symbols), self.num_sf)
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(symbols),))
        self.observation_space = spaces.Box(low=-2, high=2, shape=self.observation_shape, dtype=np.float64)
        
        # episode
        self._done = False
        self._total_reward = 0.0
        self._prev_prev_equity = 0.0
        self._prev_equity = 0.0
        self._step_count = 0
        
    def reset(self):
        self._done = False
        self._total_reward = 0.0
        self._step_count = 0
        self._process_data()
        return self._get_observation()
    
    def step(self, action):
        # action = array of size len(self.symbols)
        # each action accounts for a symbol (realise more than one trade may be opened for a given currency pair)
        self._done = False
        self._step_count += 1
        self._collect_profits()
        trade_actions_reward = self._action_loop(action) # we are taking a collection of actions, one for each symbol
        account_health_reward = self._account_health_reward(trade_actions_reward)
        
        print('\n', self._step_count ,' : ', datetime.now())
        print('trade_act_reward = ', trade_actions_reward, '\nacc_health_reward = ', account_health_reward)
        
        if self._done:
            account =_mt5.account_info()
            info = {
                'total_profit' : account.profit,
                'total_reward' : self._total_reward
            }
            
        else:
            # wait for window size minutes
            window_size_minutes = self.window_size * 60 # 1 minute = 60 secs
            time.sleep(window_size_minutes)
            
            try:
                # update previous equity values
                account =_mt5.account_info()
                self._prev_prev_equity = self._prev_equity
                self._prev_equity = account.equity
            except ValueError as err:
                print('Error', err)
            finally:
                info = {}
        
        # get new data and proceed to the next step for observation
        self._process_data()
        observation = self._get_observation()
        return observation, account_health_reward, self._done, info
    
    def _get_observation(self):
        return self._signal_features[:self.window_size]

    def _account_health_reward(self, trade_reward):
        account = _mt5.account_info()
        acccount_balance = account.balance
        initial_balance = acccount_balance - account.profit
        equity = account.equity
        
        if acccount_balance >= initial_balance * 2:
            self._done = True
            trade_reward = acccount_balance
            
        elif acccount_balance <= initial_balance * 0.75:
            self._done = True
            trade_reward = -initial_balance
            
        elif equity <= initial_balance * 0.5:
            self._done = True
            trade_reward = -initial_balance
            
        elif equity >= initial_balance * 1.5:
            trade_reward += initial_balance * 0.125
            
        elif equity <= initial_balance * 0.75:
            trade_reward -= initial_balance * 0.125
        
        elif equity > self._prev_equity and self._prev_equity > self._prev_prev_equity:
            trade_reward += initial_balance * 0.025
            
        elif equity < self._prev_equity and self._prev_equity < self._prev_prev_equity:
            trade_reward -= initial_balance * 0.025
        
        # calulate reward and give it based on mostly on current account balance and lightly on equit
        health_reward = trade_reward * abs(acccount_balance * 0.66 + equity * 0.33) / (initial_balance)
        health_reward /= 100
        
        self._total_reward += health_reward
        return health_reward
    
    def _process_data(self):
        timesteps_from_now = self.n_candles # Most recent 24 hours = 1440 minutes, candles.
        now = datetime.now()

        strategy_signal_features = pd.DataFrame()
        signal_space_arr = [] # np.zeros((self.observation_shape))
        # fetch the 1-minute forex data for the specified currency pairs
        for i, symbol in enumerate(self.symbols):
            # get the data from MetaTrader5
            candlestick_data = _mt5.copy_rates_from(symbol, self.time_frame, now, timesteps_from_now)

            # convert the data to a pandas DataFrame then drop tick_volume, spread and real_volume columns
            df = pd.DataFrame(candlestick_data)
            df.drop(['tick_volume', 'spread', 'real_volume'], axis='columns', inplace=True, errors='ignore')
            df = df.iloc[::-1]
            
            if self.tech_indicator_strategy_group == 'GROUP_A':
                strategy_signal_features = strategy_group_A(df)
            else:
                strategy_signal_features = strategy_group_A(df)
            
            # set the 'time' column as the index
            strategy_signal_features.set_index("time", inplace=True)
            signal_space_arr.append(strategy_signal_features.values)

        # set num_sf to the number of strategy signal_features
        self.num_sf = len(strategy_signal_features.columns)
        
        # Create the ndarray from the list
        ndarray = np.stack(signal_space_arr)

        # When you stack the arrays, the shape becomes (number of symbols, timesteps from now, number of features), 
        ndarray = ndarray.reshape(len(self.symbols), timesteps_from_now, self.num_sf)
        ndarray_trs = np.transpose(ndarray, (1, 0, 2)) # needs to match observation shape, timesteps is the first index, so we transpose it
        self._signal_features = ndarray_trs 
        
    def _action_loop(self, actions):
        reward = 1.0
        threshold = 0.83
        neg_threshold = -threshold
        # do nothing if neg_threshold < action < threshold
        # buy if threshold <= action <= 1
        # sell if -1 <= action <= neg_threshold
        # exit if threshold * 0.5 <= abs(action) <= threshold * 0.8
        
        for i in range(len(self.symbols)):
            if (abs(actions[i]) >= threshold * 0.5) and (abs(actions[i]) <= threshold * 0.8):
                # close all the open positions of this currency pair
                symbol = self.symbols[i]
                success, profit = self._close_all_open_positions(symbol)
                if not success:
                    print('Failed: ', _mt5.last_error())
                    # raise ValueError('Failed to close a trade!')
                reward += profit
                
            elif actions[i] >= threshold:
                # close the all the open sell positions of this currency pair and open a buy
                symbol = self.symbols[i]
                success, ticket_id = self._open_position(i, order_type=_mt5.ORDER_TYPE_BUY, lot_multiplier=abs(actions[i]))
                if not success:
                    print('Failed: ', _mt5.last_error())
                    # raise ValueError('Failed to open a buy trade!')
                reward += 1.0
                
            elif actions[i] <= neg_threshold:
                # close the all the open buy positions of this currency pair and open a buy
                symbol = self.symbols[i]
                success, ticket_id = self._open_position(i, order_type=_mt5.ORDER_TYPE_SELL, lot_multiplier=abs(actions[i]))
                if not success:
                    print('Failed: ', _mt5.last_error())
                    # raise ValueError('Failed to open a sell trade!')
                reward += 1.0
                
            else:
                # right now, do nothing for this currency pair. There is a reward of 1 for passing
                reward += 1.0
                
        return reward
        
        
    def _open_position(self, symbol_id, order_type, lot_multiplier=0.0):
        sl = 260
        symbol = self.symbols[symbol_id]
        point = _mt5.symbol_info(symbol).point
        price = _mt5.symbol_info_tick(symbol).ask
        
        open_positions = _mt5.positions_get(symbol=symbol)
        if len(open_positions) >= 7:
            return True, 0.0
        
        stop_loss = price - sl * point
        take_profit = price + 1250 * point
        
        if order_type == _mt5.ORDER_TYPE_SELL:
            stop_loss = price + sl * point
            take_profit = price - 1250 * point
            
        lot = 0.01
        if lot_multiplier >= 0.96:
            lot *= 9
        elif lot_multiplier >= 0.92:
            lot *= 6
        elif lot_multiplier >= 0.88:
            lot *= 3
            
        request = {
            "action": _mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 992,
            "type_time": _mt5.ORDER_TIME_GTC,
            "type_filling": _mt5.ORDER_FILLING_DEFAULT
        }

        # send a trading request
        result = _mt5.order_send(request)
        if result.retcode != _mt5.TRADE_RETCODE_DONE:
            print(result)
            return False, 0.0
        
        return True, 0.0
    
    def _collect_profits(self) -> None:
        try:
            account = _mt5.account_info()
            for symbol in self.symbols:
                i = 0
                open_positions = _mt5.positions_get(symbol=symbol)
                for open_trade in open_positions:
                    if i % 2 == 0:
                        continue
                    elif open_trade.profit >= account.balance * 0.077:
                        self._close_position(open_trade)
                    i += 1
        except ValueError as err:
            print(symbol, ' : Failed to collect profits')
    
    def _close_all_open_positions(self, symbol):
        success = True
        profit = 0.0
        length_of_trade = 0.0
        proportional_value = 1.0
        
        try:
            open_positions = _mt5.positions_get(symbol=symbol)
            for open_trade in open_positions:
                time_diff = datetime.fromtimestamp(open_trade.time) - datetime.now()
                length_of_trade = time_diff.total_seconds() / 3600
                profit = open_trade.profit
                proportional_value += profit * (1 + length_of_trade)
                if length_of_trade >= 36:
                    print(symbol, 'Closed length of: ', length_of_trade)
                    success = self._close_position(open_trade)
                    
        except ValueError as err:
            print('Closing trades error: ', err)
            success = False
        
        return success, proportional_value
    
    def _close_position(self, open_trade)->bool:
        tick = _mt5.symbol_info_tick(open_trade.symbol)
        
        request = {
            "action": _mt5.TRADE_ACTION_DEAL,
            "position": open_trade.ticket,
            "symbol": open_trade.symbol,
            "volume": open_trade.volume,
            "type": _mt5.ORDER_TYPE_BUY if open_trade.type == _mt5.ORDER_TYPE_SELL else _mt5.ORDER_TYPE_SELL,
            "price": tick.ask if open_trade.type == _mt5.ORDER_TYPE_SELL else tick.bid,
            "deviation": 20,
            "magic": 992,
            "type_time": _mt5.ORDER_TIME_GTC,
            "type_filling": _mt5.ORDER_FILLING_IOC
        }

        # send the trading close request
        result = _mt5.order_send(request)
        if result.retcode != _mt5.TRADE_RETCODE_DONE:
            print(result)
            return False
        return True
        
        