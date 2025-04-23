import json
import yfinance as yf
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle
import os

# backend
# loading single coin models
def coin_model_instance():
    window_size = 60
    model_coin = Sequential([
        LSTM(128, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dense(units=1)
    ])
    return model_coin


# def reload_model(coin):
#     window_size = 60
#     model_paths = [
#         '/Users/eyash.p24/Desktop/miscellaneous/Intel Project/app/checkpoints/btc_checkpoint.weights.h5',
#         '/Users/eyash.p24/Desktop/miscellaneous/Intel Project/app/checkpoints/eth_checkpoint.weights.h5',
#         '/Users/eyash.p24/Desktop/miscellaneous/Intel Project/app/checkpoints/ltc_checkpoint.weights.h5'
#     ]

#     model_multicoin = Sequential(
#         [LSTM(50, return_sequences=True, input_shape=(window_size, 5)),
#         Dropout(0.2),
#         LSTM(50, return_sequences=False),
#         Dropout(0.2),
#         Dense(units=5)]
#     )
#     model_multicoin.load_weights('/Users/eyash.p24/Desktop/miscellaneous/Intel Project/app/checkpoints/merge_checkpointv2.weights.h5')

#     coin_id = -1
#     if coin == 'BTC':
#         coin_id = 0
#     elif coin == "LTC":
#         coin_id = 1
#     elif coin == 'ETH':
#         coin_id = 2

#     coin_model = coin_model_instance()
#     coin_model.load_weights(model_paths[coin_id])
#     # print(model_multicoin.summary())
#     # print(coin_model.summary())
#     return model_multicoin, coin_model


# def reload_scaler(coin):
#     coin_paths = [
#         '/Users/eyash.p24/Desktop/miscellaneous/Intel Project/app/checkpoints/btc_scaler.pkl',
#         '/Users/eyash.p24/Desktop/miscellaneous/Intel Project/app/checkpoints/ltc_scaler.pkl',
#         '/Users/eyash.p24/Desktop/miscellaneous/Intel Project/app/checkpoints/eth_scaler.pkl',
#     ]

#     # multi-coin
#     with open('/Users/eyash.p24/Desktop/miscellaneous/Intel Project/app/checkpoints/merge_scalerv2.pkl', 'rb') as f:
#         merge_scaler = pickle.load(f)
#         f.close()
    
#     # single coin
#     coin_id = -1
#     if coin == 'BTC':
#         coin_id = 0
#     elif coin == "LTC":
#         coin_id = 1
#     elif coin == 'ETH':
#         coin_id = 2
    
#     with open(coin_paths[coin_id], 'rb') as f_coin:
#         coin_scaler = pickle.load(f_coin)
#         f_coin.close()

#     return merge_scaler, coin_scaler

def reload_model(coin):
    window_size = 60

    # Define relative paths
    base_path = os.path.dirname(__file__)
    checkpoint_dir = os.path.join(base_path, "checkpoints")

    model_paths = {
        'BTC': os.path.join(checkpoint_dir, 'btc_checkpoint.weights.h5'),
        'ETH': os.path.join(checkpoint_dir, 'eth_checkpoint.weights.h5'),
        'LTC': os.path.join(checkpoint_dir, 'ltc_checkpoint.weights.h5')
    }

    merge_model_path = os.path.join(checkpoint_dir, 'merge_checkpointv2.weights.h5')

    # Load multicoin model
    model_multicoin = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, 5)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(units=5)
    ])
    model_multicoin.load_weights(merge_model_path)

    # Load single-coin model
    coin = coin.upper()
    if coin not in model_paths:
        raise ValueError(f"Unsupported coin: {coin}")

    coin_model = coin_model_instance()
    coin_model.load_weights(model_paths[coin])

    return model_multicoin, coin_model

def reload_scaler(coin):
    # Get the current directory (where this file is located)
    base_path = os.path.dirname(__file__)
    checkpoint_dir = os.path.join(base_path, "checkpoints")

    coin_paths = {
        "BTC": os.path.join(checkpoint_dir, "btc_scaler.pkl"),
        "LTC": os.path.join(checkpoint_dir, "ltc_scaler.pkl"),
        "ETH": os.path.join(checkpoint_dir, "eth_scaler.pkl")
    }

    # Load multi-coin scaler
    merge_scaler_path = os.path.join(checkpoint_dir, "merge_scalerv2.pkl")
    with open(merge_scaler_path, 'rb') as f:
        merge_scaler = pickle.load(f)

    # Load specific coin scaler
    coin_scaler_path = coin_paths.get(coin.upper())
    if not coin_scaler_path:
        raise ValueError(f"Invalid coin: {coin}")
    
    with open(coin_scaler_path, 'rb') as f_coin:
        coin_scaler = pickle.load(f_coin)

    return merge_scaler, coin_scaler

def fetch_current_data(coin='BTC'):
    btc = yf.Ticker('BTC-USD')
    ltc = yf.Ticker('LTC-USD')
    eth = yf.Ticker('ETH-USD')

    btc_hist = btc.history(period='60D')
    ltc_hist = ltc.history(period='60D')
    eth_hist = eth.history(period='60D')

    coin_list = [btc_hist, ltc_hist, eth_hist]
    for c in coin_list:
        if 'Stock Splits' in c.columns:
            c.drop(columns=['Stock Splits'], inplace=True)
        if 'Dividends' in c.columns:
            c.drop(columns=['Dividends'], inplace=True)

    '''
    For single coin, require only close price data.
    For multi-coin, require all 5 features
    '''

    btc_data = btc_hist
    eth_data = eth_hist
    ltc_data = ltc_hist

    if coin == "BTC":
        return btc_data
    elif coin == "ETH":
        return eth_data
    elif coin == 'LTC':
        return ltc_data


def forecast(forecast_days=[1, 5, 10, 30], coin='BTC', target_col=3):
    # Fetch initial inputs
    X_merge_init = fetch_current_data(coin)  # shape (60, 5)
    current_price_value = X_merge_init['Close'].iloc[-1]
    X_coin_init = X_merge_init['Close']      # assuming shape (60, 5) too

    # Reload scalers and models
    merg_scaler, coin_scaler = reload_scaler(coin)
    merge_model, coin_model = reload_model(coin)

    # Constants
    total_features = X_merge_init.shape[-1]

    # Convert to NumPy arrays
    X_input_multicoin = X_merge_init.to_numpy()
    X_input_coin = X_merge_init['Close'].to_numpy()

    # Ensure shapes are correct
    X_input_multicoin = X_input_multicoin.reshape(1, 60, total_features)
    X_input_coin = X_input_coin.reshape(1, 60, 1)

    # Store predictions
    forecast_result_multicoin = {}
    forecast_result_single = {}
    predicted_closes_multicoin = []
    predicted_closes_single = []

    for _ in range(1, max(forecast_days) + 1):
        # Predict scaled close values
        pred_scaled_multicoin = merge_model.predict(X_input_multicoin, verbose=0)
        pred_scaled_coin = coin_model.predict(X_input_coin, verbose=0)

        # print(pred_scaled_multicoin)
        # print(X_input_coin.shape)

        # Inverse transform
        pred_full_multicoin = merg_scaler.inverse_transform(pred_scaled_multicoin)
        pred_full_coin = coin_scaler.inverse_transform(pred_scaled_coin)

        # print(pred_full_multicoin)

        pred_close_multicoin = pred_full_multicoin[:, target_col][0]
        pred_close_coin = pred_full_coin[0]

        predicted_closes_multicoin.append(pred_close_multicoin)
        predicted_closes_single.append(pred_close_coin)

        # Recursive input update
        next_input_multi = X_input_multicoin.copy()
        new_step_multi = pred_scaled_multicoin.flatten().copy()
        X_input_multicoin = np.concatenate([next_input_multi, new_step_multi.reshape(1, 1, total_features)], axis=1)

        next_input_coin = X_input_coin.copy()
        new_step_coin = pred_scaled_coin.flatten().copy()
        X_input_coin = np.concatenate([next_input_coin, new_step_coin.reshape(1, 1, 1)], axis=1)


    # Extract forecasts
    for d in forecast_days:
        forecast_result_multicoin[f"{d}d"] = predicted_closes_multicoin[d - 1]
        forecast_result_single[f"{d}d"] = predicted_closes_single[d - 1]

    # Return predictions and the latest observed timestep
    return forecast_result_multicoin, forecast_result_single, current_price_value


def generate_recommendation(predictions, current_price, current_hold=0, sell_threshold_per=20):
    """
    Generate investment recommendation based on forecasted prices, 
    current price, and user holdings, with an upper holding threshold.
    """
    sell_threshold = current_price * sell_threshold_per
    returns = {k: ((v - current_price) / current_price) * 100 for k, v in predictions.items()}

    all_negative = all(r < 0 for r in returns.values())
    immediate_gain = returns["1d"] > 1
    rebound_after_drop = returns["1d"] < -1 and returns["5d"] > 3
    long_term_gain = returns["30d"] > 5
    low_volatility = max(returns.values()) < 2

    # If user is holding too much, recommend selling on any kind of gain
    if current_hold >= sell_threshold:
        if all_negative:
            return "You hold too much — Sell now to minimize loss"
        elif immediate_gain or long_term_gain:
            best_day = max(returns, key=returns.get)
            return f"You hold too much — Sell part or all on {best_day} to book profit"
        elif low_volatility:
            return "You hold too much and market is flat — consider reducing your position"
    
    # Case 1: All returns are negative
    if all_negative:
        if current_hold > 0:
            return "Sell your holdings to avoid further loss"
        else:
            return "Avoid buying now — market looks negative"

    # Case 2: Strong rebound pattern detected
    if rebound_after_drop:
        if current_hold > 0:
            return "Hold your position, expect a rebound by day 5"
        else:
            return "Buy now to benefit from rebound, sell on day 5"

    # Case 3: Immediate gain and strong 30-day return
    if immediate_gain and long_term_gain:
        if current_hold > 0:
            return "Hold and consider increasing your position"
        else:
            return "Buy now and hold for long-term gain"

    # Case 4: Short-term spike followed by dip
    if returns["1d"] > 0 and returns["5d"] < 0:
        if current_hold > 0:
            return "Sell on day 1 to take quick profit"
        else:
            return "Buy now and sell on day 1 for quick gain"

    # Case 5: Minor short-term gain but strong long-term growth
    if returns["1d"] < 1 and long_term_gain:
        if current_hold > 0:
            return "Hold and consider buying more"
        else:
            return "Buy now and hold for long-term return"

    # Case 6: All returns low
    if low_volatility:
        if current_hold > 0:
            return "Hold — not enough momentum to sell or buy more"
        else:
            return "Wait — market is not promising right now"

    # Default: Recommend best day to sell for profit
    best_day = max(returns, key=returns.get)
    if current_hold > 0:
        return f"Hold and sell on {best_day} for best return"
    else:
        return f"Buy now and sell on {best_day} for profit"