import json 
import websocket
import os
import pandas as pd
import time


assets = ["BTC", "ETH", "LTC"]
assets_hourly = [coins.lower() + "usdt@ticker_1h" for coins in assets]
assets_hourly = "/".join(assets_hourly)
# print(assets_hourly)

assets_daily = [coins.lower() + "usdt@ticker_1d" for coins in assets]
assets_daily = "/".join(assets_daily)
# print(assets_daily)


cav_dir_hourly = "../Dataset/hourly"
cav_dir_daily = "../Dataset/daily24th"
csv_files_name = ["Bitcoin.csv", "Ethereum.csv", "Litecoin.csv"]

csv_files_hourly = [os.path.join(cav_dir_hourly, coins) for coins in csv_files_name]
# print(csv_files_hourly)

csv_files_daily = [os.path.join(cav_dir_daily, coins) for coins in csv_files_name]
# print(csv_files_daily)


for file in csv_files_hourly:
    os.makedirs(os.path.dirname(file), exist_ok=True)
    if not os.path.exists(file):
        with open(file, "w") as f:
            f.write("Date,Open,High,Low,Close,Volume\n")
            f.close()
    else:
        print(f"{file} already exists.")

for file in csv_files_daily:
    os.makedirs(os.path.dirname(file), exist_ok=True)
    if not os.path.exists(file):
        with open(file, "w") as f:
            f.write("Date,Open,High,Low,Close,Volume\n")
            f.close()
    else:
        print(f"{file} already exists.")

source = ""

def retrieve_csv_file(symbol: str, hourly=True):
    # print(symbol == "btcusdt")
    if hourly:
        csv_file_in_use = csv_files_hourly
    else:
        csv_file_in_use = csv_files_daily

    if symbol:
        if symbol ==  "btcusdt":
            return csv_file_in_use[0]
        elif symbol ==  "ethusdt":  
            return csv_file_in_use[1]
        elif symbol ==  "ltcusdt":
            return csv_file_in_use[2]
        
    else:
        raise ValueError("Invalid symbol")
    
def on_open(ws):
    print("Connection opened")

def on_message(ws, message, hourly=True):
    message = json.loads(message)
    global source
    source = message
    # print(message)

    coin_data = message['data']

    symbol = coin_data["s"].lower()
    csv_save_file = retrieve_csv_file(str(symbol), hourly)
    timestamp = coin_data["E"]
    open_price = coin_data["o"]
    high_price = coin_data["h"]
    low_price = coin_data["l"]
    close_price = coin_data["c"]
    volume = coin_data["v"]

    # timestamp = pd.to_datetime(timestamp, unit='ms')

    # Create a DataFrame from the coin_data
    df = pd.DataFrame({
        "Date": [timestamp],
        "Open": [open_price],
        "High": [high_price],
        "Low": [low_price],
        "Close": [close_price],
        "Volume": [volume]
    })

    # Append the DataFrame to the CSV file
    df.to_csv(f"{csv_save_file}", mode="a", header=False, index=False)


def on_close(ws, close_status_code, close_msg):
    print("Connection closed")  

def on_error(ws, error):
    print("Error:", error)

socket_hourly = f"wss://stream.binance.com:9443/stream?streams={assets_hourly}"
socket_daily = f"wss://stream.binance.com:9443/stream?streams={assets_daily}"


def run_websocket_hourly(socket_name):
    ws = websocket.create_connection(socket_name)
    # print("WebSocket connection opened")

    # ws = websocket.WebSocketApp(
    #     socket_hourly, 
    #     on_open=on_open,
    #     on_close=on_close,
    #     on_error=on_error,
    #     on_message=on_message
    # )
    
    result = ws.recv()
    on_message(ws, result)

    print("Received:", result)
    ws.close()

def run_websocket_daily(socket_name):
    ws = websocket.create_connection(socket_name)
    # print("WebSocket connection opened")

    # ws = websocket.WebSocketApp(
    #     socket_daily, 
    #     on_open=on_open,
    #     on_close=on_close,
    #     on_error=on_error,
    #     on_message=on_message
    # )
        
    result = ws.recv()
    on_message(ws, result, hourly=False)
    print("Received:", result)
    ws.close()


count = 0
socket = "wss://stream.binance.com:9443/stream?streams="
while True:
    if count % 24 == 0:
        print("\n\nDaily report:")
        # runs the websocket for daily data
        for coins in assets:
            socket_name = socket + coins.lower() + "usdt@ticker_1d"
            run_websocket_daily(socket_name)
        
    count += 1
    print("\n\nHourly report:")
    for coins in assets:
        socket_name = socket + coins.lower() + "usdt@ticker_1h"
        run_websocket_hourly(socket_name)

    time.sleep(3600) 