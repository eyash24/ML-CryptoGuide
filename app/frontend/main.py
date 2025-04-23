import streamlit as st
import pandas as pd
import requests

st.title("Welcome to CryptoGuide")

welcome_text = ("""
In this guide, users can forecast 3 different coins using inbuilt models: **Bitcoin, Ethereum, and Litecoin**.
The program also gives out tailored recommendations based on forecasts of both:
- A **Single-Coin Model** (trained individually)
- A **Multi-Coin Model** (trained jointly)

---

### Disclaimer
This project is for ***educational purposes only***. It is **not** intended for investment advice.

There may be discrepancy between the model forecasts.
- **Single-Coin Model** captures trends for that specific coin.
- **Multi-Coin Model** learns from the combined crypto market.

""")

st.markdown(welcome_text)

# Coin 
coin_map = {
    "Bitcoin": "BTC",
    "Ethereum": "ETH",
    "Litecoin": "LTC"
}

coin_ui = st.selectbox("Select coin to forecast:", ["---Select a Coin---"] + list(coin_map.keys()))
if coin_ui != "---Select a Coin---":
    current_amt = st.number_input("Enter your current holdings:", min_value=0.0, format="%.4f")

    if st.button("Forecast & Recommend!"):
        coin = coin_map[coin_ui]
        # Replace with your actual FastAPI URL
        api_url = f"http://localhost:8000/forecast/{coin}/{current_amt}"

        try:
            response = requests.get(api_url)
            data = response.json()

            if "error" in data and data["error"]:
                st.error(f"{data['error']}")
            else:
                st.subheader("Forecast (Closing Prices)")

                forecast_df = pd.DataFrame({
                    "Model": ["Multi-Coin", "Single-Coin"],
                    "1 Day": [
                        round(data["forecast_multicoin"]["1d"], 4),
                        round(data["forecast_coin"]["1d"], 4)
                    ],
                    "5 Days": [
                        round(data["forecast_multicoin"]["5d"], 4),
                        round(data["forecast_coin"]["5d"], 4)
                    ],
                    "10 Days": [
                        round(data["forecast_multicoin"]["10d"], 4),
                        round(data["forecast_coin"]["10d"], 4)
                    ],
                    "30 Days": [
                        round(data["forecast_multicoin"]["30d"], 4),
                        round(data["forecast_coin"]["30d"], 4)
                    ]
                })

                st.table(forecast_df)

                st.subheader("Recommendations")
                st.markdown(f"**Multi-Coin Model**: {data['rec_multicoin']}")
                st.markdown(f"**{coin} Model**: {data['rec_single']}")

        except Exception as e:
            st.error(f"Something went wrong while calling the backend: {e}")
