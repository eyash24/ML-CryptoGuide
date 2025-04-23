from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import numpy as np
from utils import forecast, generate_recommendation

api = FastAPI()

class Response(BaseModel):
    coin: Optional[str] = None
    forecast_multicoin: Optional[dict] = None
    forecast_coin: Optional[dict] = None
    current_hold: Optional[float] = None
    rec_multicoin: Optional[str] = None
    rec_single: Optional[str] = None
    error: Optional[str] = None

def numpy_to_python(obj):
    # Recursive conversion of numpy types to native types
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(i) for i in obj]
    elif isinstance(obj, (np.generic, np.ndarray)):
        return obj.item()
    else:
        return obj.item()

@api.get("/forecast/{coin}/{current_hold}", response_model=Response)
def get_optimal_prediction(coin: str, current_hold: float):
    try:
        forecast_multicoin, forecast_single, current_values = forecast(coin=coin)
        current_price = float(current_values)  # cast numpy.float32 to native float if needed

        # convert forecast dictionaries (if they contain numpy types)
        forecast_multicoin = numpy_to_python(forecast_multicoin)
        forecast_single = numpy_to_python(forecast_single)

        rec_multicoin = generate_recommendation(forecast_multicoin, current_price, current_hold)
        rec_single = generate_recommendation(forecast_single, current_price, current_hold)

        return Response(
            coin=coin,
            forecast_multicoin=forecast_multicoin,
            forecast_coin=forecast_single,
            current_hold=current_hold,
            rec_multicoin=rec_multicoin,
            rec_single=rec_single
        )

    except Exception as e:
        return Response(error=str(e))
