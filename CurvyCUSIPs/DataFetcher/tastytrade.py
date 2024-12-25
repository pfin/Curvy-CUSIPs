import asyncio
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import ujson as json
from functools import reduce


from typing import Dict, List, Optional, Set, Literal
from datetime import datetime, time
from pandas.tseries.offsets import BDay

from tastytrade import Session, DXLinkStreamer
from tastytrade.dxfeed import Quote, Candle
from tastytrade.instruments import Future, FutureOption


def get_futures_streamer_symbols(tasty_session: Session, symbols: str) -> Dict[str, str]:
    if len(symbols) == 0:
        return {}
    product_codes = [symbol[1:][:-2] for symbol in symbols]
    symbol_to_streamer_symbol = {
        contract.symbol: contract.streamer_symbol for contract in Future.get_futures(tasty_session, product_codes=product_codes)
    }
    return {streamer_symbol: symbol for symbol, streamer_symbol in symbol_to_streamer_symbol.items() if symbol in symbols}


def get_futures_options_streamer_symbols(tasty_session: Session, symbols: str) -> Dict[str, str]:
    if len(symbols) == 0:
        return {}
    return {fo.streamer_symbol: fo.symbol for fo in FutureOption.get_future_options(session=tasty_session, symbols=symbols)}


def get_historical_data(
    tasty_session: Session,
    start_date: datetime,
    end_date: datetime,
    symbols: List[str],
    interval: Optional[str] = "1d",
    extended_trading_hours: Optional[bool] = True,
    timeout: Optional[int] = 10,
    cols_to_return: Optional[str] = ["close"],
) -> pd.DataFrame:

    def _process_single_candle(
        candle: Candle,
    ) -> Dict[str, str | int | float]:
        return {
            "index": candle.index,
            "streamer_symbol": str(candle.event_symbol).split("{")[0],
            "time": candle.time,
            "timeString": datetime.fromtimestamp(candle.time / 1000.0).strftime("%Y-%m-%d : %H:%M:%S"),
            "count": candle.count,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "vwap": candle.vwap,
            "bid_volume": candle.bid_volume,
            "ask_volume": candle.ask_volume,
            "implied_volatility": candle.imp_volatility,
            "open_interest": candle.open_interest,
        }

    futures_streamer_symbols = []
    futures_options_streamer_symbols = []
    other_streamer_symbols = []

    for symbol in symbols:
        if "/" in symbol:
            if " " in symbol:
                futures_options_streamer_symbols.append(symbol)
            else:
                futures_streamer_symbols.append(symbol)
        else:
            other_streamer_symbols.append(symbol)

    futures_streamer_symbol_dict = get_futures_streamer_symbols(tasty_session, futures_streamer_symbols)
    futures_options_streamer_symbol_dict = get_futures_options_streamer_symbols(tasty_session, futures_options_streamer_symbols)

    async def wrapper():
        async with DXLinkStreamer(tasty_session) as streamer:
            subs = list(futures_streamer_symbol_dict.keys()) + list(futures_options_streamer_symbol_dict.keys()) + other_streamer_symbols
            await streamer.subscribe_candle(
                symbols=subs,
                interval=interval,
                start_time=start_date - BDay(1),
                extended_trading_hours=extended_trading_hours,
            )

            historical_data: Dict[str, List[Dict[str, float | str]]] = {}

            async def get_next_event():
                async for event in streamer.listen(Candle):
                    return event
                return None

            async def unsubscribe_all():
                unsubscribe_tasks = [streamer.unsubscribe(Candle, sub) for sub in subs]
                await asyncio.gather(*unsubscribe_tasks)

            try:
                while True:
                    try:
                        event_result = await asyncio.wait_for(get_next_event(), timeout=timeout)
                        if event_result is None:
                            print("Stream closed, exiting.")
                            break

                        processed = _process_single_candle(event_result)
                        if processed["time"] > datetime.combine(end_date, time.max).timestamp() * 1000:
                            continue

                        if processed["streamer_symbol"] in historical_data:
                            historical_data[processed["streamer_symbol"]].append(processed)
                        else:
                            historical_data[processed["streamer_symbol"]] = [processed]

                    except asyncio.TimeoutError:
                        print(f"No data received for {timeout} seconds, exiting.")
                        break
                    except Exception as e:
                        print(f"error: {e}")
                        break

            finally:
                await unsubscribe_all()

        return historical_data

    results = asyncio.run(wrapper())

    dfs = []
    for streamer_symbol, event_results in results.items():
        formatted_results = []
        for event_result in event_results:
            curr_dict = {
                "Date": datetime.strptime(event_result["timeString"], "%Y-%m-%d : %H:%M:%S"),
            }
            for col in cols_to_return:
                curr_dict[f"{streamer_symbol}_{col}"] = event_result[col]
            formatted_results.append(curr_dict)
        dfs.append(pd.DataFrame(formatted_results))

    df = reduce(lambda left, right: pd.merge(left, right, on="Date", how="outer"), dfs)
    df = df.rename(columns=futures_streamer_symbol_dict | futures_options_streamer_symbol_dict)
    return df
