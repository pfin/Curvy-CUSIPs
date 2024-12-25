import asyncio
import sys
import urllib.parse
from datetime import datetime
from functools import reduce
from typing import Any, Callable, Optional, TypeAlias, List, Tuple, Dict

import httpx
import tqdm
import tqdm.asyncio
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay, CustomBusinessDay
from termcolor import colored

sys.path.insert(0, "../")
from CurvyCUSIPs.utils.ShelveDBWrapper import ShelveDBWrapper

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


def github_headers(path: str):
    return {
        "authority": "raw.githubusercontent.com",
        "method": "GET",
        "path": path,
        "scheme": "https",
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "no-cache",
        "dnt": "1",
        "pragma": "no-cache",
        "priority": "u=0, i",
        "sec-ch-ua": '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    }


async def fetch_from_github(
    client: httpx.AsyncClient,
    url: str,
    max_retries: Optional[int] = 3,
    backoff_factor: Optional[int] = 1,
    process_data_func: Optional[Callable[[JSON], Any]] = None,
    uid: Optional[Any] = None,
):
    retries = 0
    headers = github_headers(urllib.parse.urlparse(url).path)
    try:
        while retries < max_retries:
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                res_json = response.json()
                if process_data_func:
                    try:
                        if uid:
                            return uid, process_data_func(res_json)
                        return process_data_func(res_json)
                    except Exception as e:
                        raise ValueError(f"ERROR IN JSON DATA PROCESSING FUNC: {e}")
                if uid:
                    return uid, res_json
                return res_json
            except httpx.HTTPStatusError as e:
                if response.status_code == 404:
                    print(colored(f"404 when fetching {url}: {e}", "red"))
                    if uid:
                        return uid, None
                    return None
                # print(colored(f"Bad Status when fetching {url}: {e}", "red"))
                retries += 1
                wait_time = backoff_factor * (2 ** (retries - 1))
                # print(colored(f"Throttled when fetching {url}. Waiting for {wait_time} seconds before retrying. Error: {e}"), "red")
                await asyncio.sleep(wait_time)
            except Exception as e:
                # print(colored(f"Error when fetching {url}: {e}", "red"))
                retries += 1
                wait_time = backoff_factor * (2 ** (retries - 1))
                # print(colored(f"Throttled when fetching {url}. Waiting for {wait_time} seconds before retrying. Error: {e}", "red"))
                await asyncio.sleep(wait_time)
    except:
        print(colored(f"Max Retries Reached for {url}", "red"))
        if uid:
            return uid, None
        return None


def setup_s490_vol_cube_db(dates: List[datetime]):
    urls = [f"https://raw.githubusercontent.com/yieldcurvemonkey/VolCube420/refs/heads/main/{dt.strftime("%Y-%m-%d")}.json" for dt in dates]

    async def build_tasks(client: httpx.AsyncClient):
        process_vol_cube_github = lambda json_data: {strike_offsets: records for strike_offsets, records in json_data.items()}
        tasks = [
            fetch_from_github(client=client, url=url, process_data_func=process_vol_cube_github, uid=url.split("/")[-1].split(".")[0]) for url in urls
        ]
        return await tqdm.asyncio.tqdm.gather(*tasks, desc="FETCHING VOL CUBE DATA FROM GITHUB...")

    async def fetch_all():
        async with httpx.AsyncClient() as client:
            all_data = await build_tasks(client=client)
            return all_data

    return asyncio.run(fetch_all()) 


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "init":
        vol_cube_db = ShelveDBWrapper("../db/s490_swaption_vol_cube", create=True)
        vol_cube_db.open()
        bdates = [
            ts.to_pydatetime()
            for ts in pd.date_range(
                start=datetime(2024, 1, 1), end=datetime.today(), freq=CustomBusinessDay(calendar=USFederalHolidayCalendar())
            )
        ]
        vol_cubes: List[Tuple[str, Dict[str, List[Dict[str, float]]]]] = setup_s490_vol_cube_db(bdates)
        for date_str, vol_cube in vol_cubes:
            try:
                vol_cube_db.set(date_str, vol_cube)
                print(colored(f"Wrote {date_str} Vol CUBE to DB", "green"))
            except Exception as e:
                print(colored(f"Failed to write {date_str} Vol CUBE to DB: {e}", "red"))

    else:
        vol_cube_db = ShelveDBWrapper("../db/s490_swaption_vol_cube")
        vol_cube_db.open()
        most_recent_db_dt = datetime.strptime(max(vol_cube_db.keys()), "%Y-%m-%d")
        bday_offset = ((datetime.today() - BDay(1)) - most_recent_db_dt).days
        if bday_offset > 0:
            bdates = [
                ts.to_pydatetime()
                for ts in pd.date_range(
                    start=most_recent_db_dt, end=datetime.today(), freq=CustomBusinessDay(calendar=USFederalHolidayCalendar())
                )
            ]
            vol_cubes: List[Tuple[str, Dict[str, List[Dict[str, float]]]]] = setup_s490_vol_cube_db(bdates)
            for date_str, vol_cube in vol_cubes:
                try:
                    vol_cube_db.set(date_str, vol_cube)
                    print(colored(f"Wrote {date_str} Vol CUBE to DB", "green"))
                except Exception as e:
                    print(colored(f"Failed to write {date_str} Vol CUBE to DB: {e}", "red"))

    most_recent_db_dt = datetime.strptime(max(vol_cube_db.keys()), "%Y-%m-%d")
    bday_offset = ((datetime.today() - BDay(1)) - most_recent_db_dt).days
    if bday_offset > 0:
        print(colored(f"Most recemt entry: {most_recent_db_dt} - DB is not to date - ping yieldcurvemonkey@gmail.com for update", "yellow"))
