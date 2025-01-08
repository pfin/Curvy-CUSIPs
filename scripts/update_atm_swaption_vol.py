import asyncio
import sys
import urllib.parse
from datetime import datetime
from functools import reduce
from typing import Any, Callable, Optional, TypeAlias

import httpx
import tqdm
import tqdm.asyncio
from pandas.tseries.offsets import BDay
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
                        return process_data_func(res_json)
                    except Exception as e:
                        raise ValueError(f"ERROR IN JSON DATA PROCESSING FUNC: {e}")
                return res_json
            except httpx.HTTPStatusError as e:
                if response.status_code == 404:
                    print(colored(f"404 when fetching {url}: {e}", "red"))
                    return None
                print(colored(f"Bad Status when fetching {url}: {e}", "red"))
                retries += 1
                wait_time = backoff_factor * (2 ** (retries - 1))
                print(colored(f"Throttled when fetching {url}. Waiting for {wait_time} seconds before retrying. Error: {e}"), "red")
                await asyncio.sleep(wait_time)
            except Exception as e:
                print(colored(f"Error when fetching {url}: {e}", "red"))
                retries += 1
                wait_time = backoff_factor * (2 ** (retries - 1))
                print(colored(f"Throttled when fetching {url}. Waiting for {wait_time} seconds before retrying. Error: {e}", "red"))
                await asyncio.sleep(wait_time)
    except:
        print(colored(f"Max Retries Reached for {url}", "red"))
        return None


def setup_s490_atm_vol_db():
    urls = [
        f"https://raw.githubusercontent.com/yieldcurvemonkey/VolCube420/refs/heads/main/atm_timeseries/{yr}.json"
        for yr in [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017]
    ]

    async def build_tasks(client: httpx.AsyncClient):
        # process_atm_vol_github = lambda json_data: {
        #     datetime.strptime(date_str, "%Y-%m-%d"): pd.DataFrame(records) for date_str, records in json_data.items()
        # }
        process_atm_vol_github = lambda json_data: {date_str: records for date_str, records in json_data.items()}
        tasks = [fetch_from_github(client=client, url=url, process_data_func=process_atm_vol_github) for url in urls]
        return await tqdm.asyncio.tqdm.gather(*tasks, desc="FETCHING ATM VOL FROM GITHUB...")

    async def fetch_all():
        async with httpx.AsyncClient() as client:
            all_data = await build_tasks(client=client)
            return all_data

    return reduce(lambda a, b: {**a, **b}, asyncio.run(fetch_all()))


if __name__ == "__main__":
    # kind of ineff b/c we rewrite db every update
    atm_vol_db = ShelveDBWrapper("../db/s490_swaption_atm_vol", create=True)
    atm_vol_db.open()
    atm_vol_dict = setup_s490_atm_vol_db()
    print(atm_vol_dict)
    for date_str, records in atm_vol_dict.items():
        atm_vol_db.set(date_str, records)
    print(colored("Wrote ATM Vols to DB", "green"))

    most_recent_db_dt = datetime.strptime(max(atm_vol_db.keys()), "%Y-%m-%d")
    bday_offset = ((datetime.today() - BDay(1)) - most_recent_db_dt).days
    if bday_offset > 0:
        print(colored(f"Most recemt entry: {most_recent_db_dt} - DB is not to date - ping yieldcurvemonkey@gmail.com for update", "yellow"))
