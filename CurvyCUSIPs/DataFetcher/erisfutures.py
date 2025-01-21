import asyncio
import re
import warnings
from datetime import datetime
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
import calendar
import httpx
import numpy as np
import pandas as pd
import requests
import scipy
import scipy.interpolate
from functools import reduce
from io import BytesIO
import tqdm
import tqdm.asyncio
import ssl
from pandas.errors import DtypeWarning
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay, CustomBusinessDay

from CurvyCUSIPs.DataFetcher.base import DataFetcherBase

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)

ERIS_TENORS = [
    "1D",
    "1W",
    "1M",
    "3M",
    "6M",
    "9M",
    "12M",
    "18M",
    "2Y",
    "3Y",
    "4Y",
    "5Y",
    "6Y",
    "7Y",
    "8Y",
    "9Y",
    "10Y",
    "12Y",
    "15Y",
    "20Y",
    "25Y",
    "30Y",
    "40Y",
    "50Y",
]


class ErisFuturesDataFetcher(DataFetcherBase):
    def __init__(
        self,
        global_timeout: int = 10,
        proxies: Optional[Dict[str, str]] = None,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
    ):
        super().__init__(
            global_timeout=global_timeout,
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            error_verbose=error_verbose,
        )
        self.eris_ftp_urls = "https://files.erisfutures.com/ftp"
        self.eris_ftp_headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "DNT": "1",
            "Host": "files.erisfutures.com",
            "Referer": "https://files.erisfutures.com/ftp/",
            "Sec-CH-UA": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        }

    async def _fetch_eris_ftp_files_helper(
        self,
        client: httpx.AsyncClient,
        date: datetime,
        workbook_type: Literal["EOD_ParCouponCurve_SOFR"],
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
    ):
        td = datetime.today() - date
        archives_path = f"archives/{date.year}/{date.month:02}-{calendar.month_name[date.month]}"
        file_name = f"Eris_{date.strftime('%Y%m%d')}_{workbook_type}.csv"
        if td.days < 90:
            eris_ftp_formatted_url = f"{self.eris_ftp_urls}/{file_name}"
        else:
            eris_ftp_formatted_url = f"{self.eris_ftp_urls}/{archives_path}/{file_name}"

        print(eris_ftp_formatted_url)
        retries = 0
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            while retries < max_retries:
                try:
                    async with client.stream(
                        method="GET",
                        url=eris_ftp_formatted_url,
                        headers=self.eris_ftp_headers,
                        follow_redirects=True,
                        timeout=self._global_timeout,
                    ) as response:
                        response.raise_for_status()
                        buffer = BytesIO()
                        async for chunk in response.aiter_bytes():
                            buffer.write(chunk)
                        buffer.seek(0)

                    return buffer, file_name

                except httpx.HTTPStatusError as e:
                    self._logger.error(f"ERIS FTP - Bad Status for {workbook_type}-{date}: {response.status_code}")
                    if response.status_code == 404:
                        return None, None

                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"ERIS FTP - Throttled. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self._logger.error(f"ERIS FTP - Error for {workbook_type}-{date}: {e}")
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"ERIS FTP - Throttled. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

            raise ValueError(f"ERIS FTP - Max retries exceeded for {workbook_type}-{date}")

        except Exception as e:
            self._logger.error(e)
            return None, None

    def _read_file(self, file_buffer: BytesIO, file_name: str) -> Tuple[Union[str, datetime], pd.DataFrame]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DtypeWarning)

            if file_name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_buffer)
            elif file_name.lower().endswith(".csv"):
                df = pd.read_csv(file_buffer, low_memory=False)
            else:
                return None

            try:
                datetime.strptime(file_name.split("_")[1], "%Y%m%d")
                key = datetime.strptime(file_name.split("_")[1], "%Y%m%d")
            except:
                key = file_name

            return key, df

    async def _fetch_and_read_eris_ftp_file(
        self,
        semaphore: asyncio.Semaphore,
        client: httpx.AsyncClient,
        date: datetime,
        workbook_type: Literal["EOD_ParCouponCurve_SOFR"],
    ):
        async with semaphore:
            buffer, file_name = await self._fetch_eris_ftp_files_helper(
                client=client, 
                date=date, 
                workbook_type=workbook_type
            )
            if not buffer or not file_name:
                return None, None

        key, df = await asyncio.to_thread(self._read_file, buffer, file_name)
        return key, df

    async def build_tasks(
        self,
        client: httpx.AsyncClient,
        dates: List[datetime],
        max_concurrent_tasks: int,
        workbook_type: Literal["EOD_ParCouponCurve_SOFR"],
    ):
        tasks = []
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        for date in dates:
            task = asyncio.create_task(
                self._fetch_and_read_eris_ftp_file(
                    semaphore=semaphore,
                    client=client,
                    date=date,
                    workbook_type=workbook_type,
                )
            )
            tasks.append(task)

        return await tqdm.asyncio.tqdm.gather(*tasks, desc="FETCHING ERIS FTP Files...")

    async def run_fetch_all(
        self,
        dates: List[datetime],
        max_concurrent_tasks: int,
        max_keepalive_connections: int,
        workbook_type: Literal["EOD_ParCouponCurve_SOFR"],
    ):
        # Configure httpx client with proper settings
        limits = httpx.Limits(
            max_connections=max_concurrent_tasks,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=30.0  # Added explicit keepalive expiry
        )
        
        # Configure transport with proper proxy settings if needed
        transport = None
        if self._httpx_proxies:
            proxies = {}
            for scheme, proxy_url in self._httpx_proxies.items():
                if scheme in ('http', 'https'):
                    proxies[f"{scheme}://"] = proxy_url
            if proxies:
                transport = httpx.AsyncHTTPTransport(
                    proxies=proxies,
                    retries=3,  # Added retry configuration
                    verify=False
                )
            
        async with httpx.AsyncClient(
            limits=limits,
            transport=transport,
            verify=False,
            timeout=httpx.Timeout(30.0),  # Added explicit timeout
            follow_redirects=True
        ) as client:
            all_data = await self.build_tasks(
                client=client,
                dates=dates,
                max_concurrent_tasks=max_concurrent_tasks,
                workbook_type=workbook_type,
            )
            return all_data

    def fetch_eris_ftp_timeseries(
        self,
        start_date: datetime,
        end_date: datetime,
        workbook_type: Literal["EOD_ParCouponCurve_SOFR"] = "EOD_ParCouponCurve_SOFR",
        max_concurrent_tasks: Optional[int] = 64,
        max_keepalive_connections: Optional[int] = 5,
        verbose: bool = False,
    ) -> Dict[datetime, pd.DataFrame]:
        """Fetch Eris futures data for a date range.
        
        Args:
            start_date: Start date to fetch data for
            end_date: End date to fetch data for
            workbook_type: Type of workbook to fetch
            max_concurrent_tasks: Maximum number of concurrent tasks
            max_keepalive_connections: Maximum number of keepalive connections
            verbose: Whether to print verbose output
            
        Returns:
            Dict mapping dates to DataFrames containing the futures data
        """
        bdates = pd.date_range(
            start=start_date,
            end=end_date,
            freq=CustomBusinessDay(calendar=USFederalHolidayCalendar())
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DtypeWarning)
            results: List[Tuple[str, pd.DataFrame]] = asyncio.run(
                self.run_fetch_all(
                    dates=bdates,
                    max_concurrent_tasks=max_concurrent_tasks,
                    max_keepalive_connections=max_keepalive_connections,
                    workbook_type=workbook_type,
                )
            )
            
            if results is None or len(results) == 0:
                if verbose:
                    print('"fetch_eris_ftp_timeseries" --- empty results')
                return {}

            return dict(results)
