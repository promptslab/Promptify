import os
from datetime import datetime
from email.utils import formatdate, parsedate_to_datetime

import requests
from appdirs import user_cache_dir


def download(url, destination_file):
    headers = {}

    path = user_cache_dir("promptify")

    if not os.path.isdir(path):
        os.makedirs(path)

    destination_file = os.path.join(path, destination_file)

    if os.path.exists(destination_file):
        mtime = os.path.getmtime(destination_file)
        headers["if-modified-since"] = formatdate(mtime, usegmt=True)

    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    if response.status_code == requests.codes.not_modified:
        return

    if response.status_code == requests.codes.ok:
        with open(destination_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=1048576):
                f.write(chunk)

        if last_modified := response.headers.get("last-modified"):
            new_mtime = parsedate_to_datetime(last_modified).timestamp()
            os.utime(destination_file, times=(datetime.now().timestamp(), new_mtime))
    return destination_file
