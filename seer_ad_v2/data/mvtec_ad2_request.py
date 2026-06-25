from __future__ import annotations

import copy
import http.cookiejar
import json
import re
import shutil
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from seer_ad_v2.data.mvtec_ad2_discovery import MVTEC_AD2_CATEGORIES


DATASET_PAGE = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2"
FORM_METADATA_URL = (
    "https://software.mvtec.com/acton/openapi/form/v1/43208/"
    "4a0ddcb8-1b65-41fd-b8db-cd4b404a728d?noStyle=1"
)
FORM_SUBMIT_URL = "https://software.mvtec.com/acton/forms/userSubmit.jsp"
NON_COMMERCIAL_TEXT = (
    "I am aware that the datasets are for non-commercial use only."
)
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".7z")
FULL_ARCHIVE_NAMES = (
    "mvtec_ad_2.tar.gz",
    "mvtec-ad-2.tar.gz",
    "mvtec_ad2.tar.gz",
    "mvtec-ad2.tar.gz",
)


class FormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.action = ""
        self.inputs: list[dict[str, str]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        values = {key: value or "" for key, value in attrs}
        if tag == "form":
            self.action = values.get("action", "")
        elif tag == "input":
            self.inputs.append(values)


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag != "a":
            return
        values = {key: value or "" for key, value in attrs}
        if values.get("href"):
            self.links.append(values["href"])


def parse_form_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    template = str(payload.get("processedTemplate", ""))
    parser = FormParser()
    parser.feed(template)
    fields = []
    hidden = {}
    for item in parser.inputs:
        name = item.get("name", "")
        if not name:
            continue
        field_type = item.get("type", "text").lower()
        if field_type == "hidden":
            hidden[name] = item.get("value", "")
        else:
            fields.append(
                {
                    "name": name,
                    "type": field_type,
                    "value": item.get("value", ""),
                }
            )
    return {
        "action": urllib.parse.urljoin(FORM_METADATA_URL, parser.action),
        "fields": fields,
        "hidden": hidden,
        "invalid_domain": bool(
            (payload.get("formProperties") or {}).get("invalidDomain")
        ),
    }


def validate_email(value: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value.strip()))


def build_submission(
    form: dict[str, Any],
    *,
    first_name: str,
    last_name: str,
    email: str,
    job_title: str = "",
    accept_non_commercial: bool,
    newsletter: bool = False,
) -> dict[str, str]:
    if not first_name.strip() or not last_name.strip():
        raise ValueError("First and last name are required.")
    if not validate_email(email):
        raise ValueError("A valid email address is required.")
    if not accept_non_commercial:
        raise ValueError(
            "Explicit non-commercial-use acceptance is required."
        )
    payload = {
        str(key): str(value)
        for key, value in (form.get("hidden") or {}).items()
    }
    payload.update(
        {
            "First Name": first_name.strip(),
            "Last Name": last_name.strip(),
            "E-Mail-Adresse": email.strip(),
            "Job Title": job_title.strip(),
            "Non Commercial Use": NON_COMMERCIAL_TEXT,
            "ao_refurl": DATASET_PAGE,
            "ao_target": DATASET_PAGE,
        }
    )
    if newsletter:
        payload["Optincampus"] = "true"
    else:
        payload.pop("Optincampus", None)
    return payload


def extract_archive_links(html: str, base_url: str) -> list[str]:
    parser = LinkParser()
    parser.feed(html)
    links = []
    for raw_link in parser.links:
        link = urllib.parse.urljoin(base_url, raw_link)
        path = urllib.parse.urlparse(link).path.lower()
        if any(path.endswith(suffix) for suffix in ARCHIVE_SUFFIXES):
            links.append(link)
    return list(dict.fromkeys(links))


def select_dataset_archive_link(links: list[str]) -> str:
    official_links = [link for link in links if is_official_download_url(link)]
    by_name: dict[str, list[str]] = {}
    for link in official_links:
        name = urllib.parse.unquote(
            urllib.parse.urlparse(link).path.rsplit("/", 1)[-1]
        ).lower()
        by_name.setdefault(name, []).append(link)
    for name in FULL_ARCHIVE_NAMES:
        matches = by_name.get(name, [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Multiple official full archives named {name}.")
    if len(official_links) == 1:
        return official_links[0]
    names = ", ".join(sorted(by_name)) or "none"
    raise ValueError(
        "Could not identify one official full MVTec AD 2 archive; "
        f"found: {names}."
    )


def select_category_archive_links(links: list[str]) -> dict[str, str]:
    official_links = [link for link in links if is_official_download_url(link)]
    by_name: dict[str, list[str]] = {}
    for link in official_links:
        name = urllib.parse.unquote(
            urllib.parse.urlparse(link).path.rsplit("/", 1)[-1]
        ).lower()
        by_name.setdefault(name, []).append(link)
    selected = {}
    for category in MVTEC_AD2_CATEGORIES:
        name = f"{category}.tar.gz"
        matches = by_name.get(name, [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected one official archive named {name}; "
                f"found {len(matches)}."
            )
        selected[category] = matches[0]
    return selected


def redact_email(value: str) -> str:
    local, separator, domain = value.partition("@")
    if not separator:
        return "***"
    visible = local[:1]
    return f"{visible}***@{domain}"


def is_official_download_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    return bool(
        parsed.scheme == "https"
        and (
            host == "mvtec.com"
            or host.endswith(".mvtec.com")
            or host == "mydrive.ch"
            or host.endswith(".mydrive.ch")
        )
    )


def build_opener(
    cookies: http.cookiejar.CookieJar | None = None,
) -> urllib.request.OpenerDirector:
    cookies = cookies or http.cookiejar.CookieJar()
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cookies)
    )


def clone_opener(
    opener: urllib.request.OpenerDirector,
) -> urllib.request.OpenerDirector:
    source = next(
        (
            handler.cookiejar
            for handler in opener.handlers
            if isinstance(handler, urllib.request.HTTPCookieProcessor)
        ),
        None,
    )
    cookies = http.cookiejar.CookieJar()
    if source is not None:
        for cookie in source:
            cookies.set_cookie(copy.copy(cookie))
    return build_opener(cookies)


def fetch_form(
    opener: urllib.request.OpenerDirector | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    opener = opener or build_opener()
    request = urllib.request.Request(
        FORM_METADATA_URL,
        headers={
            "Referer": DATASET_PAGE,
            "Origin": "https://www.mvtec.com",
            "User-Agent": "Lite-SEER-AD/1.0",
        },
    )
    with opener.open(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    form = parse_form_metadata(payload)
    if form["invalid_domain"] or not form["fields"]:
        raise RuntimeError("The official download form could not be loaded.")
    return form


def submit_form(
    opener: urllib.request.OpenerDirector,
    form: dict[str, Any],
    payload: dict[str, str],
    timeout: int = 60,
) -> dict[str, Any]:
    action = str(form.get("action") or FORM_SUBMIT_URL)
    if urllib.parse.urlparse(action).netloc != "software.mvtec.com":
        raise ValueError(f"Refusing unexpected form action: {action}")
    request = urllib.request.Request(
        action,
        data=urllib.parse.urlencode(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": DATASET_PAGE,
            "Origin": "https://www.mvtec.com",
            "User-Agent": "Lite-SEER-AD/1.0",
        },
        method="POST",
    )
    with opener.open(request, timeout=timeout) as response:
        final_url = response.geturl()
        html = response.read().decode("utf-8", errors="replace")
    return {
        "final_url": final_url,
        "archive_links": extract_archive_links(html, final_url),
        "response_bytes": len(html.encode("utf-8")),
    }


def download_archive(
    opener: urllib.request.OpenerDirector,
    url: str,
    path: Path,
    timeout: int = 120,
) -> dict[str, Any]:
    if not is_official_download_url(url):
        raise ValueError(f"Refusing non-official download URL: {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f"{path.name}.part")
    partial_path.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "Referer": DATASET_PAGE,
            "User-Agent": "Lite-SEER-AD/1.0",
        },
    )
    bytes_written = 0
    try:
        with opener.open(request, timeout=timeout) as response:
            final_url = response.geturl()
            if not is_official_download_url(final_url):
                raise ValueError(
                    "Refusing redirected non-official download URL: "
                    f"{final_url}"
                )
            with partial_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    bytes_written += len(chunk)
        partial_path.replace(path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise
    return {
        "archive_path": str(path.resolve()),
        "archive_bytes": bytes_written,
        "download_host": urllib.parse.urlparse(final_url).hostname or "",
    }


def split_byte_ranges(total_bytes: int, workers: int) -> list[tuple[int, int]]:
    if total_bytes < 1:
        raise ValueError("total_bytes must be positive")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    workers = min(workers, total_bytes)
    base, remainder = divmod(total_bytes, workers)
    ranges = []
    start = 0
    for index in range(workers):
        length = base + (1 if index < remainder else 0)
        end = start + length - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


def probe_archive_size(
    opener: urllib.request.OpenerDirector,
    url: str,
    timeout: int = 60,
) -> int:
    if not is_official_download_url(url):
        raise ValueError(f"Refusing non-official download URL: {url}")
    request = urllib.request.Request(
        url,
        headers={
            "Referer": DATASET_PAGE,
            "User-Agent": "Lite-SEER-AD/1.0",
            "Range": "bytes=0-0",
        },
    )
    with opener.open(request, timeout=timeout) as response:
        final_url = response.geturl()
        if not is_official_download_url(final_url):
            raise ValueError(
                f"Refusing redirected non-official download URL: {final_url}"
            )
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes 0-0/(\d+)", content_range)
        if response.status != 206 or not match:
            raise RuntimeError(
                "Official archive endpoint did not honor a byte-range probe."
            )
        response.read(1)
    return int(match.group(1))


def download_archive_ranged(
    opener: urllib.request.OpenerDirector,
    url: str,
    path: Path,
    *,
    workers: int = 8,
    timeout: int = 120,
) -> dict[str, Any]:
    total_bytes = probe_archive_size(opener, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == total_bytes:
        return {
            "archive_path": str(path.resolve()),
            "archive_bytes": total_bytes,
            "download_host": urllib.parse.urlparse(url).hostname or "",
            "download_workers": workers,
            "ranged_download": True,
            "reused_complete_archive": True,
        }
    ranges = split_byte_ranges(total_bytes, workers)

    def download_segment(
        segment_opener: urllib.request.OpenerDirector,
        index: int,
        start: int,
        end: int,
    ) -> Path:
        segment = path.with_name(f"{path.name}.part.{index:03d}")
        expected = end - start + 1
        existing = segment.stat().st_size if segment.is_file() else 0
        if existing > expected:
            segment.unlink()
            existing = 0
        if existing == expected:
            return segment
        request_start = start + existing
        request = urllib.request.Request(
            url,
            headers={
                "Referer": DATASET_PAGE,
                "User-Agent": "Lite-SEER-AD/1.0",
                "Range": f"bytes={request_start}-{end}",
            },
        )
        with segment_opener.open(request, timeout=timeout) as response:
            final_url = response.geturl()
            if not is_official_download_url(final_url):
                raise ValueError(
                    "Refusing redirected non-official download URL: "
                    f"{final_url}"
                )
            content_range = response.headers.get("Content-Range", "")
            expected_range = f"bytes {request_start}-{end}/{total_bytes}"
            if response.status != 206 or content_range != expected_range:
                raise RuntimeError(
                    "Unexpected ranged response: "
                    f"{response.status} {content_range!r}; "
                    f"expected 206 {expected_range!r}."
                )
            with segment.open("ab") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        if segment.stat().st_size != expected:
            raise RuntimeError(
                f"Segment {index} has {segment.stat().st_size} bytes; "
                f"expected {expected}."
            )
        return segment

    segments: list[Path | None] = [None] * len(ranges)
    openers = [clone_opener(opener) for _ in ranges]
    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = {
            executor.submit(
                download_segment,
                openers[index],
                index,
                start,
                end,
            ): index
            for index, (start, end) in enumerate(ranges)
        }
        for future in as_completed(futures):
            index = futures[future]
            segments[index] = future.result()

    partial = path.with_name(f"{path.name}.part")
    partial.unlink(missing_ok=True)
    with partial.open("wb") as destination:
        for segment in segments:
            if segment is None:
                raise RuntimeError("A ranged archive segment is missing.")
            with segment.open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    if partial.stat().st_size != total_bytes:
        raise RuntimeError(
            f"Merged archive has {partial.stat().st_size} bytes; "
            f"expected {total_bytes}."
        )
    partial.replace(path)
    for segment in segments:
        if segment is not None:
            segment.unlink()
    return {
        "archive_path": str(path.resolve()),
        "archive_bytes": total_bytes,
        "download_host": urllib.parse.urlparse(url).hostname or "",
        "download_workers": len(ranges),
        "ranged_download": True,
        "reused_complete_archive": False,
    }


def download_category_archives(
    opener: urllib.request.OpenerDirector,
    links: dict[str, str],
    directory: Path,
    *,
    workers: int = 4,
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    directory.mkdir(parents=True, exist_ok=True)

    def download(
        download_opener: urllib.request.OpenerDirector,
        category: str,
        url: str,
    ) -> dict[str, Any]:
        result = download_archive(
            download_opener,
            url,
            directory / f"{category}.tar.gz",
        )
        return {"category": category, **result}

    results = []
    openers = {
        category: clone_opener(opener)
        for category in links
    }
    with ThreadPoolExecutor(max_workers=min(workers, len(links))) as executor:
        futures = {
            executor.submit(
                download,
                openers[category],
                category,
                url,
            ): category
            for category, url in links.items()
        }
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda row: str(row["category"]))
