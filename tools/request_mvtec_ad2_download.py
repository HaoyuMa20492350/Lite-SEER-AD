from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.data.mvtec_ad2_request import (
    build_opener,
    build_submission,
    download_archive,
    download_archive_ranged,
    download_category_archives,
    fetch_form,
    redact_email,
    select_category_archive_links,
    select_dataset_archive_link,
    submit_form,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or explicitly submit the official MVTec AD 2 download "
            "request. Submission requires personal details and consent."
        )
    )
    parser.add_argument("--first-name", default="")
    parser.add_argument("--last-name", default="")
    parser.add_argument("--email", default="")
    parser.add_argument("--job-title", default="")
    parser.add_argument("--accept-non-commercial", action="store_true")
    parser.add_argument("--newsletter", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--archive-out",
        default="",
        help=(
            "After submission, prefer the official full-dataset archive and "
            "download it to this path. Omit to inspect the returned links only."
        ),
    )
    parser.add_argument(
        "--category-archives-dir",
        default="",
        help=(
            "Download the eight official per-category archives to this "
            "directory. This is usually faster than the full archive."
        ),
    )
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument(
        "--range-workers",
        type=int,
        default=1,
        help=(
            "Use this many resumable HTTP byte ranges for --archive-out. "
            "Values above 1 enable parallel ranged download."
        ),
    )
    parser.add_argument(
        "--out",
        default=(
            "tables/mvtec_ad2_feature_first_readiness/"
            "download_request_status.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.archive_out and args.category_archives_dir:
        raise ValueError(
            "Choose either --archive-out or --category-archives-dir."
        )
    opener = build_opener()
    form = fetch_form(opener)
    public_fields = [field["name"] for field in form["fields"]]
    report = {
        "official_form_reachable": True,
        "required_user_fields": [
            "First Name",
            "Last Name",
            "E-Mail-Adresse",
            "Non Commercial Use",
        ],
        "optional_user_fields": ["Job Title", "Optincampus"],
        "available_form_fields": public_fields,
        "newsletter_default": False,
        "submission_requested": bool(args.submit),
        "submitted": False,
    }
    if args.submit:
        payload = build_submission(
            form,
            first_name=args.first_name,
            last_name=args.last_name,
            email=args.email,
            job_title=args.job_title,
            accept_non_commercial=args.accept_non_commercial,
            newsletter=args.newsletter,
        )
        result = submit_form(opener, form, payload)
        links = list(result["archive_links"])
        report.update(
            {
                "submitted": True,
                "email": redact_email(args.email),
                "newsletter": bool(args.newsletter),
                "response_bytes": result["response_bytes"],
                "final_page_host": (
                    urllib.parse.urlparse(result["final_url"]).hostname or ""
                ),
                "archive_link_count": len(links),
                "archive_hosts": sorted(
                    {
                        urllib.parse.urlparse(link).hostname or ""
                        for link in links
                    }
                ),
            }
        )
        if args.archive_out:
            archive_link = select_dataset_archive_link(links)
            report["selected_archive_name"] = urllib.parse.unquote(
                urllib.parse.urlparse(archive_link).path.rsplit("/", 1)[-1]
            )
            downloader = (
                download_archive_ranged
                if args.range_workers > 1
                else download_archive
            )
            if args.range_workers > 1:
                report.update(
                    downloader(
                        opener,
                        archive_link,
                        Path(args.archive_out),
                        workers=args.range_workers,
                    )
                )
            else:
                report.update(
                    downloader(opener, archive_link, Path(args.archive_out))
                )
        elif args.category_archives_dir:
            category_links = select_category_archive_links(links)
            report["selected_category_archives"] = sorted(category_links)
            report["downloads"] = download_category_archives(
                opener,
                category_links,
                Path(args.category_archives_dir),
                workers=args.download_workers,
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
