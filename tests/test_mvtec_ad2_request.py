from __future__ import annotations

import pytest

from seer_ad_v2.data.mvtec_ad2_request import (
    NON_COMMERCIAL_TEXT,
    build_submission,
    download_archive,
    extract_archive_links,
    is_official_download_url,
    parse_form_metadata,
    redact_email,
    select_category_archive_links,
    select_dataset_archive_link,
    split_byte_ranges,
)


def sample_form() -> dict:
    return {
        "processedTemplate": """
        <form method="POST" action="//software.mvtec.com/acton/forms/userSubmit.jsp">
          <input name="First Name" type="text" />
          <input name="Last Name" type="text" />
          <input name="E-Mail-Adresse" type="text" />
          <input name="Job Title" type="text" />
          <input name="Non Commercial Use" type="checkbox" value="yes" />
          <input name="Optincampus" type="checkbox" value="true" />
          <input name="ao_a" type="hidden" value="43208" />
          <input name="ao_f" type="hidden" value="form-id" />
        </form>
        """,
        "formProperties": {"invalidDomain": False},
    }


def test_parse_and_build_submission_requires_consent() -> None:
    form = parse_form_metadata(sample_form())
    assert form["action"] == (
        "https://software.mvtec.com/acton/forms/userSubmit.jsp"
    )
    assert form["hidden"]["ao_a"] == "43208"
    with pytest.raises(ValueError, match="non-commercial"):
        build_submission(
            form,
            first_name="A",
            last_name="B",
            email="a@example.com",
            accept_non_commercial=False,
        )

    payload = build_submission(
        form,
        first_name=" A ",
        last_name=" B ",
        email="a@example.com",
        accept_non_commercial=True,
    )
    assert payload["First Name"] == "A"
    assert payload["Non Commercial Use"] == NON_COMMERCIAL_TEXT
    assert "Optincampus" not in payload


def test_extract_archive_links_and_redaction() -> None:
    html = """
      <a href="/downloads/MVTec_AD_2.tar.gz">dataset</a>
      <a href="notes.html">notes</a>
      <a href="/downloads/MVTec_AD_2.tar.gz">duplicate</a>
    """
    assert extract_archive_links(html, "https://example.com/page") == [
        "https://example.com/downloads/MVTec_AD_2.tar.gz"
    ]
    assert redact_email("haoyu@example.com") == "h***@example.com"


def test_select_dataset_archive_prefers_full_package() -> None:
    links = [
        "https://www.mydrive.ch/download/mvtec_ad_2.tar.gz",
        "https://www.mydrive.ch/download/can.tar.gz",
        "https://www.mydrive.ch/download/fabric.tar.gz",
    ]
    assert select_dataset_archive_link(links) == links[0]


def test_select_dataset_archive_rejects_ambiguous_category_packages() -> None:
    with pytest.raises(ValueError, match="Could not identify"):
        select_dataset_archive_link(
            [
                "https://www.mydrive.ch/download/can.tar.gz",
                "https://www.mydrive.ch/download/fabric.tar.gz",
            ]
        )


def test_select_all_category_archives() -> None:
    links = [
        f"https://www.mydrive.ch/download/{category}.tar.gz"
        for category in (
            "can",
            "fabric",
            "fruit_jelly",
            "rice",
            "sheet_metal",
            "vial",
            "wallplugs",
            "walnuts",
        )
    ]
    selected = select_category_archive_links(links)
    assert selected["can"].endswith("/can.tar.gz")
    assert selected["walnuts"].endswith("/walnuts.tar.gz")


def test_split_byte_ranges_is_contiguous_and_complete() -> None:
    ranges = split_byte_ranges(11, 4)
    assert ranges == [(0, 2), (3, 5), (6, 8), (9, 10)]
    assert sum(end - start + 1 for start, end in ranges) == 11


def test_download_rejects_non_official_host(tmp_path) -> None:
    with pytest.raises(ValueError, match="non-official"):
        download_archive(
            object(),  # type: ignore[arg-type]
            "https://example.com/dataset.tar.gz",
            tmp_path / "dataset.tar.gz",
        )
    assert is_official_download_url(
        "https://downloads.mvtec.com/MVTec-AD2.tar.gz"
    )
    assert is_official_download_url(
        "https://www.mydrive.ch/share/MVTec-AD2.tar.gz"
    )
    assert not is_official_download_url(
        "http://downloads.mvtec.com/MVTec-AD2.tar.gz"
    )
    assert not is_official_download_url(
        "https://mvtec.com.example.org/MVTec-AD2.tar.gz"
    )
