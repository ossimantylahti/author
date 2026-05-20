#!/usr/bin/env python3
"""Hakee Odoon Suomen asiakasreferenssit sivuilta 1-19 ja tallentaa CSV/Excel-muotoon."""

from __future__ import annotations

import csv
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

BASE_URL = "https://www.odoo.com"
URL_TEMPLATE = "https://www.odoo.com/customers/country/finland-69/page/{page}"
TOTAL_PAGES = 19
CSV_OUTPUT = "odoo_suomi_referenssit.csv"
XLSX_OUTPUT = "odoo_suomi_referenssit.xlsx"
TIMEOUT_SECONDS = 20
REQUEST_DELAY_RANGE = (0.5, 1.0)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fi-FI,fi;q=0.9,en-US;q=0.8,en;q=0.7",
}

CARD_SELECTOR = "div.col-md-4.col-xl-3.col-12.mb-4"
INDUSTRY_SELECTOR = ".badge.mt-3.text-bg-secondary"

COLUMNS = [
    "yrityksen_nimi",
    "yrityksen_toimiala",
    "yrityksen_kotisivu",
    "referenssin_kuvausteksti",
    "lahdesivu",
    "odoo_referenssilinkki",
]


@dataclass
class ReferenceRow:
    yrityksen_nimi: str
    yrityksen_toimiala: str
    yrityksen_kotisivu: str
    referenssin_kuvausteksti: str
    lahdesivu: str
    odoo_referenssilinkki: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "yrityksen_nimi": self.yrityksen_nimi,
            "yrityksen_toimiala": self.yrityksen_toimiala,
            "yrityksen_kotisivu": self.yrityksen_kotisivu,
            "referenssin_kuvausteksti": self.referenssin_kuvausteksti,
            "lahdesivu": self.lahdesivu,
            "odoo_referenssilinkki": self.odoo_referenssilinkki,
        }


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def fetch_page_html(session: requests.Session, url: str) -> Optional[str]:
    try:
        response = session.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[VIRHE] Sivun haku epäonnistui: {url} -> {exc}")
        return None
    return response.text


def extract_internal_reference_link(card: Tag) -> str:
    for link in card.select("a[href]"):
        href = link.get("href", "")
        absolute = urljoin(BASE_URL, href)
        if urlparse(absolute).netloc.endswith("odoo.com") and "/customers/" in absolute:
            return absolute
    return ""


def extract_company_homepage(card: Tag) -> str:
    for link in card.select("a[href]"):
        href = link.get("href", "")
        absolute = urljoin(BASE_URL, href)
        parsed = urlparse(absolute)
        if parsed.scheme in {"http", "https"} and not parsed.netloc.endswith("odoo.com"):
            return absolute
    return ""


def parse_reference_cards(html: str, source_url: str) -> List[ReferenceRow]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select(CARD_SELECTOR)
    rows: List[ReferenceRow] = []

    for card in cards:
        name = ""
        for selector in ("h5", "h4", "h3", "h2", ".h5", ".h4"):
            node = card.select_one(selector)
            if node:
                name = clean_text(node.get_text(" ", strip=True))
                if name:
                    break
        if not name:
            link = card.select_one("a[href]")
            if link:
                name = clean_text(link.get_text(" ", strip=True))

        industry_node = card.select_one(INDUSTRY_SELECTOR)
        industry = clean_text(industry_node.get_text(" ", strip=True)) if industry_node else ""

        homepage = extract_company_homepage(card)
        internal_ref = extract_internal_reference_link(card)

        description = ""
        for selector in ("p", ".text-muted", ".o_text_overflow"):
            node = card.select_one(selector)
            if node:
                candidate = clean_text(node.get_text(" ", strip=True))
                if candidate and candidate != industry:
                    description = candidate
                    break

        rows.append(
            ReferenceRow(
                yrityksen_nimi=name,
                yrityksen_toimiala=industry,
                yrityksen_kotisivu=homepage,
                referenssin_kuvausteksti=description,
                lahdesivu=source_url,
                odoo_referenssilinkki=internal_ref,
            )
        )

    return rows


def save_as_csv(rows: List[ReferenceRow], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)


def save_as_excel_if_possible(rows: List[ReferenceRow], path: str) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("[INFO] pandas ei saatavilla, Excel-tallennus ohitetaan.")
        return

    try:
        pd.DataFrame([row.to_dict() for row in rows], columns=COLUMNS).to_excel(path, index=False)
        print(f"[OK] Excel tallennettu: {path}")
    except ImportError:
        print("[INFO] openpyxl ei saatavilla, Excel-tallennus ohitetaan.")


def scrape_all_pages() -> List[ReferenceRow]:
    all_rows: List[ReferenceRow] = []
    session = requests.Session()

    for page in range(1, TOTAL_PAGES + 1):
        url = URL_TEMPLATE.format(page=page)
        print(f"[INFO] Haetaan sivu {page}/{TOTAL_PAGES}: {url}")

        html = fetch_page_html(session, url)
        if html:
            page_rows = parse_reference_cards(html, url)
            all_rows.extend(page_rows)
            print(f"[INFO] Sivu {page}: {len(page_rows)} referenssiä")

        time.sleep(random.uniform(*REQUEST_DELAY_RANGE))

    return all_rows


def main() -> None:
    rows = scrape_all_pages()

    save_as_csv(rows, CSV_OUTPUT)
    print(f"[OK] CSV tallennettu: {CSV_OUTPUT}")

    save_as_excel_if_possible(rows, XLSX_OUTPUT)

    print(f"[VALMIS] Referenssejä yhteensä: {len(rows)}")


if __name__ == "__main__":
    main()
