"""Capture screenshots of the three Loan Desk tabs.

Assumes Streamlit is running on http://127.0.0.1:8766.
Writes docs/screenshots/{01_decision,02_book,03_rules}.png.
"""
from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
URL = "http://127.0.0.1:8767"
TABS = ["Make a decision", "The whole book", "Set the rules"]
FILES = ["01_decision.png", "02_book.png", "03_rules.png"]


def no_spinner(page, t=40):
    end = time.time() + t
    while time.time() < end:
        if page.locator('[data-testid="stStatusWidget"] :text("Running")').count() == 0 \
           and page.locator('[data-testid="stSpinner"]').count() == 0:
            return
        time.sleep(0.4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(viewport={"width": 1320, "height": 2200}, device_scale_factor=2)
        page = ctx.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_selector('button[role="tab"]', timeout=60_000, state="visible")
        no_spinner(page, 60)
        time.sleep(2)
        for i, (tab, fname) in enumerate(zip(TABS, FILES)):
            print(f"Tab: {tab}")
            page.locator('button[role="tab"]').nth(i).click(force=True, timeout=8_000)
            no_spinner(page, 60)
            time.sleep(3.5)
            page.screenshot(path=str(OUT / fname), full_page=True)
        for f in sorted(OUT.glob("*.png")):
            print(f"  {f.name}  {f.stat().st_size // 1024} KB")
        b.close()


if __name__ == "__main__":
    main()
