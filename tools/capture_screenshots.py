"""
Capture README screenshots by driving the app in a real browser.

Run against the deployed app or a local instance:

    python tools/capture_screenshots.py https://<app>.streamlit.app
    python tools/capture_screenshots.py http://localhost:8501

A fresh browser context is used with no stored credentials, so reaching the app
at all doubles as a check that it is publicly accessible.

This is a development tool. It is not imported by the app and is not part of the
test suite; playwright is deliberately not in requirements-dev.txt because it
pulls a browser download that CI does not need.
"""

import pathlib
import sys

from playwright.sync_api import sync_playwright

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "assets" / "screenshots"

VIEWPORT = {"width": 1600, "height": 1000}

# Streamlit renders progressively over a websocket; these waits are generous
# because a cold Community Cloud instance can take a while to wake.
LOAD_TIMEOUT_MS = 120_000
SETTLE_MS = 3_000
TRAIN_TIMEOUT_MS = 180_000

TARGET_COLUMN = "price"


def settle(page, ms=SETTLE_MS):
    """Let Streamlit finish its current rerun before touching the DOM again."""
    page.wait_for_timeout(ms)


def shot(page, name):
    """Save a full-page screenshot and report it."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    print(f"  saved {path.name} ({path.stat().st_size // 1024} KB)")


def open_tab(page, label):
    """Click a Streamlit tab by its visible label."""
    tab = page.get_by_role("tab", name=label)
    tab.click()
    settle(page)


def main(url):
    # On Streamlit Community Cloud the app runs inside an iframe served at
    # `/~/+/`; the bare URL returns only the Cloud wrapper. Targeting the inner
    # document directly also keeps the Cloud viewer badge out of the shots.
    if "streamlit.app" in url and not url.endswith("/~/+"):
        url = url + "/~/+/"

    print(f"Capturing {url}")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        # A brand-new context carries no cookies or session, so if the app is
        # private this will land on a login page instead of the app.
        context = browser.new_context(viewport=VIEWPORT)
        page = context.new_page()

        page.goto(url, wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)

        # The app's own H1 only appears once Streamlit has connected and run.
        page.wait_for_selector("text=AI Regression Studio", timeout=LOAD_TIMEOUT_MS)
        settle(page, 5_000)

        if "login" in page.url or "/-/auth" in page.url:
            raise SystemExit(
                f"App is NOT public - redirected to {page.url}\n"
                "Set sharing to public in the Streamlit Cloud app settings."
            )

        print(f"  reachable without sign-in (final url: {page.url})")

        # --- Tab 1: upload / demo data ---
        shot(page, "01-data-upload")

        page.get_by_role("button", name="Load demo dataset").click()
        settle(page, 5_000)
        shot(page, "02-demo-loaded")

        # --- Tab 2: explorer, pick target and features ---
        open_tab(page, "Data Explorer")

        # Target defaults to a price-like column via the naming heuristic, so it
        # usually needs no interaction. Select the features explicitly.
        page.get_by_text("Smart Feature Selection").click()
        settle(page, 4_000)
        shot(page, "03-data-explorer")

        # --- Tab 3: training ---
        open_tab(page, "Model Training")
        settle(page, 3_000)
        shot(page, "04-preprocessing-steps")

        page.get_by_role("button", name="Launch Model Training").click()
        page.wait_for_selector("text=Training Complete", timeout=TRAIN_TIMEOUT_MS)
        settle(page, 3_000)
        shot(page, "05-training-complete")

        # --- Tab 4: results ---
        open_tab(page, "Results Dashboard")
        settle(page, 4_000)
        shot(page, "06-results-dashboard")

        # Global SHAP importance sits behind a button so it is not computed on
        # every rerun.
        try:
            page.get_by_role("button", name="Compute SHAP importance").click()
            page.wait_for_selector("text=Mean impact on predictions", timeout=TRAIN_TIMEOUT_MS)
            settle(page, 3_000)
            shot(page, "07-shap-global-importance")
        except Exception as exc:
            print(f"  skipped global SHAP: {exc}")

        # --- Tab 5: prediction + per-prediction SHAP ---
        open_tab(page, "Prediction Lab")
        settle(page, 3_000)

        page.get_by_role("button", name="Make Prediction").click()
        settle(page, 8_000)
        shot(page, "08-prediction-lab")

        context.close()
        browser.close()

    print(f"\nDone. Screenshots in {OUT_DIR}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1].rstrip("/"))
