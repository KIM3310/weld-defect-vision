from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_static_site_is_cloudflare_and_adsense_ready() -> None:
    adsense_client = "ca-pub-4973160293737562"
    ads_txt_record = "google.com, pub-4973160293737562, DIRECT, f08c47fec0942fa0\n"

    assert (ROOT / "site/ads.txt").read_text() == ads_txt_record

    wrangler = json.loads((ROOT / "wrangler.jsonc").read_text())
    assert wrangler["name"] == "weld-defect-vision"
    assert wrangler["pages_build_output_dir"] == "./site"

    makefile = (ROOT / "Makefile").read_text()
    assert (
        "npx --yes wrangler@latest pages deploy site --project-name weld-defect-vision"
        in makefile
    )

    for relative in ["site/index.html", "site/privacy.html", "site/terms.html"]:
        html = (ROOT / relative).read_text()
        assert f'name="google-adsense-account" content="{adsense_client}"' in html

    loader = (
        "https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"
        f"?client={adsense_client}"
    )
    assert loader not in (ROOT / "site/index.html").read_text()
    for relative in [
        "site/guide.html",
        "site/architecture.html",
        "site/verification.html",
    ]:
        assert loader in (ROOT / relative).read_text()
    for relative in [
        "site/publisher.html",
        "site/privacy.html",
        "site/terms.html",
    ]:
        assert loader not in (ROOT / relative).read_text()

    robots = (ROOT / "site/robots.txt").read_text()
    sitemap = (ROOT / "site/sitemap.xml").read_text()
    assert "Sitemap: https://weld-defect-vision.pages.dev/sitemap.xml" in robots
    for route in [
        "guide.html",
        "architecture.html",
        "verification.html",
        "publisher.html",
        "privacy.html",
        "terms.html",
    ]:
        assert f"https://weld-defect-vision.pages.dev/{route}" in sitemap
