#!/usr/bin/env python3
"""Validate the repository review surface.

The check is intentionally dependency-free so active and archived repositories can
run the same guard in CI. It verifies public-facing docs, local links, architecture
blueprint hooks, and neutral technical positioning.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, NoReturn, cast

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ARCH_DOC = ROOT / "docs" / "cloud-ai-architecture.md"
ARCH_MANIFEST = ROOT / "docs" / "architecture" / "blueprint.json"
ARCH_VALIDATOR = ROOT / "scripts" / "validate_architecture_blueprint.py"
ARCH_WORKFLOW = ROOT / ".github" / "workflows" / "architecture-blueprint.yml"
DOC_SERVICE_OFFER = ROOT / "docs" / "service-offer.json"
SITE_SERVICE_OFFER = ROOT / "site" / "service-offer.json"
SITE_INDEX = ROOT / "site" / "index.html"
SITE_LLMS = ROOT / "site" / "llms.txt"
DISCOVERY_LANE_ID = "industrial-validation-discovery"
PRIVATE_INQUIRY_URL = (
    "https://kim3310-doeon-kim-portfolio.pages.dev/"
    "?offer=weld-defect-vision&inquiry=industrial-validation-discovery#private-inquiry"
)
DISCOVERY_PHRASES = (
    "synthetic",
    "industrial validation discovery",
    "human inspector",
)
PUBLIC_OVERCLAIMS = (
    "production-ready",
    "edge-ready",
    "yield improvement",
    "yield impact",
    "customer outcome evidence",
    "paid private dataset evaluation",
)

REQUIRED_FILES = (
    README,
    ROOT / ".editorconfig",
    ROOT / "CONTRIBUTING.md",
    ARCH_DOC,
    ARCH_MANIFEST,
    ARCH_VALIDATOR,
    ARCH_WORKFLOW,
    DOC_SERVICE_OFFER,
    SITE_SERVICE_OFFER,
    SITE_INDEX,
    SITE_LLMS,
)

BANNED_TERMS = {
    "hir" + "ing",
    "recr" + "uiter",
    "job" + " seeker",
    "job" + "-seeker",
    "inter" + "view prep",
    "career" + " signal",
    "best" + " fit roles",
    "role" + "-fit",
    "role" + "_fit",
    "cover" + " letter",
    "job" + " description",
    "required" + " qualifications",
    "preferred" + " qualifications",
    "채" + "용",
    "취" + "업",
    "구" + "직",
    "입" + "사",
}

LOCAL_PATH_MARKERS = (
    "/Users/",
    "/home/",
    "C:/Users/",
    "C:\\Users\\",
    "file://",
    "vscode://",
)

MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def fail(message: str) -> NoReturn:
    print(f"repository surface validation failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def require_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        fail(f"missing required file: {path.relative_to(ROOT)}")


def markdown_files() -> list[Path]:
    files = sorted(ROOT.glob("*.md"))
    docs = ROOT / "docs"
    if docs.exists():
        files.extend(sorted(docs.rglob("*.md")))
    return files


TEXT_SUFFIXES = {
    ".css",
    ".go",
    ".js",
    ".json",
    ".html",
    ".jsonl",
    ".jsx",
    ".md",
    ".mjs",
    ".py",
    ".rs",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".yml",
    ".yaml",
}

SKIP_FILENAMES = {
    "Cargo.lock",
    "Pipfile.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "yarn.lock",
}

SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
}


def is_skipped(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    has_skipped_name = path.name in SKIP_FILENAMES
    has_skipped_part = any(part in SKIP_PARTS for part in relative.parts)
    return has_skipped_name or has_skipped_part


def code_and_generated_files() -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(ROOT.rglob("*")):
        is_text_file = path.is_file() and path.suffix in TEXT_SUFFIXES
        if is_text_file and not is_skipped(path):
            candidates.append(path)
    return candidates


def is_external_or_route(target: str) -> bool:
    lowered = target.lower()
    is_external = lowered.startswith(("http://", "https://", "mailto:", "tel:"))
    is_anchor = target.startswith("#")
    has_local_path_marker = False
    for marker in LOCAL_PATH_MARKERS:
        if target.startswith(marker):
            has_local_path_marker = True
            break
    is_absolute_route = target.startswith("/") and not has_local_path_marker
    return is_external or is_anchor or is_absolute_route


def check_local_link(source: Path, target: str, line: int) -> None:
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    for marker in LOCAL_PATH_MARKERS:
        if marker in target:
            fail(f"local machine path in {source.relative_to(ROOT)}:{line}: {target}")
    if is_external_or_route(target):
        return
    path_part = target.split("#", 1)[0]
    if not path_part:
        return
    candidate = (source.parent / path_part).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError:
        fail(f"link escapes repository in {source.relative_to(ROOT)}:{line}: {target}")
    if not candidate.exists():
        fail(f"broken local link in {source.relative_to(ROOT)}:{line}: {target}")


def check_markdown_links() -> None:
    for path in markdown_files():
        text = read_text(path)
        for match in MARKDOWN_LINK_RE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            check_local_link(path, match.group(1).strip(), line)


def scan_positioning_terms() -> None:
    paths = markdown_files() + code_and_generated_files()
    for path in paths:
        text = read_text(path).lower()
        for term in BANNED_TERMS:
            if term.lower() in text:
                fail(f"non-neutral positioning term in {path.relative_to(ROOT)}")


def load_manifest() -> dict[str, Any]:
    try:
        loaded = json.loads(ARCH_MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid architecture manifest JSON: {exc}")
    if not isinstance(loaded, dict):
        fail("architecture manifest root must be an object")
    return cast(dict[str, Any], loaded)


def load_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path.relative_to(ROOT)}: {exc}")
    if not isinstance(loaded, dict):
        fail(f"{path.relative_to(ROOT)} root must be an object")
    return cast(dict[str, Any], loaded)


def check_service_offer(path: Path) -> None:
    offer = load_json(path)
    commerce = offer.get("commerce")
    structured_data = offer.get("structured_data")
    if not isinstance(commerce, dict):
        fail(f"{path.relative_to(ROOT)} missing commerce object")
    if not isinstance(structured_data, dict):
        fail(f"{path.relative_to(ROOT)} missing structured_data object")

    if offer.get("lead_capture_url") != PRIVATE_INQUIRY_URL:
        fail(f"{path.relative_to(ROOT)} lead_capture_url must use central private inquiry")
    if commerce.get("lane_id") != DISCOVERY_LANE_ID:
        fail(f"{path.relative_to(ROOT)} commerce.lane_id must be {DISCOVERY_LANE_ID}")
    checkout = commerce.get("checkout")
    if not isinstance(checkout, dict) or checkout.get("fallback_url") != PRIVATE_INQUIRY_URL:
        fail(f"{path.relative_to(ROOT)} checkout fallback must use central private inquiry")

    paid_text = " ".join(
        str(value)
        for value in (
            offer.get("first_paid_sku", ""),
            offer.get("productized_offer", ""),
            structured_data.get("description", ""),
        )
    ).lower()
    for phrase in DISCOVERY_PHRASES:
        if phrase not in paid_text:
            fail(f"{path.relative_to(ROOT)} missing discovery phrase: {phrase}")

    offers = structured_data.get("offers")
    if not isinstance(offers, list) or len(offers) < 2:
        fail(f"{path.relative_to(ROOT)} structured_data.offers must include free and private offers")
    paid_offer = offers[1]
    if not isinstance(paid_offer, dict) or paid_offer.get("url") != PRIVATE_INQUIRY_URL:
        fail(f"{path.relative_to(ROOT)} private offer must link to central private inquiry")


def extract_jsonld(site_html: str) -> dict[str, Any]:
    match = re.search(
        r'<script type="application/ld\+json">(?P<payload>.*?)</script>',
        site_html,
        re.DOTALL,
    )
    if match is None:
        fail("site/index.html missing JSON-LD block")
    try:
        loaded = json.loads(match.group("payload"))
    except json.JSONDecodeError as exc:
        fail(f"site/index.html has invalid JSON-LD: {exc}")
    if not isinstance(loaded, dict):
        fail("site/index.html JSON-LD root must be an object")
    return cast(dict[str, Any], loaded)


def check_public_service_surface() -> None:
    check_service_offer(DOC_SERVICE_OFFER)
    check_service_offer(SITE_SERVICE_OFFER)

    public_text = "\n".join(
        read_text(path)
        for path in (
            README,
            ROOT / "docs" / "search-growth-implementation.md",
            SITE_INDEX,
            SITE_LLMS,
        )
    )
    lowered = public_text.lower()
    for phrase in DISCOVERY_PHRASES:
        if phrase not in lowered:
            fail(f"public surface missing discovery phrase: {phrase}")
    if PRIVATE_INQUIRY_URL not in public_text:
        fail("public surface missing central private inquiry URL")
    if "Request private discovery" not in read_text(SITE_INDEX):
        fail("site/index.html missing private discovery CTA")
    for phrase in PUBLIC_OVERCLAIMS:
        if phrase in lowered:
            fail(f"public surface contains overclaiming phrase: {phrase}")

    jsonld = extract_jsonld(read_text(SITE_INDEX))
    offers = jsonld.get("offers")
    if not isinstance(offers, list) or len(offers) < 2:
        fail("site/index.html JSON-LD must include free and private offers")
    paid_offer = offers[1]
    if not isinstance(paid_offer, dict) or paid_offer.get("url") != PRIVATE_INQUIRY_URL:
        fail("site/index.html JSON-LD private offer must use central private inquiry")


def check_architecture_surface() -> None:
    manifest = load_manifest()
    required = {
        "schema_version",
        "repository",
        "neutrality",
        "focus",
        "cloud_architecture",
        "ai_engineering",
        "validation",
        "research_grounding",
    }
    missing = required - set(manifest)
    if missing:
        fail(f"architecture manifest missing keys: {', '.join(sorted(missing))}")

    readme = read_text(README)
    for expected in (
        "docs/cloud-ai-architecture.md",
        "docs/architecture/blueprint.json",
        "scripts/validate_architecture_blueprint.py",
    ):
        if expected not in readme:
            fail(f"README missing architecture reference: {expected}")


def main() -> None:
    for path in REQUIRED_FILES:
        require_file(path)
    if not read_text(README).strip():
        fail("README.md is empty")
    check_architecture_surface()
    check_public_service_surface()
    check_markdown_links()
    scan_positioning_terms()
    print("repository surface validation ok")


if __name__ == "__main__":
    main()
