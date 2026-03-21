"""Streamlit dashboard for Weld Defect Vision.

Provides an interactive UI for:
- Uploading weld images for inspection
- Viewing detection results and severity scores
- Browsing inspection history
- Downloading HTML reports
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.classifier import DefectClassifier, DefectType
from app.models.severity import SeverityLevel, SeverityScorer
from app.preprocessing.pipeline import PreprocessingPipeline
from app.reporting.generator import ReportGenerator

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Weld Defect Vision",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
}
.badge-critical { color: #991b1b; font-weight: 700; }
.badge-high     { color: #92400e; font-weight: 700; }
.badge-medium   { color: #854d0e; font-weight: 700; }
.badge-low      { color: #166534; font-weight: 700; }
.badge-none     { color: #1e40af; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state["history"] = []
if "classifier" not in st.session_state:
    st.session_state["classifier"] = None
if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = None
if "scorer" not in st.session_state:
    st.session_state["scorer"] = None
if "reporter" not in st.session_state:
    st.session_state["reporter"] = None


# ---------------------------------------------------------------------------
# Service initialisation (cached)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading AI model...")
def load_services() -> tuple[DefectClassifier, PreprocessingPipeline, SeverityScorer, ReportGenerator]:
    classifier = DefectClassifier(model_path=None, demo_mode=False)
    pipeline = PreprocessingPipeline(target_size=(224, 224), apply_clahe=True)
    scorer = SeverityScorer()
    reporter = ReportGenerator()
    return classifier, pipeline, scorer, reporter


# ---------------------------------------------------------------------------
# Severity colour helpers
# ---------------------------------------------------------------------------

_SEVERITY_COLOURS = {
    SeverityLevel.CRITICAL: "#ef4444",
    SeverityLevel.HIGH: "#f97316",
    SeverityLevel.MEDIUM: "#eab308",
    SeverityLevel.LOW: "#22c55e",
    SeverityLevel.NONE: "#3b82f6",
}


def severity_colour(level: SeverityLevel) -> str:
    return _SEVERITY_COLOURS.get(level, "#94a3b8")


# ---------------------------------------------------------------------------
# Synthetic image generator (for demo tab)
# ---------------------------------------------------------------------------


def make_synthetic_image(defect: str, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.ones((256, 256, 3), dtype=np.uint8) * 160
    arr[90:160, :] = 200  # weld bead

    if defect == DefectType.POROSITY.value:
        for _ in range(rng.integers(10, 25)):
            cx, cy = rng.integers(10, 246), rng.integers(90, 159)
            r = int(rng.integers(3, 9))
            y_g, x_g = np.ogrid[-r : r + 1, -r : r + 1]
            m = x_g * x_g + y_g * y_g <= r * r
            y0, x0 = max(0, cy - r), max(0, cx - r)
            y1, x1 = min(256, cy + r + 1), min(256, cx + r + 1)
            patch = m[: y1 - y0, : x1 - x0]
            arr[y0:y1, x0:x1][patch] = 30

    elif defect == DefectType.CRACK.value:
        x = rng.integers(30, 226)
        for dy in range(80):
            jitter = int(rng.integers(-2, 3))
            xc = min(255, max(0, x + jitter))
            arr[90 + dy, max(0, xc - 1) : xc + 2] = 20
            x = xc

    elif defect == DefectType.UNDERCUT.value:
        arr[89:92, :] = 50
        arr[159:162, :] = 50

    elif defect == DefectType.SPATTER.value:
        for _ in range(rng.integers(20, 50)):
            sx, sy = rng.integers(5, 251), rng.integers(60, 200)
            r = int(rng.integers(1, 5))
            arr[max(0, sy - r) : sy + r, max(0, sx - r) : sx + r] = 240

    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("🔍 Weld Defect Vision")
st.caption("AI-powered welding inspection system for shipbuilding quality assurance")

classifier, pipeline, scorer, reporter = load_services()
mode_label = "Demo Mode" if classifier.demo_mode else "Model Mode"
st.sidebar.info(f"**Inference mode:** {mode_label}")

tab_inspect, tab_demo, tab_history, tab_about = st.tabs(
    ["📷 Inspect", "🧪 Demo", "📋 History", "ℹ️ About"]
)

# ============================================================
# TAB: Inspect
# ============================================================
with tab_inspect:
    st.subheader("Upload Weld Image for Inspection")

    col_upload, col_meta = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Choose a weld image (JPEG / PNG)",
            type=["jpg", "jpeg", "png"],
            key="uploader",
        )
    with col_meta:
        joint_id = st.text_input("Weld Joint ID", placeholder="e.g. J-2024-0042")
        notes = st.text_area("Inspector Notes", placeholder="Optional notes...", height=88)

    if uploaded is not None:
        raw_bytes = uploaded.read()
        pil_image = Image.open(io.BytesIO(raw_bytes))

        col_img, col_result = st.columns([1, 1])
        with col_img:
            st.image(pil_image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Running inspection..."):
            pre_result = pipeline.process_bytes(raw_bytes)
            detection = classifier.predict(pre_result.image)
            severity = scorer.score(detection, pre_result.image)
            report = reporter.generate(
                detection=detection,
                severity=severity,
                image_filename=uploaded.name,
                preprocessing_info=pre_result.to_dict(),
                inspector_notes=notes,
                weld_joint_id=joint_id,
            )
            st.session_state["history"].append(report)

        with col_result:
            st.markdown(f"**Report ID:** `{report.report_id}`")
            st.markdown(f"**Timestamp:** {report.timestamp}")

            defect_label = detection.defect_type.value.replace("_", " ").title()
            colour = severity_colour(severity.level)
            st.markdown(
                f"<h3 style='color:{colour}'>{'✗ ' if detection.is_defect else '✓ '}{defect_label}</h3>",
                unsafe_allow_html=True,
            )

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Confidence", f"{detection.confidence:.1%}")
            mc2.metric("Severity Score", f"{severity.score:.1f}/100")
            mc3.metric("Level", severity.level.value.upper())

            st.progress(min(1.0, severity.score / 100.0))

        st.subheader("Detection Details")
        det_col, sev_col = st.columns(2)

        with det_col:
            st.markdown("**Class Probabilities**")
            probs = detection.class_probabilities
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            for cls, prob in sorted_probs:
                label = cls.replace("_", " ").title()
                st.progress(prob, text=f"{label}: {prob:.1%}")

        with sev_col:
            st.markdown("**Severity Assessment**")
            st.markdown(f"- **Score:** {severity.score:.2f} / 100")
            st.markdown(f"- **Level:** `{severity.level.value.upper()}`")
            st.markdown(f"- **Acceptable:** {'Yes' if severity.is_acceptable else 'No'}")
            if severity.contributing_factors:
                st.markdown("**Contributing Factors:**")
                for factor in severity.contributing_factors:
                    st.markdown(f"  - {factor}")

        st.subheader("Recommended Action")
        st.info(severity.recommended_action)

        html_report = reporter.render_html(report, image_bytes=raw_bytes)
        st.download_button(
            label="Download HTML Report",
            data=html_report,
            file_name=f"{report.report_id}.html",
            mime="text/html",
        )

# ============================================================
# TAB: Demo
# ============================================================
with tab_demo:
    st.subheader("Synthetic Defect Demo")
    st.markdown(
        "Generate synthetic weld images with simulated defects to explore the system capabilities."
    )

    defect_options = [d.value for d in DefectType]
    selected_defect = st.selectbox(
        "Select defect type to simulate",
        defect_options,
        format_func=lambda x: x.replace("_", " ").title(),
    )
    seed = st.slider("Image seed (variation)", 0, 99, 0)

    if st.button("Generate & Inspect", type="primary"):
        synth_image = make_synthetic_image(selected_defect, seed=seed)
        buf = io.BytesIO()
        synth_image.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.image(synth_image, caption="Synthetic weld image", use_container_width=True)

        with st.spinner("Running inspection on synthetic image..."):
            pre = pipeline.process(synth_image)
            det = classifier.predict(pre.image)
            sev = scorer.score(det, pre.image)
            rep = reporter.generate(
                detection=det,
                severity=sev,
                image_filename=f"synthetic_{selected_defect}.png",
                weld_joint_id="DEMO",
            )
            st.session_state["history"].append(rep)

        with col_s2:
            colour = severity_colour(sev.level)
            defect_str = det.defect_type.value.replace("_", " ").title()
            st.markdown(
                f"<h3 style='color:{colour}'>{'✗ ' if det.is_defect else '✓ '}{defect_str}</h3>",
                unsafe_allow_html=True,
            )
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Confidence", f"{det.confidence:.1%}")
            dc2.metric("Severity", f"{sev.score:.1f}/100")
            dc3.metric("Level", sev.level.value.upper())
            st.info(sev.recommended_action)

        html_rep = reporter.render_html(rep, image_bytes=raw_bytes)
        st.download_button(
            label="Download HTML Report",
            data=html_rep,
            file_name=f"{rep.report_id}.html",
            mime="text/html",
            key="demo_dl",
        )

# ============================================================
# TAB: History
# ============================================================
with tab_history:
    st.subheader("Inspection History")
    history: list = st.session_state.get("history", [])

    if not history:
        st.info("No inspections yet. Upload an image or try the Demo tab.")
    else:
        summary = reporter.generate_batch_summary(history)
        hc1, hc2, hc3, hc4 = st.columns(4)
        hc1.metric("Total Inspections", summary["total"])
        hc2.metric("Defects Found", summary["defects_found"])
        hc3.metric("Pass Rate", f"{summary['pass_rate']:.1%}")
        hc4.metric("Critical", summary["by_severity"].get("critical", 0))

        st.markdown("---")
        for rep in reversed(history):
            colour = severity_colour(rep.severity.level)
            defect_str = rep.detection.defect_type.value.replace("_", " ").title()
            icon = "✗" if rep.detection.is_defect else "✓"
            with st.expander(
                f"{icon} {rep.report_id} — {defect_str} "
                f"({rep.severity.level.value.upper()}, {rep.timestamp})"
            ):
                st.json(rep.to_dict())

        if st.button("Clear History"):
            st.session_state["history"] = []
            st.rerun()

# ============================================================
# TAB: About
# ============================================================
with tab_about:
    st.subheader("About Weld Defect Vision")
    st.markdown(
        """
**Weld Defect Vision** is an AI-powered inspection system for detecting and classifying welding
defects in shipbuilding and industrial applications.

### Supported Defect Types

| Defect | ISO 6520-1 | Risk Level |
|--------|-----------|-----------|
| Crack | Group 1 | Critical |
| Incomplete Fusion | Group 4 | High |
| Undercut | Group 5 | Medium-High |
| Porosity | Group 2 | Medium |
| Overlap | Group 5 | Low-Medium |
| Spatter | Group 6 | Low |

### Architecture

- **Backbone**: ResNet-18 with ImageNet pre-training
- **Head**: Fine-tuned 2-layer classifier (256-dim)
- **Preprocessing**: CLAHE contrast enhancement in LAB colour space
- **Severity**: ISO 5817 / AWS D1.1 aligned scoring (0–100)
- **Reporting**: IIW-style JSON + HTML inspection reports

### Tech Stack

- Python 3.12 · PyTorch · torchvision · OpenCV
- FastAPI (inference API) · Streamlit (dashboard)
- ruff (linting) · mypy (type checking) · pytest (testing)

### Demo Mode

When no trained model checkpoint is available, the system operates in **demo mode**,
using image statistics (intensity distribution, contrast, pixel ratios) as heuristics
for classification. This allows full pipeline demonstration without training data.
"""
    )
