# Reviewer Evidence Map - Weld Defect Vision

Updated: 2026-05-29

This document is the short path for a technical reviewer, engineering leader, product evaluator, or buyer who wants to understand what this repository proves without wandering through every file.

## One-Line Proof

**B2B industrial AI validation.** YOLOv8 defect workflow with model governance, serving, and operator-readable evidence.

## Audience and Commercial Angle

| Lens | Answer |
|---|---|
| Primary reviewer | Manufacturing quality teams, welding inspection groups, industrial AI, and edge deployment reviewers. |
| Technical signal | Can the project be explained, verified, bounded, and extended like a real product surface? |
| Buyer signal | Is there a narrow operational pain, a runnable proof path, and a risk-aware pilot shape? |
| Stack signal | Python, Docker |

## Seven-Minute Review Route

1. Read the README `Product and Review Surface` and `Reviewer Fast Path` sections.
2. Open `docs/monetization-playbook.md` to understand the buyer, offer ladder, and GTM hypothesis.
3. Run or inspect the strongest local quality gate below.
4. Inspect CI workflow definitions and test fixtures before deeper implementation review.
5. Check the risk boundaries so claims stay credible and not overextended.

## Verification Commands

| Purpose | Command |
|---|---|
| Test suite | `python -m pytest` |

## CI and Automation Surface

- .github/workflows/architecture-blueprint.yml
- .github/workflows/ci.yml
- .github/workflows/dependency-review.yml
- .github/workflows/export-onnx.yml
- .github/workflows/repository-health.yml
- .github/workflows/repository-surface.yml
- .github/workflows/secret-scan.yml

## Evidence Inventory

- pytest/ruff-style local verification path
- containerized delivery path
- pytest passes
- Model card exists
- API detection path is documented

## Commercialization Snapshot

| Offer | Pricing hypothesis |
|---|---|
| Inspection PoC | $5k-$15k PoC |
| Model validation study | $20k-$70k validation |
| Edge-serving readiness assessment | $3k-$15k/month model ops support |

## Risk Boundaries

- Human inspection required
- Site-specific validation
- No production acceptance without criteria

## Metrics That Matter

- mAP/recall by defect
- False-negative review
- Operator review time

## Review Verdict

This repository should be evaluated as part of the broader KIM3310 portfolio: it is strongest when the reviewer sees the link between a concrete implementation, a documented verification path, and an externally credible operating story.
