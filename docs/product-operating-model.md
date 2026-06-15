# Product Operating Model

## Public Demo

- URL: https://kim3310.github.io/weld-defect-vision/
- Scope: credential-free synthetic demo surface for quick technical reader inspection.

Repository: `weld-defect-vision`
Last updated: 2026-06-03 KST

## Enterprise Product Position

Weld defect detection using YOLOv8 fine-tuning with FastAPI inference server

This repository is packaged as a concrete system surface, not a loose code sample. The enterprise value is a narrow proof that can be inspected, run, tested, and converted into a scoped implementation motion.

## Audience And Service Path

| Area | Position |
| --- | --- |
| Target users | Industrial inspection and quality teams |
| Service wedge | Inspection AI proof-of-concept package |
| System signal | Vision governance, model serving, validation notes |
| Delivery shape | Fixed-scope pilot, integration sprint, and handoff-ready operating pack |
| Expansion path | Add customer-specific adapters, policy controls, observability, and support SLAs after the pilot proves value |

## Enterprise Trust Boundary

- Keep credentials out of the repository and require environment-based configuration for live integrations.
- Treat generated screenshots, fixtures, and sample data as non-customer proof assets unless explicitly approved.
- Keep CI, repository-surface validation, architecture manifest checks, and secret scanning green before presenting the repo externally.
- Use the architecture blueprint as the source of truth for cloud, AI, data, and operational boundaries.
- Document any unsupported production assumption before a customer or evaluator sees the demo.

## Operating Model

| Function | Standard |
| --- | --- |
| Local verification | `make verify` |
| Runtime stack | Python |
| Demo readiness | README, architecture docs, and proof assets should explain the first five minutes of evaluation. |
| Support handoff | Capture setup, known limits, recovery steps, and customer-specific extension points before a production test. |
| Release discipline | Do not ship dependency mega-bumps, workflow edits, or demo URL changes without rerunning repository validators and project checks. |

## Debug And Reliability Checklist

1. Start with the README quickstart and the local verification command above.
2. Confirm `.github/workflows` checks match the local command path.
3. Confirm architecture and repository-surface validators pass after docs, workflow, or positioning changes.
4. Inspect public demos and homepage metadata before linking the repo from the portfolio.
5. Record any failing external dependency as an explicit operating limitation instead of hiding it.

## Service Next Step

Turn the repo into a customer-facing offer by pairing the proof surface with one discovery question, one measurable success metric, and one bounded follow-up path. The smallest viable package should be easy to verify and bounded enough to deliver without custom platform work.
