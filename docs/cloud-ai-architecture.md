# Cloud + AI Architecture Blueprint: weld-defect-vision

This blueprint is a neutral technical operating model for the repository. It describes the cloud architecture surface, AI engineering controls, and validation path without making external deployment or certification claims.

## Operating Model

- **Domain:** applied model pipelines and evidence-backed inference
- **Current proof surface:** Repository-local proof surface for applied model pipelines and evidence-backed inference, backed by Python service or lab runtime, Container build surface, Local compose environment.
- **Status:** active
- **Primary stack:** Python service or lab runtime, Container build surface, Local compose environment, GitHub Actions validation
- **Architecture axes:** cloud architecture, AI engineering, reliability, security, operator experience

## Cloud Architecture

Operating model: artifact registries, batch and online inference paths, edge/managed serving options, and model monitoring hooks.

### Deployment and control patterns

- Containerized runtime path suitable for repeatable local, staging, or managed service deployment
- Model serving envelope with artifact traceability, inference monitoring, and quality drift hooks

### Landing-zone controls

- identity boundary and least-privilege service access
- environment separation for local, staging, and managed runtime paths
- secret storage outside source and deterministic fallback for missing credentials
- observability hooks for logs, metrics, traces, and audit events
- rollback path for deployment, schema, and model changes

### Reliability controls

- bounded retries with explicit failure states
- health/readiness checks before operator-facing flows are trusted
- idempotent data or artifact writes where repeat execution is possible
- cost and quota guardrails for hosted services and model adapters

## AI Engineering

Operating model: dataset checks, model evaluation, explainability artifacts, serving boundaries, and human-readable confidence evidence.

### Engineering patterns

- Track dataset splits, model artifacts, inference outputs, and explainability evidence as first-class assets
- Pair quality metrics with failure modes, human review states, and rollback-ready serving boundaries
- Separate deterministic checks from model-generated output so the system remains testable without external credentials
- Capture prompts, inputs, outputs, and decision metadata as inspectable artifacts instead of hidden side effects
- Gate model-assisted actions with policy, confidence, and fallback states before they reach an operator path

### Evaluation controls

- deterministic fixtures for CI-safe verification
- golden output or schema checks for generated artifacts
- trace capture for prompts, tool calls, inputs, and outputs
- quality gates that fail closed when evidence is missing

### Model risk controls

- dataset leakage
- untracked model artifacts
- silent quality drift
- unsafe decision automation

## Architecture Map

| Layer | What must be explicit | Evidence to keep current |
| --- | --- | --- |
| Runtime | entrypoints, adapters, timeouts, retries | health checks, typed contracts, smoke tests |
| Data | schemas, freshness, retention, lineage | fixtures, migrations, export samples |
| AI | prompts, tools, retrieval, policies, evals | golden traces, scorecards, fallback cases |
| Cloud | identity, network, secrets, deploy target | IaC, workflow logs, environment notes |
| Operations | SLOs, incident flow, rollback, handoff | runbooks, audit events, release notes |

## Research Grounding

The repository is aligned with these research directions as design references, not as claims of equivalence:

- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization](https://arxiv.org/abs/1610.02391)
- [Guidelines for Human-AI Interaction](https://doi.org/10.1145/3290605.3300233)

## Validation

Run the repository-local architecture guard:

```bash
python3 scripts/validate_architecture_blueprint.py
```

The CI workflow `.github/workflows/architecture-blueprint.yml` runs the same check when the blueprint, validation script, or workflow changes.

## Extension Backlog

- Add or update architecture diagrams when runtime boundaries change.
- Keep AI evaluation fixtures close to the code path they validate.
- Promote cloud changes through reproducible IaC or documented deployment commands.
- Keep fallback behavior useful when hosted adapters, model providers, or external credentials are unavailable.
- Record operational assumptions as explicit contracts rather than hidden README prose.
