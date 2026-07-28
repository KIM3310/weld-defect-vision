# Ad-Supported Resource and Aggregate Data Architecture

Repository: `weld-defect-vision`

## Public Resource Model

Free industrial vision validation model-card worksheet for weld-defect review boundaries.

- Audience: manufacturing AI validation teams
- Central resource: https://kim3310-doeon-kim-portfolio.pages.dev/resources/weld-defect-vision/
- Live system: https://weld-defect-vision.pages.dev/
- Advertising boundary: ads allowed only on public validation resources; image uploads, inference results, defect records, and dashboards are ad-free
- Current ad state: code-ready on the central resource; serving depends on Google AdSense site approval and consent policy.

## Readiness Utility

The central resource turns the repository architecture into a practical review checklist:

- **Architecture Summary:** Repository-local proof surface for applied model pipelines and evidence-backed inference, backed by Python service or lab runtime, Container build surface, Local compose environment.
- **Runtime And Data Flow:** Primary domain: applied model pipelines and evidence-backed inference.
- **Cloud Or Local Deployment Boundary:** Operating model: artifact registries, batch and online inference paths, edge/managed serving options, and model monitoring hooks
- **Deployment patterns:** Containerized runtime path suitable for repeatable local, staging, or managed service deployment Model serving envelope with artifact traceability, inference monitoring, and quality drift hooks
- **Control boundaries:** identity boundary and least-privilege service access environment separation for local, staging, and managed runtime paths secret storage outside source and deterministic fallback for missing credentials observability hooks for logs, metrics, traces, and audit events rollback path...

The checklist state remains in the visitor's browser and is not transmitted.

## Aggregate Data Boundary

- Data asset: anonymous aggregate industrial vision validation interest and worksheet usage counts
- Sensitivity class: industrial-high-trust
- Allowed events: `resource_view`, `resource_cta_click`, `architecture_doc_open`, `privacy_support_open`
- Prohibited fields: `raw_input`, `file`, `upload`, `url`, `referrer`, `title`, `user_id`, `session_id`, `ip_address`, `payment_detail`
- Consent defaults to off.
- DNT and Global Privacy Control fail closed.
- Events are reduced to repository, allowlisted event, public surface, and consent-policy version.
- Personal, sensitive, raw, event-level, or re-identifiable data is never offered for sale.

## Storage Path

```text
Public resource
  -> consent and privacy-signal gate
  -> Cloudflare Pages event API
  -> rate-limited daily aggregate counter
  -> public benchmark response
  -> Firebase public aggregate data mart
```

Cloudflare D1 holds operational counters. Firestore project `kim3310-free-tools` is the deny-by-default public aggregate data mart. Private inquiries remain isolated from telemetry.
