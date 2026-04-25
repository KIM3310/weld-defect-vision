# Security Policy

## Supported Versions

Security fixes are applied to the default branch. Consumers should run the latest commit or latest tagged release when available.

## Reporting a Vulnerability

Do not open a public issue for suspected vulnerabilities. Use GitHub private vulnerability reporting if it is enabled for this repository, or contact the repository owner through their GitHub profile.

Please include:

- A clear description of the issue and affected component
- Reproduction steps or a minimal proof of concept
- Potential impact and any known mitigations
- Whether sensitive data, credentials, or model artifacts may be exposed

## Security Expectations

- Never commit API keys, dataset credentials, model registry tokens, or cloud credentials.
- Treat medical, industrial, and customer images as sensitive data.
- Keep dependency locks and CI verification current before release.
- Validate all uploaded files before model inference.
- Run local verification before merging:

```bash
python -m ruff check .
python -m pytest -q
```
