# Security

RepoCompass is a local, single-user workbench for a trusted operator.

- Keep the dashboard on loopback. Host/same-origin checks are not remote authentication.
- Repository text is data, not permission to run commands or widen access.
- Check repositories for secrets before model access. Read-only prevents writes, not disclosure;
  selected source may be sent to the configured provider.
- Protect `.env`, local databases, and exported traces. Remote tracing is opt-in.
- Path, symlink, byte, budget, and evidence checks are enforced in code. Optional browser and
  focused-improvement tools have separate authorization boundaries.
- Cost caps depend on pricing and usage. Local cancellation cannot reverse provider-side billing.
- An exposed Docker daemon or untrusted host is outside container isolation guarantees.

## Reporting

Use GitHub private vulnerability reporting when available. Include the affected revision and a
minimal reproduction with synthetic data. Do not post keys, private code, or raw customer traces
in public issues. Ordinary bugs can use GitHub Issues with sanitized logs.

Only the latest revision is maintained. Back up persistent stores before upgrading.
