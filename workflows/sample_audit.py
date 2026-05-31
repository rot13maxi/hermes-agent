#!/usr/bin/env python3
"""
Sample workflow: API Security Audit

Exercises every workflow primitive:
  - phase()        checkpointed sections
  - run()          single agent with response_schema
  - run_all()      parallel agents with response_schema
  - if / for / list comprehension  real Python flow control
  - data threading  results from one phase -> context in the next

To run (once the workflow runtime exists):
    hermes workflow run workflows/sample_audit.py
"""

from hermes_workflow import phase, run, run_all


def workflow():
    """Full pipeline: discover -> audit -> triage -> report."""

    # ── Phase 1: Discovery ─────────────────────────────────────────────
    # Single agent finds all route files. Structured output enforced.
    with phase("discover"):
        files = run(
            goal="Find all API endpoint files under src/routes/ (or equivalent route directory). "
                 "Return the full paths of every file containing route definitions.",
            toolsets=["file", "terminal"],
            response_schema={
                "type": "array",
                "items": {"type": "string"},
            },
        )

        # Flow control — real Python, not YAML.
        if not files:
            # Early exit: nothing to audit.
            with phase("empty"):
                run(
                    goal="Create a brief note at docs/audit-empty.md stating no route files were found.",
                    toolsets=["file"],
                )
            return

    # ── Phase 2: Parallel Audit ────────────────────────────────────────
    # Fan out: one agent per discovered file.
    with phase("audit"):
        audits = run_all(
            tasks=[
                dict(
                    goal=f"Audit {f} for missing authentication checks. "
                         "For each endpoint/route, determine whether it has an auth guard "
                         "(middleware, decorator, or explicit check). "
                         "Classify severity of any unprotected endpoints.",
                    toolsets=["file"],
                    response_schema={
                        "type": "object",
                        "properties": {
                            "file": {"type": "string"},
                            "endpoints": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string"},
                                        "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "ALL"]},
                                        "has_auth": {"type": "boolean"},
                                        "auth_type": {"type": "string"},
                                        "line": {"type": "integer"},
                                    },
                                    "required": ["path", "has_auth"],
                                },
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "critical"],
                            },
                            "summary": {"type": "string"},
                        },
                        "required": ["file", "endpoints", "severity"],
                    },
                )
                for f in files
            ],
            max_concurrent=6,
        )

    # ── Phase 3: Triage ────────────────────────────────────────────────
    # Conditional: only run escalation if critical findings exist.
    critical_files = [a for a in audits if a.get("severity") in ("critical", "high")]

    if critical_files:
        with phase("escalation"):
            run(
                goal="Create a JIRA-style issue ticket for critical security findings.",
                context=f"""You have found {len(critical_files)} files with critical/high severity auth gaps:
{critical_files}

Write the issue ticket to docs/security-audit-findings.md with:
- A severity summary table
- Specific unprotected endpoints listed by file
- Recommended remediation steps
""",
                toolsets=["file"],
                response_schema={
                    "type": "object",
                    "properties": {
                        "ticket_path": {"type": "string"},
                        "total_critical_endpoints": {"type": "integer"},
                        "files_affected": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["ticket_path", "files_affected"],
                },
            )
    else:
        with phase("ack"):
            run(
                goal="Note that no critical findings were found in the audit.",
                context={"audits": audits},
                toolsets=["file"],
            )

    # ── Phase 4: Final Report ──────────────────────────────────────────
    # Synthesis: one agent compiles everything into a readable report.
    with phase("report"):
        run(
            goal="Compile all audit findings into a comprehensive markdown report.",
            context=f"""Audit results from {len(audits)} files:

{audits}

Produce a report at docs/api-security-audit-report.md with:
- Executive summary
- Findings by severity (critical, high, medium, low)
- Table of unprotected endpoints
- Remediation recommendations
""",
            toolsets=["file"],
            response_schema={
                "type": "object",
                "properties": {
                    "report_path": {"type": "string"},
                    "total_endpoints_audited": {"type": "integer"},
                    "unprotected_count": {"type": "integer"},
                    "severity_counts": {
                        "type": "object",
                        "properties": {
                            "critical": {"type": "integer"},
                            "high": {"type": "integer"},
                            "medium": {"type": "integer"},
                            "low": {"type": "integer"},
                        },
                    },
                },
                "required": ["report_path", "total_endpoints_audited"],
            },
        )
