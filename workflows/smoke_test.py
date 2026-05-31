#!/usr/bin/env python3
"""
Smoke test workflow: minimal end-to-end test of phase(), run(), run_all().

Phases:
  1. hello — single agent says hi with structured output
  2. parallel — two agents work in parallel
  3. synthesize — combine results
"""

from hermes_workflow import phase, run, run_all


def workflow():
    # Phase 1: Single agent with structured output
    with phase("hello"):
        greeting = run(
            goal="Return a greeting for the name 'Alice'.",
            response_schema={
                "type": "object",
                "properties": {
                    "greeting": {"type": "string"},
                },
                "required": ["greeting"],
            },
        )

    # Phase 2: Parallel agents
    with phase("parallel"):
        results = run_all(
            tasks=[
                {
                    "goal": "List two interesting facts about Python programming.",
                    "response_schema": {
                        "type": "object",
                        "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
                        "required": ["facts"],
                    },
                },
                {
                    "goal": "List two interesting facts about Rust programming.",
                    "response_schema": {
                        "type": "object",
                        "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
                        "required": ["facts"],
                    },
                },
            ],
            max_concurrent=2,
        )

    # Phase 3: Synthesize
    with phase("synthesize"):
        run(
            goal="Write a short summary combining the results.",
            context=f"Greeting: {greeting}\nParallel results: {results}",
        )
