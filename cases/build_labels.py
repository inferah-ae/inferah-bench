"""Regenerate cases/labels.json from the generators and validate it against
cases/labels.schema.json. Deterministic: same code -> same file.

    .venv/bin/python -m cases.build_labels
"""
from __future__ import annotations

import json
import pathlib

import jsonschema

from cases.generators import build_all

HERE = pathlib.Path(__file__).parent


def main():
    labels = [lbl for _, _, lbl in build_all()]
    schema = json.loads((HERE / "labels.schema.json").read_text())
    jsonschema.validate(labels, schema)
    out = HERE / "labels.json"
    out.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n")
    n_explain = sum(1 for l in labels if l["expected"]["action"] == "explain")
    n_abstain = sum(1 for l in labels if l["expected"]["action"] == "abstain")
    print(f"wrote {out} — {len(labels)} cases "
          f"({n_explain} explain / {n_abstain} abstain / "
          f"{len(labels) - n_explain - n_abstain} no_driver), schema OK")


if __name__ == "__main__":
    main()
