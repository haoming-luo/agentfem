# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only result inventory from manifest metadata; never opens field arrays."""
from pathlib import Path


def result_menu(record, path):
    metadata = record.get("metadata") or {}
    output = metadata.get("field_output") or {}
    plan = metadata.get("output_plan") or {}
    artifacts = record.get("artifacts") or {}
    files = []
    for name, value in artifacts.items():
        if not isinstance(value, str):
            continue
        selected = Path(value).expanduser()
        selected = selected if selected.is_absolute() else Path(path).parent / selected
        files.append({"name": name, "path": str(selected.resolve()), "exists": selected.is_file()})
    primary = (metadata.get("visualization") or {}).get("primary_artifact")
    recommended = artifacts.get(primary) or output.get("recommended_visualization_artifact")
    if recommended is None and plan:
        recommended = artifacts.get("field_history")
    histories = []
    for item in record.get("history_records", ()):
        interval = item.get("abscissa_range")
        if interval is None and item.get("abscissa"):
            interval = [item["abscissa"][0], item["abscissa"][-1]]
        histories.append({key: item.get(key) for key in (
            "name", "unit", "abscissa_name", "abscissa_unit", "sample_count", "value_shape")}
            | {"abscissa_range": interval})
    claims = (record.get("verification") or {}).get("claims") or ()
    return {"recommended_visualization": recommended,
        "geometry": output.get("geometry", (plan.get("field") or {}).get("configuration")),
        "frame_count": output.get("frame_count", plan.get("frame_count")),
        "frame_times": output.get("frame_times"),
        "saved_load_factors": plan.get("saved_load_factors"),
        "fields": [{key: item.get(key) for key in ("name", "location", "value_shape", "unit")}
            for item in record.get("field_records", ())],
        "histories": histories, "files": files,
        "missing_files": [item["name"] for item in files if not item["exists"]],
        "failed_requirements": [item for item in claims if item.get("status") == "failed"],
        "source": "recorded_metadata; missing values are unknown; no field arrays loaded"}


def format_menu(menu):
    lines = []
    for item in menu["fields"]:
        lines.append(f"  field {item['name']}: {item['location'] or 'unknown location'}; "
            f"shape={item['value_shape']}; unit={item['unit'] or 'unknown'}")
    for item in menu["histories"]:
        lines.append(f"  history {item['name']}: {item['sample_count']} samples; "
            f"{item['abscissa_name']}={item['abscissa_range']} {item['abscissa_unit'] or 'unknown unit'}; "
            f"value unit={item['unit'] or 'unknown'}")
    lines.append(f"  saved field frames: {menu['frame_count'] if menu['frame_count'] is not None else 'unknown'}")
    if menu['missing_files']:
        lines.append("  MISSING files: " + ", ".join(menu['missing_files']))
    return lines
