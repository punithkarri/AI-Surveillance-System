from pathlib import Path
import pandas as pd

from utils.helpers import ensure_folder, get_timestamp_string
from config.settings import REPORT_OUTPUT_DIR, THREAT_CLASS_SCORES


def summarize_events(events):
    summary = {}
    for event in events:
        label = event.get("label") or event.get("object_name") or "unknown"
        threat = int(event.get("threat", event.get("confidence", 0)))
        if label not in summary:
            summary[label] = {"count": 1, "max_threat": threat, "risk": "HIGH" if label in THREAT_CLASS_SCORES else "LOW"}
        else:
            summary[label]["count"] += 1
            summary[label]["max_threat"] = max(summary[label]["max_threat"], threat)
    return summary


def export_csv_report(summary, alert_count, screenshot_count):
    rows = []
    for label, data in summary.items():
        rows.append({
            "Event": label,
            "Count": data["count"],
            "Max Threat": data["max_threat"],
            "Risk": data["risk"],
        })
    df = pd.DataFrame(rows)
    header = {
        "Report generated": [get_timestamp_string()],
        "Total alerts": [alert_count],
        "Screenshots": [screenshot_count],
    }
    header_df = pd.DataFrame(header)
    result = header_df.to_csv(index=False) + "\n" + df.to_csv(index=False)
    return result.encode("utf-8")


def export_txt_report(summary, alert_count, screenshot_count):
    lines = ["AI SURVEILLANCE REPORT", "=" * 40, f"Generated: {get_timestamp_string()}", ""]
    lines.append(f"Total alerts: {alert_count}")
    lines.append(f"Screenshots: {screenshot_count}")
    lines.append("")
    lines.append("DETECTED EVENTS:")
    lines.append("-")
    for label, data in summary.items():
        lines.append(f"{label.upper()}")
        lines.append(f"  Count: {data['count']}")
        lines.append(f"  Max Threat: {data['max_threat']}%")
        lines.append(f"  Risk: {data['risk']}")
        lines.append("")
    return "\n".join(lines).encode("utf-8")


def save_report_files(summary, alert_count, screenshot_count):
    report_dir = ensure_folder(REPORT_OUTPUT_DIR)
    timestamp = get_timestamp_string()
    csv_path = report_dir / f"surveillance_report_{timestamp}.csv"
    txt_path = report_dir / f"surveillance_report_{timestamp}.txt"
    csv_path.write_bytes(export_csv_report(summary, alert_count, screenshot_count))
    txt_path.write_bytes(export_txt_report(summary, alert_count, screenshot_count))
    return csv_path, txt_path
