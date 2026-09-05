"""Readable edit decisions and local publishing notes, without cloud uploads."""
import csv
from pathlib import Path


def write_edit_package(timeline, job):
    job = Path(job)
    with (job / "edit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["shot", "source", "source_start_seconds", "source_end_seconds", "timeline_start_seconds",
                  "duration_seconds", "frames", "fps", "role", "selection_reason"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, shot in enumerate(timeline["shots"], 1):
            writer.writerow({"shot": index, "source": shot["source"], "source_start_seconds": shot["start"],
                             "source_end_seconds": shot["end"], "timeline_start_seconds": shot["timeline_in"],
                             "duration_seconds": shot["duration"], "frames": shot["frames"], "fps": timeline["fps"],
                             "role": shot["role"], "selection_reason": shot["selection_reason"]})
    notes = ["# Your FPV Sesh export", "", f"{timeline['duration']:.2f} seconds / {len(timeline['shots'])} moments / {timeline['fps']} fps.",
             "", "Open final_4k.mp4 for the landscape master. Selected social versions are in social/; each comes from the original flight intervals.",
             "Preview files are smaller editing copies. Use the final files for posting.",
             "", "Review the opening, each trick's exit, and any cropped version before posting. The app reports estimated motion patterns, not verified stunt names.",
             "", "edit.csv lists every source cut for manual finishing in another editor. timeline.json retains exact rational times and frame mappings.",
             "", "Suggested caption to customize", "", "FPV sesh — favorite lines from the flight. 🚁", ""]
    (job / "publish-notes.md").write_text("\n".join(notes), encoding="utf-8")
