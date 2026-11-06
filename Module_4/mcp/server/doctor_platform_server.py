"""The booking platform's own MCP server (stdio). Doctors upload their schedule, patients' bookings are stored next to it.

DEMO SERVER, BOTH FILES ARE PLAIN CSV so you can open them:
    schedule.csv   what the doctors uploaded (doctor_id, doctor, date, time, status, description)
    bookings.csv   what the agent wrote (booking_id, doctor_id, doctor, date, time, patient_name, phone, reason)

This server is deliberately naive, like a first version a team would ship:
  - book_slot writes whatever it is given. It does not check the schedule.
  - It exposes the staff tools (list_bookings, cancel_booking) to every client, because the staff dashboard
    uses the same server. The public agent is handed the whole toolbox.
  - get_availability returns the doctors' description cells unchanged.
"""
import csv
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

HERE = Path(__file__).resolve().parent
# A video can start the server with its own files: doctor_platform_server.py --schedule <path> --bookings <path>
SCHEDULE = Path(sys.argv[sys.argv.index("--schedule") + 1]) if "--schedule" in sys.argv else HERE / "schedule.csv"
BOOKINGS = Path(sys.argv[sys.argv.index("--bookings") + 1]) if "--bookings" in sys.argv else HERE / "bookings.csv"
BOOKING_FIELDS = ["booking_id", "doctor_id", "doctor", "date", "time", "patient_name", "phone", "reason"]

mcp = FastMCP("doctor-platform")


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_bookings(rows: list[dict]) -> None:
    with open(BOOKINGS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BOOKING_FIELDS)
        w.writeheader()
        w.writerows(rows)


def matches(row: dict, doctor: str) -> bool:
    """Match on the doctor id (D01) or on part of the name (rao)."""
    d = doctor.strip().lower()
    return d in (row["doctor_id"].lower(), row["doctor"].lower()) or (d != "" and d in row["doctor"].lower())


@mcp.tool()
def get_availability(doctor: str, date: str = "") -> str:
    """Show the free appointment slots of a doctor. `doctor` is a doctor id (D01) or a name (Rao). `date` is optional (YYYY-MM-DD)."""
    taken = {(b["doctor_id"], b["date"], b["time"]) for b in read_rows(BOOKINGS)}
    lines = []
    for r in read_rows(SCHEDULE):
        if not matches(r, doctor) or (date and r["date"] != date):
            continue
        if (r["doctor_id"], r["date"], r["time"]) in taken:
            continue
        note = f" | note: {r['description']}" if r["description"] else ""
        lines.append(f"{r['doctor_id']} {r['doctor']} | {r['date']} {r['time']} | free{note}")
    return "\n".join(lines) or "No free slots found."


@mcp.tool()
def book_slot(doctor_id: str, date: str, time: str, patient_name: str, phone: str, reason: str = "") -> str:
    """Book an appointment: writes one row to bookings.csv. Needs the doctor id (D01), date (YYYY-MM-DD), time (HH:MM), the patient's name and phone."""
    rows = read_rows(BOOKINGS)
    doctor = next((r["doctor"] for r in read_rows(SCHEDULE) if r["doctor_id"] == doctor_id), doctor_id)
    new_id = f"B{len(rows) + 1:03d}"
    while any(r["booking_id"] == new_id for r in rows):
        new_id = f"B{int(new_id[1:]) + 1:03d}"
    rows.append({"booking_id": new_id, "doctor_id": doctor_id, "doctor": doctor, "date": date, "time": time,
                 "patient_name": patient_name, "phone": phone, "reason": reason})
    write_bookings(rows)
    return f"Booked {new_id}: {doctor} on {date} at {time} for {patient_name}."


@mcp.tool()
def list_bookings(doctor: str = "", date: str = "") -> str:
    """Staff dashboard: list the bookings with the patients' details. Optional filters: doctor (id or name) and date."""
    out = []
    for b in read_rows(BOOKINGS):
        if (doctor and not matches(b, doctor)) or (date and b["date"] != date):
            continue
        out.append(" | ".join(b[k] for k in BOOKING_FIELDS))
    return "\n".join(out) or "No bookings found."


@mcp.tool()
def cancel_booking(booking_id: str) -> str:
    """Staff dashboard: delete one booking from bookings.csv by its id (for example B003)."""
    rows = read_rows(BOOKINGS)
    kept = [r for r in rows if r["booking_id"] != booking_id]
    if len(kept) == len(rows):
        return f"No booking with id {booking_id}."
    write_bookings(kept)
    return f"Cancelled booking {booking_id}."


if __name__ == "__main__":
    mcp.run(transport="stdio")
