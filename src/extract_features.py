from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "BUT_ReverbDB_rel_19_06_RIR-Only"
OUTPUT_CSV = ROOT / "data" / "processed" / "rir_paths.csv"
TARGET_NAME = "IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav"


def collect_rir_files(base_dir: Path) -> list[dict[str, str]]:
	rows = []

	for wav_path in base_dir.rglob(TARGET_NAME):
		if wav_path.parent.name != "RIR":
			continue

		relative_path = wav_path.relative_to(base_dir)
		room_id = relative_path.parts[0]

		rows.append(
			{
				"path": str(wav_path),
				"room_id": room_id,
			}
		)

	rows.sort(key=lambda row: row["path"])
	return rows


def save_rows(rows: list[dict[str, str]], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with output_path.open("w", newline="") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=["path", "room_id"])
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	rows = collect_rir_files(RAW_DIR)
	save_rows(rows, OUTPUT_CSV)

	print(f"found {len(rows)} file")
	print(f"CSV saved in: {OUTPUT_CSV}")

	for row in rows[:10]:
		print(f"{row['room_id']} | {row['path']}")


if __name__ == "__main__":
	main()
