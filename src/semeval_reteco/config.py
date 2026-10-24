from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "reteco" / "track2_recor"

DOMAINS = ("biology", "drones", "earth_science", "economics", "hardware", "law",
    "medicalsciences", "politics", "psychology", "robotics", "sustainable_living")

SPLITS = ("train", "dev")

EMB_DIR = ROOT / "data" / "emb"