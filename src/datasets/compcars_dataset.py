import csv, json
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset

class CompCarsCls(Dataset):
    """
    Full-car multi-task dataset.
    Targets: make_id, model_id, year, view, type_id (remapped to contiguous indices).
    - Unknown 'year' handling:
        year_unknown="map_unk" -> adds an extra UNK year class and maps blanks to it (default)
        year_unknown="drop"    -> drops samples with unknown year
    """
    def __init__(
        self,
        root: str,
        csv_file: str,
        transform=None,
        use_bbox: bool = False,
        vocabs_path: Optional[str] = None,
        year_unknown: str = "map_unk",  
    ):
        assert year_unknown in {"map_unk", "drop"}
        self.root = Path(root)
        self.csv_file = Path(csv_file)
        self.transform = transform
        self.use_bbox = use_bbox
        self.year_unknown = year_unknown

        # load rows
        rows = []
        with open(self.csv_file, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)

        # Optionally drop rows with unknown year (blank cell in CSV)
        if self.year_unknown == "drop":
            rows = [r for r in rows if r.get("year", "").strip().isdigit()]

        self.rows = rows

        # load vocabs (built from TRAIN)
        if vocabs_path is None:
            vocabs_path = self.root / "indexes" / "vocabs.json"
        vocabs = json.loads(Path(vocabs_path).read_text(encoding="utf-8"))

        # Convert to_idx keys (JSON) back to int
        self.to_idx: Dict[str, Dict[int, int]] = {}
        for k, v in vocabs.items():
            self.to_idx[k] = {int(kk): vv for kk, vv in v["to_idx"].items()}

        # Class cardinalities
        self.card = {k: len(v["values"]) for k, v in vocabs.items()}

        # Handle unknown year mapping by appending an UNK class
        self.year_unk_idx: Optional[int] = None
        if self.year_unknown == "map_unk":
            self.year_unk_idx = self.card["year"]  
            self.card["year"] += 1

        # Expose class counts for heads
        self.n_make  = self.card["make_id"]
        self.n_model = self.card["model_id"]
        self.n_year  = self.card["year"]
        self.n_view  = self.card["view"]
        self.n_type  = self.card["type_id"]

    def __len__(self): 
        return len(self.rows)

    def __getitem__(self, idx: int):
        r: Dict[str, Any] = self.rows[idx]
        img = Image.open(self.root / r["rel_path"]).convert("RGB")

        if self.use_bbox:
            x1, y1, x2, y2 = map(int, [r["x1"], r["y1"], r["x2"], r["y2"]])
            if (x2 > x1) and (y2 > y1):
                img = img.crop((x1, y1, x2, y2))

        if self.transform:
            img = self.transform(img)

        # view: map -1 -> 0 before lookup 
        view_raw = int(r["view"])
        view_val = 0 if view_raw == -1 else view_raw

        # targets 
        make_idx  = self.to_idx["make_id"][int(r["make_id"])]
        model_idx = self.to_idx["model_id"][int(r["model_id"])]
        type_idx  = self.to_idx["type_id"][int(r["type_id"])]

        # Year handling
        year_str = r.get("year", "").strip()
        if self.year_unknown == "drop":
            year_idx = self.to_idx["year"][int(year_str)]
            year_raw = int(year_str)
        else:
            if year_str.isdigit():
                year_idx = self.to_idx["year"][int(year_str)]
                year_raw = int(year_str)
            else:
                year_idx = self.year_unk_idx 
                year_raw = -1

        targets = {
            "make":  torch.tensor(make_idx,  dtype=torch.long),
            "model": torch.tensor(model_idx, dtype=torch.long),
            "year":  torch.tensor(year_idx,  dtype=torch.long),
            "view":  torch.tensor(self.to_idx["view"][view_val], dtype=torch.long),
            "type":  torch.tensor(type_idx,  dtype=torch.long),
        }

        meta = {
            "rel_path": r["rel_path"],
            "make_name":  r.get("make_name", ""),
            "model_name": r.get("model_name", ""),
            "type_name":  r.get("type_name", ""),
            "raw": {
                "make_id": int(r["make_id"]),
                "model_id": int(r["model_id"]),
                "year": year_raw,     # None if unknown & map_unk
                "view": view_raw,
                "type_id": int(r["type_id"]),
            }
        }
        return img, targets, meta
