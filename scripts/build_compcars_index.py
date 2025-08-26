import csv, json
from pathlib import Path
from scipy.io import loadmat
import numpy as np

ROOT = Path("data/compcars")
OUT_DIR = ROOT / "indexes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

train_list = ROOT / "train_test_split" / "classification" / "train.txt"
test_list  = ROOT / "train_test_split" / "classification" / "test.txt"

def read_pairs(list_txt):
    with open(list_txt) as f:
        return [ln.strip() for ln in f if ln.strip()]

def _parse_int_or_none(tok):
    try:
        return int(tok)
    except (TypeError, ValueError):
        return None
    
def _first_data_key(d, preferred):
    """Pick the first present key from `preferred`, else first non-magic key."""
    for k in preferred:
        if k in d:
            return k
    for k in d.keys():
        if not k.startswith("__"):
            return k
    raise KeyError("No usable keys found in .mat dict")

def _cellstr_to_list(arr):
    """
    Convert a MATLAB cellstr (loaded via loadmat) into a Python list[str],
    robust against shapes/orientations and nested char arrays.
    """
    a = np.array(arr, dtype=object).squeeze()
    # If it's a single string already:
    if isinstance(a, str):
        return [a]
    out = []
    # Flatten so we can iterate regardless of (N,1) vs (1,N)
    for elem in np.ravel(a):
        if isinstance(elem, str):
            out.append(elem)
            continue
        e = np.array(elem)
        # Char array (e.g., dtype '<U1' or 'S1') -> join characters
        if e.dtype.kind in ("U", "S"):
            out.append("".join(e.squeeze().tolist()))
        elif e.size == 1:
            # Scalar (numeric or object) -> string
            out.append(str(e.item()))
        else:
            # Last resort: stringify
            out.append(str(e))
    # Strip whitespace just in case
    return [s.strip() for s in out]

def _mat_lookups():
    """Load once and cache: type names + make/model names (1-based IDs)."""
    look = getattr(_mat_lookups, "_cache", None)
    if look is None:
        # Adjust these paths if yours live elsewhere
        types_mat = loadmat(ROOT / "misc" / "car_type.mat")
        mm_mat    = loadmat(ROOT / "misc" / "make_model_name.mat")

        # Be flexible about key names
        type_key   = _first_data_key(types_mat, ["types", "type_names", "car_type", "car_types"])
        makes_key  = _first_data_key(mm_mat,    ["make_names", "makes", "make"])
        models_key = _first_data_key(mm_mat,    ["model_names", "models", "model"])

        type_names  = _cellstr_to_list(types_mat[type_key])
        make_names  = _cellstr_to_list(mm_mat[makes_key])
        model_names = _cellstr_to_list(mm_mat[models_key])

        look = {"type_names": type_names, "make_names": make_names, "model_names": model_names}
        _mat_lookups._cache = look
    return look

def _name_from_1based(arr, idx):
    # Safe 1-based lookup
    if not isinstance(idx, int) or idx <= 0 or idx > len(arr):
        return ""
    return arr[idx - 1]

def row_from_rel(rel):
    # rel like: make_id/model_id/year/img.jpg
    parts = rel.split("/")
    make_id  = _parse_int_or_none(parts[0])
    model_id = _parse_int_or_none(parts[1])
    year     = _parse_int_or_none(parts[2])  # "unknown" -> None

    # label file has viewpoint + bbox
    lab = ROOT / "label" / rel.replace(".jpg",".txt")
    with open(lab) as f:
        v_raw = int(f.readline().strip())  # -1,1..5
        _ = f.readline().strip()           # bbox count (unused)
        x1,y1,x2,y2 = map(int, f.readline().split())
    v = 0 if v_raw == -1 else v_raw       # normalize unknown view

    # type_id from attributes.txt keyed by model_id
    attrs = getattr(row_from_rel, "_attrs", None)
    if attrs is None:
        attrs = {}
        with open(ROOT/"misc"/"attributes.txt") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#") or ln.lower().startswith("model_id"):
                    continue
                cols = ln.split()
                if len(cols) < 6:
                    continue  # skip truncated lines
                mid, vmax, disp, doors, seats, typ = cols[:6]
                try:
                    attrs[int(mid)] = int(float(typ))
                except ValueError:
                    continue
        row_from_rel._attrs = attrs

    type_id = attrs.get(model_id, 0)  # 0 means unknown/no type

    # lookups for names
    look = _mat_lookups()
    def _name_from_1based(arr, idx):
        # arr is python list; idx is 1-based; return "" if out-of-range/None/0
        if not isinstance(idx, int) or idx <= 0 or idx > len(arr):
            return ""
        return arr[idx-1]

    look = _mat_lookups()
    make_name  = _name_from_1based(look["make_names"], make_id)
    model_name = _name_from_1based(look["model_names"], model_id)
    type_name  = _name_from_1based(look["type_names"], type_id)


    return {
        "split": "",  # fill later
        "rel_path": f"image/{rel}",
        "make_id": make_id,
        "model_id": model_id,
        "year": year,                 # None if unknown
        "view": v,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "type_id": type_id,          # 0 if unknown
        "make_name": make_name,
        "model_name": model_name,
        "type_name": type_name       # "" if unknown/0
    }

def main():
    rows = []
    for rel in read_pairs(train_list):
        r = row_from_rel(rel); r["split"]="train"; rows.append(r)
    for rel in read_pairs(test_list):
        r = row_from_rel(rel); r["split"]="test"; rows.append(r)

    # build vocabs from TRAIN only (contiguous indices), skipping None
    def build_vocab(key):
        vals = sorted({int(r[key]) for r in rows if r["split"]=="train" and r[key] is not None})
        to_idx = {v:i for i,v in enumerate(vals)}
        return {"values": vals, "to_idx": to_idx}

    vocabs = {
        "make_id": build_vocab("make_id"),
        "model_id": build_vocab("model_id"),
        "year": build_vocab("year"),
        # view: map -1->0, and 1..5 stay as 1..5; then build contiguous
        "view": {"values":[0,1,2,3,4,5], "to_idx": {v:i for i,v in enumerate([0,1,2,3,4,5])}},
        "type_id": build_vocab("type_id"),
    }

    # write CSV
    out_csv = OUT_DIR / "index.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # write vocabs
    (OUT_DIR/"vocabs.json").write_text(json.dumps(vocabs))

    print(f"Wrote {out_csv}")
    print(f"Wrote {OUT_DIR/'vocabs.json'}")

if __name__ == "__main__":
    main()
