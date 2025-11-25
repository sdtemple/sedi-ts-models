### Get results from k-fold experiments ###


### Imports ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
import sys

### File locations ###

file_prefix, idx, num_final_epochs = sys.argv[1:] 
num_final_epochs = int(num_final_epochs)
J = num_final_epochs  # change J to use a different number of final epochs
idx = int(idx)
file_results = f'{file_prefix}{idx}.txt'
LOG_PATH = Path(rf"{file_results}")
OUT_DIR = LOG_PATH.parent / "parsed_logs"
OUT_DIR.mkdir(exist_ok=True)

### Helper functions ###

def try_cast(v: str):
    v = v.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        return int(v)
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return v

def parse_log(path: Path):
    model_params = {}
    training_params = {}
    data_params = {}
    folds = {}
    current_fold = None
    state = None  # None, 'model', 'training'

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("Model parameters"):
            state = "model"
            continue
        if line.startswith("Training parameters"):
            state = "training"
            continue
        if line.startswith("Data parameters"):
            state = "data"
            continue
        if line.startswith("KFold:"):
            state = None
            current_fold = line.split(":", 1)[1].strip()
            folds[current_fold] = []
            continue

        if state == "model":
            if ":" in line:
                k, v = line.split(":", 1)
                model_params[k.strip()] = try_cast(v)
            continue
        if state == "training":
            if ":" in line:
                k, v = line.split(":", 1)
                training_params[k.strip()] = try_cast(v)
            continue
        if state == "data":
            if ":" in line:
                k, v = line.split(":", 1)
                data_params[k.strip()] = try_cast(v)
            continue

        # parse epoch lines for current fold
        if current_fold and line.startswith("Epoch"):
            # parts like: Epoch1/200,TrainLoss:0.4994,ValLoss:0.5521,ValKGE(std):0.7130,ValKGE(orig):0.3626
            parts = [p.strip() for p in line.split(",") if p.strip()]
            row = {"fold": current_fold}
            m = re.match(r"Epoch\s*(\d+)\s*/\s*(\d+)", parts[0], re.I)
            if m:
                row["epoch"] = int(m.group(1))
                row["epoch_total"] = int(m.group(2))
            else:
                # fallback: keep raw first part
                row["epoch_label"] = parts[0]

            for p in parts[1:]:
                if ":" not in p:
                    continue
                key, val = p.split(":", 1)
                key = key.strip()
                row[key] = try_cast(val)
            folds[current_fold].append(row)

    return model_params, training_params, data_params, folds



### Initial parsing of logs ###

model_params, training_params, data_params, folds = parse_log(LOG_PATH)

# save params
pd.Series(model_params).to_frame("value").to_csv(OUT_DIR / "model_params.csv")
pd.Series(training_params).to_frame("value").to_csv(OUT_DIR / "training_params.csv")
pd.Series(data_params).to_frame("value").to_csv(OUT_DIR / "data_params.csv")

# save per-fold CSVs and one combined CSV
all_frames = []
for fold_name, rows in folds.items():
    if not rows:
        continue
    df = pd.DataFrame(rows).sort_values(["epoch"])
    # normalize column names: replace spaces and keep parentheses if desired
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    fname = f"fold_{re.sub(r'[^0-9A-Za-z_-]+', '_', fold_name)}.csv"
    df.to_csv(OUT_DIR / fname, index=False)
    all_frames.append(df)

if all_frames:
    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(OUT_DIR / "combined_folds.csv", index=False)

print(f"Parsed model params: {len(model_params)} keys")
print(f"Parsed training params: {len(training_params)} keys")
print(f"Parsed folds: {len(folds)} -> saved to {OUT_DIR}")

### Compute summary statistics ###

try:

    # Kling Gupta Efficiency

    col = "ValKGE(orig)"
    summaries = []
    for fold_name, rows in folds.items():
        if not rows:
            continue
        df_fold = pd.DataFrame(rows).sort_values("epoch")
        if col not in df_fold.columns:
            continue
        series = df_fold[col].dropna()
        last_series = series.iloc[-J:] if len(series) else series
        desc = last_series.describe()
        summaries.append({
            "fold": fold_name,
            "n_used": int(len(last_series)),
            "mean": float(desc["mean"]) if "mean" in desc else None,
            "min": float(desc["min"]) if "min" in desc else None,
            "25%": float(desc.get("25%")) if desc.get("25%") is not None else None,
            "50%": float(desc.get("50%")) if desc.get("50%") is not None else None,
            "75%": float(desc.get("75%")) if desc.get("75%") is not None else None,
            "max": float(desc["max"]) if "max" in desc else None,
        })

    summary_df = pd.DataFrame(summaries).set_index("fold").sort_index()

    # display summary with 4 decimal places (without modifying summary_df)
    with pd.option_context('display.float_format', '{:.4f}'.format):

        # save a CSV with numeric values formatted to 4 decimal places
        summary_df.to_csv(OUT_DIR / f"valkge_orig_summary.csv", float_format="%.4f")

except Exception as e:
    print(f"Failed to compute KGE summary: {e}")
    f=open(OUT_DIR / f"valkge_orig_summary.csv", "w")
    f.write(f"Error computing KGE summary: {e}\n")
    f.close()



# Beta component

try:

    col = "ValBeta(orig)"
    summaries = []
    for fold_name, rows in folds.items():
        if not rows:
            continue
        df_fold = pd.DataFrame(rows).sort_values("epoch")
        if col not in df_fold.columns:
            continue
        series = df_fold[col].dropna()
        last_series = series.iloc[-J:] if len(series) else series
        desc = last_series.describe()
        summaries.append({
            "fold": fold_name,
            "n_used": int(len(last_series)),
            "mean": float(desc["mean"]) if "mean" in desc else None,
            "min": float(desc["min"]) if "min" in desc else None,
            "25%": float(desc.get("25%")) if desc.get("25%") is not None else None,
            "50%": float(desc.get("50%")) if desc.get("50%") is not None else None,
            "75%": float(desc.get("75%")) if desc.get("75%") is not None else None,
            "max": float(desc["max"]) if "max" in desc else None,
        })

    summary_df = pd.DataFrame(summaries).set_index("fold").sort_index()

    # display summary with 4 decimal places (without modifying summary_df)
    with pd.option_context('display.float_format', '{:.4f}'.format):

        # save a CSV with numeric values formatted to 4 decimal places
        summary_df.to_csv(OUT_DIR / f"valbeta_orig_summary.csv", float_format="%.4f")

except Exception as e:
    print(f"Failed to compute Beta summary: {e}")
    f=open(OUT_DIR / f"valbeta_orig_summary.csv", "w")
    f.write(f"Error computing Beta summary: {e}\n")
    f.close()





# Alpha component

try:

    col = "ValAlpha(orig)"
    summaries = []
    for fold_name, rows in folds.items():
        if not rows:
            continue
        df_fold = pd.DataFrame(rows).sort_values("epoch")
        if col not in df_fold.columns:
            continue
        series = df_fold[col].dropna()
        last_series = series.iloc[-J:] if len(series) else series
        desc = last_series.describe()
        summaries.append({
            "fold": fold_name,
            "n_used": int(len(last_series)),
            "mean": float(desc["mean"]) if "mean" in desc else None,
            "min": float(desc["min"]) if "min" in desc else None,
            "25%": float(desc.get("25%")) if desc.get("25%") is not None else None,
            "50%": float(desc.get("50%")) if desc.get("50%") is not None else None,
            "75%": float(desc.get("75%")) if desc.get("75%") is not None else None,
            "max": float(desc["max"]) if "max" in desc else None,
        })

    summary_df = pd.DataFrame(summaries).set_index("fold").sort_index()

    # display summary with 4 decimal places (without modifying summary_df)
    with pd.option_context('display.float_format', '{:.4f}'.format):

        # save a CSV with numeric values formatted to 4 decimal places
        summary_df.to_csv(OUT_DIR / f"valalpha_orig_summary.csv", float_format="%.4f")

except Exception as e:
    print(f"Failed to compute Alpha summary: {e}")
    f=open(OUT_DIR / f"valalpha_orig_summary.csv", "w")
    f.write(f"Error computing Alpha summary: {e}\n")
    f.close()



# Pearson correlation component

try:

    col = "ValPearson(orig)"
    summaries = []
    for fold_name, rows in folds.items():
        if not rows:
            continue
        df_fold = pd.DataFrame(rows).sort_values("epoch")
        if col not in df_fold.columns:
            continue
        series = df_fold[col].dropna()
        last_series = series.iloc[-J:] if len(series) else series
        desc = last_series.describe()
        summaries.append({
            "fold": fold_name,
            "n_used": int(len(last_series)),
            "mean": float(desc["mean"]) if "mean" in desc else None,
            "min": float(desc["min"]) if "min" in desc else None,
            "25%": float(desc.get("25%")) if desc.get("25%") is not None else None,
            "50%": float(desc.get("50%")) if desc.get("50%") is not None else None,
            "75%": float(desc.get("75%")) if desc.get("75%") is not None else None,
            "max": float(desc["max"]) if "max" in desc else None,
        })

    summary_df = pd.DataFrame(summaries).set_index("fold").sort_index()

    # display summary with 4 decimal places (without modifying summary_df)
    with pd.option_context('display.float_format', '{:.4f}'.format):

        # save a CSV with numeric values formatted to 4 decimal places
        summary_df.to_csv(OUT_DIR / f"valr_orig_summary.csv", float_format="%.4f")

except Exception as e:
    print(f"Failed to compute Pearson summary: {e}")
    f=open(OUT_DIR / f"valr_orig_summary.csv", "w")
    f.write(f"Error computing Pearson summary: {e}\n")
    f.close()





### Concatenate results into one summary file ###

out_path = OUT_DIR / f"combined_results_{idx}.txt"

# concatenate like `cat` for the three CSVs (no extra newlines or headers added)
files = ["model_params.csv", 
         "training_params.csv",
         "data_params.csv", 
         "valkge_orig_summary.csv",
         "valbeta_orig_summary.csv",
         "valalpha_orig_summary.csv",
         "valr_orig_summary.csv",
         ]
paths = [OUT_DIR / f for f in files if (OUT_DIR / f).exists()]

with open(out_path, "wb") as out_f:
    for p in paths:
        out_f.write(f'{p}\n'.encode())
        out_f.write(p.read_bytes())
        out_f.write(b'\n')

print(f"Concatenated {len(paths)} files -> {out_path}")




### Clean up the folder by removing intermediate CSVs ###

# remove all files in OUT_DIR except those matching "combined_results*.csv"
removed = []
kept = []

for p in OUT_DIR.iterdir():
    if not p.is_file():
        continue
    if p.name.startswith("combined_results") and p.suffix == ".txt":
        kept.append(p)
    else:
        try:
            p.unlink()
            removed.append(p)
        except Exception as e:
            print(f"Failed to remove {p}: {e}")

print(f"Removed {len(removed)} files:")
for f in removed:
    print(" ", f)
print(f"Kept {len(kept)} files:")
for f in kept:
    print(" ", f)
