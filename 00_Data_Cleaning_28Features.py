import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METALS = [
    "Fe", "Cr", "Al", "Ti", "Ta", "Mo", "Nb", "Zr", "Hf", "Si", "V",
    "Ni", "Co", "Mn", "Cu", "Y", "W",
]
ELEMENTS = METALS + ["N"]
PROCESS = ["Temp", "Bias", "Pressure", "Ar_N_ratio"]
DESCRIPTORS = [
    "DeltaSmix", "DeltaHmix", "DeltaR", "DeltaElectronegativity", "E_beta",
]
FEATURES = METALS + ["Has_N", "N_M_ratio"] + PROCESS + DESCRIPTORS
KEY = ELEMENTS + PROCESS
NUMERIC = list(dict.fromkeys(ELEMENTS + PROCESS + DESCRIPTORS + [
    "Has_N", "N_M_ratio", "Hardness", "ResidualStress",
    "UseFor_HardnessModel", "UseFor_StressModel",
]))
EXPECTED = {"Hardness": 446, "ResidualStress": 90}


def load_table(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        book = pd.ExcelFile(path)
        sheet = "Database" if "Database" in book.sheet_names else book.sheet_names[0]
        return pd.read_excel(path, sheet_name=sheet)
    raise ValueError(f"Unsupported input type: {path.suffix}")


def numeric(frame):
    frame = frame.copy()
    for name in NUMERIC:
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    for name in METALS:
        if name not in frame:
            frame[name] = 0.0
    frame[METALS] = frame[METALS].fillna(0.0)
    return frame


def derive(frame):
    frame = numeric(frame)
    metal_total = frame[METALS].sum(axis=1)
    ratio = pd.to_numeric(frame.get("N_M_ratio"), errors="coerce")
    has_n = pd.to_numeric(frame.get("Has_N"), errors="coerce")
    if "N" not in frame:
        frame["N"] = np.nan
    frame["N"] = pd.to_numeric(frame["N"], errors="coerce")
    frame["N"] = frame["N"].fillna(ratio * metal_total).fillna(0.0)
    calculated_ratio = pd.Series(
        np.where(metal_total.gt(0), frame["N"] / metal_total, np.nan),
        index=frame.index,
    )
    frame["Has_N"] = has_n.fillna(frame["N"].gt(0).astype(float))
    frame["N_M_ratio"] = ratio.fillna(calculated_ratio)
    return frame


def prepare_source(frame):
    return derive(frame)


def prepare_supplemental(frame):
    frame = derive(frame)
    if "UseFor_HardnessModel" not in frame:
        frame["UseFor_HardnessModel"] = frame.get(
            "Hardness", pd.Series(np.nan, index=frame.index)
        ).notna().astype(float)
    if "UseFor_StressModel" not in frame:
        frame["UseFor_StressModel"] = frame.get(
            "ResidualStress", pd.Series(np.nan, index=frame.index)
        ).notna().astype(float)
    return frame


def validate(frame):
    missing = [name for name in FEATURES if name not in frame]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    hardness = (
        pd.to_numeric(frame.get("UseFor_HardnessModel"), errors="coerce").eq(1)
        & pd.to_numeric(frame.get("Hardness"), errors="coerce").notna()
    )
    stress = (
        pd.to_numeric(frame.get("UseFor_StressModel"), errors="coerce").eq(1)
        & pd.to_numeric(frame.get("ResidualStress"), errors="coerce").notna()
    )
    active = hardness | stress
    checked = frame.loc[active]
    if checked[ELEMENTS].lt(0).any().any():
        raise ValueError("Elemental fractions must be nonnegative")
    if checked[ELEMENTS].sum(axis=1).le(0).any():
        raise ValueError("Total composition must be greater than zero")
    for name in DESCRIPTORS:
        if checked[name].isna().any():
            raise ValueError(f"Missing descriptor values: {name}")


def key_set(frame):
    keys = frame.reindex(columns=KEY).copy()
    for name in KEY:
        keys[name] = pd.to_numeric(keys[name], errors="coerce")
    keys[ELEMENTS] = keys[ELEMENTS].fillna(0.0)
    return {
        tuple(None if pd.isna(value) else round(float(value), 12) for value in row)
        for row in keys.itertuples(index=False, name=None)
    }


def validate_supplemental(source, supplemental):
    if key_set(source).intersection(key_set(supplemental)):
        raise ValueError("Composition-process collision detected")
    if supplemental.duplicated(KEY).any():
        raise ValueError("Duplicate composition-process rows detected")


def select_training(frame, target, flag):
    required = FEATURES + [target, flag]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    data = frame.loc[:, required].copy()
    for name in required:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    data = data.loc[data[flag].eq(1) & data[target].notna()].copy()
    data[METALS] = data[METALS].fillna(0.0)
    data[flag] = 1
    return data.reset_index(drop=True)


def clean_training_data(database, supplemental, output_dir):
    source = prepare_source(load_table(database))
    validate(source)
    frames = [source]
    if supplemental is not None:
        extra = prepare_supplemental(load_table(supplemental))
        validate(extra)
        validate_supplemental(source, extra)
        frames.append(extra)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    hardness = select_training(combined, "Hardness", "UseFor_HardnessModel")
    stress = select_training(combined, "ResidualStress", "UseFor_StressModel")
    if len(hardness) != EXPECTED["Hardness"] or len(stress) != EXPECTED["ResidualStress"]:
        raise ValueError(f"Unexpected training-table sizes: hardness={len(hardness)}, stress={len(stress)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hardness_path = output_dir / "Hardness_Training_Database_28Features.csv"
    stress_path = output_dir / "Residual_Stress_Training_Database_28Features.csv"
    hardness.to_csv(hardness_path, index=False)
    stress.to_csv(stress_path, index=False)
    return hardness_path, stress_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--supplemental")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    clean_training_data(args.database, args.supplemental, args.output_dir)


if __name__ == "__main__":
    main()
