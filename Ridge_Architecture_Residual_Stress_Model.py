import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "BaselineStressMagnitude_GPa",
    "LayerNumber",
    "InterfaceDensity_um-1",
    "TotalThickness_um",
    "TransitionLayerAdded",
]
TARGET_COLUMN = "SigmaMultilayer_GPa"
RIDGE_ALPHA = 8.0
EXPECTED_ROWS = 24
DEFAULT_INPUT = Path(__file__).with_name("Ridge_Architecture_Training_Data.csv")
DEFAULT_MODEL = Path(__file__).with_name("Ridge_Architecture_Residual_Stress_Model.joblib")


def load_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("input must be a CSV or Excel file")


def train_model(data: pd.DataFrame) -> Pipeline:
    required = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} training rows, found {len(data)}")
    frame = data.loc[:, required].copy()
    for name in required:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    target = frame[TARGET_COLUMN].abs()
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )
    model.fit(frame[FEATURE_COLUMNS], target)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL,
    )
    args = parser.parse_args()
    model = train_model(load_data(args.input))
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    print(f"Saved {args.model_out}")


if __name__ == "__main__":
    main()
