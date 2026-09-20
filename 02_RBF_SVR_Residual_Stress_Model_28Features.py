from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


FEATURES = [
    "Fe", "Cr", "Al", "Ti", "Ta", "Mo", "Nb", "Zr", "Hf", "Si", "V",
    "Ni", "Co", "Mn", "Cu", "Y", "W", "Has_N", "N_M_ratio", "Temp",
    "Bias", "Pressure", "Ar_N_ratio", "DeltaSmix", "DeltaHmix", "DeltaR",
    "DeltaElectronegativity", "E_beta",
]
FROZEN_PARAMETERS = {
    "C": 192.2617074855396,
    "epsilon": 0.08033417994820687,
    "gamma": 0.015496043764111123,
}
TARGET = "ResidualStress"
SELECTION_COLUMN = "UseFor_StressModel"
EXPECTED_TRAINING_ROWS = 90
DEFAULT_MODEL = Path(__file__).with_name(
    "RBF_SVR_Residual_Stress_Model_28Features.joblib"
)


def build_fixed_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("median_imputer", SimpleImputer(strategy="median")),
            ("standard_scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", **FROZEN_PARAMETERS)),
        ]
    )


def select_final_training_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required = FEATURES + [TARGET, SELECTION_COLUMN]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    data = frame.loc[:, required].apply(pd.to_numeric, errors="coerce")
    data = data.loc[data[SELECTION_COLUMN].eq(1) & data[TARGET].notna()].copy()
    data.loc[:, FEATURES[:17]] = data.loc[:, FEATURES[:17]].fillna(0.0)
    if len(data) != EXPECTED_TRAINING_ROWS:
        raise ValueError(
            f"Unexpected final residual-stress training-table size: {len(data)}"
        )
    return data.loc[:, FEATURES], data[TARGET]


def predict_file(input_path: Path, output_path: Path, model_path: Path) -> None:
    frame = pd.read_excel(input_path) if input_path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(input_path)
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"Prediction input is missing the 28 feature fields: {missing}")
    model = joblib.load(model_path)
    frame["Stress_pred_GPa"] = model.predict(frame.loc[:, FEATURES])
    if output_path.suffix.lower() == ".xlsx":
        frame.to_excel(output_path, index=False)
    else:
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input CSV/XLSX containing the 28 features")
    parser.add_argument("--output", type=Path, required=True, help="Output prediction CSV/XLSX")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    predict_file(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
