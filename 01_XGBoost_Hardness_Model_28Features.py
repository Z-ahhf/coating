from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


FEATURES = [
    "Fe", "Cr", "Al", "Ti", "Ta", "Mo", "Nb", "Zr", "Hf", "Si", "V",
    "Ni", "Co", "Mn", "Cu", "Y", "W", "Has_N", "N_M_ratio", "Temp",
    "Bias", "Pressure", "Ar_N_ratio", "DeltaSmix", "DeltaHmix", "DeltaR",
    "DeltaElectronegativity", "E_beta",
]
FROZEN_PARAMETERS = {
    "colsample_bytree": 0.8385045019271884,
    "gamma": 1.0,
    "learning_rate": 0.03062751474852941,
    "max_depth": 5,
    "min_child_weight": 1,
    "n_estimators": 398,
    "reg_alpha": 0.00118772873199637,
    "reg_lambda": 0.4100527416687401,
    "subsample": 0.712044516043649,
}
TARGET = "Hardness"
SELECTION_COLUMN = "UseFor_HardnessModel"
EXPECTED_TRAINING_ROWS = 446
DEFAULT_MODEL = Path(__file__).with_name(
    "XGBoost_Hardness_Model_28Features.joblib"
)


def build_fixed_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("median_imputer", SimpleImputer(strategy="median")),
            (
                "xgb",
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=20260630,
                    n_jobs=1,
                    tree_method="hist",
                    **FROZEN_PARAMETERS,
                ),
            ),
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
            f"Unexpected final hardness training-table size: {len(data)}"
        )
    return data.loc[:, FEATURES], data[TARGET]


def predict_file(input_path: Path, output_path: Path, model_path: Path) -> None:
    frame = pd.read_excel(input_path) if input_path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(input_path)
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"Prediction input is missing the 28 feature fields: {missing}")
    model = joblib.load(model_path)
    frame["H_pred_GPa"] = model.predict(frame.loc[:, FEATURES])
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
