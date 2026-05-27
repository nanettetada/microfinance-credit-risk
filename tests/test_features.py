from src.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    generate,
    train_val_test_split,
)
from src.features import build_preprocessor


def test_generate_shape_and_columns():
    df = generate(n=500)
    assert len(df) == 500
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["default"]:
        assert col in df.columns
    assert df["default"].isin([0, 1]).all()


def test_train_val_test_split_disjoint():
    df = generate(n=1000)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)
    assert len(X_train) + len(X_val) + len(X_test) == 1000
    # No overlapping indices
    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)


def test_preprocessor_handles_unseen_category():
    df = generate(n=500)
    X = df.drop(columns=["default"])
    pre = build_preprocessor().fit(X)
    X_new = X.iloc[:3].copy()
    X_new.loc[X_new.index[0], "purpose"] = "BRAND_NEW_PURPOSE"
    transformed = pre.transform(X_new)
    assert transformed.shape[0] == 3
