"""
solution.py — The Data Lies Until Grouped
"""
import pandas as pd

def build_sample_data():
    return pd.DataFrame({
        "team": ["A","A","A","A","B","B","B","B"],
        "region": ["North","North","South","South","North","North","South","South"],
        "month": [1,2,1,2,1,2,1,2],
        "sales": [100,120,80,90,200,220,150,140],
    })

def add_group_features(df):
    group_cols = ["team", "region"]
    df = df.sort_values(group_cols + ["month"]).copy()
    df["group_mean_sales"] = df.groupby(group_cols)["sales"].transform("mean")
    df["rolling_avg_3"] = df.groupby(group_cols)["sales"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    df["rank_within_group"] = (
        df.groupby(group_cols)["sales"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    return df

def grouped_summary(df):
    return (
        df.groupby(["team", "region"])
        .agg(
            mean_sales=("sales", "mean"),
            max_sales=("sales", "max"),
            min_sales=("sales", "min"),
            count=("sales", "size"),
        )
        .reset_index()
    )

def main():
    df = build_sample_data()
    print("Original dataframe:")
    print(df)
    print("\nGrouped summary:")
    print(grouped_summary(df))
    print("\nDataframe with group features:")
    print(add_group_features(df))

if __name__ == "__main__":
    main()
