import pandas as pd

from src import config


def main():
    df = pd.read_csv(config.DATA_PATH, sep=";")
    frac = 200 / len(df)
    sample_df = df.groupby(config.TARGET, group_keys=False).sample(
        frac=frac, random_state=config.RANDOM_STATE
    )
    sample_df.to_csv(
        config.PROJECT_ROOT / "tests" / "data" / "sample.csv", sep=";", index=False
    )


if __name__ == "__main__":
    main()
