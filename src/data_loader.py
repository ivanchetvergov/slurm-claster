import pandas as pd
from pathlib import Path


_CLEAN_COLS = [
    "JobIDRaw", "UID", "JobName", "Partition", "ReqNodes", "ReqCPUS",
    "Submit", "Start", "End", "ElapsedRaw", "TimelimitRaw", "State",
]


class SlurmDataLoader:

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)

    def load(self):
        self.df = pd.read_csv(self.csv_path, sep="|", low_memory=False)
        self.df.columns = self.df.columns.str.strip()
        return self

    def filter(self, uid: int, job_name: str) -> pd.DataFrame:
        mask = (self.df["UID"] == uid) & (self.df["JobName"] == job_name)
        result = self.df[mask].copy()

        if result.empty:
            raise ValueError(f"Нет записей для UID={uid}, JobName='{job_name}'")

        clean_cols = [c for c in _CLEAN_COLS if c in result.columns]
        out = self.csv_path.parent / f"cleared_{uid}_{job_name}.csv"
        result[clean_cols].to_csv(out, sep="|", index=False)
        print(f"Сохранено {len(result)} записей → {out}")

        return result[clean_cols]
