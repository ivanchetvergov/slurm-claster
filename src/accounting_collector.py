import subprocess
import time
from pathlib import Path


SACCT_FIELDS = (
    "JobID", "JobIDRaw", "JobName", "User", "Partition", "State", "QOS", "Priority",
    "Reason", "Flags", "Submit", "Eligible", "Start", "End", "Elapsed", "ElapsedRaw",
    "Planned", "PlannedCPURAW", "ReqNodes", "ReqCPUS", "ReqMem", "AllocNodes",
    "AllocCPUS", "AllocTRES", "NTasks", "Timelimit", "TimelimitRaw", "CPUTimeRAW",
    "NodeList", "ExitCode",
)


class AccountingCollector:

    def __init__(self, output_file: str | Path, poll_interval: int = 10):
        self.output_file = Path(output_file)
        self.poll_interval = poll_interval

    def wait(self, job_ids: list[str]) -> None:
        print(f"Ожидание {len(job_ids)} джобов...")
        pending = set(job_ids)
        while pending:
            pending &= set(subprocess.run(
                ["squeue", "--noheader", "--format=%i", "--jobs", ",".join(pending)],
                capture_output=True, text=True,
            ).stdout.strip().splitlines())
            if pending:
                print(f"  осталось: {len(pending)}")
                time.sleep(self.poll_interval)
        print("Все джобы завершены.")

    def collect(self, job_ids: list[str]) -> Path:
        result = subprocess.run(
            ["sacct", "--jobs", ",".join(job_ids), "--allocations",
             "--format", ",".join(SACCT_FIELDS), "--parsable2", "--noheader"],
            capture_output=True, text=True, check=True,
        )
        with open(self.output_file, "w") as f:
            f.write("|".join(SACCT_FIELDS) + "\n")
            f.write(result.stdout)
        print(f"Статистика → {self.output_file}")
        return self.output_file
