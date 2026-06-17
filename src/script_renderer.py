import subprocess
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from job_generator import JobRequest


def _slurm_time(secs: int) -> str:
    d, rem = divmod(secs, 86400)
    h, rem = divmod(rem, 3600)
    m, s   = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:{s:02d}"


class ScriptRenderer:

    def __init__(self, template_dir: Path, output_dir: Path, partition: str = "debug"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.partition = partition
        env = Environment(loader=FileSystemLoader(str(template_dir)), trim_blocks=True, lstrip_blocks=True)
        self.template = env.get_template("job.sh.j2")

    def render(self, job_id: int, request: JobRequest) -> Path:
        content = self.template.render(
            job_id=job_id,
            partition=self.partition,
            elapsed_sec=request.elapsed_sec,
            timelimit_str=_slurm_time(request.timelimit_sec),
            nodes=request.nodes,
        )
        path = self.output_dir / f"job_{job_id:04d}.sh"
        path.write_text(content)
        return path

    def submit(self, script_path: Path) -> str:
        """Submits script and returns the assigned job ID."""
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True, text=True, check=True,
        )
        tokens = result.stdout.strip().split()
        if not tokens:
            raise ValueError(f"Unexpected sbatch output: {result.stdout!r}")
        return tokens[-1]
