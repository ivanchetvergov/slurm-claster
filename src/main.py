#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from data_loader import SlurmDataLoader
from stats_calculator import StatsCalculator
from job_generator import JobGenerator
from script_renderer import ScriptRenderer
from accounting_collector import AccountingCollector


BASE_DIR = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",       required=True)
    p.add_argument("--uid",       required=True, type=int)
    p.add_argument("--job",       required=True)
    p.add_argument("--n",         required=True, type=int)
    p.add_argument("--partition", default="debug")
    p.add_argument("--output",    default="output")
    p.add_argument("--seed",      type=int, default=None)
    p.add_argument("--scale",    type=int, default=1,
                   help="делитель для elapsed/timelimit")
    p.add_argument("--max-time", type=int, default=3600,
                   help="жёсткий потолок timelimit в секундах (по умолчанию 3600)")
    p.add_argument("--dry-run",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n[1/5] Загрузка {args.csv}")
    df = SlurmDataLoader(args.csv).load().filter(uid=args.uid, job_name=args.job)

    print("\n[2/5] Статистика")
    stats = StatsCalculator(df).compute()
    print(stats)

    print(f"\n[3/5] Генерация {args.n} заявок")
    requests = JobGenerator(stats, seed=args.seed, time_scale=args.scale, max_seconds=args.max_time).generate(args.n)

    scripts_dir = BASE_DIR.parent / "scripts"
    output_dir  = BASE_DIR.parent / "output"
    renderer = ScriptRenderer(
        template_dir=BASE_DIR / "templates",
        output_dir=scripts_dir,
        partition=args.partition,
    )

    print(f"\n[4/5] Скрипты → {scripts_dir}")
    job_ids = []
    for i, req in enumerate(requests, start=1):
        script = renderer.render(job_id=i, request=req)
        if args.dry_run:
            coef = req.timelimit_sec / req.elapsed_sec
            print(f"  [{i:03d}] nodes={req.nodes}  elapsed={req.elapsed_sec}s  limit={req.timelimit_sec}s  coef={coef:.1f}x  ({script.name})")
        else:
            out = renderer.submit(script)
            job_ids.append(out)
            print(f"  [{i:03d}] nodes={req.nodes}  elapsed={req.elapsed_sec}s  limit={req.timelimit_sec}s  → {out}")

    if args.dry_run:
        print("\ndry-run, sbatch не вызывался.")
        return

    print("\n[5/5] Сбор статистики")
    output_dir.mkdir(parents=True, exist_ok=True)
    collector = AccountingCollector(output_dir / "sacct_results.csv")
    collector.wait(job_ids)
    collector.collect(job_ids)
    print("\nГотово.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)
