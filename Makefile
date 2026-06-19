UID    = 50109
JOB    = sphere.slrm
N      = 200
SCALE    = 2000
PART     = debug
SEED     = 42
MAX_TIME = 3600

CSV     = data/acct_0921-0923.csv
CLEARED = data/cleared_$(UID)_$(JOB).csv
JOBIDS   = output/job_ids.txt
RESULTS  = slurm_results
SCRIPTS  = scripts

_PY_BASE = python3 src/main.py \
     --uid $(UID) --job $(JOB) \
     --n $(N) --scale $(SCALE) --partition $(PART) --seed $(SEED) --max-time $(MAX_TIME)

PY       = $(_PY_BASE) --csv $(CLEARED)
PY_FULL  = $(_PY_BASE) --csv $(CSV)

.PHONY: dry dry-full run run-full submit collect show-scripts plots analyze clean clean-scripts clean-plots clean-results help

help:
	@echo "Цели:"
	@echo "  dry            — dry-run на очищенных данных ($(CLEARED))"
	@echo "  dry-full       — dry-run на полных данных ($(CSV))"
	@echo "  run            — полный запуск на очищенных данных"
	@echo "  run-full       — полный запуск на полных данных"
	@echo "  submit         — отправить уже сгенерированные скрипты через sbatch"
	@echo "  collect        — дождаться и собрать sacct после make submit"
	@echo "  show-scripts   — показать список сгенерированных скриптов"
	@echo "  plots          — графики по оригинальным данным"
	@echo "  analyze        — графики по результатам sacct"
	@echo "  clean-scripts  — удалить сгенерированные job_*.sh"
	@echo "  clean-plots    — удалить plots/*.png"
	@echo "  clean-results  — удалить sacct_results.csv"
	@echo "  clean          — всё вышеперечисленное"
	@echo ""
	@echo "Переменные (можно переопределить):"
	@echo "  N=$(N)  SCALE=$(SCALE)  PART=$(PART)  SEED=$(SEED)"
	@echo ""
	@echo "Примеры:"
	@echo "  make dry N=5        — сгенерировать 5 скриптов"
	@echo "  make dry N=100      — сгенерировать 100 скриптов"

dry: $(CLEARED)
	$(PY) --dry-run

dry-full:
	$(PY_FULL) --dry-run

run: $(CLEARED)
	nohup $(PY) > run.log 2>&1 & echo "PID: $$!"
	@echo "Лог: run.log"

run-full:
	nohup $(PY_FULL) > run.log 2>&1 & echo "PID: $$!"
	@echo "Лог: run.log"

submit:
	@scripts=$$(ls $(SCRIPTS)/job_*.sh 2>/dev/null); \
	[ -z "$$scripts" ] && echo "Нет скриптов в $(SCRIPTS)/" && exit 1; \
	mkdir -p output; rm -f $(JOBIDS); \
	for s in $$scripts; do sbatch $$s | awk '{print $$NF}' | tee -a $(JOBIDS); done; \
	echo "Job IDs → $(JOBIDS)"

collect:
	@[ -f $(JOBIDS) ] || (echo "Нет $(JOBIDS) — сначала запусти make submit" && exit 1)
	python3 -c "\
import sys, datetime; sys.path.insert(0,'src'); \
from pathlib import Path; \
from accounting_collector import AccountingCollector; \
ids = open('$(JOBIDS)').read().split(); \
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S'); \
p = Path('$(RESULTS)') / f'sacct_{ts}.csv'; p.parent.mkdir(exist_ok=True); \
ac = AccountingCollector(p); ac.wait(ids); ac.collect(ids)"

show-scripts:
	@ls -lh $(SCRIPTS)/job_*.sh 2>/dev/null || echo "Нет скриптов в $(SCRIPTS)/"

plots: $(CLEARED)
	python3 src/plots.py --uid $(UID) --job $(JOB) --scale $(SCALE) --n $(N) --seed $(SEED)

$(CLEARED):
	@echo "Нет $(CLEARED) — сначала запусти 'make dry' или 'make run'" && exit 1

analyze:
	$(eval LATEST := $(shell ls -t $(RESULTS)/sacct_*.csv 2>/dev/null | head -1))
	@[ -z "$(LATEST)" ] && echo "Нет файлов в $(RESULTS)/ — сначала запусти make run/collect" && exit 1 || true
	python3 src/analyze.py \
	    --sacct $(LATEST) --subsample $(CLEARED) \
	    --scale $(SCALE) --output-dir plots

clean-scripts:
	rm -f $(SCRIPTS)/job_*.sh $(JOBIDS)

clean-plots:
	rm -f plots/*.png

clean-results:
	rm -rf $(RESULTS) run.log

clean: clean-scripts clean-plots clean-results
