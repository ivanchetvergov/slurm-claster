UID    = 50109
JOB    = sphere.slrm
N      = 30
SCALE  = 100
PART   = debug
SEED   = 42

CSV     = data/acct_0921-0923.csv
CLEARED = data/cleared_$(UID)_$(JOB).csv
SACCT   = src/output/sacct_results.csv
SCRIPTS = src/output

_PY_BASE = python src/main.py \
     --uid $(UID) --job $(JOB) \
     --n $(N) --scale $(SCALE) --partition $(PART) --seed $(SEED)

PY       = $(_PY_BASE) --csv $(CLEARED)
PY_FULL  = $(_PY_BASE) --csv $(CSV)

.PHONY: dry dry-full run run-full plots analyze clean clean-scripts clean-plots clean-results help

help:
	@echo "Цели:"
	@echo "  dry            — dry-run на очищенных данных ($(CLEARED))"
	@echo "  dry-full       — dry-run на полных данных ($(CSV))"
	@echo "  run            — полный запуск на очищенных данных"
	@echo "  run-full       — полный запуск на полных данных"
	@echo "  plots          — графики по оригинальным данным"
	@echo "  analyze        — графики по результатам sacct"
	@echo "  clean-scripts  — удалить сгенерированные job_*.sh"
	@echo "  clean-plots    — удалить plots/*.png"
	@echo "  clean-results  — удалить sacct_results.csv"
	@echo "  clean          — всё вышеперечисленное"
	@echo ""
	@echo "Переменные (можно переопределить):"
	@echo "  N=$(N)  SCALE=$(SCALE)  PART=$(PART)  SEED=$(SEED)"

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

plots: $(CLEARED)
	python src/plots.py --uid $(UID) --job $(JOB) --scale $(SCALE) --n $(N) --seed $(SEED)

$(CLEARED):
	@echo "Нет $(CLEARED) — сначала запусти 'make dry' или 'make run'" && exit 1

analyze: $(SACCT)
	python src/analyze.py \
	    --sacct $(SACCT) --subsample $(CLEARED) \
	    --scale $(SCALE) --output plots/analysis.png

$(SACCT):
	@echo "Нет $(SACCT) — сначала запусти 'make run'" && exit 1

clean-scripts:
	rm -f $(SCRIPTS)/job_*.sh $(SCRIPTS)/job_*.out

clean-plots:
	rm -f plots/*.png

clean-results:
	rm -f $(SACCT) run.log

clean: clean-scripts clean-plots clean-results
