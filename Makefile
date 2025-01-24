test: fix run_dtp run_penn run_dse run_samnn

run_dtp:
	@. .venv/bin/activate && python DynamicTaskPrioritization/test_dtp.py

run_penn:
	@. .venv/bin/activate && python PredictiveEntanglementNN/test_penn.py

run_dse:
	@. .venv/bin/activate && python DataShapeErosion/test_dse.py

run_samnn:
	@. .venv/bin/activate && python SelfAssemblingModularNN/test_samnn.py

install:
	@. .venv/bin/activate && python -m venv .venv
	@. .venv/bin/activate && pip install -r requirements.txt

clean:
	@rm -rf .venv
	@rm -rf __pycache__
	@rm -rf */__pycache__

fix:
	@. .venv/bin/activate && black .
	@. .venv/bin/activate && isort .