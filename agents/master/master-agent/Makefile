# Master Orchestration Agent Makefile

# Virtual environment
VENV := ../venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Check if virtual environment exists, create if not
$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

# Install dependencies
install-deps: $(VENV)
	$(PIP) install -r requirements.txt

# Start the master orchestration agent
start-master: install-deps
	@echo "Starting Master Orchestration Agent..."
	$(PYTHON) enhanced_main.py

# Start the basic master agent (fallback)
start-basic: install-deps
	@echo "Starting Basic Master Agent..."
	$(PYTHON) main.py

# Clean virtual environment
clean:
	rm -rf $(VENV)

# Show status
status:
	@echo "Master Orchestration Agent Status"
	@echo "================================="
	@echo "Virtual Environment: $(VENV)"
	@if [ -d "$(VENV)" ]; then echo "Status: Ready"; else echo "Status: Not installed"; fi
	@echo "Main Script: enhanced_main.py"

# Help
help:
	@echo "Available commands:"
	@echo "  start-master  - Start the enhanced master orchestration agent"
	@echo "  start-basic   - Start the basic master agent"
	@echo "  install-deps  - Install Python dependencies"
	@echo "  status        - Show agent status"
	@echo "  clean         - Remove virtual environment"
	@echo "  help          - Show this help message"

.PHONY: start-master start-basic install-deps clean status help