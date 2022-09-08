PY_VERSION := 3
VIRTUALENV_EXE := python3 -m virtualenv -p python3
VIRTUALENV_DIR := .venv
ACTIVATE := $(VIRTUALENV_DIR)/bin/activate
AMS_EXAMPLE ?= /ams_wrapper/

.PHONY: default  ams_test\
	venv  ams_coverage clean_pyc clean \

default: ams_test

venv: $(ACTIVATE)
	@echo -n "Using "
	@. $(ACTIVATE); python3 --version

$(ACTIVATE): requirements.txt
	@echo "Updating virtualenv dependencies in: $(VIRTUALENV_DIR)..."
	@test -d $(VIRTUALENV_DIR) || $(VIRTUALENV_EXE) $(VIRTUALENV_DIR)
	@. $(ACTIVATE); pip$(PY_VERSION) install --upgrade pip==21.3.1
	@. $(ACTIVATE); pip$(PY_VERSION) install -vUqq setuptools
	@. $(ACTIVATE); pip$(PY_VERSION) install -qq -r requirements.txt 
	@. $(ACTIVATE); pip$(PY_VERSION) install -qq -r  $(AMS_EXAMPLE)/requirements.txt 
	@touch $(ACTIVATE)


ams_coverage: $(ACTIVATE)
	@echo "Computing unit test coverage for ams..."
	@. $(ACTIVATE); test -d $(AMS_EXAMPLE)tests/unit/test_images  || ($(AMS_EXAMPLE)tests/unit/get_test_images.sh && mv test_images $(AMS_EXAMPLE)tests/unit)
	@. $(ACTIVATE); pytest --cov-config=$(AMS_EXAMPLE).coveragerc --cov=src $(AMS_EXAMPLE)tests/unit --cov-report=html --cov-fail-under=78

ams_test: $(ACTIVATE)
	echo "Running ams wrapper unit tests"
	test -d $(VIRTUALENV_DIR) || $(VIRTUALENV_EXE) $(VIRTUALENV_DIR)
	@. $(ACTIVATE); test -d $(AMS_EXAMPLE)tests/unit/test_images || ($(AMS_EXAMPLE)tests/unit/get_test_images.sh && mv test_images $(AMS_EXAMPLE)tests/unit)
	@. $(ACTIVATE); pytest  $(AMS_EXAMPLE)tests/unit

ams_clean:
	@echo "Removing ams virtual env files and test images ..."
	@rm -rf $(VIRTUALENV_DIR)
	@rm -rf $(AMS_EXAMPLE)tests/unit/test_images

clean_pyc:
	@echo "Removing .pyc files..."
	@find . -name '*.pyc' -exec rm -f {} \;

clean: clean_pyc
	@echo "Removing virtual env files..."
	@rm -rf $(VIRTUALENV_DIR)