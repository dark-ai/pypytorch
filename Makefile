PYTHON = python3
TARGET = pypytorch
VERSION = 0.0.1

install: setup.py
	$(PYTHON) setup.py bdist_wheel &&\
		cd dist &&\
		pip uninstall $(TARGET) &&\
		pip install $(TARGET)-$(VERSION)-py3-none-any.whl

clean: build dist
	-$(RM) -rf build dist $(TARGET).egg-info

.PHONY: install clean