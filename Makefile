PYTHON = python
TARGET = pypytorch
VERSION = 0.0.1a0
PIP = pip

install: setup.py
	$(PYTHON) setup.py bdist_wheel &&\
		cd dist &&\
		$(PIP) uninstall $(TARGET) -y &&\
		$(PIP) install $(TARGET)-$(VERSION)-py3-none-any.whl

clean: build dist
	-$(RM) -rf build dist $(TARGET).egg-info

.PHONY: install clean

