# needed to be able to call other targets within given one:
THIS_FILE := $(lastword $(MAKEFILE_LIST))

clean:
	find . -name '*.pyc' -type f -delete

install:
	pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl
	pip install -r requirements.txt

.PHONY: clean install
