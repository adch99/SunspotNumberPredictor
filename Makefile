default:
	python bin/main.py 2> logs/errors.txt
	less logs/errors.txt

clean:
	rm src/__pycache__/*.pyc