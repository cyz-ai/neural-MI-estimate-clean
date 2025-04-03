rm -rf `find -type d -name .ipynb_checkpoints`
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf