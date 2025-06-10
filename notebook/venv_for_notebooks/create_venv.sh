virtualenv .venv
source .venv/bin/activate
echo "Installing modules inside virtual environment ... \n"
pip install -r ./notebook/venv_for_notebooks/create_venv_requirements.txt 
python -m pip install graphviz