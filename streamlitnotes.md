1. Generate requirements.txt
https://stackoverflow.com/questions/30363976/how-should-i-generate-requirements-txt-pip-freeze-not-a-good-way

- install pipreqs library
~$ conda install -c conda-forge pipreqs

- go into project directory
~$ pipreqs --force

- this will generate the requirements.txt base on your project files


2. https://www.geeksforgeeks.org/a-beginners-guide-to-streamlit/

3. if encounter "ModuleNotFound Error":
- https://discuss.streamlit.io/t/no-module-named-sklearn/9218/19
~$ pip install scikit-learn==version_use_to_export_pickle_file