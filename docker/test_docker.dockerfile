FROM python:3.11-slim
RUN pip install pandas
# check if the pandas is installed successfully
RUN python -c "import pandas ; print(pandas.__version__)"
RUN echo "pandas is installed successfully"
