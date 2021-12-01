FROM jupyter/scipy-notebook

RUN pip install scikit-learn
RUN pip3 install qiskit
RUN pip3 install pylatexenc
RUN pip3 install plotly
WORKDIR /app
ADD . /app



