#FROM huecker.io/library/alpine:latest
FROM pytorch/pytorch
#RUN pip install opencv-python
#RUN apt-get update && apt-get install libgl1
RUN pip install jupyter
RUN pip install jupyterlab
RUN pip install notebook
EXPOSE 9999
ENV NAME vgolikov

RUN pip install scikit-learn
RUN pip install pandas
RUN pip install netCDF4
RUN pip install matplotlib
RUN pip install pendulum
RUN conda install -c conda-forge wrf-python=1.3.4.1
RUN pip install transformers
RUN pip install SciPy
RUN pip install optuna

RUN conda install -c conda-forge esmpy
RUN pip install pyproj
RUN pip install global-land-mask
RUN pip install cartopy
RUN pip install addict
COPY . /home
WORKDIR /home/

#CMD ["python", "experiments/conv2d/main.py"]
