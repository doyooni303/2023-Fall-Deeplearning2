# base Image
FROM nvcr.io/nvidia/pytorch:22.12-py3

# default
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y vim \
    && apt-get install -y git \
    && apt-get install -y g++ && apt-get install -y curl

# install basic libraries via requirements.txt
RUN pip install --upgrade pip
RUN pip install matplotlib seaborn scikit-learn scipy pandas numpy ipdb xlutils tabulate sktime==0.4.1

# fonts
RUN apt-get install -y fonts-nanum
RUN rm -rf ~/.cache/matplotlib/*

# requirements.txt
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]
