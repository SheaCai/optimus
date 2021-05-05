FROM python:3.6

COPY . /optimus

WORKDIR /optimus

RUN pip3 install -r requirements.txt

ENTRYPOINT [ "/bin/bash" ]
