FROM amazon/aws-lambda-python:3.8

ARG FUNCTION_DIR="/var/task/"

COPY ./ ${FUNCTION_DIR}

# Setup Python environment
RUN yum install gcc gcc72-c++ g++ libsndfile -y
RUN yum groupinstall 'Development Tools' -y
RUN pip install -r requirements.txt
RUN cd libsvm-3.25/python && python setup.py install
# Grab the zappa handler.py and put it in the working directory
RUN ZAPPA_HANDLER_PATH=$( \
    python -c "from zappa import handler; print (handler.__file__)" \
    ) \
    && echo $ZAPPA_HANDLER_PATH \
    && cp $ZAPPA_HANDLER_PATH ${FUNCTION_DIR}

CMD [ "handler.lambda_handler" ]