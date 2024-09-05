# Download the base image for arm64
FROM --platform=amd64 public.ecr.aws/lambda/python:3.10

# Copy function code
COPY . ${LAMBDA_TASK_ROOT}

# Install dependencies using requirements.txt
RUN pip install flask -t ${LAMBDA_TASK_ROOT}
RUN pip install aws-lambda-wsgi -t ${LAMBDA_TASK_ROOT}
RUN pip install aws-lambda-powertools -t ${LAMBDA_TASK_ROOT}
RUN pip install transformers -t ${LAMBDA_TASK_ROOT}
RUN pip install torch -t ${LAMBDA_TASK_ROOT}
RUN pip install nltk -t ${LAMBDA_TASK_ROOT}
RUN pip install sentencepiece -t ${LAMBDA_TASK_ROOT}