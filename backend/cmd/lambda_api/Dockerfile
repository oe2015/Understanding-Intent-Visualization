# Download the base image for arm64
FROM --platform=amd64 public.ecr.aws/lambda/python:3.11

# Copy the requirements.txt file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies using requirements.txt
RUN pip install -r requirements.txt -t ${LAMBDA_TASK_ROOT} --no-cache-dir 

# Copy function code
COPY *.py ${LAMBDA_TASK_ROOT}/

# List all the files in the current directory
RUN ls -al

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.lambda_handler" ]
