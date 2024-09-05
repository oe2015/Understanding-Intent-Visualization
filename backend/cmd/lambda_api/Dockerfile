# Base image as a build arg
ARG BASE_LAMBDA_IMAGE

# download base image
FROM ${BASE_LAMBDA_IMAGE}

# Copy function code
COPY . ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.lambda_handler" ]
