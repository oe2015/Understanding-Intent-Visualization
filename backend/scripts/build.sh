#!/bin/bash

source "${BASH_SOURCE%/*}/common.sh"

if [ -z "$1" ] ; then
    c_echo $RED "Need to provide the workspace as the first argument"
    exit 1
fi

# Clean up the workspace
c_echo $GREEN "Cleaning unecessary cache and temps"
find . -name '.DS_Store' -type f -delete
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

# Building lambda_handlers
c_echo $GREEN "
--------------------------------------------------------------------------------
--                          Building lambda_handlers                          --
--------------------------------------------------------------------------------
"

mkdir -p $LAMBDAHANDLERS_PATH

# for all cmds build the target binaries
for cmdpath in ./cmd/lambda_*; do
    APP_NAME=$(basename $cmdpath)

    echo 
    c_echo $GREEN "#------------------  Building $APP_NAME  ------------------#"
    mkdir -p $LAMBDAHANDLERS_PATH/$APP_NAME
    rsync -irvcP --delete --out-format='Changed file: %i %n%L' $cmdpath/ \
          --log-file=/tmp/rsync.log \
          --log-file-format='Changed file: %i %n%L' \
          --exclude "tests/" \
          --exclude "tmp/" \
          $LAMBDAHANDLERS_PATH/$APP_NAME; (( exit_status = exit_status || $? ))
    echo
    rsync -irvcP --out-format='Changed file: %i %n%L' ./configs/$1.json \
          --log-file=/tmp/rsync.log \
          --log-file-format='Changed file: %i %n%L' \
          $LAMBDAHANDLERS_PATH/$APP_NAME/config.json; (( exit_status = exit_status || $? ))

    echo

    # Return error if any of the above commands failed
    if [ "$exit_status" -eq 1 ]; then
        c_echo $RED "Build Failed!!!"
        exit 1
    fi
    
    # Remove the rsync log file
    rm /tmp/rsync.log

    # Generate hash files for the codebase, lambda_handlers and lambda_layers
    hn=$(find $LAMBDAHANDLERS_PATH/$APP_NAME/* -type f | sort | tr '\n' '\0' | xargs -0 cat | openssl md5 -r |  awk '{print $1}')
    hp=$(cat $LAMBDAHANDLERS_PATH/$APP_NAME.hash 2>&1 || echo UNK)
    if [ "$hn" = "$hp" ]; then
        c_echo $YELLOW "Skipping image creation for $APP_NAME as contents haven't changed"
        continue
    fi

    # Write out the hash
    printf '%b' "$hn" > $LAMBDAHANDLERS_PATH/$APP_NAME.hash

    # Getting AWS Account and Region and create name for the docker image
    AWS_ACCOUNT=$(jq -r '."aws_account"' ./environments/$1.tfvars.json)
    AWS_REGION=$(jq -r '."aws_region"' ./environments/$1.tfvars.json)
    LAMBDA_NAME="frappe-$(echo "$APP_NAME" | sed 's/_/-/g')-${1}-$AWS_REGION"    
    REPO_NAME="$AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$LAMBDA_NAME"
    c_echo $GREEN "ECR Repo: $REPO_NAME"
    
    # AWS ECR Login
    c_echo $GREEN "Logging into AWS ECR"
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com; (( exit_status = exit_status || $? ))

    # Return error if any of the above commands failed
    if [ "$exit_status" -eq 1 ]; then
        c_echo $RED "Build Failed!!!"
        exit 1
    fi

    # Build and push the docker image using base image as argument
    c_echo $GREEN "Building docker image for $APP_NAME"
    docker build -t $REPO_NAME:latest \
                 --progress=plain \
                 -f $LAMBDAHANDLERS_PATH/$APP_NAME/Dockerfile \
                 $LAMBDAHANDLERS_PATH/$APP_NAME; (( exit_status = exit_status || $? ))
    
    # Return error if any of the above commands failed
    if [ "$exit_status" -eq 1 ]; then
        c_echo $RED "Build Failed!!!"
        exit 1
    fi

    # Push the docker image
    c_echo $GREEN "Pushing docker image for $APP_NAME"
    docker push $REPO_NAME:latest; (( exit_status = exit_status || $? ))

    # Return error if any of the above commands failed
    if [ "$exit_status" -eq 1 ]; then
        c_echo $RED "Build Failed!!!"
        exit 1
    fi
    
done

echo

# Write out the tf vars for this version of the lambdas
c_echo $GREEN "
--------------------------------------------------------------------------------
--       Finalizing Terraform Vars ignore.lambda_handlers.auto.tfvars         --
--------------------------------------------------------------------------------
"

echo "#--------------------------------------------------------------" > $TERRAFORM_PATH/ignore.lambda_handlers.auto.tfvars
echo "# This file is auto generated by build.sh. Do not edit." >> $TERRAFORM_PATH/ignore.lambda_handlers.auto.tfvars
echo "#--------------------------------------------------------------" >> $TERRAFORM_PATH/ignore.lambda_handlers.auto.tfvars
echo "build_version = \"$APP_VERSION\"" >> $TERRAFORM_PATH/ignore.lambda_handlers.auto.tfvars
echo "commit = \"$COMMIT\"" >> $TERRAFORM_PATH/ignore.lambda_handlers.auto.tfvars
