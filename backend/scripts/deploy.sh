#!/bin/bash

source "${BASH_SOURCE%/*}/common.sh"

# Description:
#   Deploys a deployment to an environment
#
# Usage:
# ./scripts/deploy.sh --interactive
# ./scripts/deploy.sh <environment>
# ./scripts/deploy.sh --environment <environment>
# ./scripts/deploy.sh --account <account> --deployment <deployment>

# Parse arguments and set variables using getops
arg0=$(basename "$0" .sh)
blnk=$(echo "$arg0" | sed 's/./ /g')

# Print usage information
usage_info()
{
    c_echo $YELLOW "Usage:"
    c_echo $YELLOW "  $arg0 <environment> [--auto]"
    c_echo $YELLOW "  $arg0 --interactive [--auto]"
    c_echo $YELLOW "  $arg0 --environment <environment> [--auto]"
    c_echo $YELLOW "  $arg0 --account <account> --deployment <deployment> [--auto]"
}

# Print usage in case of bad arguments
usage()
{
    #exec 1>2   # Send standard output to standard error
    usage_info
    exit 1
}

# Print error message and exit
error()
{
    echo "$arg0: $*" >&2
    exit 1
}

# Print help message
help()
{
    usage_info
    echo
    c_echo $YELLOW "Options:"
    echo "  {-i|--interactive}               -- Interactive mode"
    echo "  {-e|--environment} <environment> -- Environment to deploy to"
    echo "  {-a|--account} <account>         -- Account to deploy to"
    echo "  {-d|--deployment} <deployment>   -- Deployment to deploy"
    echo "  {-y|--auto}                      -- Auto approve"
    echo "  {-h|--help}                      -- Display this help and exit"
    echo
    echo
    c_echo $YELLOW "Available Deployments:"
    ls -d deployments/* | grep -v backend | sed 's/deployments\///g' | sort
    echo
    c_echo $YELLOW "Available Environments:"
    ls -d environments/* | grep -v backend | sed 's/.tfvars.json//g' | sed 's/environments\///g'
    echo
    c_echo $YELLOW "Available Accounts:"
    ls -d accounts/* | grep -v backend | sed 's/.tfvars.json//g' | sed 's/accounts\///g'
    exit 0
}

# Parse command line flags using getopts
flags()
{
    # No Arguments
    [ $# -eq 0 ] && usage

    # if argument is not a flag, then modify command with environment
    if [ "${1:0:1}" != "-" ]; then
        ENVIRONMENT=$1
        shift
        set -- "$@" "-e" "$ENVIRONMENT"
    fi
    
    # Parse Arguments
    while test $# -gt 0
    do
        case "$1" in
        (-i|--interactive)
            export INTERACTIVE=true
            shift;;
        (-e|--environment)
            shift
            [ $# = 0 ] && error "No environment specified"
            export ENVIRONMENT="$1"

            # Ensure that environment is valid
            if [ ! -f "environments/$ENVIRONMENT.tfvars.json" ]; then
                error "Invalid environment $ENVIRONMENT"
            fi
            shift;;
        (-a|--account)
            shift
            [ $# = 0 ] && error "No account specified"
            export ACCOUNT="$1"

            # Ensure that account is valid
            if [ ! -f "accounts/$ACCOUNT.tfvars.json" ]; then
                error "Invalid account $ACCOUNT"
            fi
            shift;;
        (-d|--deployment)
            shift
            [ $# = 0 ] && error "No deployment specified"
            export DEPLOYMENT="$1"

            # Ensure that deployment is valid
            if [ ! -d "deployments/$DEPLOYMENT" ]; then
                error "Invalid deployment $DEPLOYMENT"
            fi
            shift;;
        (-y|--auto)
            export AUTO_APPROVE="auto"
            shift;;
        (-h|--help)
            help;;
        (-*)
            error "Unknown option $1";;
        esac
    done

    # Ensure that account, deployment and environment flags are not passed with interactive flag
    if [ -n "$INTERACTIVE" ] && ([ -n "$ACCOUNT" ] || [ -n "$DEPLOYMENT" ] || [ -n "$ENVIRONMENT" ]); then
        error "--interactive flag must be passed alone"
    fi
    
    # Make environment terraform if only environment is passed
    if [ -n "$ENVIRONMENT" ] && [ -z "$ACCOUNT" ] && [ -z "$DEPLOYMENT" ]; then
        export DEPLOYMENT="terraform"
    fi
}

flags "$@"

# Confguring interactive mode
interactive()
{
    # Get all deployments except backend and list terraform at the top
    DEPLOYMENTS=$(ls -d deployments/* | grep -v backend | sed 's/deployments\///g' | sort)

    # Ask for deployment
    c_echo $GREEN "Which deployment would you like to deploy?"
    select DEPLOYMENT in $DEPLOYMENTS; do test -n "$DEPLOYMENT" && break; c_echo $RED ">>> Invalid Selection"; done

    # Ask for environment if deployment is environment based
    if [[ " ${ENVIRONMENT_DEPLOYMENTS[*]} " =~ " ${DEPLOYMENT} " ]]; then
        ENVIRONMENTS=$(ls -d environments/* | sed 's/.tfvars.json//g' | sed 's/environments\///g')
        c_echo $GREEN "On which environment would you like to deploy?"
        select ENVIRONMENT in $ENVIRONMENTS; do test -n "$ENVIRONMENT" && break; c_echo $RED ">>> Invalid Selection"; done

        # Ask for confirmation and default to yes
        echo
        c_echo $YELLOW "Are you sure you want to deploy $DEPLOYMENT to $ENVIRONMENT? (y/n) (Default: y)"
        read -r CONFIRMATION
        if [[ -z $CONFIRMATION ]]; then
            CONFIRMATION="y"
        fi
        if [[ $CONFIRMATION != "y" ]]; then
            c_echo $RED "Exiting..."
            exit 1
        fi
    fi

    # Ask for account if deployment is account based
    if [[ " ${ACCOUNT_DEPLOYMENTS[*]} " =~ " ${DEPLOYMENT} " ]]; then
        ACCOUNTS=$(ls -d accounts/* | grep -v backend | sed 's/.tfvars.json//g' | sed 's/accounts\///g')
        c_echo $GREEN "On which account would you like to deploy?"
        select ACCOUNT in $ACCOUNTS; do test -n "$ACCOUNT" && break; echo ">>> Invalid Selection"; done

        # Ask for confirmation and default to yes
        echo
        c_echo $YELLOW "Are you sure you want to deploy $DEPLOYMENT to $ACCOUNT? (y/n) (Default: y)"
        read -r CONFIRMATION
        if [[ -z $CONFIRMATION ]]; then
            CONFIRMATION="y"
        fi
        if [[ $CONFIRMATION != "y" ]]; then
            c_echo $RED "Exiting..."
            exit 1
        fi
    fi
}

# If --interactive is passed, run interactive mode
if [ -n "$INTERACTIVE" ]; then
    c_echo $YELLOW "Running in interactive mode..."
    interactive
fi

# Do the build and deployment only if the deployment is terraform
if [[ $DEPLOYMENT == *"terraform"* ]]; then
    echo
    c_echo $GREEN "Building for ${ENVIRONMENT}"

    scripts/build.sh $ENVIRONMENT
    if [ "$?" -ne "0" ]; then
        c_echo $RED "Build failed"
        exit 1
    fi
    echo
    c_echo $GREEN "Build succeeded"

    echo
    c_echo $GREEN "Deploying to ${ENVIRONMENT}"

    deployments/terraform/terraform-apply.sh $ENVIRONMENT $AUTO_APPROVE
    if [ "$?" -ne "0" ]; then
        c_echo $RED "Deployment failed"
        exit 1
    fi
    echo
    c_echo $GREEN "Deployment succeeded"
fi

# Do the the deployment only if the deployment is environment based and not terraform
if [[ " ${ENVIRONMENT_DEPLOYMENTS[*]} " =~ " ${DEPLOYMENT} " ]] && [[ $DEPLOYMENT != *"terraform"* ]]; then
    echo
    c_echo $GREEN "Deploying ${DEPLOYMENT} to ${ENVIRONMENT}"

    deployments/${DEPLOYMENT}/${DEPLOYMENT}-apply.sh $ENVIRONMENT $AUTO_APPROVE
    if [ "$?" -ne "0" ]; then
        c_echo $RED "Deployment failed"
        exit 1
    fi
    echo
    c_echo $GREEN "Deployment succeeded"
fi

# Do the the deployment only if the deployment is account based
if [[ " ${ACCOUNT_DEPLOYMENTS[*]} " =~ " ${DEPLOYMENT} " ]]; then
    echo
    c_echo $GREEN "Deploying ${DEPLOYMENT} to ${ACCOUNT}"

    deployments/${DEPLOYMENT}/${DEPLOYMENT}-apply.sh $ACCOUNT $AUTO_APPROVE
    if [ "$?" -ne "0" ]; then
        c_echo $RED "Deployment failed"
        exit 1
    fi
    echo
    c_echo $GREEN "Deployment succeeded"
fi


