# ##############################################################################
# Snippet to require usage of a 'non-default' workspace.
# ##############################################################################
data "local_file" "is_default_workspace" {
  count    = terraform.workspace == "default" ? 1 : 0
  filename = "ERROR: Default workspace is not allowed."
}

# ##############################################################################
# Backend
# ##############################################################################
terraform {
  backend "s3" {
    key                  = "terraform.tfstat"
    workspace_key_prefix = "frappe"
  }
}

# ##############################################################################
# Terraform and Provider Requirements
# ##############################################################################
terraform {
  required_version = "~>1.9.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~>5.64.0"
    }
  }
}

provider "aws" {
  region              = var.aws_region
  profile             = var.aws_profile
  allowed_account_ids = [var.aws_account]

  default_tags {
    # tags holds the default tags applied to all resources.
    # Individual resources should override Name and Function,
    # (as well as anything else they find appropriate).
    tags = {
      Name          = "FRAPPE - ${terraform.workspace} - ${var.aws_region}"
      Group         = "FRAPPE - ${terraform.workspace}"
      Owner         = "FRAPPE Team"
      Purpose       = "FRAming Persuasion and Propaganda Explorer - ${terraform.workspace} - ${var.aws_region}"
      Function      = "App"
      Environment   = terraform.workspace
      Criticality   = "High"
      TTL           = "2021-06-01"
      MangedBy      = "terraform"
      Business_Unit = "Infrastructure"
      Region        = var.aws_region
    }
  }
}

data "aws_caller_identity" "current" {}

# ##############################################################################
# Variables
# ##############################################################################
variable "aws_account_name" {
  type = string
  validation {
    condition     = length(var.aws_account_name) > 0
    error_message = "Use the -var-file flag with the terraform command to specify the account configuration."
  }
}

variable "aws_account" {
  type = string
  validation {
    condition     = length(var.aws_account) > 0
    error_message = "Use the -var-file flag with the terraform command to specify the account configuration."
  }
}

variable "aws_region" {
  type = string
  validation {
    condition     = length(var.aws_region) > 0
    error_message = "Use the -var-file flag with the terraform command to specify the account configuration."
  }
}

variable "aws_profile" {
  type = string
  validation {
    condition     = length(var.aws_profile) > 0
    error_message = "Use the -var-file flag with the terraform command to specify the account configuration."
  }
}

variable "build_version" {}

variable "commit" {}
