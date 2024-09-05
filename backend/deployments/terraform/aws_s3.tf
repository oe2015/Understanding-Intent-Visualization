# ##############################################################################
# FRAPPE Default
# ##############################################################################
resource "aws_s3_bucket" "default" {
  bucket        = "frappe-${terraform.workspace}-${var.aws_region}"
  force_destroy = terraform.workspace == "production" ? false : true

  tags = {
    Name = "FRAPPE - S3 - ${terraform.workspace} - ${var.aws_region}"
  }
}

resource "aws_s3_bucket_versioning" "default" {
  bucket = aws_s3_bucket.default.id
  versioning_configuration {
    status = "Disabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "default" {
  bucket = aws_s3_bucket.default.id

  rule {
    id = "frappe-tmp-${terraform.workspace}"
    filter {
      prefix = "tmp/"
    }
    expiration {
      days = 3
    }
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "default" {
  bucket = aws_s3_bucket.default.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "default" {
  bucket = aws_s3_bucket.default.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
