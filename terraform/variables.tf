variable "databricks_profile" {
  description = "Local Databricks CLI profile name"
  type        = string
  default     = "azure-video"
}

# ---- Naming -----------------------------------------------------------------
variable "name_prefix" {
  description = "Short name used as a prefix for all resources"
  type        = string
  default     = "lumen"
}

variable "catalog_name" {
  description = "Existing UC catalog to host lumen_bronze/silver/gold schemas"
  type        = string
  default     = "classic_stable_89j9qf"
}

# ---- Lakebase Autoscale -----------------------------------------------------
variable "lakebase_project_id" {
  description = "Lakebase Autoscale project id (lowercase, kebab)"
  type        = string
  default     = "ecommerce-search-demo"
}

variable "lakebase_pg_version" {
  description = "Postgres major version on Lakebase Autoscale (16 or 17)"
  type        = number
  default     = 17
}

variable "lakebase_branch_id" {
  description = "Branch name used by the app"
  type        = string
  default     = "production"
}

variable "lakebase_endpoint_id" {
  description = "Read/write endpoint id"
  type        = string
  default     = "primary"
}

variable "lakebase_min_cu" {
  description = "Autoscale minimum CU (1 CU ≈ 2 GB RAM on Lakebase Autoscale)"
  type        = number
  default     = 1.0
}

variable "lakebase_max_cu" {
  description = "Autoscale maximum CU (max-min must be <= 16)"
  type        = number
  default     = 2.0
}

variable "lakebase_suspend_seconds" {
  description = "Idle-suspend timeout in seconds. Demo: keep warm — use 604800s (7d)."
  type        = number
  default     = 604800
}

variable "lakebase_app_database" {
  description = "Application database created inside the Lakebase branch (lowercase, no underscores)"
  type        = string
  default     = "appdb"
}

# ---- Embedding endpoint -----------------------------------------------------
variable "embedding_endpoint_name" {
  description = "Foundation-model endpoint used for batch + query embeddings"
  type        = string
  default     = "databricks-bge-large-en"
}

# ---- App --------------------------------------------------------------------
variable "app_name" {
  description = "Databricks App name (lowercase, kebab)"
  type        = string
  default     = "lumen-recommender"
}

variable "app_compute_size" {
  description = "MEDIUM or LARGE"
  type        = string
  default     = "MEDIUM"
}

variable "app_workspace_source_path" {
  description = "Workspace path that holds the uploaded app source (set by null_resource)"
  type        = string
  default     = "/Workspace/Apps/lumen-recommender"
}
