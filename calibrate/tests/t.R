# -----------------------------------------------------------------------------
# Fit a Generalized Additive Model (GAM) to synthetic classification data
#
# This script performs the following steps:
# 1. Loads the 'mgcv' library for GAMs.
# 2. Reads the synthetic data created by the Python script.
# 3. Defines a GAM formula for a logistic model (binary outcome).
#    - variable_one: UNPENALIZED main effect (P-spline basis).
#    - variable_two: PENALIZED main effect (P-spline basis).
#    - interaction:  PENALIZED interaction effect.
# 4. Fits the GAM to the data.
# 5. Prints a summary of the fitted model to the console.
# 6. Saves the entire fitted GAM object to a file ('gam_fit.rds')
#    so it can be reloaded and analyzed later without re-fitting.
# -----------------------------------------------------------------------------

# --- 1. Preamble: Load Libraries and Define Filenames ---

# Ensure the 'mgcv' package is installed and loaded.
if (!require(mgcv)) {
  install.packages("mgcv")
  library(mgcv)
}

# Define file paths for clarity
input_csv_file <- 'synthetic_classification_data.csv'
output_rds_file <- 'gam_model_fit.rds'


# --- 2. Load Data ---

# Check if the input file exists before trying to read it.
if (!file.exists(input_csv_file)) {
  stop(paste("Error: Input file not found. Please run the Python script first to create '",
             input_csv_file, "'", sep=""))
}

cat("Loading data from '", input_csv_file, "'...\n\n", sep="")
data <- read.csv(input_csv_file)


# --- 3. Define and Fit the GAM ---

# Define the model formula. This is the core of the specification.
# We are fitting a logistic regression model (outcome is 0 or 1).
# The formula specifies three distinct components:
#
#   s(variable_one, bs = "ps", k = 10, fx = TRUE):
#     - An unpenalized smooth of variable_one.
#     - `bs = "ps"` specifies a P-spline basis.
#     - `k = 10` sets the basis dimension (degrees of freedom).
#     - `fx = TRUE` is the key: it fixes the degrees of freedom, making it UNPENALIZED.
#
#   s(variable_two, bs = "ps", k = 20):
#     - A standard PENALIZED P-spline smooth of variable_two.
#     - `k = 20` sets the upper limit on flexibility. The final edf will be lower.
#
#   ti(variable_one, variable_two, bs = c("ps", "ps"), k = c(10, 10)):
#     - A PENALIZED tensor product interaction (`ti`) of the two variables.
#     - `ti()` is used to model the "pure" interaction, separate from main effects.
#     - `bs = c("ps", "ps")` specifies P-spline marginal bases for each variable.
#
gam_formula <- outcome ~ s(variable_one, bs = "ps", k = 10, fx = TRUE) +
                           s(variable_two, bs = "ps", k = 20) +
                           ti(variable_one, variable_two, bs = c("ps", "ps"), k = c(10, 10))

cat("Fitting the GAM... This may take a moment.\n")
# Fit the model using REML for smoothness selection, which is generally robust.
# The family is 'binomial' because our outcome is 0 or 1.
gam_fit <- gam(
  gam_formula,
  family = binomial(),
  data = data,
  method = "REML" # Restricted Maximum Likelihood is a good default
)


# --- 4. Inspect the Model Fit ---

cat("\n--- GAM Model Summary ---\n")
# The summary shows the Effective Degrees of Freedom (edf).
# - For the unpenalized `s(variable_one)`, edf will be very close to k-1 = 9.
# - For the penalized terms, edf will be less than their maximum possible value.
print(summary(gam_fit))


# --- 5. Save the Model Object ---

cat("\nSaving the fitted GAM object to '", output_rds_file, "'...\n", sep="")
# saveRDS is the standard R function for saving a single R object to a file.
saveRDS(gam_fit, file = output_rds_file)

cat("Script finished successfully.\n\n")


# --- 6. How to Reload the Model Later ---
#
# To load the model back into R in a new session, simply run:
#
# reloaded_gam <- readRDS('gam_model_fit.rds')
#
# # You can then inspect it or make predictions:
# summary(reloaded_gam)
# plot(reloaded_gam, pages = 1)
#
# -----------------------------------------------------------------------------
