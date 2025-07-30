if (!require(mgcv)) {
  install.packages("mgcv")
  library(mgcv)
}

# Define file paths for clarity
input_csv_file <- 'synthetic_classification_data.csv'
output_rds_file <- 'gam_model_fit.rds'


# --- 2. Load Data ---
cat("Loading data from '", input_csv_file, "'...\n\n", sep="")
data <- read.csv(input_csv_file)


# --- 3. Define and Fit the GAM ---
# - variable_one: unpenalized (fx=TRUE), cubic (default), 8 INTERNAL knots (k=6)
# - variable_two: penalized, cubic (default), 8 INTERNAL knots (k=6)
# - interaction: penalized, cubic (default), 8 INTERNAL knots for each marginal (k=c(6,6))
#
# Note on knots: For bs="ps" with degree=3, number of internal knots = k - degree + 1.
# Therefore, to get 8 internal knots, k must be 6.
# This results in 10 total unique knot locations (2 boundary + 8 internal).

gam_formula <- outcome ~ s(variable_one, bs = "ps", k = 6, fx = TRUE) +
                           s(variable_two, bs = "ps", k = 6) +
                           ti(variable_one, variable_two, bs = c("ps", "ps"), k = c(6, 6))

cat("Fitting the GAM... This may take a moment.\n")
# Fit the model using REML for smoothness selection
# The family is 'binomial' because our outcome is 0 or 1.
gam_fit <- gam(
  gam_formula,
  family = binomial(),
  data = data,
  method = "REML" # Restricted Maximum Likelihood is a good default
)


# --- 4. Inspect the Model Fit ---

cat("\n--- GAM Model Summary ---\n")
print(summary(gam_fit))


# --- 5. Save the Model Object ---

cat("\nSaving the fitted GAM object to '", output_rds_file, "'...\n", sep="")
saveRDS(gam_fit, file = output_rds_file)

cat("Script finished successfully.\n\n")
