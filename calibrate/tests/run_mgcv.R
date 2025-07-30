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
# The goal is to create a model with a specific number of parameters:
# - Intercept: 1
# - variable_one Main Effect: 10 coefficients
# - variable_two Main Effect: 11 coefficients
# - Interaction Effect: 121 coefficients (11 * 11)
# - Total: 1 + 10 + 11 + 121 = 143 parameters
# All splines are specified as cubic B-splines (degree=3), which is achieved with m=3.

# Note on knots and basis dimension 'k':
# For cubic P-splines (bs="ps", m=3), the number of parameters is k-1 due to a sum-to-zero constraint.
# The number of internal knots is k - degree - 1 = k - 3 - 1 = k - 4.
# To get 8 internal knots, we need k = 8 + 4 = 12.

# - variable_one: Unpenalized (fx=TRUE), cubic. To get 10 parameters, we need a basis dimension k=11 (k-1=10).
#   This results in k-4 = 11-4 = 7 internal knots, a slight deviation from the target to meet the parameter count.
# - variable_two: Penalized, cubic. To get 11 parameters, we need k=12 (k-1=11). This results in k-4 = 12-4 = 8 internal knots.
# - interaction: Penalized, cubic. To get 121 parameters (11*11), the marginal bases must each have k=12
#   (since (k1-1)*(k2-1) = 11*11). This is consistent with 8 internal knots for each marginal.

gam_formula <- outcome ~ s(variable_one, bs = "ps", k = 11, fx = TRUE, m = 3) +
                           s(variable_two, bs = "ps", k = 12, m = 3) +
                           ti(variable_one, variable_two, bs = c("ps", "ps"), k = c(12, 12), m = 3)

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
