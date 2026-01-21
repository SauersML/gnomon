library(bigsnpr)
library(bigstatsr)
library(data.table)

# Ensure R.utils is available for snp_asGeneticPos
if (!require("R.utils", quietly = TRUE)) install.packages("R.utils", repos="https://cloud.r-project.org")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
    stop("Usage: Rscript run_ldpred2.R <bfile_train> <pheno_file> <bfile_val> <out_prefix>")
}

bfile_train <- args[1]
pheno_file <- args[2]  # Expecting FID IID Pheno (no header)
bfile_val <- args[3]   # Used for hyperparam tuning (LDpred2-auto uses it internally or we use grid)
out_prefix <- args[4]

# Avoid nested parallelism on CI runners.
# bigsnpr/bigstatsr can use parallelism internally, and some CI environments
# already set parallel backends, which triggers `?assert_cores`.
options(bigstatsr.ncores = 1)
Sys.setenv(OMP_NUM_THREADS = "1")
Sys.setenv(OPENBLAS_NUM_THREADS = "1")
Sys.setenv(MKL_NUM_THREADS = "1")

# Read Genotypes
rds_train <- snp_readBed2(paste0(bfile_train, ".bed"), backingfile = tempfile())
obj.bigSNP.train <- snp_attach(rds_train)
G <- obj.bigSNP.train$genotypes
CHR <- obj.bigSNP.train$map$chromosome
POS <- obj.bigSNP.train$map$physical.pos
NCORES <- 1

# Read Phenotype
pheno <- fread(pheno_file, header=FALSE)
y_train <- pheno$V3

# Run GWAS (Marginal effects)
# We assume quantitative trait for simplicity in simulation, or binary?
# The user's simulation is binary (0/1).
# big_univLogReg for binary
gwas_train <- big_univLogReg(G, y_train, covar.train = NULL, ncores = NCORES)
chisq <- y_train # just dummy for now
beta <- gwas_train$estim
se <- gwas_train$std.err
lpval <- -predict(gwas_train)

# LD Correlation Matrix
# Compute LD on a subset of SNPs/Indivs for speed (or full if small)
# Using restricted chromosome info from map
POS2 <- snp_asGeneticPos(CHR, POS, dir = "tmp-data") # Need map? defaulting
# Just use simple window
corr <- snp_cor(G, ncores = NCORES, size = 3 / 1000) # 3 cM? or 1000 SNPs

# LDpred2-auto
# Run
ldpred_auto <- snp_ldpred2_auto(
  corr, 
  df_beta = data.frame(beta = beta, beta_se = se, n_eff = length(y_train)),
  h2_init = 0.5,
  vec_p_init = seq_log(1e-4, 0.9, length.out = 30),
  ncores = NCORES
)

# Get final beta (mean over chains)
beta_auto <- rowMeans(sapply(ldpred_auto, function(auto) auto$beta_est))

# Save Betas
# ID A1 Beta
map <- obj.bigSNP.train$map
out_df <- data.frame(
  ID = map$marker.ID,
  A1 = map$allele1,
  Beta = beta_auto
)

write.table(out_df, paste0(out_prefix, ".scores"), quote=FALSE, row.names=FALSE, col.names=FALSE)
