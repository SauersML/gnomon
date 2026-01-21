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

map_dir <- Sys.getenv("LDPRED2_MAP_DIR", unset = "tmp-data")
dir.create(map_dir, recursive = TRUE, showWarnings = FALSE)

cat("LDpred2 diagnostics:\n")
cat(paste0("  cwd=", getwd(), "\n"))
cat(paste0("  tempdir=", tempdir(), "\n"))
cat(paste0("  args=", paste(args, collapse = " "), "\n"))
cat(paste0("  map_dir=", map_dir, "\n"))
cat(paste0("  map_dir exists=", dir.exists(map_dir), "\n"))
tmp_write_ok <- FALSE
tmp_write_err <- ""
try({
  file.create(file.path(map_dir, "__write_test"))
  tmp_write_ok <- file.exists(file.path(map_dir, "__write_test"))
  if (tmp_write_ok) file.remove(file.path(map_dir, "__write_test"))
}, silent = TRUE)
cat(paste0("  map_dir writeable=", tmp_write_ok, "\n"))

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

uniq_y <- sort(unique(y_train))
is_binary <- length(uniq_y) <= 2 && all(uniq_y %in% c(0, 1))
cat(paste0("  y_unique=", paste(uniq_y, collapse=","), "\n"))
if (is_binary) {
  y_tab <- table(y_train)
  cat(paste0("  y_counts_0=", ifelse(!is.na(y_tab["0"]), y_tab["0"], 0), "\n"))
  cat(paste0("  y_counts_1=", ifelse(!is.na(y_tab["1"]), y_tab["1"], 0), "\n"))
  n0_chk <- sum(y_train == 0)
  n1_chk <- sum(y_train == 1)
  if (n0_chk == 0 || n1_chk == 0) {
    stop("Binary phenotype must have both controls (0) and cases (1).")
  }
}

n_train <- length(y_train)

maf <- snp_MAF(G)
mac <- 2 * n_train * maf
min_mac <- as.numeric(Sys.getenv("LDPRED2_MIN_MAC", unset = "20"))

keep_maf <- is.finite(maf) & maf > 0 & maf < 1
keep_mac <- is.finite(mac) & mac >= min_mac
ind_col_maf <- which(keep_maf & keep_mac)
cat(paste0("  snps_total=", ncol(G), "\n"))
cat(paste0("  snps_after_maf_mac=", length(ind_col_maf), "\n"))
if (length(ind_col_maf) == 0) {
  stop("No SNPs left after MAF/MAC filtering.")
}

# Run GWAS (Marginal effects)
if (is_binary) {
  gwas_train <- big_univLogReg(G, y_train, ind.col = ind_col_maf, covar.train = NULL, ncores = NCORES)
} else {
  gwas_train <- big_univLinReg(G, y_train, ind.col = ind_col_maf, covar.train = NULL, ncores = NCORES)
}
chisq <- y_train # just dummy for now
beta <- gwas_train$estim
se <- gwas_train$std.err
lpval <- -predict(gwas_train)

ok_beta <- is.finite(beta)
ok_se <- is.finite(se) & se > 0
ind_col_ok <- ind_col_maf[which(ok_beta & ok_se)]
cat(paste0("  snps_after_se_filter=", length(ind_col_ok), "\n"))
cat(paste0("  snps_dropped_bad_beta=", sum(!ok_beta), "\n"))
cat(paste0("  snps_dropped_bad_se=", sum(!ok_se), "\n"))
if (length(ind_col_ok) == 0) {
  stop("No SNPs left after filtering invalid beta/beta_se.")
}

beta <- beta[ok_beta & ok_se]
se <- se[ok_beta & ok_se]

# LD Correlation Matrix
# Compute LD on a subset of SNPs/Indivs for speed (or full if small)
# Using restricted chromosome info from map
POS2 <- tryCatch({
  snp_asGeneticPos(CHR, POS, dir = map_dir)
}, error = function(e) {
  cat("snp_asGeneticPos failed. Diagnostics:\n")
  cat(paste0("  map_dir=", map_dir, "\n"))
  cat(paste0("  map_dir exists=", dir.exists(map_dir), "\n"))
  if (dir.exists(map_dir)) {
    cat("  map_dir listing:\n")
    print(list.files(map_dir, all.files = TRUE))
  }
  stop(e)
})
# Just use simple window
corr <- snp_cor(G, ind.col = ind_col_ok, ncores = NCORES, size = 3 / 1000) # 3 cM? or 1000 SNPs

# LDpred2-auto
# Run
ldpred_auto <- snp_ldpred2_auto(
  corr, 
  df_beta = data.frame(beta = beta, beta_se = se, n_eff = if (is_binary) {
    n0 <- sum(y_train == 0)
    n1 <- sum(y_train == 1)
    4 / (1 / n0 + 1 / n1)
  } else {
    length(y_train)
  }),
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
  ID = map$marker.ID[ind_col_ok],
  A1 = map$allele1[ind_col_ok],
  Beta = beta_auto
)

write.table(out_df, paste0(out_prefix, ".scores"), quote=FALSE, row.names=FALSE, col.names=FALSE)
