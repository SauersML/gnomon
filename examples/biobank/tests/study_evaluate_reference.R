# Reference values for test_study_evaluate_reference.py from R's own implementations.
# Rscript study_evaluate_reference.R <dir> <horizon>: reads <dir>/*.csv, writes <dir>/r_values.csv (name, value)
# and <dir>/r_vectors.csv (one column per per-row reference vector). Base stats and survival always; pROC,
# timeROC and riskRegression only when installed (their values are then absent, and the tests say so).
args <- commandArgs(TRUE)
dir <- args[1]
h <- as.numeric(args[2])
suppressMessages(library(survival))
has <- function(p) suppressWarnings(requireNamespace(p, quietly = TRUE))
values <- list()
vectors <- list()
put <- function(name, value) {
  value <- as.numeric(value)
  if (length(value) != 1) stop(sprintf("reference value %s has length %d", name, length(value)))
  values[[name]] <<- value
}
tight <- glm.control(epsilon = 1e-14, maxit = 200)

# ---- binary: calibration, loess, pROC ----
b <- read.csv(file.path(dir, "binary.csv"))
lp <- log(b$p1) - log1p(-b$p1)
f0 <- glm(b$y ~ 1 + offset(lp), family = binomial, control = tight)
put("cal_int", coef(f0)[1]); put("cal_int_se", sqrt(vcov(f0)[1, 1]))
f1 <- glm(b$y ~ lp, family = binomial, control = tight)
put("cal_slope", coef(f1)[2]); put("cal_slope_se", sqrt(vcov(f1)[2, 2]))
lo <- loess(y ~ p1, data = b, span = 0.75, degree = 2, family = "gaussian",
            control = loess.control(surface = "direct"))
vectors$loess_fit <- predict(lo, newdata = data.frame(p1 = b$p1))
put("ici", mean(abs(b$p1 - vectors$loess_fit)))
if (has("pROC")) {
  r1 <- pROC::roc(b$y, b$p1, levels = c(0, 1), direction = "<", quiet = TRUE)
  r2 <- pROC::roc(b$y, b$p2, levels = c(0, 1), direction = "<", quiet = TRUE)
  put("auc", pROC::auc(r1)); put("auc_var", pROC::var(r1, method = "delong"))
  test <- pROC::roc.test(r1, r2, method = "delong", paired = TRUE)
  put("d_auc", pROC::auc(r1) - pROC::auc(r2)); put("d_auc_z", test$statistic)
}

# ---- weighted calibration and loess (IPCW weights, known-status rows) ----
wb <- read.csv(file.path(dir, "weighted.csv"))
wlp <- log(wb$p) - log1p(-wb$p)
g0 <- suppressWarnings(glm(wb$y ~ 1 + offset(wlp), family = quasibinomial, weights = wb$w, control = tight))
g1 <- suppressWarnings(glm(wb$y ~ wlp, family = quasibinomial, weights = wb$w, control = tight))
put("wcal_int", coef(g0)[1]); put("wcal_slope", coef(g1)[2])
wlo <- loess(y ~ p, data = wb, weights = w, span = 0.75, degree = 2, family = "gaussian",
             control = loess.control(surface = "direct"))
vectors$wloess_fit <- predict(wlo, newdata = data.frame(p = wb$p))

# ---- survival: Aalen-Johansen, concordance, reverse KM, brute-force competing-risk definitions ----
brute <- function(s, g_left, g_at_h) {
  # Blanche definition-2 IPCW AUC and the IPCW Brier with deaths as known non-cases, pair by pair.
  case <- s$code == 1 & s$t <= h
  died <- s$code == 2 & s$t <= h
  past <- s$t > h
  w <- ifelse(case | died, 1 / g_left, ifelse(past, 1 / g_at_h, 0))
  ctrl <- died | past
  num <- 0
  for (i in which(case)) num <- num + w[i] * sum(w[ctrl] * ((s$p1[i] > s$p1[ctrl]) + 0.5 * (s$p1[i] == s$p1[ctrl])))
  auc <- num / (sum(w[case]) * sum(w[ctrl]))
  brier <- mean(w * (as.numeric(case) - s$p1)^2)
  c(auc, brier)
}
for (kind in c("cont", "tied")) {
  s <- read.csv(file.path(dir, paste0("surv_", kind, ".csv")))
  aj <- survfit(Surv(t, factor(code, levels = 0:2)) ~ 1, data = s)
  sm <- summary(aj, times = h, extend = TRUE)
  col <- which(aj$states == "1")
  put(paste0("aj_", kind), sm$pstate[1, col]); put(paste0("aj_se_", kind), sm$std.err[1, col])
  # Wolbers' C, unweighted: a death recoded as censored after every time stays comparable with every case.
  big <- max(s$t) + 1
  tt <- ifelse(s$code == 2, big, s$t)
  cc <- concordance(Surv(tt, as.integer(s$code == 1)) ~ s$p1, reverse = TRUE, ymax = h)
  put(paste0("c_harrell_", kind), cc$concordance)
  # Single-event versions (deaths are censorings): Harrell and Uno over the same restricted range.
  single <- Surv(s$t, as.integer(s$code == 1))
  put(paste0("c_single_harrell_", kind), concordance(single ~ s$p1, reverse = TRUE, ymax = h)$concordance)
  put(paste0("c_single_uno_", kind),
      concordance(single ~ s$p1, reverse = TRUE, ymax = h, timewt = "n/G2")$concordance)
  # The reverse Kaplan-Meier R computes by flipping the status, at the horizon.
  rk <- survfit(Surv(pmin(s$t, h), as.integer(s$t <= h & s$code == 0)) ~ 1)
  put(paste0("g_h_", kind), summary(rk, times = h, extend = TRUE)$surv)
  vectors[[paste0("g_left_", kind)]] <- c(1, rk$surv)[findInterval(s$t, rk$time, left.open = TRUE) + 1]
  # The definitions themselves, on the censoring survival the Python side used (s$g_left, s$g_h).
  bf <- brute(s, s$g_left, s$g_h[1])
  put(paste0("brute_auc_", kind), bf[1]); put(paste0("brute_brier_", kind), bf[2])
  if (has("timeROC")) {
    tr <- timeROC::timeROC(T = s$t, delta = s$code, marker = s$p1, cause = 1, weighting = "marginal",
                           times = h, iid = TRUE)
    put(paste0("timeroc_auc2_", kind), tr$AUC_2[length(tr$AUC_2)])
    put(paste0("timeroc_auc2_se_", kind), tr$inference$vect_sd_2[length(tr$inference$vect_sd_2)])
  }
  if (has("riskRegression") && has("prodlim")) {
    suppressMessages(library(prodlim))
    sc <- riskRegression::Score(list(m = as.matrix(s$p1)), formula = Hist(t, code) ~ 1, data = s,
                                times = h, cause = 1, metrics = c("auc", "brier"), cens.model = "km",
                                null.model = FALSE, se.fit = TRUE)
    put(paste0("rr_auc_", kind), sc$AUC$score$AUC[1]); put(paste0("rr_auc_se_", kind), sc$AUC$score$se[1])
    put(paste0("rr_brier_", kind), sc$Brier$score$Brier[1])
  }
}

# ---- Cox censoring model: the design matrix comes from evaluate.censoring_design ----
cx <- read.csv(file.path(dir, "cox.csv"))
X <- as.matrix(cx[, grepl("^x", names(cx))])
for (ridge in c(0, 1)) {
  fit <- if (ridge == 0) {
    coxph(Surv(cx$t, cx$d) ~ X, ties = "breslow", control = coxph.control(eps = 1e-10, toler.chol = 1e-13, iter.max = 100))
  } else {
    coxph(Surv(cx$t, cx$d) ~ ridge(X, theta = ridge, scale = FALSE), ties = "breslow",
          control = coxph.control(eps = 1e-10, toler.chol = 1e-13, iter.max = 100))
  }
  base <- basehaz(fit, centered = TRUE)
  risk <- exp(predict(fit, type = "lp"))
  at_h <- base$hazard[findInterval(h, base$time)]
  vectors[[paste0("cox_g_", ridge)]] <- exp(-risk * at_h)
}

write.csv(data.frame(name = names(values), value = unlist(values)), file.path(dir, "r_values.csv"),
          row.names = FALSE)
write.csv(as.data.frame(vectors[c("loess_fit")]), file.path(dir, "r_loess.csv"), row.names = FALSE)
write.csv(as.data.frame(vectors[c("wloess_fit")]), file.path(dir, "r_wloess.csv"), row.names = FALSE)
write.csv(as.data.frame(vectors[c("g_left_cont", "g_left_tied")]), file.path(dir, "r_gleft.csv"), row.names = FALSE)
write.csv(as.data.frame(vectors[c("cox_g_0", "cox_g_1")]), file.path(dir, "r_cox.csv"), row.names = FALSE)
