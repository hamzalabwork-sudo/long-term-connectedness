# Reference run of the CRAN ConnectednessApproach code (sourced from the package's R files,
# because the package's dependency tree does not install on R 4.1).
#
# Usage: Rscript validate_r.R panel.csv outdir rsrc_dir nlag nfore kappa1 kappa2 p1 p2 ...
#   p1 p2 ... are the interior partition edges in radians (pi+1e-5 and 0 are added).
args <- commandArgs(trailingOnly = TRUE)
infile <- args[1]; outdir <- args[2]; rsrc <- args[3]
nlag <- as.integer(args[4]); nfore <- as.integer(args[5])
k1 <- as.numeric(args[6]); k2 <- as.numeric(args[7])
partition <- c(pi + 0.00001, as.numeric(args[-(1:7)]), 0)

.libPaths(c("~/R/library", .libPaths()))
suppressMessages({ library(methods); library(MASS); library(zoo); library(progress) })
source(file.path(rsrc, "tools.R"))                 # frequencyConnectedness::getIndeces/getPartition
for (f in c("BayesPrior.R", "TVPVAR.R", "IRF.R", "FEVD.R", "ConnectednessTable.R",
            "TimeConnectedness.R", "ConnectednessApproach.R")) source(file.path(rsrc, f))
src <- readLines(file.path(rsrc, "FrequencyConnectedness.R"))
eval(parse(text = gsub("frequencyConnectedness::getPartition", "getPartition", src, fixed = TRUE)))

df <- read.csv(infile, check.names = FALSE)
x <- zoo(as.matrix(df[, -1]), order.by = as.POSIXct(df[, 1], tz = "UTC", format = "%Y-%m-%d %H:%M:%S"))
k <- ncol(x); tt <- nrow(x)
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

t0 <- Sys.time()
prior <- BayesPrior(x, nlag = nlag)
fit <- TVPVAR(x, configuration = list(l = c(k1, k2), nlag = nlag, prior = prior))
cat("TVPVAR seconds:", as.numeric(Sys.time() - t0, units = "secs"), "\n")
# B_t is (k, k*nlag, T); write row t as c(B_t[,,t]) (column-major)
write.csv(t(apply(fit$B_t, 3, c)), file.path(outdir, "B_t.csv"), row.names = FALSE)
write.csv(t(apply(fit$Q_t, 3, c)), file.path(outdir, "Q_t.csv"), row.names = FALSE)
write.csv(data.frame(b = prior$bprior), file.path(outdir, "prior_b.csv"), row.names = FALSE)
write.csv(prior$Q, file.path(outdir, "prior_Q.csv"), row.names = FALSE)

tc <- TimeConnectedness(Phi = fit$B_t, Sigma = fit$Q_t, nfore = nfore, generalized = TRUE)
write.csv(data.frame(TCI = tc$TCI[, 1], tc$NET, check.names = FALSE), file.path(outdir, "time.csv"), row.names = FALSE)

fc <- FrequencyConnectedness(Phi = fit$B_t, Sigma = fit$Q_t, nfore = nfore, partition = partition,
                             generalized = TRUE, scenario = "ABS")
write.csv(fc$TCI, file.path(outdir, "freq_TCI.csv"), row.names = FALSE)
nb <- dim(fc$NET)[3]
for (b in 1:nb) write.csv(fc$NET[, , b], file.path(outdir, sprintf("freq_NET_%d.csv", b - 1)), row.names = FALSE)

# the full wrapper as an applied user would call it, to confirm the same path
dca <- ConnectednessApproach(x, nlag = nlag, nfore = nfore, model = "TVP-VAR", connectedness = "Frequency",
                             VAR_config = list(TVPVAR = list(kappa1 = k1, kappa2 = k2, prior = "BayesPrior")),
                             Connectedness_config = list(FrequencyConnectedness = list(
                               partition = partition, generalized = TRUE, scenario = "ABS")))
cat("wrapper vs direct, max |TCI diff|:", max(abs(dca$TCI - fc$TCI)), "\n")
cat("band names:", colnames(fc$TCI), "\n")
cat("total seconds:", as.numeric(Sys.time() - t0, units = "secs"), "\n")
