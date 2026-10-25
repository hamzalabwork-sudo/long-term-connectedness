# Convert the example datasets of ConnectednessApproach (.rda, zoo objects) to CSV.
# Missing .rda files are downloaded from the CRAN mirror on GitHub.
#
# Usage (from work/):  Rscript ../code/convert_package_data.R pkgdata/rda pkgdata/csv
args <- commandArgs(trailingOnly = TRUE)
rda_dir <- ifelse(length(args) >= 1, args[1], "pkgdata/rda")
csv_dir <- ifelse(length(args) >= 2, args[2], "pkgdata/csv")
dir.create(rda_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(csv_dir, showWarnings = FALSE, recursive = TRUE)
suppressMessages(library(zoo))

sets <- c("aaacgo2022", "acg2020", "bcg2022", "bgu2021", "cegg2022", "cg2021", "cgg2022", "cgp2024",
          "cgs2021", "cgs2022", "dy2009", "dy2012", "g2020", "gcat2022", "gg2018", "gghm2022",
          "jcggh2022", "lw2021")
base <- "https://raw.githubusercontent.com/cran/ConnectednessApproach/master/data/"

dump1 <- function(x, name) {
  x <- as.zoo(x)
  v <- as.matrix(coredata(x))
  storage.mode(v) <- "double"
  d <- data.frame(date = as.character(index(x)), v, check.names = FALSE)
  write.csv(d, file.path(csv_dir, paste0(name, ".csv")), row.names = FALSE)
  cat(sprintf("%-12s k=%2d T=%5d\n", name, ncol(v), nrow(v)))
}

for (s in sets) {
  f <- file.path(rda_dir, paste0(s, ".rda"))
  if (!file.exists(f)) download.file(paste0(base, s, ".rda"), f, mode = "wb", quiet = TRUE)
  e <- new.env()
  nm <- load(f, envir = e)
  x <- get(nm[1], envir = e)
  tryCatch({
    if (is.list(x) && !is.zoo(x) && !is.data.frame(x)) {
      for (i in seq_along(x)) dump1(x[[i]], paste0(s, "_", ifelse(is.null(names(x)[i]), i, names(x)[i])))
    } else {
      dump1(x, s)
    }
  }, error = function(err) cat(s, "ERROR", conditionMessage(err), "\n"))
}
