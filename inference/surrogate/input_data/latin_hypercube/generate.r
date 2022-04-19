#!/usr/bin/env Rscript

library(MaxPro)

args = commandArgs(trailingOnly=TRUE)

p <- 5 # v, mu, FvK, b2, Fext
n <- strtoi(args[1]) # number of samples

ret <- MaxProLHD(n, p)

D <- ret[["Design"]]

colnames(D) <- c("v", "mu", "FvK", "b2", "Fext")

write.table(D,
            file=sprintf("samples-%d.csv", n),
            row.names=FALSE,
            sep=",",
            quote=FALSE)
