set.seed(1138)

n = 30
p = 10
L = 1
V = 1
beta_true = rep(0, p)

B = 100
b_1s = numeric(B)
susie.fits.easy = list() # logistic SuSiE fits
logistic.fits.easy = list() # logistic regression fits
for (i in 1:B) {
  beta_true[1] = rnorm(1, 0, sqrt(V))
  b_1s[i] = beta_true[1]
  # make independent N(0, 1) covariates
  X = matrix(rnorm(n * p), nrow = n)
  # make response
  Y = rbinom(n, 1, exp(X %*% beta_true) / (1 + exp(X %*% beta_true)))
  susie.fits.easy[[i]] = susie_logistic_VB(Y, X, L, V)
  logistic.fits.easy[[i]] = glm(Y ~ X, family = "binomial")
}
