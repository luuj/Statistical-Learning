#!/usr/bin/R

library(rlang)
library(dplyr)
library(data.table)
library(ggplot2)
library(grid)
library(gridExtra)
require(lme4)
require(plyr)

args = (commandArgs(TRUE))

#### ===========================
#### SIMULATE DATASET
#### ===========================
set.seed(235)

## low dim for testing
low_M <- as.integer(args[1])
X_low <- matrix(rnorm(low_M*5000), 5000, low_M)
beta <- rep(10, 20)
eps <- rnorm(5000)
y_low <- X_low[, 1:20] %*% beta + eps


#### ===========================
#### FUNCTIONS
#### ===========================
## define objective function
obj_func <- function(beta_hat, y, X, lambda) {
  N <- length(y)
  f = 1/(2*N) * sum((y - X %*% beta_hat)^2) + lambda * sum(abs(beta_hat))
  return(f)
}

## 1. Gradient descent
run_GD <- function(beta_hat, y, X, lambda = 0.2295, step_size = 0.01,
                   iterMax = 100, tol = 1e-3) {
  N <- length(y)
  M <- dim(X)[2]
  
  beta_prev <- beta_hat
  obj = obj_func(beta_hat, y, X, lambda = lambda)
  
  beta_path <- c(beta_hat)
  obj_path <- c(obj)

  
  i <- 0
  
  while ( (max(beta_hat - beta_prev) > tol) && (i < iterMax) || (i == 0) ) {
    beta_prev <- beta_hat
    
    beta_hat = beta_hat - step_size *(-1/N*t(X)%*%(y-X%*%beta_hat) + lambda*sign(beta_hat))
    obj <- obj_func(beta_hat, y, X, lambda = lambda)

    i <- i + 1
    beta_path <- c(beta_path, beta_hat)
    obj_path <- c(obj_path, obj)
  }

  return(list(beta_hat = beta_hat, 
              beta_path = beta_path, 
              log_obj_path = log(obj_path), 
              iter = i))

}

## 2. Proximal gradient descent (ISTA)
sft_thresh <- function(y, lambda){
  theta_hat <- ifelse(y > lambda, 
                      yes = y - lambda, 
                      no = ifelse(abs(y) <= lambda, 
                                  yes = 0, 
                                  no = y + lambda))
  return(theta_hat)
}

run_PGD <- function(beta_hat, y, X, lambda = 0.2295, step_size = 0.01,
                    iterMax = 100, tol = 1e-3) {
  N <- length(y)
  M <- dim(X)[2]
  
  beta_prev <- beta_hat
  obj = obj_func(beta_hat, y, X, lambda = lambda)
  
  beta_path <- c(beta_hat)
  obj_path <- c(obj)
  
  i <- 0
  
  while ( (max(beta_hat - beta_prev) > tol) && (i < iterMax) || (i == 0) ) {
    beta_prev <- beta_hat
    
    yt = beta_hat - step_size *(-1/N*t(X)%*%(y-X%*%beta_hat) + lambda*sign(beta_hat))
    beta_hat <- sft_thresh(yt, lambda*step_size)

    obj <- obj_func(beta_hat, y, X, lambda = lambda)
    
    beta_path <- c(beta_path, beta_hat)
    obj_path <- c(obj_path, obj)
    
    i <- i + 1
  }
  
  return(list(beta_hat = beta_hat, 
              beta_path = beta_path, 
              log_obj_path = log(obj_path), 
              iter = i))
}

## 3-4. FISTA (classic: lambda; new: t/t+3)
run_APGD <- function(beta_hat, y, X, lambda = 0.2295, step_size = 0.01,
                     type = "classic", iterMax = 100, tol = 1e-3) {
  N <- length(y)
  M <- dim(X)[2]
  
  beta_prev <- beta_hat
  obj = obj_func(beta_hat, y, X, lambda = lambda)
  
  beta_path <- c(beta_hat)
  obj_path <- c(obj)

  
  i <- 0
  lambda_t <- 1
  
  while ( (max(beta_hat - beta_prev) > tol) && (i < iterMax) || (i == 0) ) {
    beta_prev <- beta_hat
    
    yt = beta_hat - step_size *(-1/N*t(X)%*%(y-X%*%beta_hat) + lambda*sign(beta_hat))
    beta_hat <- sft_thresh(yt, lambda*step_size)
    
    if (type == "classic") {
      lambda_0 <- lambda_t
      lambda_t <- (1 + sqrt(1 + 4*(lambda_0)^2)) / 2
      beta_hat = beta_hat + (lambda_0 - 1) / lambda_t * (beta_hat - beta_prev)
    } else if (type == "new") {
      beta_hat = beta_hat + (i / (i+3)) * (beta_hat - beta_prev)
    }
    
    obj <- obj_func(beta_hat, y, X, lambda = lambda)
  
    beta_path <- c(beta_path, beta_hat)
    obj_path <- c(obj_path, obj)
    
    i <- i + 1
  }
  
  return(list(beta_hat = beta_hat, 
              beta_path = beta_path, 
              log_obj_path = log(obj_path), 
              iter = i))
  
}

## 4. Nesterov smoothing:
huber_grad <- function(z, mu){
  gradient <- ifelse(abs(z) <= mu, yes = z / mu, no = sign(z))
  return(sum(gradient))
}

run_AGD_smooth <- function(beta_hat, y, X, lambda = 0.2295, step_size = 0.01, huber_mu = 1,
                           iterMax = 100, tol = 1e-3) {
  N <- length(y)
  M <- dim(X)[2]
  
  beta_prev <- beta_hat
  obj = obj_func(beta_hat, y, X, lambda = lambda)
  
  beta_path <- c(beta_hat)
  obj_path <- c(obj)

  i <- 0
  lambda_t <- 1
  
  while ( (max(beta_hat - beta_prev) > tol) && (i < iterMax) || (i == 0) ) {
    beta_prev <- beta_hat
    
    yt = beta_hat - step_size * (-1/N*t(X)%*%(y-X%*%beta_hat) + lambda * huber_grad(beta_hat, huber_mu))
    
    lambda_0 <- lambda_t
    lambda_t <- (1 + sqrt(1 + 4*(lambda_0)^2)) / 2
    beta_hat <- yt + (lambda_0 - 1) / lambda_t * (yt - beta_prev)
    
    obj <- obj_func(beta_hat, y, X, lambda = lambda)
  
    beta_path <- c(beta_path, beta_hat)
    obj_path <- c(obj_path, obj)
    
    i <- i + 1
  }
  
  return(list(beta_hat = beta_hat, 
              beta_path = beta_path, 
              log_obj_path = log(obj_path), 
              iter = i))

}

## 5. ADMM
run_ADMM <- function(beta_hat, y, X, rho, iterMax = 100, tol = 1e-3, lambda = 0.2295) {
  N <- length(y)
  M <- dim(X)[2]
  
  beta_prev <- beta_hat
  obj = obj_func(beta_hat, y, X, lambda = lambda)
  
  beta_path <- c(beta_hat)
  obj_path <- c(obj)

  i <- 0
  lambda_t <- rep(1, M)
  z_t <- beta_hat
  
  while ( (max(beta_hat - beta_prev) > tol) && (i < iterMax) || (i == 0) ) {
    beta_prev <- beta_hat
    
    # update z
    z_t = sft_thresh(beta_hat + 1 / rho * lambda_t, lambda / rho)
    
    # update lambda
    lambda_0 <- lambda_t
    lambda_t <- rho * (beta_hat - z_t)
    
    # update beta
    beta_hat = solve(t(X) %*% X + rho) %*% (t(X) %*% y + rho*(z_t - lambda_t))

    # record things / update iteration
    obj <- obj_func(beta_hat, y, X, lambda = lambda)
    beta_path <- c(beta_path, beta_hat)
    obj_path <- c(obj_path, obj)
    i <- i + 1
  }
  
  return(list(beta_hat = beta_hat, 
              beta_path = beta_path, 
              log_obj_path = log(obj_path), 
              iter = i))

}

### 7. Group lasso
block_sft_thresh <- function(y_vec, lambda){
  
  theta_hat <- ifelse(sum(y_vec^2) > lambda, 
                      yes = (1 - lambda / sqrt(sum(y_vec^2))) * y_vec, 
                      no = rep(0, length(y_vec)))
  return(theta_hat)
}

run_groupLASSO_APGD <- function(beta_hat, y, X, type = "classic", iterMax = 100, tol = 1e-3, lambda = 0.2295, step_size = 0.01) {
  N <- length(y)
  M <- dim(X)[2]
  
  beta_prev <- beta_hat
  obj = obj_func(beta_hat, y, X, lambda = lambda)
  
  beta_path <- c(beta_hat)
  obj_path <- c(obj)

  i <- 0
  lambda_t <- 1
  
  while ( (max(beta_hat - beta_prev) > tol) && (i < iterMax) || (i == 0) ) {
    beta_prev <- beta_hat
    
    yt = beta_hat - step_size *(-1/N*t(X)%*%(y-X%*%beta_hat) + lambda*sign(beta_hat))
    
    ## substitute with block soft thresholding
    for (j in 1:(M/4)) {
      vec_j <- c(4*j-3, 4*j-2, 4*j-1, 4*j)
      beta_hat[vec_j] = block_sft_thresh(yt[vec_j], lambda*step_size)
    }
    
    ## vectorized version (not working)
    # beta_hat <- unlist(sapply(1:(M/4), function(x) {
    #   vec_j <- c(4*x-3, 4*x-2, 4*x-1, 4*x)
    #   block_sft_thresh(yt[vec_j], lambda*step_size)
    #   }))
    
    if (type == "classic") {
      lambda_0 <- lambda_t
      lambda_t <- (1 + sqrt(1 + 4*(lambda_0)^2)) / 2
      beta_hat = beta_hat + (lambda_0 - 1) / lambda_t * (beta_hat - beta_prev)
    } else if (type == "new") {
      beta_hat = beta_hat + (i / (i+3)) * (beta_hat - beta_prev)
    }
    
    obj <- obj_func(beta_hat, y, X, lambda = lambda)
  
    beta_path <- c(beta_path, beta_hat)
    obj_path <- c(obj_path, obj)
    
    i <- i + 1
  }
  
  return(list(beta_hat = beta_hat, 
              beta_path = beta_path, 
              log_obj_path = log(obj_path), 
              iter = i))
}


#### ===========================
#### RUN OPTIMIZATION
#### ===========================
### low dimensional
GD_out <- run_GD(rep(0, low_M), y_low, X_low, step_size = 0.1)
PGD_out <- run_PGD(rep(0, low_M), y_low, X_low, step_size = 0.1)
APGD_out_1 <- run_APGD(rep(0, low_M), y_low, X_low, type = "classic", step_size = 0.1)
APGD_out_2 <- run_APGD(rep(0, low_M), y_low, X_low, type = "new", step_size = 0.1)
AGD_smooth_out <- run_AGD_smooth(rep(0, low_M), y_low, X_low, 
                                 step_size = 0.01, huber_mu = 2)
ADMM_out <- run_ADMM(rep(0, low_M), y_low, X_low, rho = 0.1)
group_lasso_out <- run_groupLASSO_APGD(rep(0, low_M), y_low, X_low, type = "classic", step_size = 0.1, lambda = 0.459)

low_dim_out_list <- list(GD_out, PGD_out, APGD_out_1, APGD_out_2, AGD_smooth_out, ADMM_out, group_lasso_out)

#### ===========================
#### VISUALIZATION
#### ===========================
LD_out_list <- lapply(low_dim_out_list, 
       function(x) {data.frame(iter = seq(1, x$iter+1), log_obj = x$log_obj_path)})

dt_plot <- bind_rows(LD_out_list, .id = "id")
dt_plot$method <- mapvalues(dt_plot$id, 
          from=1:7,
          to=c("GD", "PGD", "FISTA (classic)", "FISTA (new)", "Nesterov smoothing", "ADMM", "Group Lasso APGD"))

p <- ggplot(dt_plot, aes(x=iter, y=log_obj)) + 
  geom_line(aes(color=method), size = 1) + 
  scale_colour_discrete("Estimation type") +
  labs(x = "Iteration", y = expression(log(f(beta))), title = "Trajectories of objective value across estimation methods")

ggsave(paste0("HW3_2_LD_", low_M, ".png"), p, height = 5, width = 7)
