package com.curioloop.yum4j.ssm.kalman.filter;

public enum ForecastErrorStrategy {
    AUTO,
    CHOLESKY_SOLVE,
    LU_SOLVE,
    UNIVARIATE,
    CHOLESKY_THEN_LU,
    AUTO_THEN_LU,
    LU_THEN_CHOLESKY
}