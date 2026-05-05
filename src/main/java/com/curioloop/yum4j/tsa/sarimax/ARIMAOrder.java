package com.curioloop.yum4j.tsa.sarimax;

/**
 * Non-seasonal ARIMA order $(p, d, q)$.
 */
public value record ARIMAOrder(int autoregressive, int integration, int movingAverage) {

    public ARIMAOrder {
        if (autoregressive < 0) {
            throw new IllegalArgumentException("autoregressive order must be non-negative");
        }
        if (integration < 0) {
            throw new IllegalArgumentException("integration order must be non-negative");
        }
        if (movingAverage < 0) {
            throw new IllegalArgumentException("movingAverage order must be non-negative");
        }
    }

    public static ARIMAOrder of(int autoregressive, int integration, int movingAverage) {
        return new ARIMAOrder(autoregressive, integration, movingAverage);
    }

    public boolean isZero() {
        return autoregressive == 0 && integration == 0 && movingAverage == 0;
    }
}