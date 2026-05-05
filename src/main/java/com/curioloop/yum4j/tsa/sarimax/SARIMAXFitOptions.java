package com.curioloop.yum4j.tsa.sarimax;

import com.curioloop.yum4j.kalman.mle.MLECovariance;
import com.curioloop.yum4j.optim.Minimizer;

import java.util.Objects;

/**
 * Fit controls for {@link SARIMAX#fit()}.
 */
public final class SARIMAXFitOptions {

    public static final SARIMAXFitOptions DEFAULT = builder().build();

    private final Minimizer<?, ?, ?> optimizer;
    private final MLECovariance covarianceType;

    private SARIMAXFitOptions(Builder builder) {
        this.optimizer = builder.optimizer;
        this.covarianceType = builder.covarianceType;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Returns the configured optimizer instance, or {@code null} to use the default
     * MLE optimizer.
     */
    public Minimizer<?, ?, ?> optimizer() {
        return optimizer;
    }

    public MLECovariance covarianceType() {
        return covarianceType;
    }

    public static final class Builder {
        private Minimizer<?, ?, ?> optimizer;
        private MLECovariance covarianceType = MLECovariance.OPG;

        private Builder() {
        }

        public Builder optimizer(Minimizer<?, ?, ?> optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        public Builder covarianceType(MLECovariance covarianceType) {
            this.covarianceType = Objects.requireNonNull(covarianceType, "covarianceType");
            return this;
        }

        public SARIMAXFitOptions build() {
            return new SARIMAXFitOptions(this);
        }
    }
}