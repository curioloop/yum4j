package com.curioloop.yum4j.tsa.varmax;

import com.curioloop.yum4j.ssm.kalman.mle.MLEResults;
import com.curioloop.yum4j.ssm.kalman.mle.FixedParameters;
import com.curioloop.yum4j.optim.Minimizer;

import java.util.Arrays;
import java.util.Objects;

public final class VARMAXFitOptions {

    public static final VARMAXFitOptions DEFAULT = builder().build();

    private final Minimizer<?, ?, ?> optimizer;
    private final MLEResults.Covariance covarianceType;
    private final double[] startParams;
    private final FixedParameters fixedParameters;

    private VARMAXFitOptions(Builder builder) {
        this.optimizer = builder.optimizer;
        this.covarianceType = builder.covarianceType;
        this.startParams = builder.startParams == null ? null : builder.startParams.clone();
        this.fixedParameters = builder.fixedParameters;
    }

    public static Builder builder() {
        return new Builder();
    }

    public Minimizer<?, ?, ?> optimizer() {
        return optimizer;
    }

    public MLEResults.Covariance covarianceType() {
        return covarianceType;
    }

    public double[] startParams() {
        return startParams == null ? null : startParams.clone();
    }

    public FixedParameters fixedParameters() {
        return fixedParameters;
    }

    public Builder toBuilder() {
        return new Builder()
            .optimizer(optimizer)
            .covarianceType(covarianceType)
            .startParams(startParams)
            .fixedParameters(fixedParameters);
    }

    public static final class Builder {
        private Minimizer<?, ?, ?> optimizer;
        private MLEResults.Covariance covarianceType = MLEResults.Covariance.OPG;
        private double[] startParams;
        private FixedParameters fixedParameters = FixedParameters.none();

        private Builder() {
        }

        public Builder optimizer(Minimizer<?, ?, ?> optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        public Builder covarianceType(MLEResults.Covariance covarianceType) {
            this.covarianceType = Objects.requireNonNull(covarianceType, "covarianceType");
            return this;
        }

        public Builder startParams(double... startParams) {
            this.startParams = startParams == null ? null : Arrays.copyOf(startParams, startParams.length);
            return this;
        }

        public Builder fixedParameters(FixedParameters fixedParameters) {
            this.fixedParameters = fixedParameters == null ? FixedParameters.none() : fixedParameters;
            return this;
        }

        public VARMAXFitOptions build() {
            return new VARMAXFitOptions(this);
        }
    }
}
