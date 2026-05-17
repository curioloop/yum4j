package com.curioloop.yum4j.ssm.kalman.mle;

public final class FixedParameterResults {

    private FixedParameterResults() {
    }

    public static int freeParameterCount(int parameterCount, FixedParameters fixedParameters) {
        return parameterCount - resolve(fixedParameters, parameterCount).size();
    }

    public static int dfModel(int parameterCount, FixedParameters fixedParameters, int additionalDegreesOfFreedom) {
        return freeParameterCount(parameterCount, fixedParameters) + additionalDegreesOfFreedom;
    }

    public static double[] adjustedCovariance(double[] covariance, FixedParameters fixedParameters) {
        FixedParameters fixed = fixedParameters == null ? FixedParameters.none() : fixedParameters;
        if (fixed.isEmpty()) {
            return covariance;
        }
        double[] adjusted = covariance.clone();
        int dimension = (int) Math.round(Math.sqrt(adjusted.length));
        for (int index : fixed.validate(dimension).indices()) {
            for (int col = 0; col < dimension; col++) {
                adjusted[index * dimension + col] = Double.NaN;
                adjusted[col * dimension + index] = Double.NaN;
            }
        }
        return adjusted;
    }

    private static FixedParameters resolve(FixedParameters fixedParameters, int parameterCount) {
        return (fixedParameters == null ? FixedParameters.none() : fixedParameters).validate(parameterCount);
    }
}