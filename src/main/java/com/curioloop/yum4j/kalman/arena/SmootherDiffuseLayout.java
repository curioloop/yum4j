package com.curioloop.yum4j.kalman.arena;

public value record SmootherDiffuseLayout(int diffuseEstimatorBase,
                                    int diffuseEstimatorLength,
                                    int diffuse1EstimatorCovBase,
                                    int diffuse1EstimatorCovLength,
                                    int diffuse2EstimatorCovBase,
                                    int diffuse2EstimatorCovLength,
                                    int totalLength) {

    public static SmootherDiffuseLayout create(int scalarWidth, int kStates, int nobs) {
        int offset = 0;

        int diffuseEstimatorBase = offset;
        int diffuseEstimatorLength = scalarWidth * kStates * (nobs + 1);
        offset += diffuseEstimatorLength;
        int diffuse1EstimatorCovBase = offset;
        int diffuse1EstimatorCovLength = scalarWidth * kStates * kStates * (nobs + 1);
        offset += diffuse1EstimatorCovLength;
        int diffuse2EstimatorCovBase = offset;
        int diffuse2EstimatorCovLength = scalarWidth * kStates * kStates * (nobs + 1);
        offset += diffuse2EstimatorCovLength;

        return new SmootherDiffuseLayout(
            diffuseEstimatorBase,
            diffuseEstimatorLength,
            diffuse1EstimatorCovBase,
            diffuse1EstimatorCovLength,
            diffuse2EstimatorCovBase,
            diffuse2EstimatorCovLength,
            offset);
    }
}