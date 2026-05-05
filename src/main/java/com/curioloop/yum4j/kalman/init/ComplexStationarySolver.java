package com.curioloop.yum4j.kalman.init;

import com.curioloop.yum4j.kalman.model.StateSpaceModel;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.cmplx.ZLAS;

final class ComplexStationarySolver {

    private ComplexStationarySolver() {}

    static int requiredDoubleCount(int kStates, int kPosdef) {
        int matrix1Length = 2 * kStates * kStates;
        int matrix2Length = Math.max(matrix1Length, 2 * kStates * kPosdef);
        int matrix3Length = matrix1Length;
        int workLength = requiredWorkLength(kStates);
        int wLength = 2 * kStates;
        int rworkLength = kStates;
        return matrix1Length + matrix2Length + matrix3Length + workLength + wLength + rworkLength;
    }

    static int requiredPivotCount(int kStates) {
        return kStates;
    }

    static double[] solveCov(StateSpaceModel stateSpace, int t) {
        int kStates = stateSpace.stateCount();
        double[] stationaryCov = new double[2 * kStates * kStates];
        double[] backing = new double[requiredDoubleCount(kStates, stateSpace.stateDisturbanceCount())];
        int[] pivots = new int[requiredPivotCount(kStates)];
        return solveCov(stateSpace, t, backing, pivots, stationaryCov, 0) ? stationaryCov : null;
    }

    static boolean solveCov(StateSpaceModel stateSpace,
                            int t,
                            double[] stationaryBacking,
                            int[] stationaryPivots,
                            double[] target,
                            int targetOffset) {
        return solveCov(stateSpace, t, stationaryBacking, 0, stationaryPivots, target, targetOffset, stateSpace.stateCount());
    }

    static boolean solveCov(StateSpaceModel stateSpace,
                            int t,
                            double[] stationaryBacking,
                            int stationaryBackingOffset,
                            int[] stationaryPivots,
                            double[] target,
                            int targetOffset) {
        return solveCov(stateSpace, t, stationaryBacking, stationaryBackingOffset,
            stationaryPivots, target, targetOffset, stateSpace.stateCount());
    }

    static boolean solveCov(StateSpaceModel stateSpace,
                            int t,
                            double[] stationaryBacking,
                            int[] stationaryPivots,
                            double[] target,
                            int targetOffset,
                            int targetLd) {
        return solveCov(stateSpace, t, stationaryBacking, 0, stationaryPivots, target, targetOffset, targetLd);
    }

    static boolean solveCov(StateSpaceModel stateSpace,
                            int t,
                            double[] stationaryBacking,
                            int stationaryBackingOffset,
                            int[] stationaryPivots,
                            double[] target,
                            int targetOffset,
                            int targetLd) {
        int kStates = stateSpace.stateCount();
        int kPosdef = stateSpace.stateDisturbanceCount();
        if (kStates == 0) {
            return true;
        }

        int transitionLength = 2 * kStates * kStates;
        int matrix2Length = Math.max(transitionLength, 2 * kStates * kPosdef);
        int transitionOffset = stationaryBackingOffset;
        int matrix2Offset = transitionOffset + transitionLength;
        int matrix3Offset = matrix2Offset + matrix2Length;
        int workOffset = matrix3Offset + transitionLength;

        validateWorkspace(stateSpace, stationaryBacking, stationaryBackingOffset, stationaryPivots);
        StationarySupport.copyComplexMatrix(
            stateSpace.transitionData(),
            stateSpace.transitionOffset(t),
            2 * stateSpace.transitionLeadingDimension(),
            stationaryBacking,
            transitionOffset,
            kStates);
        System.arraycopy(stationaryBacking, transitionOffset, stationaryBacking, matrix3Offset, transitionLength);
        if (!isDiscreteStable(stationaryBacking, matrix3Offset, workOffset, kStates)) {
            return solveCovViaRealEmbedding(stateSpace, t, target, targetOffset, targetLd);
        }
        StationarySupport.copyComplexMatrix(
            stateSpace.transitionData(),
            stateSpace.transitionOffset(t),
            2 * stateSpace.transitionLeadingDimension(),
            stationaryBacking,
            transitionOffset,
            kStates);

        StationarySupport.buildSelectedStateCov(stateSpace, t, stationaryBacking, matrix2Offset, target, targetOffset, targetLd);
        for (int row = 0; row < kStates; row++) {
            int rowOffset = targetOffset + row * targetLd * 2;
            for (int col = 0; col < 2 * kStates; col += 2) {
                target[rowOffset + col] = -target[rowOffset + col];
                target[rowOffset + col + 1] = -target[rowOffset + col + 1];
            }
        }
        if (!solveDiscreteLyapunov(stationaryBacking, transitionOffset, matrix2Offset,
                stationaryPivots, target, targetOffset, targetLd, kStates)) {
            return solveCovViaRealEmbedding(stateSpace, t, target, targetOffset, targetLd);
        }
        StationarySupport.enforceHermitian(target, targetOffset, kStates, targetLd);
        if (StationarySupport.allFiniteSquare(target, targetOffset, kStates, targetLd, 2)
                && covarianceResidualWithinTolerance(stateSpace, t,
                stationaryBacking, matrix2Offset, matrix3Offset,
                target, targetOffset, targetLd)) {
            return true;
        }
        return solveCovViaRealEmbedding(stateSpace, t, target, targetOffset, targetLd);
    }

    static double[] solveMean(StateSpaceModel stateSpace, int t) {
        int kStates = stateSpace.stateCount();
        double[] stationaryMean = new double[2 * kStates];
        double[] backing = new double[requiredDoubleCount(kStates, stateSpace.stateDisturbanceCount())];
        int[] pivots = new int[requiredPivotCount(kStates)];
        return solveMean(stateSpace, t, backing, pivots, stationaryMean, 0) ? stationaryMean : null;
    }

    static boolean solveMean(StateSpaceModel stateSpace,
                             int t,
                             double[] stationaryBacking,
                             int[] stationaryPivots,
                             double[] target,
                             int targetOffset) {
        return solveMean(stateSpace, t, stationaryBacking, 0, stationaryPivots, target, targetOffset);
    }

    static boolean solveMean(StateSpaceModel stateSpace,
                             int t,
                             double[] stationaryBacking,
                             int stationaryBackingOffset,
                             int[] stationaryPivots,
                             double[] target,
                             int targetOffset) {
        int kStates = stateSpace.stateCount();
        if (kStates == 0 || stateSpace.stateInterceptData() == null) {
            return true;
        }

        validateWorkspace(stateSpace, stationaryBacking, stationaryBackingOffset, stationaryPivots);
        System.arraycopy(stateSpace.stateInterceptData(), stateSpace.stateInterceptOffset(t), target, targetOffset, 2 * kStates);
        StationarySupport.copyComplexMatrix(
            stateSpace.transitionData(),
            stateSpace.transitionOffset(t),
            2 * stateSpace.transitionLeadingDimension(),
            stationaryBacking,
            stationaryBackingOffset,
            kStates);
        for (int i = 0; i < 2 * kStates * kStates; i += 2) {
            int off = stationaryBackingOffset + i;
            stationaryBacking[off] = -stationaryBacking[off];
            stationaryBacking[off + 1] = -stationaryBacking[off + 1];
        }
        for (int i = 0; i < kStates; i++) {
            stationaryBacking[stationaryBackingOffset + (i * kStates + i) * 2] += 1.0;
        }

        if (ZLAS.zgetrf(kStates, kStates, stationaryBacking, stationaryBackingOffset >> 1, kStates, stationaryPivots) != 0) {
            return solveMeanViaRealEmbedding(stateSpace, t, target, targetOffset);
        }
        ZLAS.zgetrs(BLAS.Trans.NoTrans, kStates, 1, stationaryBacking, stationaryBackingOffset >> 1, kStates, stationaryPivots, target, targetOffset >> 1, 1);
        if (StationarySupport.allFinite(target, targetOffset, 2 * kStates)) {
            return true;
        }
        return solveMeanViaRealEmbedding(stateSpace, t, target, targetOffset);
    }

    private static boolean isDiscreteStable(double[] stationaryBacking,
                                            int transitionOffset,
                                            int workOffset,
                                            int kStates) {
        int rworkLength = Math.max(1, kStates);
        int rworkOffset = stationaryBacking.length - rworkLength;
        int wOffset = rworkOffset - 2 * kStates;
        int workLength = wOffset - workOffset;
        if (workLength < 1) {
            return false;
        }

        if (ZLAS.zgees('N', 'N', null, kStates, stationaryBacking, transitionOffset >> 1, kStates,
            stationaryBacking, wOffset, null, 0, kStates,
            stationaryBacking, workOffset, workLength, stationaryBacking, rworkOffset, null) != 0) {
            return false;
        }

        for (int i = 0; i < kStates; i++) {
            if (Math.hypot(stationaryBacking[wOffset + i * 2], stationaryBacking[wOffset + i * 2 + 1]) >= 1.0) {
                return false;
            }
        }
        return true;
    }

    private static boolean solveDiscreteLyapunov(double[] stationaryBacking,
                                                 int transitionOffset,
                                                 int matrix2Offset,
                                                 int[] stationaryPivots,
                                                 double[] rhs,
                                                 int rhsOffset,
                                                 int rhsLd,
                                                 int kStates) {
        int transitionLength = 2 * kStates * kStates;
        System.arraycopy(stationaryBacking, transitionOffset, stationaryBacking, matrix2Offset, transitionLength);
        for (int i = 0; i < kStates; i++) {
            stationaryBacking[matrix2Offset + (i * kStates + i) * 2] += 1.0;
            stationaryBacking[transitionOffset + (i * kStates + i) * 2] -= 1.0;
        }

        if (ZLAS.zgetrf(kStates, kStates, stationaryBacking, matrix2Offset >> 1, kStates, stationaryPivots) != 0) {
            return false;
        }

        ZLAS.zgetrs(BLAS.Trans.NoTrans, kStates, kStates,
            stationaryBacking, matrix2Offset >> 1, kStates, stationaryPivots, rhs, rhsOffset >> 1, rhsLd);
        StationarySupport.conjugateTransposeSquare(rhs, rhsOffset, kStates, rhsLd);
        ZLAS.zgetrs(BLAS.Trans.NoTrans, kStates, kStates,
            stationaryBacking, matrix2Offset >> 1, kStates, stationaryPivots, rhs, rhsOffset >> 1, rhsLd);

        StationarySupport.conjugateTransposeSquare(stationaryBacking, transitionOffset, kStates);
        ZLAS.zgetrs(BLAS.Trans.Conj, kStates, kStates,
            stationaryBacking, matrix2Offset >> 1, kStates, stationaryPivots,
            stationaryBacking, transitionOffset >> 1, kStates);
        StationarySupport.conjugateTransposeSquare(stationaryBacking, transitionOffset, kStates);

        return solveContinuousLyapunov(stationaryBacking, transitionOffset, matrix2Offset, rhs, rhsOffset, rhsLd, kStates, 2.0);
    }

    private static boolean solveContinuousLyapunov(double[] stationaryBacking,
                                                   int transitionOffset,
                                                   int matrix2Offset,
                                                   double[] rhs,
                                                   int rhsOffset,
                                                   int rhsLd,
                                                   int kStates,
                                                   double sign) {
        int matrix3Offset = matrix2Offset + 2 * kStates * kStates;
        int workOffset = matrix3Offset + 2 * kStates * kStates;
        int workLength = stationaryBacking.length - workOffset - 3 * kStates;
        int wOffset = workOffset + workLength;
        int rworkOffset = wOffset + 2 * kStates;
        if (ZLAS.zgees('V', 'N', null, kStates, stationaryBacking, transitionOffset >> 1, kStates,
            stationaryBacking, wOffset, stationaryBacking, matrix3Offset >> 1, kStates,
            stationaryBacking, workOffset, workLength, stationaryBacking, rworkOffset, null) != 0) {
            return false;
        }

        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
            sign, 0.0, rhs, rhsOffset >> 1, kStates, stationaryBacking, matrix3Offset >> 1, kStates,
            0.0, 0.0, stationaryBacking, matrix2Offset >> 1, kStates);
        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, kStates,
            1.0, 0.0, stationaryBacking, matrix3Offset >> 1, kStates,
            stationaryBacking, matrix2Offset >> 1, kStates,
            0.0, 0.0, rhs, rhsOffset >> 1, rhsLd);

        double scale = ZLAS.ztrsyl('N', 'C', 1, kStates, kStates,
            stationaryBacking, transitionOffset >> 1, kStates,
            stationaryBacking, transitionOffset >> 1, kStates,
            rhs, rhsOffset >> 1, rhsLd, null);

        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kStates,
            scale, 0.0, rhs, rhsOffset >> 1, rhsLd, stationaryBacking, matrix3Offset >> 1, kStates,
            0.0, 0.0, stationaryBacking, matrix2Offset >> 1, kStates);
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
            1.0, 0.0, stationaryBacking, matrix3Offset >> 1, kStates,
            stationaryBacking, matrix2Offset >> 1, kStates,
            0.0, 0.0, rhs, rhsOffset >> 1, rhsLd);
        return true;
    }

    private static int requiredWorkLength(int kStates) {
        if (kStates == 0) {
            return 0;
        }
        double[] query = new double[1];
        double[] w = new double[Math.max(2, 2 * kStates)];
        double[] rwork = new double[Math.max(1, kStates)];
        ZLAS.zgees('V', 'N', null, kStates, null, 0, kStates,
            w, 0, null, 0, kStates,
            query, 0, -1, rwork, null);
        return Math.max(1, (int) query[0]);
    }

    private static void validateWorkspace(StateSpaceModel stateSpace,
                                          double[] stationaryBacking,
                                          int stationaryBackingOffset,
                                          int[] stationaryPivots) {
        int requiredBacking = requiredDoubleCount(stateSpace.stateCount(), stateSpace.stateDisturbanceCount());
        if (stationaryBacking == null || stationaryBackingOffset < 0
                || stationaryBackingOffset + requiredBacking > stationaryBacking.length) {
            throw new IllegalArgumentException("stationaryBacking length "
                + (stationaryBacking == null ? 0 : stationaryBacking.length)
                + " with offset " + stationaryBackingOffset
                + " is too small for required length " + requiredBacking);
        }
        int requiredPivots = requiredPivotCount(stateSpace.stateCount());
        if (stationaryPivots == null || stationaryPivots.length < requiredPivots) {
            throw new IllegalArgumentException("stationaryPivots length "
                + (stationaryPivots == null ? 0 : stationaryPivots.length)
                + " is too small for required length " + requiredPivots);
        }
    }

    private static boolean solveCovViaRealEmbedding(StateSpaceModel stateSpace,
                                                    int t,
                                                    double[] target,
                                                    int targetOffset,
                                                    int targetLd) {
        int kStates = stateSpace.stateCount();
        int n = kStates * kStates;
        double[] rhs = new double[2 * n];
        double[] workspace = new double[Math.max(2 * kStates * kStates, 2 * kStates * stateSpace.stateDisturbanceCount())];
        StationarySupport.buildSelectedStateCov(stateSpace, t, workspace, 0, rhs, 0);

        double[] system = new double[2 * n * n];
        double[] transition = stateSpace.transitionData();
        int transitionOffset = stateSpace.transitionOffset(t);
        int transitionLd = stateSpace.transitionLeadingDimension();
        for (int i = 0; i < kStates; i++) {
            for (int j = 0; j < kStates; j++) {
                int equation = i * kStates + j;
                int diagonal = (equation * n + equation) * 2;
                system[diagonal] = 1.0;
                for (int a = 0; a < kStates; a++) {
                    int ta = transitionOffset + (i * transitionLd + a) * 2;
                    double tiaRe = transition[ta];
                    double tiaIm = transition[ta + 1];
                    for (int b = 0; b < kStates; b++) {
                        int tb = transitionOffset + (j * transitionLd + b) * 2;
                        double tjbRe = transition[tb];
                        double tjbIm = -transition[tb + 1];
                        int variable = a * kStates + b;
                        int coeff = (equation * n + variable) * 2;
                        system[coeff] -= tiaRe * tjbRe - tiaIm * tjbIm;
                        system[coeff + 1] -= tiaRe * tjbIm + tiaIm * tjbRe;
                    }
                }
            }
        }

        double[] realMatrix = new double[4 * n * n];
        double[] realRhs = new double[2 * n];
        buildRealifiedSystem(system, n, rhs, realMatrix, realRhs);
        int[] pivots = new int[2 * n];
        if (BLAS.dgetrf(2 * n, 2 * n, realMatrix, 0, 2 * n, pivots, 0) != 0) {
            return false;
        }
        BLAS.dgetrs(BLAS.Trans.NoTrans, 2 * n, 1, realMatrix, 0, 2 * n, pivots, 0, realRhs, 0, 1);
        projectRealifiedVectorToComplexMatrix(realRhs, kStates, target, targetOffset, targetLd);
        StationarySupport.enforceHermitian(target, targetOffset, kStates, targetLd);
        return true;
    }

    private static boolean solveMeanViaRealEmbedding(StateSpaceModel stateSpace,
                                                     int t,
                                                     double[] target,
                                                     int targetOffset) {
        int kStates = stateSpace.stateCount();
        double[] system = new double[2 * kStates * kStates];
        double[] transition = stateSpace.transitionData();
        int transitionOffset = stateSpace.transitionOffset(t);
        int transitionLd = stateSpace.transitionLeadingDimension();
        for (int row = 0; row < kStates; row++) {
            for (int col = 0; col < kStates; col++) {
                int source = transitionOffset + (row * transitionLd + col) * 2;
                int targetIndex = (row * kStates + col) * 2;
                system[targetIndex] = -transition[source];
                system[targetIndex + 1] = -transition[source + 1];
            }
            system[(row * kStates + row) * 2] += 1.0;
        }

        double[] rhs = new double[2 * kStates];
        System.arraycopy(stateSpace.stateInterceptData(), stateSpace.stateInterceptOffset(t), rhs, 0, 2 * kStates);
        double[] realMatrix = new double[4 * kStates * kStates];
        double[] realRhs = new double[2 * kStates];
        buildRealifiedSystem(system, kStates, rhs, realMatrix, realRhs);
        int[] pivots = new int[2 * kStates];
        if (BLAS.dgetrf(2 * kStates, 2 * kStates, realMatrix, 0, 2 * kStates, pivots, 0) != 0) {
            return false;
        }
        BLAS.dgetrs(BLAS.Trans.NoTrans, 2 * kStates, 1, realMatrix, 0, 2 * kStates, pivots, 0, realRhs, 0, 1);
        projectRealifiedVectorToComplexState(realRhs, kStates, target, targetOffset);
        return true;
    }

    private static boolean covarianceResidualWithinTolerance(StateSpaceModel stateSpace,
                                                            int t,
                                                            double[] stationaryBacking,
                                                            int workspaceOffset,
                                                            int expectedOffset,
                                                            double[] covariance,
                                                            int covarianceOffset,
                                                            int covarianceLd) {
        int kStates = stateSpace.stateCount();
        StationarySupport.buildSelectedStateCov(stateSpace, t, stationaryBacking, workspaceOffset, stationaryBacking, expectedOffset);
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
            1.0, 0.0, stateSpace.transitionData(), stateSpace.transitionOffset(t) >> 1, stateSpace.transitionLeadingDimension(),
            covariance, covarianceOffset >> 1, covarianceLd,
            0.0, 0.0, stationaryBacking, workspaceOffset >> 1, kStates);
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kStates,
            1.0, 0.0, stationaryBacking, workspaceOffset >> 1, kStates,
            stateSpace.transitionData(), stateSpace.transitionOffset(t) >> 1, stateSpace.transitionLeadingDimension(),
            -1.0, 0.0, stationaryBacking, expectedOffset >> 1, kStates);

        double maxResidual = 0.0;
        double maxScale = 0.0;
        for (int row = 0; row < kStates; row++) {
            int rowOffset = covarianceOffset + row * covarianceLd * 2;
            for (int col = 0; col < kStates; col++) {
                int index = rowOffset + col * 2;
                int expectedIndex = expectedOffset + (row * kStates + col) * 2;
                double residualRe = covariance[index] - stationaryBacking[expectedIndex];
                double residualIm = covariance[index + 1] - stationaryBacking[expectedIndex + 1];
                maxResidual = Math.max(maxResidual, Math.hypot(residualRe, residualIm));
                maxScale = Math.max(maxScale, Math.hypot(covariance[index], covariance[index + 1])
                    + Math.hypot(stationaryBacking[expectedIndex], stationaryBacking[expectedIndex + 1]));
            }
        }
        return maxResidual <= 1e-10 * Math.max(1.0, maxScale);
    }

    private static void buildRealifiedSystem(double[] complexMatrix,
                                             int dimension,
                                             double[] complexRhs,
                                             double[] realMatrix,
                                             double[] realRhs) {
        for (int row = 0; row < dimension; row++) {
            realRhs[row] = complexRhs[row * 2];
            realRhs[dimension + row] = complexRhs[row * 2 + 1];
            for (int col = 0; col < dimension; col++) {
                int complexIndex = (row * dimension + col) * 2;
                double real = complexMatrix[complexIndex];
                double imag = complexMatrix[complexIndex + 1];
                realMatrix[row * (2 * dimension) + col] = real;
                realMatrix[row * (2 * dimension) + dimension + col] = -imag;
                realMatrix[(dimension + row) * (2 * dimension) + col] = imag;
                realMatrix[(dimension + row) * (2 * dimension) + dimension + col] = real;
            }
        }
    }

    private static void projectRealifiedVectorToComplexState(double[] source,
                                                             int dimension,
                                                             double[] target,
                                                             int targetOffset) {
        for (int i = 0; i < dimension; i++) {
            target[targetOffset + i * 2] = source[i];
            target[targetOffset + i * 2 + 1] = source[dimension + i];
        }
    }

    private static void projectRealifiedVectorToComplexMatrix(double[] source,
                                                              int dimension,
                                                              double[] target,
                                                              int targetOffset,
                                                              int targetLd) {
        for (int row = 0; row < dimension; row++) {
            for (int col = 0; col < dimension; col++) {
                int sourceIndex = row * dimension + col;
                int targetIndex = targetOffset + (row * targetLd + col) * 2;
                target[targetIndex] = source[sourceIndex];
                target[targetIndex + 1] = source[dimension * dimension + sourceIndex];
            }
        }
    }
}