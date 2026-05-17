package com.curioloop.yum4j.ssm.kalman.init;

import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.linalg.blas.BLAS;

import java.util.Arrays;

final class RealStationarySolver {

    private RealStationarySolver() {}

    static int requiredDoubleCount(int kStates, int kPosdef) {
        int matrix1Length = kStates * kStates;
        int matrix2Length = Math.max(matrix1Length, kStates * kPosdef);
        int matrix3Length = matrix1Length;
        return matrix1Length + matrix2Length + matrix3Length + requiredWorkLength(kStates);
    }

    static int requiredPivotCount(int kStates) {
        return kStates;
    }

    static double[] solveCov(KalmanSSM stateSpace, int t) {
        int kStates = stateSpace.stateCount();
        double[] stationaryCov = new double[kStates * kStates];
        double[] backing = new double[requiredDoubleCount(kStates, stateSpace.stateDisturbanceCount())];
        int[] pivots = new int[requiredPivotCount(kStates)];
        return solveCov(stateSpace, t, backing, pivots, stationaryCov, 0) ? stationaryCov : null;
    }

    static boolean solveCov(KalmanSSM stateSpace,
                            int t,
                            double[] stationaryBacking,
                            int[] stationaryPivots,
                            double[] target,
                            int targetOffset) {
        return solveCov(stateSpace, t, stationaryBacking, 0, stationaryPivots, target, targetOffset, stateSpace.stateCount());
    }

    static boolean solveCov(KalmanSSM stateSpace,
                            int t,
                            double[] stationaryBacking,
                            int stationaryBackingOffset,
                            int[] stationaryPivots,
                            double[] target,
                            int targetOffset) {
        return solveCov(stateSpace, t, stationaryBacking, stationaryBackingOffset,
            stationaryPivots, target, targetOffset, stateSpace.stateCount());
    }

    static boolean solveCov(KalmanSSM stateSpace,
                            int t,
                            double[] stationaryBacking,
                            int[] stationaryPivots,
                            double[] target,
                            int targetOffset,
                            int targetLd) {
        return solveCov(stateSpace, t, stationaryBacking, 0, stationaryPivots, target, targetOffset, targetLd);
    }

    static boolean solveCov(KalmanSSM stateSpace,
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

        int transitionLength = kStates * kStates;
        int matrix2Length = Math.max(transitionLength, kStates * kPosdef);
        int transitionOffset = stationaryBackingOffset;
        int matrix2Offset = transitionOffset + transitionLength;
        int matrix3Offset = matrix2Offset + matrix2Length;
        int workOffset = matrix3Offset + transitionLength;

        validateWorkspace(stateSpace, stationaryBacking, stationaryBackingOffset, stationaryPivots);
        StationarySupport.copyRealMatrix(
            stateSpace.transitionData(),
            stateSpace.transitionOffset(t),
            stateSpace.transitionLeadingDimension(),
            stationaryBacking,
            transitionOffset,
            kStates);
        System.arraycopy(stationaryBacking, transitionOffset, stationaryBacking, matrix3Offset, transitionLength);
        if (!isDiscreteStable(stationaryBacking, matrix3Offset, workOffset, kStates)) {
            return false;
        }
        StationarySupport.copyRealMatrix(
            stateSpace.transitionData(),
            stateSpace.transitionOffset(t),
            stateSpace.transitionLeadingDimension(),
            stationaryBacking,
            transitionOffset,
            kStates);

        if (targetLd != kStates) {
            double[] contiguousTarget = new double[kStates * kStates];
            StationarySupport.buildSelectedStateCov(stateSpace, t, stationaryBacking, matrix2Offset, contiguousTarget, 0, kStates);
            if (!solveDiscreteLyapunov(stationaryBacking, transitionOffset, matrix2Offset,
                    stationaryPivots, contiguousTarget, 0, kStates, kStates)) {
                return false;
            }
            if (!StationarySupport.allFiniteSquare(contiguousTarget, 0, kStates, kStates, 1)) {
                return false;
            }
            for (int row = 0; row < kStates; row++) {
                System.arraycopy(contiguousTarget, row * kStates, target, targetOffset + row * targetLd, kStates);
            }
            return true;
        }

        StationarySupport.buildSelectedStateCov(stateSpace, t, stationaryBacking, matrix2Offset, target, targetOffset, targetLd);
        if (!solveDiscreteLyapunov(stationaryBacking, transitionOffset, matrix2Offset,
                stationaryPivots, target, targetOffset, targetLd, kStates)) {
            return false;
        }
        return StationarySupport.allFiniteSquare(target, targetOffset, kStates, targetLd, 1);
    }

    static double[] solveMean(KalmanSSM stateSpace, int t) {
        int kStates = stateSpace.stateCount();
        double[] stationaryMean = new double[kStates];
        double[] backing = new double[requiredDoubleCount(kStates, stateSpace.stateDisturbanceCount())];
        int[] pivots = new int[requiredPivotCount(kStates)];
        return solveMean(stateSpace, t, backing, pivots, stationaryMean, 0) ? stationaryMean : null;
    }

    static boolean solveMean(KalmanSSM stateSpace,
                             int t,
                             double[] stationaryBacking,
                             int[] stationaryPivots,
                             double[] target,
                             int targetOffset) {
        return solveMean(stateSpace, t, stationaryBacking, 0, stationaryPivots, target, targetOffset);
    }

    static boolean solveMean(KalmanSSM stateSpace,
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
        BLAS.dcopy(kStates, stateSpace.stateInterceptData(), stateSpace.stateInterceptOffset(t), 1, target, targetOffset, 1);
        StationarySupport.copyRealMatrix(
            stateSpace.transitionData(),
            stateSpace.transitionOffset(t),
            stateSpace.transitionLeadingDimension(),
            stationaryBacking,
            stationaryBackingOffset,
            kStates);
        BLAS.dscal(kStates * kStates, -1.0, stationaryBacking, stationaryBackingOffset, 1);
        for (int i = 0; i < kStates; i++) {
            stationaryBacking[stationaryBackingOffset + i * kStates + i] += 1.0;
        }

        if (BLAS.dgetrf(kStates, kStates, stationaryBacking, stationaryBackingOffset, kStates, stationaryPivots, 0) != 0) {
            return false;
        }
        BLAS.dgetrs(BLAS.Trans.NoTrans, kStates, 1, stationaryBacking, stationaryBackingOffset, kStates, stationaryPivots, 0, target, targetOffset, 1);
        return StationarySupport.allFinite(target, targetOffset, kStates);
    }

    private static boolean isDiscreteStable(double[] stationaryBacking,
                                            int transitionOffset,
                                            int workOffset,
                                            int kStates) {
        int wiOff = stationaryBacking.length - kStates;
        int wrOff = wiOff - kStates;
        int lwork = wrOff - workOffset;
        if (lwork < 1) {
            return false;
        }

        if (BLAS.dgees('N', 'N', null, kStates, stationaryBacking, transitionOffset, kStates,
            stationaryBacking, wrOff, stationaryBacking, wiOff, null, 0, kStates,
            stationaryBacking, workOffset, lwork, null) != 0) {
            return false;
        }

        for (int i = 0; i < kStates; i++) {
            if (Math.hypot(stationaryBacking[wrOff + i], stationaryBacking[wiOff + i]) >= 1.0) {
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
        double[] originalRhs = copySquare(rhs, rhsOffset, rhsLd, kStates);
        if (solveDiscreteLyapunovSeries(stationaryBacking, transitionOffset, matrix2Offset,
                rhs, rhsOffset, rhsLd, kStates)) {
            return true;
        }
        restoreSquare(originalRhs, rhs, rhsOffset, rhsLd, kStates);
        if (kStates <= 32 && solveDiscreteLyapunovDirect(stationaryBacking, transitionOffset, rhs, rhsOffset, rhsLd, kStates)) {
            return true;
        }
        restoreSquare(originalRhs, rhs, rhsOffset, rhsLd, kStates);
        if (solveDiscreteLyapunovBilinear(stationaryBacking, transitionOffset, matrix2Offset,
                stationaryPivots, rhs, rhsOffset, rhsLd, kStates)) {
            return true;
        }
        return false;
    }

    private static double[] copySquare(double[] source, int sourceOffset, int sourceLd, int dimension) {
        double[] copy = new double[dimension * dimension];
        for (int row = 0; row < dimension; row++) {
            System.arraycopy(source, sourceOffset + row * sourceLd, copy, row * dimension, dimension);
        }
        return copy;
    }

    private static void restoreSquare(double[] source, double[] target, int targetOffset, int targetLd, int dimension) {
        for (int row = 0; row < dimension; row++) {
            System.arraycopy(source, row * dimension, target, targetOffset + row * targetLd, dimension);
        }
    }

    private static boolean solveDiscreteLyapunovSeries(double[] stationaryBacking,
                                                       int transitionOffset,
                                                       int matrix2Offset,
                                                       double[] rhs,
                                                       int rhsOffset,
                                                       int rhsLd,
                                                       int kStates) {
        if (rhsLd != kStates) {
            return false;
        }

        int matrixLength = kStates * kStates;
        int previousOffset = matrix2Offset;
        int leftProductOffset = matrix2Offset + matrixLength;
        for (int row = 0; row < kStates; row++) {
            int rhsRow = rhsOffset + row * rhsLd;
            System.arraycopy(rhs, rhsRow, stationaryBacking, previousOffset + row * kStates, kStates);
        }

        final int maxIterations = 4096;
        final double tolerance = 1e-12;
        for (int iter = 0; iter < maxIterations; iter++) {
            Arrays.fill(stationaryBacking, leftProductOffset, leftProductOffset + matrixLength, 0.0);
            for (int row = 0; row < kStates; row++) {
                int rowBase = row * kStates;
                int transitionRow = transitionOffset + rowBase;
                for (int col = 0; col < kStates; col++) {
                    double factor = stationaryBacking[transitionRow + col];
                    if (factor == 0.0) {
                        continue;
                    }
                    int previousRow = previousOffset + col * kStates;
                    for (int inner = 0; inner < kStates; inner++) {
                        stationaryBacking[leftProductOffset + rowBase + inner] += factor * stationaryBacking[previousRow + inner];
                    }
                }
            }

            double max = 0.0;
            for (int row = 0; row < kStates; row++) {
                int rowBase = row * kStates;
                int rhsRow = rhsOffset + row * rhsLd;
                for (int col = 0; col < kStates; col++) {
                    double value = 0.0;
                    int transitionRow = transitionOffset + col * kStates;
                    for (int inner = 0; inner < kStates; inner++) {
                        value += stationaryBacking[leftProductOffset + rowBase + inner] * stationaryBacking[transitionRow + inner];
                    }
                    stationaryBacking[previousOffset + rowBase + col] = value;
                    rhs[rhsRow + col] += value;
                    max = Math.max(max, Math.abs(value));
                }
            }
            if (max < tolerance) {
                return StationarySupport.allFiniteSquare(rhs, rhsOffset, kStates, rhsLd, 1);
            }
        }
        return false;
    }

    private static boolean solveDiscreteLyapunovBilinear(double[] stationaryBacking,
                                                         int transitionOffset,
                                                         int matrix2Offset,
                                                         int[] stationaryPivots,
                                                         double[] rhs,
                                                         int rhsOffset,
                                                         int rhsLd,
                                                         int kStates) {
        for (int row = 0; row < kStates; row++) {
            System.arraycopy(stationaryBacking, transitionOffset + row * kStates,
                stationaryBacking, matrix2Offset + row * kStates, kStates);
        }
        for (int i = 0; i < kStates; i++) {
            stationaryBacking[matrix2Offset + i * kStates + i] += 1.0;
            stationaryBacking[transitionOffset + i * kStates + i] -= 1.0;
        }

        if (BLAS.dgetrf(kStates, kStates, stationaryBacking, matrix2Offset, kStates, stationaryPivots, 0) != 0) {
            return false;
        }

        BLAS.dgetrs(BLAS.Trans.NoTrans, kStates, kStates,
            stationaryBacking, matrix2Offset, kStates, stationaryPivots, 0,
            stationaryBacking, transitionOffset, kStates);

        BLAS.dgetrs(BLAS.Trans.NoTrans, kStates, kStates,
            stationaryBacking, matrix2Offset, kStates, stationaryPivots, 0,
            rhs, rhsOffset, rhsLd);
        StationarySupport.transposeSquare(rhs, rhsOffset, kStates, rhsLd);
        BLAS.dgetrs(BLAS.Trans.NoTrans, kStates, kStates,
            stationaryBacking, matrix2Offset, kStates, stationaryPivots, 0,
            rhs, rhsOffset, rhsLd);
        StationarySupport.transposeSquare(rhs, rhsOffset, kStates, rhsLd);
        for (int row = 0; row < kStates; row++) {
            BLAS.dscal(kStates, -2.0, rhs, rhsOffset + row * rhsLd, 1);
        }

        return solveContinuousLyapunov(stationaryBacking, transitionOffset, matrix2Offset, rhs, rhsOffset, rhsLd, kStates, 1.0);
    }

    private static boolean solveDiscreteLyapunovDirect(double[] stationaryBacking,
                                                       int transitionOffset,
                                                       double[] rhs,
                                                       int rhsOffset,
                                                       int rhsLd,
                                                       int kStates) {
        int systemDim = kStates * kStates;
        double[] system = new double[systemDim * systemDim];
        double[] vector = new double[systemDim];
        int[] pivots = new int[systemDim];

        for (int row = 0; row < kStates; row++) {
            for (int col = 0; col < kStates; col++) {
                int equation = row * kStates + col;
                vector[equation] = rhs[rhsOffset + row * rhsLd + col];
                system[equation * systemDim + equation] = 1.0;
                int transitionRow = transitionOffset + row * kStates;
                int transitionCol = transitionOffset + col * kStates;
                for (int i = 0; i < kStates; i++) {
                    double left = stationaryBacking[transitionRow + i];
                    if (left == 0.0) {
                        continue;
                    }
                    int variableRow = i * kStates;
                    for (int j = 0; j < kStates; j++) {
                        double right = stationaryBacking[transitionCol + j];
                        if (right == 0.0) {
                            continue;
                        }
                        system[equation * systemDim + variableRow + j] -= left * right;
                    }
                }
            }
        }

        if (BLAS.dgetrf(systemDim, systemDim, system, 0, systemDim, pivots, 0) != 0) {
            return false;
        }
        BLAS.dgetrs(BLAS.Trans.NoTrans, systemDim, 1, system, 0, systemDim, pivots, 0, vector, 0, 1);

        for (int row = 0; row < kStates; row++) {
            int rhsRow = rhsOffset + row * rhsLd;
            int vectorRow = row * kStates;
            System.arraycopy(vector, vectorRow, rhs, rhsRow, kStates);
        }
        for (int row = 0; row < kStates; row++) {
            for (int col = row + 1; col < kStates; col++) {
                double avg = 0.5 * (rhs[rhsOffset + row * rhsLd + col] + rhs[rhsOffset + col * rhsLd + row]);
                rhs[rhsOffset + row * rhsLd + col] = avg;
                rhs[rhsOffset + col * rhsLd + row] = avg;
            }
        }
        return true;
    }

    private static boolean solveContinuousLyapunov(double[] stationaryBacking,
                                                   int transitionOffset,
                                                   int matrix2Offset,
                                                   double[] rhs,
                                                   int rhsOffset,
                                                   int rhsLd,
                                                   int kStates,
                                                   double sign) {
        if (kStates == 1) {
            rhs[rhsOffset] = sign * rhs[rhsOffset] / (2.0 * stationaryBacking[transitionOffset]);
            return true;
        }

        int matrix3Offset = matrix2Offset + kStates * kStates;
        int workOffset = matrix3Offset + kStates * kStates;
        int workLength = stationaryBacking.length - workOffset;
        int lwork = Math.max(1, workLength - 2 * kStates);
        int wrOff = workOffset + lwork;
        int wiOff = wrOff + kStates;
        if (BLAS.dgees('V', 'N', null, kStates, stationaryBacking, transitionOffset, kStates,
            stationaryBacking, wrOff, stationaryBacking, wiOff, stationaryBacking, matrix3Offset, kStates,
            stationaryBacking, workOffset, lwork, null) != 0) {
            return false;
        }

        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
            sign, rhs, rhsOffset, rhsLd, stationaryBacking, matrix3Offset, kStates,
            0.0, stationaryBacking, matrix2Offset, kStates);
        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
            1.0, stationaryBacking, matrix3Offset, kStates, stationaryBacking, matrix2Offset, kStates,
            0.0, rhs, rhsOffset, rhsLd);

        double scale = BLAS.dtrsyl(false, true, 1, kStates, kStates,
            stationaryBacking, transitionOffset, kStates, stationaryBacking, transitionOffset, kStates,
            rhs, rhsOffset, rhsLd, null);

        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates,
            scale, rhs, rhsOffset, rhsLd, stationaryBacking, matrix3Offset, kStates,
            0.0, stationaryBacking, matrix2Offset, kStates);
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
            1.0, stationaryBacking, matrix3Offset, kStates, stationaryBacking, matrix2Offset, kStates,
            0.0, rhs, rhsOffset, rhsLd);
        return true;
    }

    private static int requiredWorkLength(int kStates) {
        if (kStates == 0) {
            return 0;
        }
        double[] query = new double[1];
        BLAS.dgees('V', 'N', null, kStates, null, kStates,
            null, null, null, kStates,
            query, 0, -1, null);
        return Math.max(1, (int) query[0]) + 2 * kStates;
    }

    private static void validateWorkspace(KalmanSSM stateSpace,
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
}