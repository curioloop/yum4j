package com.curioloop.yum4j.ssm.kalman.init;

import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.cmplx.ZLAS;

final class StationarySupport {

    private StationarySupport() {}

    static void buildSelectedStateCov(KalmanSSM stateSpace,
                                      int t,
                                      double[] workspace,
                                      int workspaceOffset,
                                      double[] target,
                                      int targetOffset) {
        buildSelectedStateCov(stateSpace, t, workspace, workspaceOffset, target, targetOffset, stateSpace.stateCount());
    }

    static void buildSelectedStateCov(KalmanSSM stateSpace,
                                      int t,
                                      double[] workspace,
                                      int workspaceOffset,
                                      double[] target,
                                      int targetOffset,
                                      int targetLd) {
        int kStates = stateSpace.stateCount();
        int kPosdef = stateSpace.stateDisturbanceCount();
        if (kPosdef == 0) {
            zeroSquare(target, targetOffset, kStates, targetLd, stateSpace.complex() ? 2 : 1);
            return;
        }

        double[] selection = stateSpace.selectionData();
        int selectionOffset = stateSpace.selectionOffset(t);
        int selectionLd = stateSpace.selectionLeadingDimension();
        double[] stateCov = stateSpace.stateCovarianceData();
        int stateCovOffset = stateSpace.stateCovarianceOffset(t);
        int stateCovLd = stateSpace.stateCovarianceLeadingDimension();

        if (stateSpace.complex()) {
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                1.0, 0.0, selection, selectionOffset >> 1, selectionLd,
                stateCov, stateCovOffset >> 1, stateCovLd,
                0.0, 0.0, workspace, workspaceOffset >> 1, kPosdef);
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kPosdef,
                1.0, 0.0, workspace, workspaceOffset >> 1, kPosdef,
                selection, selectionOffset >> 1, selectionLd,
                0.0, 0.0, target, targetOffset >> 1, targetLd);
            return;
        }

        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
            1.0, selection, selectionOffset, selectionLd,
            stateCov, stateCovOffset, stateCovLd,
            0.0, workspace, workspaceOffset, kPosdef);
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kPosdef,
            1.0, workspace, workspaceOffset, kPosdef,
            selection, selectionOffset, selectionLd,
            0.0, target, targetOffset, targetLd);
    }

    static void copyRealMatrix(double[] source, int sourceOffset, int sourceLd, double[] target, int n) {
        copyRealMatrix(source, sourceOffset, sourceLd, target, 0, n);
    }

    static void copyRealMatrix(double[] source, int sourceOffset, int sourceLd, double[] target, int targetOffset, int n) {
        for (int i = 0; i < n; i++) {
            System.arraycopy(source, sourceOffset + i * sourceLd, target, targetOffset + i * n, n);
        }
    }

    static void copyComplexMatrix(double[] source, int sourceOffset, int sourceLd, double[] target, int n) {
        copyComplexMatrix(source, sourceOffset, sourceLd, target, 0, n);
    }

    static void copyComplexMatrix(double[] source, int sourceOffset, int sourceLd, double[] target, int targetOffset, int n) {
        for (int i = 0; i < n; i++) {
            int sourceRow = sourceOffset + i * sourceLd;
            int targetRow = targetOffset + i * n * 2;
            for (int j = 0; j < n; j++) {
                int sourceIndex = sourceRow + j * 2;
                int targetIndex = targetRow + j * 2;
                target[targetIndex] = source[sourceIndex];
                target[targetIndex + 1] = source[sourceIndex + 1];
            }
        }
    }

    static void transposeSquare(double[] matrix, int offset, int n) {
        transposeSquare(matrix, offset, n, n);
    }

    static void transposeSquare(double[] matrix, int offset, int n, int ld) {
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                int ij = offset + i * ld + j;
                int ji = offset + j * ld + i;
                double value = matrix[ij];
                matrix[ij] = matrix[ji];
                matrix[ji] = value;
            }
        }
    }

    static void transposeSquare(double[] matrix, int n) {
        transposeSquare(matrix, 0, n);
    }

    static void conjugateTransposeSquare(double[] matrix, int offset, int n) {
        conjugateTransposeSquare(matrix, offset, n, n);
    }

    static void conjugateTransposeSquare(double[] matrix, int offset, int n, int ld) {
        for (int i = 0; i < n; i++) {
            matrix[offset + (i * ld + i) * 2 + 1] = -matrix[offset + (i * ld + i) * 2 + 1];
        }
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                int ij = offset + (i * ld + j) * 2;
                int ji = offset + (j * ld + i) * 2;
                double ijRe = matrix[ij];
                double ijIm = matrix[ij + 1];
                double jiRe = matrix[ji];
                double jiIm = matrix[ji + 1];
                matrix[ij] = jiRe;
                matrix[ij + 1] = -jiIm;
                matrix[ji] = ijRe;
                matrix[ji + 1] = -ijIm;
            }
        }
    }

    static void conjugateTransposeSquare(double[] matrix, int n) {
        conjugateTransposeSquare(matrix, 0, n);
    }

    static void enforceHermitian(double[] matrix, int offset, int n) {
        enforceHermitian(matrix, offset, n, n);
    }

    static void enforceHermitian(double[] matrix, int offset, int n, int ld) {
        for (int i = 0; i < n; i++) {
            matrix[offset + (i * ld + i) * 2 + 1] = 0.0;
            for (int j = i + 1; j < n; j++) {
                int ij = offset + (i * ld + j) * 2;
                int ji = offset + (j * ld + i) * 2;
                double avgRe = 0.5 * (matrix[ij] + matrix[ji]);
                double avgIm = 0.5 * (matrix[ij + 1] - matrix[ji + 1]);
                matrix[ij] = avgRe;
                matrix[ij + 1] = avgIm;
                matrix[ji] = avgRe;
                matrix[ji + 1] = -avgIm;
            }
        }
    }

    static void enforceHermitian(double[] matrix, int n) {
        enforceHermitian(matrix, 0, n);
    }

    static boolean allFinite(double[] values, int offset, int length) {
        for (int i = 0; i < length; i++) {
            if (!Double.isFinite(values[offset + i])) {
                return false;
            }
        }
        return true;
    }

    static boolean allFiniteSquare(double[] values, int offset, int n, int ld, int scalarWidth) {
        for (int row = 0; row < n; row++) {
            int rowOffset = offset + row * ld * scalarWidth;
            for (int col = 0; col < n * scalarWidth; col++) {
                if (!Double.isFinite(values[rowOffset + col])) {
                    return false;
                }
            }
        }
        return true;
    }

    static boolean allFinite(double[] values) {
        return allFinite(values, 0, values.length);
    }

    private static void zero(double[] values, int offset, int length) {
        for (int i = 0; i < length; i++) {
            values[offset + i] = 0.0;
        }
    }

    private static void zeroSquare(double[] values, int offset, int n, int ld, int scalarWidth) {
        for (int row = 0; row < n; row++) {
            zero(values, offset + row * ld * scalarWidth, n * scalarWidth);
        }
    }
}