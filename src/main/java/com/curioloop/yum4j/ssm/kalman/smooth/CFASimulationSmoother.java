package com.curioloop.yum4j.ssm.kalman.smooth;

import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSMSupport;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.linalg.blas.BLAS;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

public final class CFASimulationSmoother {

    private CFASimulationSmoother() {
    }

    public record PosteriorMoments(int kStates, int nobs, double[] mean, double[] covariance) {
        public int order() {
            return kStates * nobs;
        }

        public double mean(int i, int t) {
            return mean[t * kStates + i];
        }

        public double covariance(int row, int col, int t) {
            int order = order();
            int rowIndex = t * kStates + row;
            int colIndex = t * kStates + col;
            return covariance[rowIndex * order + colIndex];
        }

        public double jointCovariance(int row, int t, int col, int s) {
            int order = order();
            return covariance[(t * kStates + row) * order + s * kStates + col];
        }
    }

    private record PosteriorSystem(double[] factor, double[] mean) {
    }

    static final class Workspace {
        private double[] precision;
        private double[] rhs;
        private double[] workMatrix0;
        private double[] workMatrix1;
        private double[] workMatrix2;
        private double[] workMatrix3;
        private double[] workVector0;
        private double[] workVector1;
        private double[] workVector2;
        private int[] observedIndex;

        void ensure(int order) {
            precision = ensure(precision, order * order);
            rhs = ensure(rhs, order);
        }

        void ensure(KalmanSSM model) {
            int kEndog = model.observationDimension();
            int kStates = model.stateCount();
            int kPosdef = model.stateDisturbanceCount();
            int order = model.observationCount() * kStates;
            ensure(order);
            int matrixLength = Math.max(kStates * kStates,
                Math.max(Math.max(kEndog * kEndog, kEndog * kStates), kStates * kPosdef));
            int vectorLength = Math.max(kStates, kEndog);
            workMatrix0 = ensure(workMatrix0, matrixLength);
            workMatrix1 = ensure(workMatrix1, matrixLength);
            workMatrix2 = ensure(workMatrix2, matrixLength);
            workMatrix3 = ensure(workMatrix3, matrixLength);
            workVector0 = ensure(workVector0, vectorLength);
            workVector1 = ensure(workVector1, vectorLength);
            workVector2 = ensure(workVector2, vectorLength);
            if (observedIndex == null || observedIndex.length < kEndog) {
                observedIndex = new int[kEndog];
            }
        }

        long retainedDoubleCount() {
            return doubleCount(precision)
                + doubleCount(rhs)
                + doubleCount(workMatrix0)
                + doubleCount(workMatrix1)
                + doubleCount(workMatrix2)
                + doubleCount(workMatrix3)
                + doubleCount(workVector0)
                + doubleCount(workVector1)
                + doubleCount(workVector2);
        }

        void release() {
            precision = null;
            rhs = null;
            workMatrix0 = null;
            workMatrix1 = null;
            workMatrix2 = null;
            workMatrix3 = null;
            workVector0 = null;
            workVector1 = null;
            workVector2 = null;
            observedIndex = null;
        }

        private static double[] ensure(double[] values, int length) {
            return values == null || values.length < length ? new double[length] : values;
        }

        private static long doubleCount(double[] values) {
            return values == null ? 0L : values.length;
        }
    }

    public static void validateContract(KalmanSSM model,
                                        InitialState init,
                                        SimulationSmootherOptions options) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(init, "init");
        Objects.requireNonNull(options, "options");
        if (!options.includes(SimulationSmootherOptions.Surface.STATE)
                || options.includes(SimulationSmootherOptions.Surface.DISTURBANCE)
                || options.includes(SimulationSmootherOptions.Surface.GENERATED_OUTPUTS)) {
            throw new UnsupportedOperationException(
                "CFA simulation smoothing currently supports state-vector output only");
        }
        validateModelContract(model, init);
    }

    public static PosteriorMoments posteriorMoments(KalmanSSM model,
                                                    InitialState init) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(init, "init");
        validateModelContract(model, init);
        int order = model.observationCount() * model.stateCount();
        Workspace workspace = new Workspace();
        workspace.ensure(model);
        PosteriorSystem posterior = computePosteriorSystem(model, init, order, workspace);
        double[] covariance = new double[order * order];
        for (int i = 0; i < order; i++) {
            covariance[i * order + i] = 1.0;
        }
        BLAS.dpotrs(BLAS.Uplo.Lower, order, order,
            posterior.factor(), 0, order,
            covariance, 0, order);
        return new PosteriorMoments(model.stateCount(), model.observationCount(),
            posterior.mean().clone(), covariance);
    }

    private static void validateModelContract(KalmanSSM model,
                                              InitialState init) {
        KalmanSSMSupport.requireRealStorage(model, "model");
        if (model.observationCount() == 1) {
            throw new UnsupportedOperationException(
                "CFA simulation smoothing cannot be used with a single observation");
        }
        if (init.mayResolveDiffuse()) {
            throw new UnsupportedOperationException(
                "CFA simulation smoothing with diffuse initialization is not implemented yet");
        }
        validateInitialDistribution(model, init);
        validateObservationShocks(model);
        validateStateShocks(model);
    }

    static SimulationSmootherResult simulate(KalmanSSM model,
                                             InitialState init,
                                             Random rng,
                                             SimulationSmoother.Pool pool,
                                             SimulationSmootherOptions options) {
        Objects.requireNonNull(rng, "rng");
        int order = model.observationCount() * model.stateCount();
        double[] variates = new double[order];
        for (int i = 0; i < order; i++) {
            variates[i] = rng.nextGaussian();
        }
        return simulate(model, init, variates, pool, options);
    }

    static SimulationSmootherResult simulate(KalmanSSM model,
                                             InitialState init,
                                             double[] variates,
                                             SimulationSmoother.Pool pool,
                                             SimulationSmootherOptions options) {
        validateContract(model, init, options);
        Objects.requireNonNull(variates, "variates");
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        int kPosdef = model.stateDisturbanceCount();
        int nobs = model.observationCount();
        int order = nobs * kStates;
        if (variates.length < order) {
            throw new IllegalArgumentException("CFA simulation smoothing variates length must be at least " + order);
        }

        SimulationSmootherResult result;
        if (pool != null) {
            result = pool.borrowResult(kEndog, kStates, kPosdef, nobs,
                false, true, false);
        } else {
            result = new SimulationSmootherResult(kEndog, kStates, kPosdef, nobs,
                false, true, false);
        }

        Workspace workspace = pool == null ? new Workspace() : pool.cfaWorkspace(order);
        workspace.ensure(model);
        PosteriorSystem posterior = computePosteriorSystem(model, init, order, workspace);

        int drawBase = result.simulatedStateBase();
        System.arraycopy(variates, 0, result.simulatedState, drawBase, order);
        BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit,
            order, 1, 1.0, posterior.factor(), 0, order,
            result.simulatedState, drawBase, 1);
        for (int t = 0; t < nobs; t++) {
            int sourceOffset = t * kStates;
            int targetOffset = result.simulatedStateOffset(t);
            for (int i = 0; i < kStates; i++) {
                result.simulatedState[targetOffset + i] += posterior.mean()[sourceOffset + i];
            }
        }
        if (pool != null && !pool.retainsCfaPosteriorWorkspace()) {
            pool.releaseCfaWorkspace();
        }
        return result;
    }

    private static PosteriorSystem computePosteriorSystem(KalmanSSM model,
                                                          InitialState init,
                                                          int order,
                                                          Workspace workspace) {
        double[] precision = workspace.precision;
        double[] rhs = workspace.rhs;
        Arrays.fill(precision, 0, order * order, 0.0);
        Arrays.fill(rhs, 0, order, 0.0);
        assemblePosteriorPrecision(model, init, precision, rhs, workspace);

        double[] factor = precision;
        int info = BLAS.dpotrf(BLAS.Uplo.Lower, order, factor, 0, order);
        if (info != 0) {
            throw new IllegalStateException("CFA simulation smoothing posterior precision is not positive definite");
        }

        double[] mean = rhs;
        BLAS.dpotrs(BLAS.Uplo.Lower, order, 1, factor, 0, order, mean, 0, 1);
        return new PosteriorSystem(factor, mean);
    }

    private static void assemblePosteriorPrecision(KalmanSSM model,
                                                   InitialState init,
                                                   double[] precision,
                                                   double[] rhs,
                                                   Workspace workspace) {
        int kStates = model.stateCount();
        int nobs = model.observationCount();
        double[] initialState = workspace.workVector0;
        double[] initialCovariance = workspace.workMatrix0;
        init.resolveInto(model, 0, initialState, 0, initialCovariance, 0, null, 0);
        double[] initialPrecision = inversePositiveDefinite(initialCovariance, kStates,
            "CFA simulation smoothing requires a non-degenerate initial state distribution");
        addBlock(precision, nobs, kStates, 0, 0, initialPrecision, 1.0);
        addMatrixVectorProduct(rhs, 0, initialPrecision, kStates, initialState, 0, workspace.workVector1, 1.0);

        for (int t = 0; t < nobs - 1; t++) {
            addTransitionPrecision(model, t, precision, rhs, workspace);
        }
        for (int t = 0; t < nobs; t++) {
            addObservationPrecision(model, t, precision, rhs, workspace);
        }
    }

    private static void addTransitionPrecision(KalmanSSM model,
                                               int t,
                                               double[] precision,
                                               double[] rhs,
                                               Workspace workspace) {
        int kStates = model.stateCount();
        int nobs = model.observationCount();
        double[] transition = workspace.workMatrix0;
        copySquare(model.transitionData(), model.transitionOffset(t),
            model.transitionLeadingDimension(), kStates, transition, kStates);
        double[] stateIntercept = workspace.workVector0;
        System.arraycopy(model.stateInterceptData(), model.stateInterceptOffset(t), stateIntercept, 0, kStates);
        double[] selectedPrecision = inverseSelectedStateCovariance(model, t, workspace.workMatrix1, workspace.workMatrix2);
        double[] qiT = workspace.workMatrix2;
        multiply(selectedPrecision, kStates, kStates, transition, kStates, qiT);
        double[] tQiT = workspace.workMatrix3;
        multiplyTransposeLeft(transition, kStates, kStates, qiT, kStates, tQiT);

        addBlock(precision, nobs, kStates, t, t, tQiT, 1.0);
        addBlock(precision, nobs, kStates, t + 1, t + 1, selectedPrecision, 1.0);
        addBlock(precision, nobs, kStates, t + 1, t, qiT, -1.0);
        addBlockTranspose(precision, nobs, kStates, t, t + 1, qiT, -1.0);

        double[] qiC = workspace.workVector1;
        matrixVector(selectedPrecision, kStates, stateIntercept, 0, qiC, 0);
        addVector(rhs, (t + 1) * kStates, qiC, kStates, 1.0);
        double[] tQiC = workspace.workVector2;
        matrixTransposeVector(transition, kStates, kStates, qiC, 0, tQiC, 0);
        addVector(rhs, t * kStates, tQiC, kStates, -1.0);
    }

    private static void addObservationPrecision(KalmanSSM model,
                                                int t,
                                                double[] precision,
                                                double[] rhs,
                                                Workspace workspace) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        int nobs = model.observationCount();
        int observed = kEndog - model.missingCount(t);
        if (observed == 0) {
            return;
        }
        int[] indices = workspace.observedIndex;
        for (int i = 0, j = 0; i < kEndog; i++) {
            if (!model.isMissing(i, t)) {
                indices[j++] = i;
            }
        }

        double[] obsCov = workspace.workMatrix0;
        double[] design = workspace.workMatrix1;
        double[] demeaned = workspace.workVector0;
        int obsCovOffset = model.obsCovOffset(t);
        int obsCovLd = model.obsCovLeadingDimension();
        int designOffset = model.designOffset(t);
        int designLd = model.designLeadingDimension();
        int obsInterceptOffset = model.obsInterceptOffset(t);
        int endogOffset = model.endogOffset(t);
        for (int row = 0; row < observed; row++) {
            int sourceRow = indices[row];
            demeaned[row] = model.endogData()[endogOffset + sourceRow]
                - model.obsInterceptData()[obsInterceptOffset + sourceRow];
            System.arraycopy(model.designData(), designOffset + sourceRow * designLd,
                design, row * kStates, kStates);
            for (int col = 0; col < observed; col++) {
                obsCov[row * observed + col] = model.obsCovData()[obsCovOffset + sourceRow * obsCovLd + indices[col]];
            }
        }
        double[] obsPrecision = inversePositiveDefinite(obsCov, observed,
            "CFA simulation smoothing requires non-degenerate observation shocks");
        double[] hiZ = workspace.workMatrix2;
        multiply(obsPrecision, observed, observed, design, kStates, hiZ);
        double[] zHiZ = workspace.workMatrix3;
        multiplyTransposeLeft(design, observed, kStates, hiZ, kStates, zHiZ);
        addBlock(precision, nobs, kStates, t, t, zHiZ, 1.0);
        double[] hiY = workspace.workVector1;
        matrixVector(obsPrecision, observed, demeaned, 0, hiY, 0);
        double[] zHiY = workspace.workVector2;
        matrixTransposeVector(design, observed, kStates, hiY, 0, zHiY, 0);
        addVector(rhs, t * kStates, zHiY, kStates, 1.0);
    }

    private static double[] inverseSelectedStateCovariance(KalmanSSM model, int t, double[] selected, double[] work) {
        buildSelectedStateCovariance(model, t, selected, work);
        return inversePositiveDefinite(selected, model.stateCount(),
            "CFA simulation smoothing requires non-degenerate state shocks");
    }

    private static void buildSelectedStateCovariance(KalmanSSM model, int t, double[] selected, double[] work) {
        int kStates = model.stateCount();
        int kPosdef = model.stateDisturbanceCount();
        if (kPosdef == 0) {
            Arrays.fill(selected, 0, kStates * kStates, 0.0);
            return;
        }
        int selectionOffset = model.selectionOffset(t);
        int selectionLd = model.selectionLeadingDimension();
        int stateCovOffset = model.stateCovarianceOffset(t);
        int stateCovLd = model.stateCovarianceLeadingDimension();
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
            1.0, model.selectionData(), selectionOffset, selectionLd,
            model.stateCovarianceData(), stateCovOffset, stateCovLd,
            0.0, work, 0, kPosdef);
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kPosdef,
            1.0, work, 0, kPosdef,
            model.selectionData(), selectionOffset, selectionLd,
            0.0, selected, 0, kStates);
    }

    private static double[] inversePositiveDefinite(double[] matrix, int dimension, String message) {
        if (BLAS.dpotrf(BLAS.Uplo.Lower, dimension, matrix, 0, dimension) != 0
                || !BLAS.dpotri(BLAS.Uplo.Lower, dimension, matrix, 0, dimension)) {
            throw new UnsupportedOperationException(message);
        }
        for (int row = 0; row < dimension; row++) {
            for (int col = row + 1; col < dimension; col++) {
                matrix[row * dimension + col] = matrix[col * dimension + row];
            }
        }
        return matrix;
    }

    private static void multiply(double[] left, int rows, int inner, double[] right, int cols, double[] out) {
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, rows, cols, inner,
            1.0, left, 0, inner,
            right, 0, cols,
            0.0, out, 0, cols);
    }

    private static void multiplyTransposeLeft(double[] left, int rows, int cols, double[] right, int rightCols, double[] out) {
        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, cols, rightCols, rows,
            1.0, left, 0, cols,
            right, 0, rightCols,
            0.0, out, 0, rightCols);
    }

    private static void matrixVector(double[] matrix, int dimension,
                                     double[] vector, int vectorOffset,
                                     double[] out, int outOffset) {
        BLAS.dgemv(BLAS.Trans.NoTrans, dimension, dimension, 1.0,
            matrix, 0, dimension,
            vector, vectorOffset, 1,
            0.0, out, outOffset, 1);
    }

    private static void matrixTransposeVector(double[] matrix, int rows, int cols,
                                              double[] vector, int vectorOffset,
                                              double[] out, int outOffset) {
        BLAS.dgemv(BLAS.Trans.Trans, rows, cols, 1.0,
            matrix, 0, cols,
            vector, vectorOffset, 1,
            0.0, out, outOffset, 1);
    }

    private static void addMatrixVectorProduct(double[] target, int targetOffset,
                                               double[] matrix, int dimension,
                                               double[] vector, int vectorOffset,
                                               double scale) {
        double[] tmp = new double[dimension];
        addMatrixVectorProduct(target, targetOffset, matrix, dimension, vector, vectorOffset, tmp, scale);
    }

    private static void addMatrixVectorProduct(double[] target, int targetOffset,
                                               double[] matrix, int dimension,
                                               double[] vector, int vectorOffset,
                                               double[] tmp,
                                               double scale) {
        matrixVector(matrix, dimension, vector, vectorOffset, tmp, 0);
        addVector(target, targetOffset, tmp, dimension, scale);
    }

    private static void addVector(double[] target, int targetOffset, double[] source, int length, double scale) {
        for (int i = 0; i < length; i++) {
            target[targetOffset + i] += scale * source[i];
        }
    }

    private static void addBlock(double[] target, int nobs, int kStates,
                                 int blockRow, int blockCol,
                                 double[] block, double scale) {
        int order = nobs * kStates;
        int rowBase = blockRow * kStates;
        int colBase = blockCol * kStates;
        for (int row = 0; row < kStates; row++) {
            int targetRow = (rowBase + row) * order + colBase;
            int blockRowOffset = row * kStates;
            for (int col = 0; col < kStates; col++) {
                target[targetRow + col] += scale * block[blockRowOffset + col];
            }
        }
    }

    private static void addBlockTranspose(double[] target, int nobs, int kStates,
                                          int blockRow, int blockCol,
                                          double[] block, double scale) {
        int order = nobs * kStates;
        int rowBase = blockRow * kStates;
        int colBase = blockCol * kStates;
        for (int row = 0; row < kStates; row++) {
            int targetRow = (rowBase + row) * order + colBase;
            for (int col = 0; col < kStates; col++) {
                target[targetRow + col] += scale * block[col * kStates + row];
            }
        }
    }

    private static void validateInitialDistribution(KalmanSSM model, InitialState init) {
        int kStates = model.stateCount();
        double[] state = new double[kStates];
        double[] covariance = new double[kStates * kStates];
        double[] diffuseCovariance = new double[kStates * kStates];
        boolean diffuse = init.resolveInto(model, 0,
            state, 0,
            covariance, 0,
            diffuseCovariance, 0);
        if (diffuse) {
            throw new UnsupportedOperationException(
                "CFA simulation smoothing with diffuse initialization is not implemented yet");
        }
        requirePositiveDefinite(covariance, kStates,
            "CFA simulation smoothing requires a non-degenerate initial state distribution");
    }

    private static void validateObservationShocks(KalmanSSM model) {
        int kEndog = model.observationDimension();
        double[] covariance = new double[kEndog * kEndog];
        for (int t = 0; t < model.observationCount(); t++) {
            copySquare(model.obsCovData(), model.obsCovOffset(t), model.obsCovLeadingDimension(),
                kEndog, covariance, kEndog);
            requirePositiveDefinite(covariance, kEndog,
                "CFA simulation smoothing requires non-degenerate observation shocks");
        }
    }

    private static void validateStateShocks(KalmanSSM model) {
        int kStates = model.stateCount();
        int kPosdef = model.stateDisturbanceCount();
        double[] selectedCovariance = new double[kStates * kStates];
        double[] selectedWork = new double[kStates * kPosdef];
        for (int t = 0; t < model.observationCount(); t++) {
            buildSelectedStateCovariance(model, t, selectedCovariance, selectedWork);
            requirePositiveDefinite(selectedCovariance, kStates,
                "CFA simulation smoothing requires non-degenerate state shocks");
        }
    }

    private static void requirePositiveDefinite(double[] matrix, int dimension, String message) {
        double[] factor = matrix.clone();
        if (BLAS.dpotrf(BLAS.Uplo.Lower, dimension, factor, 0, dimension) != 0) {
            throw new UnsupportedOperationException(message);
        }
    }

    private static void copySquare(double[] source,
                                   int sourceOffset,
                                   int sourceLeadingDimension,
                                   int dimension,
                                   double[] target,
                                   int targetLeadingDimension) {
        for (int row = 0; row < dimension; row++) {
            System.arraycopy(source, sourceOffset + row * sourceLeadingDimension,
                target, row * targetLeadingDimension, dimension);
        }
    }
}