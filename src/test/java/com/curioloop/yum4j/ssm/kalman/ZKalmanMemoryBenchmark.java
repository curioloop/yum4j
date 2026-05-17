package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;

import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;

import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.ZSmootherResult;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.cmplx.ZLAS;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(0)
@State(Scope.Thread)
public class ZKalmanMemoryBenchmark {

    @Param({"256", "1024"})
    int nobs;

    @Param({"16", "32", "64"})
    int kStates;

    private int ke;

    private ModelFixture rep;
    private ModelFixture missingRep;
    private InitialState init;
    private FilterOptions fullFilterOptions;
    private FilterOptions predictedStateOnlyFilterOptions;
    private FilterOptions predictedCovarianceOnlyFilterOptions;
    private FilterOptions filteredStateOnlyFilterOptions;
    private FilterOptions filteredCovarianceOnlyFilterOptions;
    private FilterOptions smootherFilterOptions;
    private SmootherOptions fullSmootherOptions;
    private KalmanEngine.Workspace borrowedFilterWorkspace;
    private KalmanEngine.Workspace borrowedPredictedStateOnlyFilterWorkspace;
    private KalmanEngine.Workspace borrowedPredictedCovarianceOnlyFilterWorkspace;
    private KalmanEngine.Workspace borrowedFilteredStateOnlyFilterWorkspace;
    private KalmanEngine.Workspace borrowedFilteredCovarianceOnlyFilterWorkspace;
    private SmootherEngine.Workspace borrowedSmootherWorkspace;
    private ZFilterResult smootherFilterInput;
    private SmootherEngine.Workspace missingBorrowedSmootherWorkspace;
    private ZFilterResult missingSmootherFilterInput;

    private double[] gemvMatrix;
    private double[] gemvVector;
    private double[] gemvOutput;
    private double[] gemvConjOutput;
    private double[] gemmNtConjA;
    private double[] gemmNtConjB;
    private double[] gemmNtConjOut;
    private double[] gemmNnStateObsOut;
    private double[] gemmNnSquareSmallKB;
    private double[] gemmNnSquareSmallKOut;
    private double[] gemmNnObsSquareA;
    private double[] gemmNnObsSquareOut;
    private double[] gemmCnA;
    private double[] gemmCnB;
    private double[] gemmCnOut;
    private double[] gemmCnSquareOut;
    private double[] gemmCnSquareSmallKA;
    private double[] gemmCnSquareSmallKB;
    private double[] gemmCnSquareSmallKOut;
    private double[] gemmCcSquareOut;
    private double[] potrfStateBase;
    private double[] potrfStateWork;
    private double[] potrsStateFactorBase;
    private double[] potrsStateFactorWork;
    private double[] potrsStateVectorBase;
    private double[] potrsStateVectorWork;
    private double[] potrsStateRectBase;
    private double[] potrsStateRectWork;
    private double[] potrfObsBase;
    private double[] potrfObsWork;
    private double[] potrsObsVectorBase;
    private double[] potrsObsVectorWork;
    private double[] potrsObsRectBase;
    private double[] potrsObsRectWork;

    @Setup(Level.Trial)
    public void setup() {
        rep = buildBenchmarkModel(nobs, kStates);
        missingRep = buildMissingBenchmarkModel(rep);
        init = InitialState.approximateDiffuse();
        fullFilterOptions = FilterOptions.defaults();
        predictedStateOnlyFilterOptions = FilterOptions.builder()
            .retainOnly(FilterOptions.Surface.PREDICTED_STATE)
            .build();
        predictedCovarianceOnlyFilterOptions = FilterOptions.builder()
            .retainOnly(FilterOptions.Surface.PREDICTED_STATE_COVARIANCE)
            .build();
        filteredStateOnlyFilterOptions = FilterOptions.builder()
            .retainOnly(FilterOptions.Surface.FILTERED_STATE)
            .build();
        filteredCovarianceOnlyFilterOptions = FilterOptions.builder()
            .retainOnly(FilterOptions.Surface.FILTERED_STATE_COVARIANCE)
            .build();
        smootherFilterOptions = SmootherOptions.conventional().requiredFilterOptions();
        fullSmootherOptions = SmootherOptions.conventional();
        borrowedFilterWorkspace = KalmanEngine.workspace();
        borrowedPredictedStateOnlyFilterWorkspace = KalmanEngine.workspace();
        borrowedPredictedCovarianceOnlyFilterWorkspace = KalmanEngine.workspace();
        borrowedFilteredStateOnlyFilterWorkspace = KalmanEngine.workspace();
        borrowedFilteredCovarianceOnlyFilterWorkspace = KalmanEngine.workspace();
        borrowedSmootherWorkspace = SmootherEngine.workspace();
        missingBorrowedSmootherWorkspace = SmootherEngine.workspace();

        KalmanEngine.filterComplexBorrowedUnsafe(rep, init, borrowedFilterWorkspace, fullFilterOptions);
        KalmanEngine.filterComplexBorrowedUnsafe(rep, init, borrowedPredictedStateOnlyFilterWorkspace, predictedStateOnlyFilterOptions);
        KalmanEngine.filterComplexBorrowedUnsafe(rep, init, borrowedPredictedCovarianceOnlyFilterWorkspace, predictedCovarianceOnlyFilterOptions);
        KalmanEngine.filterComplexBorrowedUnsafe(rep, init, borrowedFilteredStateOnlyFilterWorkspace, filteredStateOnlyFilterOptions);
        KalmanEngine.filterComplexBorrowedUnsafe(rep, init, borrowedFilteredCovarianceOnlyFilterWorkspace, filteredCovarianceOnlyFilterOptions);
        smootherFilterInput = KalmanEngine.filterComplex(rep, init, smootherFilterOptions);
        SmootherEngine.smoothComplexBorrowedUnsafe(rep, smootherFilterInput, borrowedSmootherWorkspace, fullSmootherOptions);
        missingSmootherFilterInput = KalmanEngine.filterComplex(missingRep, init, smootherFilterOptions);
        SmootherEngine.smoothComplexBorrowedUnsafe(missingRep, missingSmootherFilterInput,
            missingBorrowedSmootherWorkspace, fullSmootherOptions);

        Random rng = new Random(91L + 31L * kStates + nobs);
        gemvMatrix = randomComplexMatrix(kStates, kStates, rng);
        gemvVector = randomComplexVector(kStates, rng);
        gemvOutput = randomComplexVector(kStates, rng);
        gemvConjOutput = randomComplexVector(kStates, rng);

        ke = Math.max(4, Math.min(8, kStates));

        gemmNtConjA = randomComplexMatrix(kStates, kStates, rng);
        gemmNtConjB = randomComplexMatrix(kStates, kStates, rng);
        gemmNtConjOut = new double[2 * kStates * kStates];
        gemmNnStateObsOut = new double[2 * kStates * ke];
        gemmNnSquareSmallKB = randomComplexMatrix(ke, kStates, rng);
        gemmNnSquareSmallKOut = new double[2 * kStates * kStates];
        gemmNnObsSquareA = randomComplexMatrix(ke, kStates, rng);
        gemmNnObsSquareOut = new double[2 * ke * ke];

        gemmCnA = randomComplexMatrix(kStates, ke, rng);
        gemmCnB = randomComplexMatrix(kStates, ke, rng);
        gemmCnOut = new double[2 * ke * ke];
        gemmCnSquareOut = new double[2 * kStates * kStates];
        gemmCnSquareSmallKA = randomComplexMatrix(ke, kStates, rng);
        gemmCnSquareSmallKB = randomComplexMatrix(ke, kStates, rng);
        gemmCnSquareSmallKOut = new double[2 * kStates * kStates];
        gemmCcSquareOut = new double[2 * kStates * kStates];

        potrfStateBase = hermitianPositiveDefinite(kStates, rng);
        potrfStateWork = new double[2 * kStates * kStates];
        potrsStateFactorBase = potrfStateBase.clone();
        ZLAS.zpotrf(BLAS.Uplo.Lower, kStates, potrsStateFactorBase, 0, kStates);
        potrsStateFactorWork = new double[2 * kStates * kStates];
        potrsStateVectorBase = randomComplexVector(kStates, rng);
        potrsStateVectorWork = new double[2 * kStates];
        potrsStateRectBase = randomComplexMatrix(kStates, kStates, rng);
        potrsStateRectWork = new double[2 * kStates * kStates];

        potrfObsBase = hermitianPositiveDefinite(ke, rng);
        potrfObsWork = new double[2 * ke * ke];
        potrsObsVectorBase = randomComplexVector(ke, rng);
        potrsObsVectorWork = new double[2 * ke];
        potrsObsRectBase = randomComplexMatrix(ke, kStates, rng);
        potrsObsRectWork = new double[2 * ke * kStates];
    }

    @Setup(Level.Invocation)
    public void setupInvocation() {
        System.arraycopy(potrfStateBase, 0, potrfStateWork, 0, potrfStateBase.length);
        System.arraycopy(potrsStateFactorBase, 0, potrsStateFactorWork, 0, potrsStateFactorBase.length);
        System.arraycopy(potrsStateVectorBase, 0, potrsStateVectorWork, 0, potrsStateVectorBase.length);
        System.arraycopy(potrsStateRectBase, 0, potrsStateRectWork, 0, potrsStateRectBase.length);

        System.arraycopy(potrfObsBase, 0, potrfObsWork, 0, potrfObsBase.length);
        System.arraycopy(potrsObsVectorBase, 0, potrsObsVectorWork, 0, potrsObsVectorBase.length);
        System.arraycopy(potrsObsRectBase, 0, potrsObsRectWork, 0, potrsObsRectBase.length);
    }

    @Benchmark
    public ZFilterResult filterBorrowedResultFull() {
        return KalmanEngine.filterComplexBorrowedUnsafe(rep, init, borrowedFilterWorkspace, fullFilterOptions);
    }

    @Benchmark
    public ZFilterResult filterBorrowedResultPredictedStateOnly() {
        return KalmanEngine.filterComplexBorrowedUnsafe(rep, init, borrowedPredictedStateOnlyFilterWorkspace,
            predictedStateOnlyFilterOptions);
    }

    @Benchmark
    public ZFilterResult filterBorrowedResultPredictedCovarianceOnly() {
        return KalmanEngine.filterComplexBorrowedUnsafe(rep, init, borrowedPredictedCovarianceOnlyFilterWorkspace,
            predictedCovarianceOnlyFilterOptions);
    }

    @Benchmark
    public ZFilterResult filterBorrowedResultFilteredStateOnly() {
        return KalmanEngine.filterComplexBorrowedUnsafe(rep, init, borrowedFilteredStateOnlyFilterWorkspace,
            filteredStateOnlyFilterOptions);
    }

    @Benchmark
    public ZFilterResult filterBorrowedResultFilteredCovarianceOnly() {
        return KalmanEngine.filterComplexBorrowedUnsafe(rep, init, borrowedFilteredCovarianceOnlyFilterWorkspace,
            filteredCovarianceOnlyFilterOptions);
    }

    @Benchmark
    public ZSmootherResult smoothBorrowedResultFull() {
        return SmootherEngine.smoothComplexBorrowedUnsafe(rep, smootherFilterInput, borrowedSmootherWorkspace, fullSmootherOptions);
    }

    @Benchmark
    public ZSmootherResult smoothBorrowedResultFullMissing() {
        return SmootherEngine.smoothComplexBorrowedUnsafe(missingRep, missingSmootherFilterInput,
            missingBorrowedSmootherWorkspace, fullSmootherOptions);
    }

    @Benchmark
    public double[] hotspotGemvNoTrans() {
        ZLAS.zcopy(kStates, gemvVector, 0, 1, gemvOutput, 0, 1);
        ZLAS.zgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0, 0.0,
            gemvMatrix, 0, kStates,
            gemvVector, 0, 1,
            1.0, 0.0, gemvOutput, 0, 1);
        return gemvOutput;
    }

    @Benchmark
    public double[] hotspotGemvConj() {
        ZLAS.zcopy(kStates, gemvVector, 0, 1, gemvConjOutput, 0, 1);
        ZLAS.zgemv(BLAS.Trans.Conj, kStates, kStates, 1.0, 0.0,
            gemvMatrix, 0, kStates,
            gemvVector, 0, 1,
            1.0, 0.0, gemvConjOutput, 0, 1);
        return gemvConjOutput;
    }

    @Benchmark
    public double[] hotspotZgemmNoTransConj() {
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kStates, 1.0, 0.0,
            gemmNtConjA, 0, kStates,
            gemmNtConjB, 0, kStates,
            0.0, 0.0, gemmNtConjOut, 0, kStates);
        return gemmNtConjOut;
    }

    @Benchmark
    public double[] hotspotZgemmNoTransNoTransStateObsRect() {
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, kStates, 1.0, 0.0,
            gemmNtConjA, 0, kStates,
            gemmCnA, 0, ke,
            0.0, 0.0, gemmNnStateObsOut, 0, ke);
        return gemmNnStateObsOut;
    }

    @Benchmark
    public double[] hotspotZgemmNoTransNoTransSquareSmallK() {
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, ke, 1.0, 0.0,
            gemmCnA, 0, ke,
            gemmNnSquareSmallKB, 0, kStates,
            0.0, 0.0, gemmNnSquareSmallKOut, 0, kStates);
        return gemmNnSquareSmallKOut;
    }

    @Benchmark
    public double[] hotspotZgemmNoTransNoTransObsSquare() {
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, kStates, 1.0, 0.0,
            gemmNnObsSquareA, 0, kStates,
            gemmCnA, 0, ke,
            0.0, 0.0, gemmNnObsSquareOut, 0, ke);
        return gemmNnObsSquareOut;
    }

    @Benchmark
    public double[] hotspotZgemmConjNoTransSquare() {
        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, kStates, 1.0, 0.0,
            gemmNtConjA, 0, kStates,
            gemmNtConjB, 0, kStates,
            0.0, 0.0, gemmCnSquareOut, 0, kStates);
        return gemmCnSquareOut;
    }

    @Benchmark
    public double[] hotspotZgemmConjNoTransRect() {
        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, ke, ke, kStates, 1.0, 0.0,
            gemmCnA, 0, ke,
            gemmCnB, 0, ke,
            0.0, 0.0, gemmCnOut, 0, ke);
        return gemmCnOut;
    }

    @Benchmark
    public double[] hotspotZgemmConjNoTransSquareSmallK() {
        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, ke, 1.0, 0.0,
            gemmCnSquareSmallKA, 0, kStates,
            gemmCnSquareSmallKB, 0, kStates,
            0.0, 0.0, gemmCnSquareSmallKOut, 0, kStates);
        return gemmCnSquareSmallKOut;
    }

    @Benchmark
    public double[] hotspotZgemmConjConjSquare() {
        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.Conj, kStates, kStates, kStates, 1.0, 0.0,
            gemmNtConjA, 0, kStates,
            gemmNtConjB, 0, kStates,
            0.0, 0.0, gemmCcSquareOut, 0, kStates);
        return gemmCcSquareOut;
    }

    @Benchmark
    public double[] hotspotZpotrfStateKalman() {
        ZLAS.zpotrf(BLAS.Uplo.Lower, kStates, potrfStateWork, 0, kStates);
        return potrfStateWork;
    }

    @Benchmark
    public double[] hotspotZpotrsStateVectorKalman() {
        ZLAS.zpotrs(BLAS.Uplo.Lower, kStates, 1,
            potrsStateFactorWork, 0, kStates,
            potrsStateVectorWork, 0, 1);
        return potrsStateVectorWork;
    }

    @Benchmark
    public double[] hotspotZpotrsStateRectKalman() {
        ZLAS.zpotrs(BLAS.Uplo.Lower, kStates, kStates,
            potrsStateFactorWork, 0, kStates,
            potrsStateRectWork, 0, kStates);
        return potrsStateRectWork;
    }

    @Benchmark
    public double[] hotspotZpotrsStateKalman() {
        ZLAS.zpotrs(BLAS.Uplo.Lower, kStates, 1,
            potrsStateFactorWork, 0, kStates,
            potrsStateVectorWork, 0, 1);
        ZLAS.zpotrs(BLAS.Uplo.Lower, kStates, kStates,
            potrsStateFactorWork, 0, kStates,
            potrsStateRectWork, 0, kStates);
        return potrsStateRectWork;
    }

    @Benchmark
    public double[] hotspotZpotrfPotrsStateKalman() {
        ZLAS.zpotrf(BLAS.Uplo.Lower, kStates, potrfStateWork, 0, kStates);
        ZLAS.zpotrs(BLAS.Uplo.Lower, kStates, 1,
            potrfStateWork, 0, kStates,
            potrsStateVectorWork, 0, 1);
        ZLAS.zpotrs(BLAS.Uplo.Lower, kStates, kStates,
            potrfStateWork, 0, kStates,
            potrsStateRectWork, 0, kStates);
        return potrsStateRectWork;
    }

    @Benchmark
    public double[] hotspotZpotrfPotrsObsKalman() {
        ZLAS.zpotrf(BLAS.Uplo.Lower, ke, potrfObsWork, 0, ke);
        ZLAS.zpotrs(BLAS.Uplo.Lower, ke, 1,
            potrfObsWork, 0, ke,
            potrsObsVectorWork, 0, 1);
        ZLAS.zpotrs(BLAS.Uplo.Lower, ke, kStates,
            potrfObsWork, 0, ke,
            potrsObsRectWork, 0, kStates);
        return potrsObsRectWork;
    }

    public static void main(String[] args) throws RunnerException {
        Options options = new OptionsBuilder()
            .include(ZKalmanMemoryBenchmark.class.getSimpleName())
            .build();
        new Runner(options).run();
    }

    private static ModelFixture buildBenchmarkModel(int nobs, int kStates) {
        int kEndog = Math.max(4, Math.min(8, kStates));
        int kPosdef = kStates;
        ModelFixture rep = ModelFixture.complex(kEndog, kStates, kPosdef, nobs);
        Random rng = new Random(17L + 13L * kStates + nobs);

        double[] design = randomComplexMatrix(kEndog, kStates, rng);
        double[] obsCov = hermitianDiagonal(kEndog, 0.2, 0.05);
        double[] transition = stableTransition(kStates, rng);
        double[] selection = identityRectangular(kStates, kPosdef);
        double[] stateCov = hermitianDiagonal(kPosdef, 0.4, 0.08);
        double[] obsIntercept = new double[2 * kEndog];
        double[] stateIntercept = new double[2 * kStates];

        double[] state = randomComplexVector(kStates, rng);
        double[] innovation = new double[2 * kStates];
        double[] observation = new double[2 * kEndog];

        for (int t = 0; t < nobs; t++) {
            fillGaussianComplex(innovation, rng, 0.1);
            fillGaussianComplex(observation, rng, 0.05);
            propagateState(transition, kStates, state, innovation);
            projectObservation(design, kEndog, kStates, state, observation);

            rep.setDesign(design, t);
            rep.setObsIntercept(obsIntercept, t);
            rep.setObsCov(obsCov, t);
            rep.setTransition(transition, t);
            rep.setStateIntercept(stateIntercept, t);
            rep.setSelection(selection, t);
            rep.setStateCov(stateCov, t);
            rep.setEndog(observation, t);
        }
        return rep;
    }

    private static ModelFixture buildMissingBenchmarkModel(ModelFixture dense) {
        ModelFixture rep = ModelFixture.copyOf(dense);
        boolean[] mask = new boolean[rep.kEndog];
        int mid = rep.kEndog / 2;

        for (int t = 0; t < rep.nobs; t++) {
            Arrays.fill(mask, false);

            if ((t & 3) == 1) {
                mask[0] = true;
            }
            if (t % 5 == 2) {
                mask[rep.kEndog - 1] = true;
            }
            if (rep.kEndog > 2 && t % 7 == 3) {
                mask[1] = true;
            }
            if (rep.kEndog > 3 && t % 11 == 4) {
                mask[mid] = true;
            }

            int missingCount = 0;
            for (boolean value : mask) {
                if (value) {
                    missingCount++;
                }
            }
            if (missingCount == rep.kEndog) {
                mask[rep.kEndog - 1] = false;
            }
            rep.setMissing(mask, t);
        }
        return rep;
    }

    private static void propagateState(double[] transition, int kStates, double[] state, double[] innovation) {
        double[] next = new double[2 * kStates];
        ZLAS.zgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0, 0.0,
            transition, 0, kStates,
            state, 0, 1,
            0.0, 0.0, next, 0, 1);
        ZLAS.zaxpy(kStates, 1.0, 0.0, innovation, 0, 1, next, 0, 1);
        ZLAS.zcopy(kStates, next, 0, 1, state, 0, 1);
    }

    private static void projectObservation(double[] design, int kEndog, int kStates, double[] state, double[] observation) {
        ZLAS.zgemv(BLAS.Trans.NoTrans, kEndog, kStates, 1.0, 0.0,
            design, 0, kStates,
            state, 0, 1,
            0.0, 0.0, observation, 0, 1);
    }

    private static double[] stableTransition(int kStates, Random rng) {
        double[] transition = new double[2 * kStates * kStates];
        for (int row = 0; row < kStates; row++) {
            for (int col = 0; col < kStates; col++) {
                int index = (row * kStates + col) * 2;
                double scale = row == col ? 0.65 : 0.015;
                transition[index] = (rng.nextDouble() * 2.0 - 1.0) * scale;
                transition[index + 1] = (rng.nextDouble() * 2.0 - 1.0) * scale;
            }
            int diag = (row * kStates + row) * 2;
            transition[diag] += 0.35;
        }
        return transition;
    }

    private static double[] hermitianDiagonal(int n, double base, double slope) {
        double[] matrix = new double[2 * n * n];
        for (int i = 0; i < n; i++) {
            matrix[(i * n + i) * 2] = base + slope * i;
        }
        return matrix;
    }

    private static double[] hermitianPositiveDefinite(int n, Random rng) {
        double[] factor = randomComplexMatrix(n, n, rng);
        double[] matrix = new double[2 * n * n];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, n, n, n, 1.0, 0.0,
            factor, 0, n,
            factor, 0, n,
            0.0, 0.0, matrix, 0, n);
        for (int i = 0; i < n; i++) {
            matrix[(i * n + i) * 2] += n;
        }
        return matrix;
    }

    private static double[] identityRectangular(int rows, int cols) {
        double[] matrix = new double[2 * rows * cols];
        int diagonal = Math.min(rows, cols);
        for (int i = 0; i < diagonal; i++) {
            matrix[(i * cols + i) * 2] = 1.0;
        }
        return matrix;
    }

    private static double[] randomComplexMatrix(int rows, int cols, Random rng) {
        double[] matrix = new double[2 * rows * cols];
        for (int i = 0; i < rows * cols; i++) {
            matrix[i * 2] = (rng.nextDouble() * 2.0 - 1.0) * 0.2;
            matrix[i * 2 + 1] = (rng.nextDouble() * 2.0 - 1.0) * 0.2;
        }
        return matrix;
    }

    private static double[] randomComplexVector(int n, Random rng) {
        double[] vector = new double[2 * n];
        fillGaussianComplex(vector, rng, 0.2);
        return vector;
    }

    private static void fillGaussianComplex(double[] vector, Random rng, double scale) {
        for (int i = 0; i < vector.length; i += 2) {
            vector[i] = rng.nextGaussian() * scale;
            vector[i + 1] = rng.nextGaussian() * scale;
        }
    }
}