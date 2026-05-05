package com.curioloop.yum4j.kalman.mle;

import com.curioloop.yum4j.kalman.filter.FilterResult;
import com.curioloop.yum4j.kalman.filter.FilterSpec;
import com.curioloop.yum4j.kalman.filter.KalmanFilter;
import com.curioloop.yum4j.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.kalman.filter.ZKalmanFilter;
import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.model.StateSpaceModel;
import com.curioloop.yum4j.kalman.smooth.KalmanSmoother;
import com.curioloop.yum4j.kalman.smooth.SmootherResult;
import com.curioloop.yum4j.kalman.smooth.SmootherSpec;
import com.curioloop.yum4j.kalman.smooth.ZKalmanSmoother;
import com.curioloop.yum4j.kalman.smooth.ZSmootherResult;
import com.curioloop.yum4j.linalg.mat.LU;
import com.curioloop.yum4j.optim.Minimizer;
import com.curioloop.yum4j.optim.Optimization;
import com.curioloop.yum4j.optim.Univariate;
import com.curioloop.yum4j.optim.lbfgsb.LBFGSBProblem;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MLEModelTest {

    private static final double TOL = 1e-8;
    private static final double[] Y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1};
    private static final FilterSpec LIKELIHOOD_SPEC = FilterSpec.full().without(
        FilterSpec.Storage.FORECAST,
        FilterSpec.Storage.PREDICTED_STATE,
        FilterSpec.Storage.FILTERED_STATE,
        FilterSpec.Storage.KALMAN_GAIN);

    @Test
    void testRealAr1LogLikelihoodMatchesDirectFilter() {
        RealAr1Model model = new RealAr1Model(Y);
        double[] params = {0.6, 0.5};

        StateSpaceModel direct = buildRealAr1StateSpace(params[0], params[1], Y);
        FilterResult directFilter = KalmanFilter.filter(direct, StateInitialization.stationary(direct), null, LIKELIHOOD_SPEC);

        assertEquals(directFilter.logLikelihood(), model.logLikelihood(params), TOL);
        assertArrayEquals(directFilter.logLikelihoodObs, model.logLikelihoodObs(params), TOL);
    }

    @Test
    void testRealAr1PredictAndOwnership() {
        RealAr1Model model = new RealAr1Model(Y);
        double[] firstParams = {0.55, 0.4};
        double[] secondParams = {0.2, 0.9};

        FilterResult prediction = model.predict(firstParams);
        assertEquals(0, prediction.filteredState.length);
        assertEquals(0, prediction.kalmanGain.length);
        assertEquals(0, prediction.logLikelihoodObs.length);
        assertTrue(prediction.forecast.length > 0);
        assertTrue(prediction.predictedState.length > 0);

        FilterResult first = model.filter(firstParams, FilterSpec.defaults());
        double firstForecast0 = first.forecast(0, 0);
        model.filter(secondParams, FilterSpec.defaults());
        assertEquals(firstForecast0, first.forecast(0, 0), TOL);
    }

    @Test
    void testRealAr1SmoothAndObjective() {
        RealAr1Model model = new RealAr1Model(Y);
        double[] params = {0.5, 0.75};

        StateSpaceModel direct = buildRealAr1StateSpace(params[0], params[1], Y);
        FilterResult directFilter = KalmanFilter.filter(direct, StateInitialization.stationary(direct), null,
            SmootherSpec.conventional().requiredFilterSpec());
        SmootherResult directSmooth = KalmanSmoother.smooth(direct, directFilter, null, SmootherSpec.conventional().stateOnly());
        SmootherResult actualSmooth = model.smooth(params, SmootherSpec.conventional().stateOnly());

        assertArrayEquals(directSmooth.smoothedState, actualSmooth.smoothedState, TOL);

        double[] score = model.score(params);
        for (double value : score) {
            assertTrue(Double.isFinite(value));
        }

        Univariate objective = model.objective();
        double[] gradient = new double[params.length];
        double objectiveValue = objective.evaluate(Arrays.copyOf(params, params.length), params.length, gradient);
        assertEquals(-model.logLikelihood(params), objectiveValue, TOL);
        for (int i = 0; i < gradient.length; i++) {
            assertEquals(-score[i], gradient[i], 1e-6);
        }

        double[] unconstrained = model.untransformParams(params);
        double transformedObjectiveValue = model.transformedObjective()
            .evaluate(Arrays.copyOf(unconstrained, unconstrained.length), unconstrained.length, null);
        assertEquals(objectiveValue, transformedObjectiveValue, 1e-6);
    }

    @Test
    void testOptimizeAcceptsConfiguredMinimizerInstance() {
        RealAr1Model model = new RealAr1Model(Y);
        LBFGSBProblem optimizer = Minimizer.lbfgsb()
            .maxIterations(3)
            .maxEvaluations(20)
            .gradientTolerance(1e-3);

        Optimization result = model.optimize(optimizer);

        assertTrue(result.evaluations() > 0);
        assertEquals(3, optimizer.maxIterations());
        assertEquals(20, optimizer.maxEvaluations());
        assertEquals(model.paramCount(), optimizer.dimension());
        assertArrayEquals(model.untransformParams(model.startParams()), optimizer.initialPoint(), 0.0);
    }

    @Test
    void testScoreObsAndOpgInformationMatrixMatchScoreAggregationAndBurnSemantics() {
        BurnedRealAr1Model model = new BurnedRealAr1Model(Y, 2);
        double[] params = {0.5, 0.75};

        double[] scoreObs = model.scoreObs(params);
        double[] score = model.score(params);
        double[] opg = model.opgInformationMatrix(params);

        assertEquals(Y.length * params.length, scoreObs.length);
        assertArrayEquals(new double[]{0.0, 0.0, 0.0, 0.0}, Arrays.copyOf(scoreObs, 4), TOL);

        double[] aggregated = new double[params.length];
        for (int row = 0; row < Y.length; row++) {
            for (int col = 0; col < params.length; col++) {
                aggregated[col] += scoreObs[row * params.length + col];
            }
        }
        assertArrayEquals(score, aggregated, 1e-6);

        double[] expectedOpg = new double[params.length * params.length];
        for (int row = 0; row < Y.length; row++) {
            int offset = row * params.length;
            for (int i = 0; i < params.length; i++) {
                double left = scoreObs[offset + i];
                for (int j = 0; j < params.length; j++) {
                    expectedOpg[i * params.length + j] += left * scoreObs[offset + j];
                }
            }
        }
        assertArrayEquals(expectedOpg, opg, 1e-10);
    }

    @Test
    void testCovarianceHelpersMatchManualInformationAlgebra() {
        RealAr1Model model = new RealAr1Model(Y);
        double[] params = {0.5, 0.75};

        double[] observedInformation = model.observedInformationMatrix(params);
        double[] opgInformation = model.opgInformationMatrix(params);

        double[] expectedOim = invertInformation(symmetrize(observedInformation, params.length), params.length);
        double[] expectedOpg = invertInformation(opgInformation, params.length);
        double[] expectedRobust = symmetrize(
            multiply(multiply(expectedOim, opgInformation, params.length), expectedOim, params.length),
            params.length);

        double[] actualOim = model.covarianceFromObservedInformation(params);
        double[] actualOpg = model.covarianceFromOpg(params);
        double[] actualRobust = model.robustCovarianceFromObservedInformation(params);

        assertArrayEquals(expectedOim, actualOim, 1e-10);
        assertArrayEquals(expectedOpg, actualOpg, 1e-10);
        assertArrayEquals(expectedRobust, actualRobust, 1e-10);
        assertArrayEquals(diagonalStdErr(actualOim), model.standardErrors(actualOim), 1e-12);
    }

    @Test
    void testObservedInformationInversionPolicyHookCanUsePseudoInverse() {
        SingularObservedInformationModel model = new SingularObservedInformationModel();
        double[] params = {1.0, 1.0};
        double[] observedInformation = {
            1.0, 1.0,
            1.0, 1.0
        };
        double[] opgInformation = {
            2.0, 0.0,
            0.0, 1.0
        };
        double[] expectedOim = {
            0.25, 0.25,
            0.25, 0.25
        };
        double[] expectedRobust = symmetrize(
            multiply(multiply(expectedOim, opgInformation, params.length), expectedOim, params.length),
            params.length);

        assertArrayEquals(observedInformation, model.observedInformationMatrix(params), 0.0);
        assertArrayEquals(expectedOim, model.covarianceFromObservedInformation(params), 1e-12);
        assertArrayEquals(expectedRobust, model.robustCovarianceFromObservedInformation(params), 1e-12);
    }

    @Test
    void testComplexAr1LogLikelihoodAndSmootherMatchDirectPath() {
        ComplexAr1Model model = new ComplexAr1Model(Y);
        double[] params = {0.4, 0.2, 0.5};

        StateSpaceModel direct = buildComplexAr1StateSpace(params[0], params[1], params[2], Y);
        ZFilterResult directFilter = ZKalmanFilter.filter(direct, StateInitialization.stationary(direct), null, LIKELIHOOD_SPEC);
        ZSmootherResult directSmooth = ZKalmanSmoother.smooth(
            direct,
            ZKalmanFilter.filter(direct, StateInitialization.stationary(direct), null,
                SmootherSpec.conventional().requiredFilterSpec()),
            null,
            SmootherSpec.conventional().stateOnly());

        assertEquals(directFilter.logLikelihood(), model.logLikelihood(params), TOL);
        assertArrayEquals(directFilter.logLikelihoodObs, model.logLikelihoodObs(params), TOL);

        ZSmootherResult actualSmooth = model.smooth(params, SmootherSpec.conventional().stateOnly());
        assertArrayEquals(directSmooth.smoothedState, actualSmooth.smoothedState, TOL);

        ZFilterResult first = model.filter(params, FilterSpec.defaults());
        double firstLikelihood0 = first.logLikelihoodObs[0];
        model.filter(new double[]{0.1, 0.3, 1.1}, FilterSpec.defaults());
        assertEquals(firstLikelihood0, first.logLikelihoodObs[0], TOL);
    }

    private static StateSpaceModel buildRealAr1StateSpace(double phi, double obsVariance, double[] y) {
        return StateSpaceModel.builder(1, 1, 1, y.length)
            .design(new double[]{1.0}, false)
            .obsIntercept(new double[]{0.0}, false)
            .obsCov(new double[]{obsVariance}, false)
            .transition(new double[]{phi}, false)
            .stateIntercept(new double[]{0.0}, false)
            .selection(new double[]{1.0}, false)
            .stateCovariance(new double[]{1.0}, false)
            .endog(Arrays.copyOf(y, y.length))
            .allObserved()
            .build();
    }

    private static StateSpaceModel buildComplexAr1StateSpace(double phiRe,
                                                             double phiIm,
                                                             double obsVariance,
                                                             double[] y) {
        return StateSpaceModel.complexBuilder(1, 1, 1, y.length)
            .design(complexScalar(1.0, 0.0), false)
            .obsIntercept(complexScalar(0.0, 0.0), false)
            .obsCov(complexScalar(obsVariance, 0.0), false)
            .transition(complexScalar(phiRe, phiIm), false)
            .stateIntercept(complexScalar(0.0, 0.0), false)
            .selection(complexScalar(1.0, 0.0), false)
            .stateCovariance(complexScalar(1.0, 0.0), false)
            .endog(embedComplexSeries(y))
            .allObserved()
            .build();
    }

    private static double[] complexScalar(double re, double im) {
        return new double[]{re, im};
    }

    private static double[] embedComplexSeries(double[] y) {
        double[] series = new double[y.length * 2];
        for (int i = 0; i < y.length; i++) {
            series[i * 2] = y[i];
            series[i * 2 + 1] = 0.0;
        }
        return series;
    }

    private static double[] invertInformation(double[] information, int dimension) {
        double[] inverse = LU.decompose(Arrays.copyOf(information, information.length), dimension)
            .inverse(new double[information.length]);
        return symmetrize(inverse, dimension);
    }

    private static double[] multiply(double[] left, double[] right, int dimension) {
        double[] product = new double[dimension * dimension];
        for (int row = 0; row < dimension; row++) {
            for (int col = 0; col < dimension; col++) {
                double sum = 0.0;
                for (int inner = 0; inner < dimension; inner++) {
                    sum += left[row * dimension + inner] * right[inner * dimension + col];
                }
                product[row * dimension + col] = sum;
            }
        }
        return product;
    }

    private static double[] symmetrize(double[] matrix, int dimension) {
        double[] symmetric = Arrays.copyOf(matrix, matrix.length);
        for (int row = 0; row < dimension; row++) {
            for (int col = row + 1; col < dimension; col++) {
                double value = 0.5 * (symmetric[row * dimension + col] + symmetric[col * dimension + row]);
                symmetric[row * dimension + col] = value;
                symmetric[col * dimension + row] = value;
            }
        }
        return symmetric;
    }

    private static double[] diagonalStdErr(double[] covariance) {
        int dimension = (int) Math.round(Math.sqrt(covariance.length));
        double[] stderr = new double[dimension];
        for (int index = 0; index < dimension; index++) {
            stderr[index] = Math.sqrt(covariance[index * dimension + index]);
        }
        return stderr;
    }

    private static final class RealAr1Model extends RealMLE {

        private final double[] y;

        private RealAr1Model(double[] y) {
            super(buildRealAr1StateSpace(0.0, 1.0, y), new double[]{0.4, 0.5}, "phi", "obsVariance");
            this.y = Arrays.copyOf(y, y.length);
        }

        @Override
        public double[] transformParams(double[] unconstrainedParams) {
            double[] params = super.transformParams(unconstrainedParams);
            params[0] = Math.tanh(params[0]);
            params[1] = Math.exp(params[1]);
            return params;
        }

        @Override
        public double[] untransformParams(double[] params) {
            double[] unconstrained = super.untransformParams(params);
            unconstrained[0] = 0.5 * Math.log((1.0 + params[0]) / (1.0 - params[0]));
            unconstrained[1] = Math.log(params[1]);
            return unconstrained;
        }

        @Override
        protected void updateModel(double[] params, StateSpaceModel model) {
            model.transition[0] = params[0];
            model.obsCov[0] = params[1];
            System.arraycopy(y, 0, model.endog, 0, y.length);
        }

        @Override
        protected StateInitialization initialization(double[] params, StateSpaceModel model) {
            return StateInitialization.stationary(model);
        }
    }

    private static final class BurnedRealAr1Model extends RealMLE {

        private final double[] y;
        private final int burn;

        private BurnedRealAr1Model(double[] y, int burn) {
            super(buildRealAr1StateSpace(0.0, 1.0, y), new double[]{0.4, 0.5}, "phi", "obsVariance");
            this.y = Arrays.copyOf(y, y.length);
            this.burn = burn;
        }

        @Override
        public double[] transformParams(double[] unconstrainedParams) {
            double[] params = super.transformParams(unconstrainedParams);
            params[0] = Math.tanh(params[0]);
            params[1] = Math.exp(params[1]);
            return params;
        }

        @Override
        public double[] untransformParams(double[] params) {
            double[] unconstrained = super.untransformParams(params);
            unconstrained[0] = 0.5 * Math.log((1.0 + params[0]) / (1.0 - params[0]));
            unconstrained[1] = Math.log(params[1]);
            return unconstrained;
        }

        @Override
        protected void updateModel(double[] params, StateSpaceModel model) {
            model.transition[0] = params[0];
            model.obsCov[0] = params[1];
            System.arraycopy(y, 0, model.endog, 0, y.length);
        }

        @Override
        protected StateInitialization initialization(double[] params, StateSpaceModel model) {
            return StateInitialization.stationary(model);
        }

        @Override
        protected int likelihoodBurn(double[] params, StateSpaceModel model) {
            return burn;
        }
    }

    private static final class ComplexAr1Model extends ComplexMLE {

        private final double[] endog;

        private ComplexAr1Model(double[] y) {
            super(buildComplexAr1StateSpace(0.0, 0.0, 1.0, y),
                new double[]{0.3, 0.1, 0.5},
                "phiRe",
                "phiIm",
                "obsVariance");
            this.endog = embedComplexSeries(y);
        }

        @Override
        protected void updateModel(double[] params, StateSpaceModel model) {
            model.transition[0] = params[0];
            model.transition[1] = params[1];
            model.obsCov[0] = params[2];
            model.obsCov[1] = 0.0;
            System.arraycopy(endog, 0, model.endog, 0, endog.length);
        }

        @Override
        protected StateInitialization initialization(double[] params, StateSpaceModel model) {
            return StateInitialization.stationary(model);
        }
    }

    private static final class SingularObservedInformationModel extends RealMLE {

        private SingularObservedInformationModel() {
            super(buildRealAr1StateSpace(0.0, 1.0, new double[]{0.0}), new double[]{1.0, 1.0}, "alpha", "beta");
        }

        @Override
        protected void updateModel(double[] params, StateSpaceModel model) {
        }

        @Override
        protected StateInitialization initialization(double[] params, StateSpaceModel model) {
            return StateInitialization.stationary(model);
        }

        @Override
        protected double[] observedInformationMatrixSurface(double[] point) {
            return new double[]{
                1.0, 1.0,
                1.0, 1.0
            };
        }

        @Override
        protected double[] opgInformationMatrixSurface(double[] point) {
            return new double[]{
                2.0, 0.0,
                0.0, 1.0
            };
        }

        @Override
        protected double[] invertObservedInformationSurface(double[] information, int dimension) {
            return pseudoInverseInformationMatrix(information, dimension);
        }
    }
}