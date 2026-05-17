package com.curioloop.yum4j.tsa.sarimax;

import com.curioloop.yum4j.ssm.kalman.mle.MLEResults;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.optim.Minimizer;
import com.curioloop.yum4j.tsa.prediction.PredictionInformationSet;
import com.curioloop.yum4j.tsa.statespace.SimulationAnchor;
import org.junit.jupiter.api.Test;

import java.lang.reflect.RecordComponent;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SARIMAXStatsmodelsSurfaceAuditTest {

    @Test
    void testCurrentOptionSurfaceMatchesSupportedOverlap() {
        Set<String> optionNames = Arrays.stream(SARIMAXOption.values())
            .map(Enum::name)
            .collect(Collectors.toSet());

        assertEquals(Set.of(
            "ENFORCE_STATIONARITY",
            "ENFORCE_INVERTIBILITY",
            "USE_EXACT_DIFFUSE"), optionNames);
    }

    @Test
    void testCurrentResultsSurfaceExposesOverlappingMetrics() throws NoSuchMethodException {
        assertNotNull(SARIMAXResults.class.getMethod("parameterNames"));
        assertNotNull(SARIMAXResults.class.getMethod("covType"));
        assertNotNull(SARIMAXResults.class.getMethod("logLikelihood"));
        assertNotNull(SARIMAXResults.class.getMethod("logLikelihoodObs"));
        assertNotNull(SARIMAXResults.class.getMethod("logLikelihoodBurn"));
        assertNotNull(SARIMAXResults.class.getMethod("scale"));
        assertNotNull(SARIMAXResults.class.getMethod("nobsEffective"));
        assertNotNull(SARIMAXResults.class.getMethod("kDiffuseStates"));
        assertNotNull(SARIMAXResults.class.getMethod("dfModel"));
        assertNotNull(SARIMAXResults.class.getMethod("dfResid"));
        assertNotNull(SARIMAXResults.class.getMethod("aic"));
        assertNotNull(SARIMAXResults.class.getMethod("aicc"));
        assertNotNull(SARIMAXResults.class.getMethod("bic"));
        assertNotNull(SARIMAXResults.class.getMethod("hqic"));
        assertNotNull(SARIMAXResults.class.getMethod("covParams"));
        assertNotNull(SARIMAXResults.class.getMethod("bseDefault"));
        assertNotNull(SARIMAXResults.class.getMethod("bse"));
        assertNotNull(SARIMAXResults.class.getMethod("bseApprox"));
        assertNotNull(SARIMAXResults.class.getMethod("bseOim"));
        assertNotNull(SARIMAXResults.class.getMethod("bseOpg"));
        assertNotNull(SARIMAXResults.class.getMethod("covParamsDefault"));
        assertNotNull(SARIMAXResults.class.getMethod("covParamsApprox"));
        assertNotNull(SARIMAXResults.class.getMethod("covParamsOim"));
        assertNotNull(SARIMAXResults.class.getMethod("covParamsOpg"));
        assertNotNull(SARIMAXResults.class.getMethod("bseRobust"));
        assertNotNull(SARIMAXResults.class.getMethod("covParamsRobust"));
        assertNotNull(SARIMAXResults.class.getMethod("bseRobustOim"));
        assertNotNull(SARIMAXResults.class.getMethod("covParamsRobustOim"));
        assertNotNull(SARIMAXResults.class.getMethod("bseRobustApprox"));
        assertNotNull(SARIMAXResults.class.getMethod("covParamsRobustApprox"));
        assertNotNull(SARIMAXResults.class.getMethod("tvalues"));
        assertNotNull(SARIMAXResults.class.getMethod("zvalues"));
        assertNotNull(SARIMAXResults.class.getMethod("pvalues"));
        assertNotNull(SARIMAXResults.class.getMethod("confInt"));
        assertNotNull(SARIMAXResults.class.getMethod("confInt", double.class));
        assertNotNull(SARIMAXResults.class.getMethod("fittedValues"));
        assertNotNull(SARIMAXResults.class.getMethod("residuals"));
        assertNotNull(SARIMAXResults.class.getMethod("standardizedForecastError"));
        assertNotNull(SARIMAXResults.class.getMethod("smooth"));
        assertNotNull(SARIMAXResults.class.getMethod("predict"));
        assertNotNull(SARIMAXResults.class.getMethod("predict", int.class, int.class, int.class, double[][].class));
        assertNotNull(SARIMAXResults.class.getMethod("predict", int.class, int.class, PredictionInformationSet.class, boolean.class));
        assertNotNull(SARIMAXResults.class.getMethod("forecast", int.class));
        assertNotNull(SARIMAXResults.class.getMethod("forecast", int.class, double[][].class));
        assertNotNull(SARIMAXResults.class.getMethod("forecast", int.class, double[][].class, boolean.class));
        assertNotNull(SARIMAXResults.class.getMethod("append", double[].class));
        assertNotNull(SARIMAXResults.class.getMethod("append", double[].class, double[][].class));
        assertNotNull(SARIMAXResults.class.getMethod("appendRefit", double[].class));
        assertNotNull(SARIMAXResults.class.getMethod("appendRefit", double[].class, double[][].class));
        assertNotNull(SARIMAXResults.class.getMethod("appendRefit", double[].class, double[][].class, SARIMAXFitOptions.class));
        assertNotNull(SARIMAXResults.class.getMethod("extend", double[].class));
        assertNotNull(SARIMAXResults.class.getMethod("extend", double[].class, double[][].class));
        assertNotNull(SARIMAXResults.class.getMethod("extendRefit", double[].class));
        assertNotNull(SARIMAXResults.class.getMethod("extendRefit", double[].class, double[][].class));
        assertNotNull(SARIMAXResults.class.getMethod("extendRefit", double[].class, double[][].class, SARIMAXFitOptions.class));
        assertNotNull(SARIMAXResults.class.getMethod("apply", double[].class));
        assertNotNull(SARIMAXResults.class.getMethod("apply", double[].class, double[][].class));
        assertNotNull(SARIMAXResults.class.getMethod("applyRefit", double[].class));
        assertNotNull(SARIMAXResults.class.getMethod("applyRefit", double[].class, double[][].class));
        assertNotNull(SARIMAXResults.class.getMethod("applyRefit", double[].class, double[][].class, SARIMAXFitOptions.class));
        assertNotNull(SARIMAXResults.class.getMethod("simulate", int.class));
        assertNotNull(SARIMAXResults.class.getMethod("simulate", int.class, Random.class));
        assertNotNull(SARIMAXResults.class.getMethod("simulate", int.class, double[][].class, Random.class,
            SimulationSmootherOptions.class));
        assertNotNull(SARIMAXResults.class.getMethod("simulate", int.class, SimulationAnchor.class, Random.class,
            SimulationSmootherOptions.class));
        assertNotNull(SARIMAXResults.class.getMethod("simulate", int.class, int.class, double[][].class, Random.class,
            SimulationSmootherOptions.class));
        assertNotNull(SARIMAXResults.class.getMethod("simulateRepetitions", int.class, int.class));
        assertNotNull(SARIMAXResults.class.getMethod("simulateRepetitions", int.class, int.class, SimulationAnchor.class,
            Random.class, SimulationSmootherOptions.class));
        assertNotNull(SARIMAXResults.class.getMethod("simulate", int.class, double[].class, double[].class, double[].class));
        assertNotNull(SARIMAXResults.class.getMethod("impulseResponses", int.class));
        assertNotNull(SARIMAXResults.class.getMethod("impulseResponses", int.class, int.class));
        assertNotNull(SARIMAXResults.class.getMethod("impulseResponses", int.class, int.class, boolean.class, boolean.class));
        assertNotNull(SARIMAXResults.class.getMethod("impulseResponses", int.class, int.class, boolean.class, boolean.class, int.class));
        assertNotNull(SARIMAXResults.class.getMethod("impulseResponses", int.class, double[].class, boolean.class, boolean.class));
        assertNotNull(SARIMAXResults.class.getMethod("impulseResponses", int.class, double[].class, boolean.class, boolean.class, int.class));
        assertNotNull(SARIMAXResults.class.getMethod("impulseResponseRepetitions", int.class, int.class, int.class, boolean.class, boolean.class));
        assertNotNull(SARIMAXResults.class.getMethod("testNormality"));
        assertNotNull(SARIMAXResults.class.getMethod("testHeteroskedasticity"));
        assertNotNull(SARIMAXResults.class.getMethod("testSerialCorrelation", int[].class));
        assertNotNull(SARIMAXResults.class.getMethod("smoothedStateWeights"));
        assertNotNull(SARIMAXResults.class.getMethod("smoothedStateWeights", SmootherOptions.class));
        assertNotNull(SARIMAXResults.class.getMethod("smoothedDecomposition"));
        assertNotNull(SARIMAXResults.class.getMethod("smoothedDecomposition", SmootherOptions.class));
        assertNotNull(SARIMAXResults.class.getMethod("news", SARIMAXResults.class));
        assertNotNull(SARIMAXResults.class.getMethod("news", SARIMAXResults.class, SmootherOptions.class));
        assertNotNull(SARIMAXResults.class.getMethod("fixedParameters"));
        assertNotNull(SARIMAXResults.class.getMethod("fixedParameterCount"));
        assertNotNull(SARIMAXResults.class.getMethod("freeParameterCount"));
        assertNotNull(SARIMAXResults.class.getMethod("isFixedParameter", int.class));

        assertNotNull(SARIMAXPrediction.class.getMethod("start"));
        assertNotNull(SARIMAXPrediction.class.getMethod("end"));
        assertNotNull(SARIMAXPrediction.class.getMethod("kind"));
        assertNotNull(SARIMAXPrediction.class.getMethod("informationSet"));
        assertNotNull(SARIMAXPrediction.class.getMethod("signalOnly"));
        assertNotNull(SARIMAXPrediction.class.getMethod("dynamic"));
        assertNotNull(SARIMAXPrediction.class.getMethod("dynamicStart"));
        assertNotNull(SARIMAXPrediction.class.getMethod("mean"));
        assertNotNull(SARIMAXPrediction.class.getMethod("seMean"));
        assertNotNull(SARIMAXPrediction.class.getMethod("confInt"));
        assertNotNull(SARIMAXPrediction.class.getMethod("confInt", double.class));
        assertNotNull(SARIMAXPrediction.class.getMethod("summaryFrame"));
        assertNotNull(SARIMAXPrediction.class.getMethod("summaryFrame", double.class));

        assertMissingMethod(SARIMAXResults.class, "predict", int.class, int.class, boolean.class, double[][].class);
        assertMissingMethod(SARIMAXResults.class, "getPrediction");
        assertMissingMethod(SARIMAXResults.class, "getPrediction", int.class, int.class);
        assertMissingMethod(SARIMAXResults.class, "getForecast", int.class);
        assertMissingMethod(SARIMAXResults.class, "append", double[].class, boolean.class);
        assertMissingMethod(SARIMAXResults.class, "append", double[].class, double[][].class, boolean.class);
        assertMissingMethod(SARIMAXResults.class, "extend", double[].class, boolean.class);
        assertMissingMethod(SARIMAXResults.class, "extend", double[].class, double[][].class, boolean.class);
        assertMissingMethod(SARIMAXResults.class, "apply", double[].class, boolean.class);
        assertMissingMethod(SARIMAXResults.class, "apply", double[].class, double[][].class, boolean.class);
        assertMissingMethod(SARIMAXPrediction.class, "predictedMean");
    }

    @Test
    void testCurrentModelSurfaceExposesExplicitCovarianceSelection() throws NoSuchMethodException {
        assertNotNull(SARIMAX.class.getMethod("covarianceFromObservedInformation", double[].class));
        assertNotNull(SARIMAX.class.getMethod("covarianceFromApproximateInformation", double[].class));
        assertNotNull(SARIMAX.class.getMethod("covarianceFromOpg", double[].class));
        assertNotNull(SARIMAX.class.getMethod("robustCovarianceFromObservedInformation", double[].class));
        assertNotNull(SARIMAX.class.getMethod("robustCovarianceFromApproximateInformation", double[].class));
    }

    @Test
    void testCurrentModelSurfaceExposesCanonicalFitSelection() throws NoSuchMethodException {
        assertNotNull(SARIMAX.class.getMethod("objective"));
        assertNotNull(SARIMAX.class.getMethod("transformedObjective"));
        assertNotNull(SARIMAX.class.getMethod("fit"));
        assertNotNull(SARIMAX.class.getMethod("fit", SARIMAXFitOptions.class));

        assertMissingMethod(SARIMAX.class, "optimize");
        assertMissingMethod(SARIMAX.class, "optimize", Minimizer.class);
        assertMissingMethod(SARIMAX.class, "optimizeTransformed", double[].class);
        assertMissingMethod(SARIMAX.class, "optimizeTransformed", double[].class, Minimizer.class);
        assertMissingMethod(SARIMAX.class, "fit", Minimizer.class);
        assertMissingMethod(SARIMAX.class, "fitConstrained", java.util.Map.class);
        assertMissingMethod(SARIMAX.class, "fitConstrained", int[].class, double[].class);
    }

    @Test
    void testFitOptionsSurfaceExposesCovarianceTypeSelection() throws NoSuchMethodException {
        Set<String> builderMethods = Arrays.stream(SARIMAXFitOptions.Builder.class.getDeclaredMethods())
            .map(Method::getName)
            .collect(Collectors.toSet());
        Minimizer<?, ?, ?> optimizer = Minimizer.subplex().maxEvaluations(20);

        assertTrue(builderMethods.contains("covarianceType"));
        assertTrue(builderMethods.contains("optimizer"));
        assertTrue(builderMethods.contains("startParams"));
        assertTrue(builderMethods.contains("fixedParameters"));
        assertFalse(builderMethods.contains("fixedParams"));
        assertFalse(builderMethods.contains("fixedParam"));
        assertNotNull(SARIMAXFitOptions.class.getMethod("optimizer"));
        assertNotNull(SARIMAXFitOptions.class.getMethod("covarianceType"));
        assertNotNull(SARIMAXFitOptions.class.getMethod("startParams"));
        assertNotNull(SARIMAXFitOptions.class.getMethod("fixedParameters"));
        assertEquals(MLEResults.Covariance.ROBUST_APPROX,
            SARIMAXFitOptions.builder().covarianceType(MLEResults.Covariance.ROBUST_APPROX).build().covarianceType());
        assertEquals("opg", SARIMAXFitOptions.DEFAULT.covarianceType().id());
        assertNull(SARIMAXFitOptions.DEFAULT.optimizer());
        assertSame(optimizer, SARIMAXFitOptions.builder().optimizer(optimizer).build().optimizer());
        assertEquals("robust_approx", MLEResults.Covariance.ROBUST_APPROX.id());
        assertMissingMethod(SARIMAXFitOptions.Builder.class, "covarianceType", String.class);
    }

    @Test
    void testCurrentSpecSurfaceExposesSupportedRepresentationKnobs() {
        Set<String> builderMethods = Arrays.stream(SARIMAXSpec.Builder.class.getDeclaredMethods())
            .map(Method::getName)
            .collect(Collectors.toSet());

        assertTrue(builderMethods.contains("autoregressiveLags"));
        assertTrue(builderMethods.contains("movingAverageLags"));
        assertTrue(builderMethods.contains("simpleDifferencing"));
        assertTrue(builderMethods.contains("hamiltonRepresentation"));
        assertTrue(builderMethods.contains("approximateDiffuseVariance"));
        assertTrue(builderMethods.contains("measurementError"));
        assertTrue(builderMethods.contains("concentrateScale"));
        assertTrue(builderMethods.contains("mleRegression"));
        assertTrue(builderMethods.contains("timeVaryingRegression"));
        assertTrue(builderMethods.contains("seasonalOrder"));
        assertTrue(builderMethods.contains("exog"));
        assertTrue(builderMethods.contains("trend"));
        assertTrue(builderMethods.contains("trendPowers"));
        assertTrue(builderMethods.contains("trendOffset"));
    }

    @Test
    void testCurrentSpecSurfaceExposesSparseLagMasksNeededByWpiSeasonal() {
        Set<String> builderMethods = Arrays.stream(SARIMAXSpec.Builder.class.getDeclaredMethods())
            .map(Method::getName)
            .collect(Collectors.toSet());
        Set<String> arimaMethods = Arrays.stream(ARIMAOrder.class.getDeclaredMethods())
            .map(Method::getName)
            .collect(Collectors.toSet());

        assertTrue(builderMethods.contains("autoregressiveLags"));
        assertTrue(builderMethods.contains("movingAverageLags"));
        assertFalse(arimaMethods.contains("autoregressiveLags"));
        assertFalse(arimaMethods.contains("movingAverageLags"));

        assertEquals(3, ARIMAOrder.class.getRecordComponents().length);
        assertTrue(Arrays.stream(ARIMAOrder.class.getRecordComponents())
            .map(RecordComponent::getType)
            .allMatch(type -> type == int.class));
    }

    @Test
    void testWpiDiffuseReferenceUsesApproximateDiffuseRepresentationSurface() {
        Set<String> builderMethods = Arrays.stream(SARIMAXSpec.Builder.class.getDeclaredMethods())
            .map(Method::getName)
            .collect(Collectors.toSet());

        double initialVariance = parseScalar("wpi1_diffuse", "initial_variance");

        assertEquals(1.0e9, initialVariance, 0.0);
        assertTrue(builderMethods.contains("approximateDiffuseVariance"));
        assertTrue(builderMethods.contains("simpleDifferencing"));
        assertTrue(builderMethods.contains("hamiltonRepresentation"));
    }

    @Test
    void testCurrentResultsSurfaceExposesSummaryTextSurface() throws NoSuchMethodException {
        Set<String> resultMethods = Arrays.stream(SARIMAXResults.class.getDeclaredMethods())
            .map(Method::getName)
            .collect(Collectors.toSet());

        assertTrue(resultMethods.contains("summary"));
        assertNotNull(SARIMAXResults.class.getMethod("summary"));
        assertNotNull(SARIMAXResults.class.getMethod("summary", double.class));
    }

    @Test
    void testImplementedStatsmodelsGapsAreNoLongerDeferred() {
        Set<String> resultMethods = Arrays.stream(SARIMAXResults.class.getDeclaredMethods())
            .map(Method::getName)
            .collect(Collectors.toSet());

        assertTrue(resultMethods.contains("news"));
        assertTrue(resultMethods.contains("appendRefit"));
        assertTrue(resultMethods.contains("extendRefit"));
        assertTrue(resultMethods.contains("applyRefit"));
        assertTrue(resultMethods.contains("simulateRepetitions"));
        assertTrue(resultMethods.contains("testNormality"));
        assertTrue(resultMethods.contains("testHeteroskedasticity"));
        assertTrue(resultMethods.contains("testSerialCorrelation"));
    }

    private static double parseScalar(String fileName, String key) {
        String fixtureName = fileName.endsWith(".json") ? fileName.substring(0, fileName.length() - ".json".length()) : fileName;
        return StatsmodelsSarimaxFixtures.dictScalar(fixtureName, key);
    }

    private static void assertMissingMethod(Class<?> type, String name, Class<?>... parameterTypes) {
        assertThrows(NoSuchMethodException.class, () -> type.getMethod(name, parameterTypes));
    }
}