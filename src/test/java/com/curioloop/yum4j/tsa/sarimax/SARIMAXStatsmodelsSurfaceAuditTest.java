package com.curioloop.yum4j.tsa.sarimax;

import com.curioloop.yum4j.kalman.mle.MLECovariance;
import com.curioloop.yum4j.optim.Minimizer;
import org.junit.jupiter.api.Test;

import java.lang.reflect.RecordComponent;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
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
        assertNotNull(SARIMAXResults.class.getMethod("standardizedForecastsError"));
        assertNotNull(SARIMAXResults.class.getMethod("smooth"));
        assertNotNull(SARIMAXResults.class.getMethod("predict", int.class, int.class, boolean.class, double[][].class));
        assertNotNull(SARIMAXResults.class.getMethod("forecast", int.class, double[][].class));
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
    void testCurrentModelSurfaceExposesMLEOptimizationSelection() throws NoSuchMethodException {
        assertNotNull(SARIMAX.class.getMethod("optimize"));
        assertNotNull(SARIMAX.class.getMethod("optimize", Minimizer.class));
        assertNotNull(SARIMAX.class.getMethod("optimizeTransformed", double[].class));
        assertNotNull(SARIMAX.class.getMethod("optimizeTransformed", double[].class, Minimizer.class));
        assertNotNull(SARIMAX.class.getMethod("fit", Minimizer.class));
    }

    @Test
    void testFitOptionsSurfaceExposesCovarianceTypeSelection() throws NoSuchMethodException {
        Set<String> builderMethods = Arrays.stream(SARIMAXFitOptions.Builder.class.getDeclaredMethods())
            .map(Method::getName)
            .collect(Collectors.toSet());
        Minimizer<?, ?, ?> optimizer = Minimizer.subplex().maxEvaluations(20);

        assertTrue(builderMethods.contains("covarianceType"));
        assertTrue(builderMethods.contains("optimizer"));
        assertNotNull(SARIMAXFitOptions.class.getMethod("optimizer"));
        assertNotNull(SARIMAXFitOptions.class.getMethod("covarianceType"));
        assertEquals("opg", SARIMAXFitOptions.DEFAULT.covarianceType().id());
        assertNull(SARIMAXFitOptions.DEFAULT.optimizer());
        assertSame(optimizer, SARIMAXFitOptions.builder().optimizer(optimizer).build().optimizer());
        assertEquals("robust_approx", MLECovariance.ROBUST_APPROX.id());
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
        assertTrue(builderMethods.contains("trendPowers"));
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

    private static double parseScalar(String fileName, String key) {
        String fixtureName = fileName.endsWith(".json") ? fileName.substring(0, fileName.length() - ".json".length()) : fileName;
        return StatsmodelsSarimaxFixtures.dictScalar(fixtureName, key);
    }
}