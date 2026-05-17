package com.curioloop.yum4j.tsa.varmax;

import com.curioloop.yum4j.ssm.kalman.mle.MLEResults;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.optim.Minimizer;
import com.curioloop.yum4j.tsa.diagnostics.InstantaneousCausality;
import com.curioloop.yum4j.tsa.prediction.PredictionInformationSet;
import com.curioloop.yum4j.tsa.statespace.SimulationAnchor;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VARMAXStatsmodelsSurfaceAuditTest {

    @Test
    void currentOptionSurfaceMatchesSupportedOverlap() {
        Set<String> optionNames = Arrays.stream(VARMAXOption.values())
            .map(Enum::name)
            .collect(Collectors.toSet());

        assertEquals(Set.of(
            "ENFORCE_STATIONARITY",
            "ENFORCE_INVERTIBILITY",
            "USE_EXACT_DIFFUSE"), optionNames);
    }

    @Test
    void currentModelSurfaceExposesCoreMLEAndStatespaceMethods() throws NoSuchMethodException {
        assertNotNull(VARMAX.class.getMethod("spec"));
        assertNotNull(VARMAX.class.getMethod("paramCount"));
        assertNotNull(VARMAX.class.getMethod("parameterNames"));
        assertNotNull(VARMAX.class.getMethod("startParams"));
        assertNotNull(VARMAX.class.getMethod("unconstrainedStartParams"));
        assertNotNull(VARMAX.class.getMethod("transformParams", double[].class));
        assertNotNull(VARMAX.class.getMethod("untransformParams", double[].class));
        assertNotNull(VARMAX.class.getMethod("logLikelihood", double[].class));
        assertNotNull(VARMAX.class.getMethod("logLikelihoodObs", double[].class));
        assertNotNull(VARMAX.class.getMethod("filter", double[].class));
        assertNotNull(VARMAX.class.getMethod("smooth", double[].class));
        assertNotNull(VARMAX.class.getMethod("predict", double[].class));
        assertNotNull(VARMAX.class.getMethod("score", double[].class));
        assertNotNull(VARMAX.class.getMethod("scoreObs", double[].class));
        assertNotNull(VARMAX.class.getMethod("opgInformationMatrix", double[].class));
        assertNotNull(VARMAX.class.getMethod("observedInformationMatrix", double[].class));
        assertNotNull(VARMAX.class.getMethod("covarianceFromObservedInformation", double[].class));
        assertNotNull(VARMAX.class.getMethod("covarianceFromApproximateInformation", double[].class));
        assertNotNull(VARMAX.class.getMethod("covarianceFromOpg", double[].class));
        assertNotNull(VARMAX.class.getMethod("robustCovarianceFromObservedInformation", double[].class));
        assertNotNull(VARMAX.class.getMethod("robustCovarianceFromApproximateInformation", double[].class));
        assertNotNull(VARMAX.class.getMethod("snapshotModel", double[].class));
        assertNotNull(VARMAX.class.getMethod("fit"));
        assertNotNull(VARMAX.class.getMethod("fit", VARMAXFitOptions.class));

        assertMissingMethod(VARMAX.class, "fit", Minimizer.class);
        assertMissingMethod(VARMAX.class, "fitConstrained", java.util.Map.class);
        assertMissingMethod(VARMAX.class, "fitConstrained", int[].class, double[].class);
    }

    @Test
    void currentResultsSurfaceExposesOverlappingMetrics() throws NoSuchMethodException {
        assertNotNull(VARMAXResults.class.getMethod("parameterNames"));
        assertNotNull(VARMAXResults.class.getMethod("covType"));
        assertNotNull(VARMAXResults.class.getMethod("logLikelihood"));
        assertNotNull(VARMAXResults.class.getMethod("logLikelihoodObs"));
        assertNotNull(VARMAXResults.class.getMethod("logLikelihoodBurn"));
        assertNotNull(VARMAXResults.class.getMethod("scale"));
        assertNotNull(VARMAXResults.class.getMethod("nobsEffective"));
        assertNotNull(VARMAXResults.class.getMethod("dfModel"));
        assertNotNull(VARMAXResults.class.getMethod("dfResid"));
        assertNotNull(VARMAXResults.class.getMethod("aic"));
        assertNotNull(VARMAXResults.class.getMethod("aicc"));
        assertNotNull(VARMAXResults.class.getMethod("bic"));
        assertNotNull(VARMAXResults.class.getMethod("hqic"));
        assertNotNull(VARMAXResults.class.getMethod("covParams"));
        assertNotNull(VARMAXResults.class.getMethod("bse"));
        assertNotNull(VARMAXResults.class.getMethod("zvalues"));
        assertNotNull(VARMAXResults.class.getMethod("pvalues"));
        assertNotNull(VARMAXResults.class.getMethod("confInt"));
        assertNotNull(VARMAXResults.class.getMethod("fittedValues"));
        assertNotNull(VARMAXResults.class.getMethod("residuals"));
        assertNotNull(VARMAXResults.class.getMethod("standardizedForecastError"));
        assertNotNull(VARMAXResults.class.getMethod("smooth"));
        assertNotNull(VARMAXResults.class.getMethod("coefficientMatricesVAR"));
        assertNotNull(VARMAXResults.class.getMethod("coefficientMatricesVMA"));
        assertNotNull(VARMAXResults.class.getMethod("predict"));
        assertNotNull(VARMAXResults.class.getMethod("predict", int.class, int.class, int.class, double[][].class));
        assertNotNull(VARMAXResults.class.getMethod("predict", int.class, int.class, PredictionInformationSet.class, boolean.class));
        assertNotNull(VARMAXResults.class.getMethod("forecast", int.class));
        assertNotNull(VARMAXResults.class.getMethod("forecast", int.class, double[][].class));
        assertNotNull(VARMAXResults.class.getMethod("forecast", int.class, double[][].class, boolean.class));
        assertNotNull(VARMAXResults.class.getMethod("append", double[][].class));
        assertNotNull(VARMAXResults.class.getMethod("append", double[][].class, double[][].class));
        assertNotNull(VARMAXResults.class.getMethod("appendRefit", double[][].class));
        assertNotNull(VARMAXResults.class.getMethod("appendRefit", double[][].class, double[][].class));
        assertNotNull(VARMAXResults.class.getMethod("appendRefit", double[][].class, double[][].class, VARMAXFitOptions.class));
        assertNotNull(VARMAXResults.class.getMethod("extend", double[][].class));
        assertNotNull(VARMAXResults.class.getMethod("extend", double[][].class, double[][].class));
        assertNotNull(VARMAXResults.class.getMethod("extendRefit", double[][].class));
        assertNotNull(VARMAXResults.class.getMethod("extendRefit", double[][].class, double[][].class));
        assertNotNull(VARMAXResults.class.getMethod("extendRefit", double[][].class, double[][].class, VARMAXFitOptions.class));
        assertNotNull(VARMAXResults.class.getMethod("apply", double[][].class));
        assertNotNull(VARMAXResults.class.getMethod("apply", double[][].class, double[][].class));
        assertNotNull(VARMAXResults.class.getMethod("applyRefit", double[][].class));
        assertNotNull(VARMAXResults.class.getMethod("applyRefit", double[][].class, double[][].class));
        assertNotNull(VARMAXResults.class.getMethod("applyRefit", double[][].class, double[][].class, VARMAXFitOptions.class));
        assertNotNull(VARMAXResults.class.getMethod("simulate", int.class));
        assertNotNull(VARMAXResults.class.getMethod("simulate", int.class, Random.class));
        assertNotNull(VARMAXResults.class.getMethod("simulate", int.class, double[][].class, Random.class,
            SimulationSmootherOptions.class));
        assertNotNull(VARMAXResults.class.getMethod("simulate", int.class, SimulationAnchor.class, Random.class,
            SimulationSmootherOptions.class));
        assertNotNull(VARMAXResults.class.getMethod("simulate", int.class, SimulationAnchor.class, double[][].class,
            Random.class, SimulationSmootherOptions.class));
        assertNotNull(VARMAXResults.class.getMethod("simulate", int.class, int.class, double[][].class, Random.class,
            SimulationSmootherOptions.class));
        assertNotNull(VARMAXResults.class.getMethod("simulateRepetitions", int.class, int.class));
        assertNotNull(VARMAXResults.class.getMethod("simulateRepetitions", int.class, int.class, SimulationAnchor.class,
            Random.class, SimulationSmootherOptions.class));
        assertNotNull(VARMAXResults.class.getMethod("simulate", int.class, double[].class, double[].class, double[].class));
        assertNotNull(VARMAXResults.class.getMethod("simulate", int.class, int.class, double[][].class,
            double[].class, double[].class, double[].class, SimulationSmootherOptions.class));
        assertNotNull(VARMAXResults.class.getMethod("impulseResponses", int.class));
        assertNotNull(VARMAXResults.class.getMethod("impulseResponses", int.class, int.class));
        assertNotNull(VARMAXResults.class.getMethod("impulseResponses", int.class, int.class, boolean.class, boolean.class));
        assertNotNull(VARMAXResults.class.getMethod("impulseResponses", int.class, int.class, boolean.class, boolean.class, int.class));
        assertNotNull(VARMAXResults.class.getMethod("impulseResponses", int.class, double[].class, boolean.class, boolean.class));
        assertNotNull(VARMAXResults.class.getMethod("impulseResponses", int.class, double[].class, boolean.class, boolean.class, int.class));
        assertNotNull(VARMAXResults.class.getMethod("impulseResponseRepetitions", int.class, int.class, int.class, boolean.class, boolean.class));
        assertNotNull(VARMAXResults.class.getMethod("fevd", int.class));
        assertNotNull(VARMAXResults.class.getMethod("testCausality", int.class, int.class, int.class));
        Method instantaneousCausality = VARMAXResults.class.getMethod("testInstantaneousCausality", int.class, int.class);
        assertNotNull(instantaneousCausality);
        assertEquals(InstantaneousCausality.class, instantaneousCausality.getReturnType());
        assertNotNull(VARMAXResults.class.getMethod("testNormality"));
        assertNotNull(VARMAXResults.class.getMethod("testHeteroskedasticity"));
        assertNotNull(VARMAXResults.class.getMethod("testSerialCorrelation", int[].class));
        assertNotNull(VARMAXResults.class.getMethod("smoothedStateWeights"));
        assertNotNull(VARMAXResults.class.getMethod("smoothedStateWeights", SmootherOptions.class));
        assertNotNull(VARMAXResults.class.getMethod("smoothedDecomposition"));
        assertNotNull(VARMAXResults.class.getMethod("smoothedDecomposition", SmootherOptions.class));
        assertNotNull(VARMAXResults.class.getMethod("news", VARMAXResults.class));
        assertNotNull(VARMAXResults.class.getMethod("news", VARMAXResults.class, SmootherOptions.class));
        assertNotNull(VARMAXResults.class.getMethod("fixedParameters"));
        assertNotNull(VARMAXResults.class.getMethod("fixedParameterCount"));
        assertNotNull(VARMAXResults.class.getMethod("freeParameterCount"));
        assertNotNull(VARMAXResults.class.getMethod("isFixedParameter", int.class));

        assertMissingMethod(VARMAXResults.class, "predict", int.class, int.class, boolean.class, double[][].class);
        assertMissingMethod(VARMAXResults.class, "getPrediction");
        assertMissingMethod(VARMAXResults.class, "getPrediction", int.class, int.class);
        assertMissingMethod(VARMAXResults.class, "getForecast", int.class);
        assertMissingMethod(VARMAXResults.class, "append", double[][].class, boolean.class);
        assertMissingMethod(VARMAXResults.class, "append", double[][].class, double[][].class, boolean.class);
        assertMissingMethod(VARMAXResults.class, "extend", double[][].class, boolean.class);
        assertMissingMethod(VARMAXResults.class, "extend", double[][].class, double[][].class, boolean.class);
        assertMissingMethod(VARMAXResults.class, "apply", double[][].class, boolean.class);
        assertMissingMethod(VARMAXResults.class, "apply", double[][].class, double[][].class, boolean.class);
    }

    @Test
    void currentPredictionSurfaceExposesStatsmodelsPredictionResultOverlap() throws NoSuchMethodException {
        assertNotNull(VARMAXPrediction.class.getMethod("start"));
        assertNotNull(VARMAXPrediction.class.getMethod("end"));
        assertNotNull(VARMAXPrediction.class.getMethod("kind"));
        assertNotNull(VARMAXPrediction.class.getMethod("informationSet"));
        assertNotNull(VARMAXPrediction.class.getMethod("signalOnly"));
        assertNotNull(VARMAXPrediction.class.getMethod("dynamic"));
        assertNotNull(VARMAXPrediction.class.getMethod("dynamicStart"));
        assertNotNull(VARMAXPrediction.class.getMethod("observationDimension"));
        assertNotNull(VARMAXPrediction.class.getMethod("mean"));
        assertNotNull(VARMAXPrediction.class.getMethod("variance"));
        assertNotNull(VARMAXPrediction.class.getMethod("seMean"));
        assertNotNull(VARMAXPrediction.class.getMethod("confInt"));
        assertNotNull(VARMAXPrediction.class.getMethod("confInt", double.class));
        assertNotNull(VARMAXPrediction.class.getMethod("summaryFrame"));
        assertNotNull(VARMAXPrediction.class.getMethod("summaryFrame", double.class));
        assertMissingMethod(VARMAXPrediction.class, "predictedMean");
    }

    @Test
    void currentSpecSurfaceExposesSupportedConstructionKnobs() {
        Set<String> builderMethods = Arrays.stream(VARMAXSpec.Builder.class.getDeclaredMethods())
            .map(Method::getName)
            .collect(Collectors.toSet());

        assertTrue(builderMethods.contains("exog"));
        assertTrue(builderMethods.contains("observationIntercept"));
        assertTrue(builderMethods.contains("trend"));
        assertTrue(builderMethods.contains("trendPowers"));
        assertTrue(builderMethods.contains("trendOffset"));
        assertTrue(builderMethods.contains("errorCovariance"));
        assertTrue(builderMethods.contains("measurementError"));
        assertTrue(builderMethods.contains("logLikelihoodBurn"));
        assertTrue(builderMethods.contains("approximateDiffuseVariance"));
        assertTrue(builderMethods.contains("include"));
        assertTrue(builderMethods.contains("exclude"));
        assertTrue(builderMethods.contains("endogNames"));
        assertTrue(builderMethods.contains("exogNames"));
    }

    @Test
    void fitOptionsSurfaceExposesCovarianceTypeSelection() throws NoSuchMethodException {
        Set<String> builderMethods = Arrays.stream(VARMAXFitOptions.Builder.class.getDeclaredMethods())
            .map(Method::getName)
            .collect(Collectors.toSet());
        Minimizer<?, ?, ?> optimizer = Minimizer.subplex().maxEvaluations(20);

        assertTrue(builderMethods.contains("covarianceType"));
        assertTrue(builderMethods.contains("optimizer"));
        assertTrue(builderMethods.contains("startParams"));
        assertTrue(builderMethods.contains("fixedParameters"));
        assertFalse(builderMethods.contains("fixedParams"));
        assertFalse(builderMethods.contains("fixedParam"));
        assertNotNull(VARMAXFitOptions.class.getMethod("optimizer"));
        assertNotNull(VARMAXFitOptions.class.getMethod("covarianceType"));
        assertNotNull(VARMAXFitOptions.class.getMethod("startParams"));
        assertNotNull(VARMAXFitOptions.class.getMethod("fixedParameters"));
        assertEquals(MLEResults.Covariance.ROBUST_APPROX,
            VARMAXFitOptions.builder().covarianceType(MLEResults.Covariance.ROBUST_APPROX).build().covarianceType());
        assertEquals("opg", VARMAXFitOptions.DEFAULT.covarianceType().id());
        assertNull(VARMAXFitOptions.DEFAULT.optimizer());
        assertSame(optimizer, VARMAXFitOptions.builder().optimizer(optimizer).build().optimizer());
        assertEquals("robust_approx", MLEResults.Covariance.ROBUST_APPROX.id());
        assertMissingMethod(VARMAXFitOptions.Builder.class, "covarianceType", String.class);
    }

    @Test
    void implementedStatsmodelsGapsAreNoLongerDeferred() {
        Set<String> resultMethods = Arrays.stream(VARMAXResults.class.getDeclaredMethods())
            .map(Method::getName)
            .collect(Collectors.toSet());

        assertTrue(resultMethods.contains("news"));
        assertTrue(resultMethods.contains("fevd"));
        assertTrue(resultMethods.contains("testCausality"));
        assertTrue(resultMethods.contains("testInstantaneousCausality"));
        assertTrue(resultMethods.contains("appendRefit"));
        assertTrue(resultMethods.contains("extendRefit"));
        assertTrue(resultMethods.contains("applyRefit"));
        assertTrue(resultMethods.contains("simulateRepetitions"));
        assertTrue(resultMethods.contains("testNormality"));
        assertTrue(resultMethods.contains("testHeteroskedasticity"));
        assertTrue(resultMethods.contains("testSerialCorrelation"));
    }

    private static void assertMissingMethod(Class<?> type, String name, Class<?>... parameterTypes) {
        assertThrows(NoSuchMethodException.class, () -> type.getMethod(name, parameterTypes));
    }
}