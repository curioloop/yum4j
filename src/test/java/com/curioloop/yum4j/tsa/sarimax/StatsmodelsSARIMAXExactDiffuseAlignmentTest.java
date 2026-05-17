package com.curioloop.yum4j.tsa.sarimax;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StatsmodelsSARIMAXExactDiffuseAlignmentTest {

    private static final String FIXTURE_RESOURCE =
        "statsmodels/tsa/statespace/tests/results/sarimax_public_alignment.json";
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper()
        .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
        .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    static Stream<ExactDiffuseReference> exactDiffuseReferences() {
        return FixtureRoot.load().exactDiffuseReferences().stream();
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("exactDiffuseReferences")
    void exactDiffuseReferenceLikelihoodMatchesStatsmodelsFixtures(ExactDiffuseReference fixture) {
        SARIMAX model = new SARIMAX(spec(fixture));

        assertArrayEquals(fixture.paramNames(), model.parameterNames(), fixture.id() + " parameterNames");
        assertEquals(fixture.logLikelihoodBurn(), model.likelihoodBurn(), fixture.id() + " burn");
        assertClose(fixture.logLikelihood(), model.logLikelihood(fixture.params()), fixture.tolerance(), fixture.id() + " llf");
        assertArrayClose(fixture.logLikelihoodObs(), model.logLikelihoodObs(fixture.params()), fixture.tolerance(),
            fixture.id() + " llfObs");
        assertEquals(fixture.logLikelihood(), Arrays.stream(fixture.logLikelihoodObs()).sum(),
            fixture.tolerance() * Math.max(1.0, Math.abs(fixture.logLikelihood())), fixture.id() + " fixture llf sum");
    }

    private static SARIMAXSpec spec(ExactDiffuseReference fixture) {
        SARIMAXSpec.Builder builder = SARIMAXSpec.builder(
                ARIMAOrder.of(fixture.order()[0], fixture.order()[1], fixture.order()[2]),
                fixture.endog())
            .seasonalOrder(SeasonalOrder.of(fixture.seasonalOrder()[0],
                fixture.seasonalOrder()[1], fixture.seasonalOrder()[2], fixture.seasonalOrder()[3]))
            .trendPowers(fixture.trendPowers())
            .include(SARIMAXOption.USE_EXACT_DIFFUSE);
        if (fixture.exog() != null) {
            builder.exog(fixture.exog());
        }
        return builder.build();
    }

    private static void assertArrayClose(double[] expected, double[] actual, double tolerance, String label) {
        assertEquals(expected.length, actual.length, label + " length");
        for (int index = 0; index < expected.length; index++) {
            assertClose(expected[index], actual[index], tolerance, label + " index " + index);
        }
    }

    private static void assertClose(double expected, double actual, double tolerance, String label) {
        double allowed = tolerance * Math.max(1.0, Math.abs(expected));
        assertTrue(Math.abs(expected - actual) <= allowed,
            () -> label + " ==> expected: <" + expected + "> but was: <" + actual + "> tolerance: <" + allowed + ">");
    }

    private record FixtureRoot(List<ExactDiffuseReference> exactDiffuseReferences) {
        private static FixtureRoot load() {
            try (InputStream input = Thread.currentThread().getContextClassLoader().getResourceAsStream(FIXTURE_RESOURCE)) {
                assertNotNull(input, "missing fixture resource " + FIXTURE_RESOURCE);
                return OBJECT_MAPPER.readValue(input, FixtureRoot.class);
            } catch (IOException ex) {
                throw new UncheckedIOException(ex);
            }
        }
    }

    private record ExactDiffuseReference(String id,
                                         int[] order,
                                         int[] seasonalOrder,
                                         int[] trendPowers,
                                         boolean useExactDiffuse,
                                         int logLikelihoodBurn,
                                         double tolerance,
                                         double[] endog,
                                         double[][] exog,
                                         double[] params,
                                         String[] paramNames,
                                         double logLikelihood,
                                         double[] logLikelihoodObs) {
        @Override
        public String toString() {
            return id;
        }
    }
}