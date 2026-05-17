package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.test.RecordedRandomGenerator;
import com.fasterxml.jackson.databind.JsonNode;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

class StatsDistributionParityTest extends StatsDistributionFixtureTestBase {

    private static final double RTOL = 1e-12;
    private static final double DENSITY_ABS_TOL = 1e-12;
    private static final double MASS_ABS_TOL = 1e-15;
    private static final double SAMPLE_ABS_TOL = 1e-10;
    private static final double STRICT_SAMPLE_ABS_TOL = 1e-12;

    @ParameterizedTest(name = "{0}")
    @MethodSource("densityCases")
    void densityGridMatchesReference(DensityCase testCase) {
        JsonNode fixture = loadFixture(testCase.fixtureFile, testCase.section);
        Distribution distribution = testCase.factory.create(fixture);
        double[] grid = gridValues(fixture);
        double[] expected = densityValues(fixture);

        for (int index = 0; index < grid.length; index++) {
            assertClose(testCase.name, expected[index], distribution.logPdf(grid[index]), index,
                testCase.absoluteTolerance);
        }

        double[] batch = new double[grid.length];
        distribution.batch(Distribution.Metric.LOG_PDF, grid, 0, 1, grid.length, batch, 0);
        for (int index = 0; index < grid.length; index++) {
            assertClose(testCase.name + " batch", expected[index], batch[index], index,
                testCase.absoluteTolerance);
        }

        double[] interleaved = new double[grid.length * 2];
        for (int index = 0; index < grid.length; index++) interleaved[index * 2] = grid[index];
        double[] strided = new double[grid.length];
        distribution.batch(Distribution.Metric.LOG_PDF, interleaved, 0, 2, grid.length, strided, 0);
        for (int index = 0; index < grid.length; index++) {
            assertClose(testCase.name + " strided batch", expected[index], strided[index], index,
                testCase.absoluteTolerance);
        }

        double[] untouched = {42.0, 43.0};
        distribution.batch(Distribution.Metric.LOG_PDF, grid, 0, 1, 0, untouched, 0);
        assertThat(untouched).as(testCase.name + " n=0 batch").containsExactly(42.0, 43.0);

        if (fixture.has("probeXs")) {
            double[] probes = asDoubleArray(fixture.get("probeXs"));
            double[] probeExpected = asDoubleArray(fixture.get("probeLogPmf"));
            for (int index = 0; index < probes.length; index++) {
                assertClose(testCase.name + " probe", probeExpected[index], distribution.logPdf(probes[index]), index,
                    testCase.absoluteTolerance);
            }
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("sampleCases")
    void sampleReplayMatchesReference(SampleCase testCase) {
        JsonNode fixture = loadFixture(testCase.fixtureFile, testCase.section);
        Distribution distribution = testCase.factory.create(fixture);
        int samples = intField(fixture, "n");

        double[] batch = new double[samples];
        distribution.sample(replayRng(fixture), samples, batch, 0, 1);
        assertSamples(testCase, fixture, batch);

        double[] strided = new double[samples * 2];
        distribution.sample(replayRng(fixture), samples, strided, 0, 2);
        double[] picked = new double[samples];
        for (int index = 0; index < samples; index++) {
            picked[index] = strided[index * 2];
            assertThat(strided[index * 2 + 1]).as(testCase.name + " untouched stride slot %d", index).isEqualTo(0.0);
        }
        assertSamples(testCase, fixture, picked);

        if (testCase.scalarReplay == ScalarReplay.FIRST) {
            double first = distribution.sample(replayRng(fixture));
            assertSample(testCase, fixture, 0, first);
        } else if (testCase.scalarReplay == ScalarReplay.ALL) {
            RecordedRandomGenerator rng = replayRng(fixture);
            for (int index = 0; index < samples; index++) {
                assertSample(testCase, fixture, index, distribution.sample(rng));
            }
        }
    }

    private static Stream<DensityCase> densityCases() {
        return Stream.of(
            density("beta logPdf", "beta.json", "logPdf", StatsDistributionParityTest::betaDistribution),
            density("exponential logPdf", "exponential.json", "logPdf", StatsDistributionParityTest::exponentialDistribution),
            density("gamma logPdf", "gamma.json", "logPdf", StatsDistributionParityTest::gammaDistribution),
            density("inverse-gamma logPdf", "invgamma.json", "logPdf", StatsDistributionParityTest::inverseGammaDistribution),
            density("laplace logPdf", "laplace.json", "logPdf", StatsDistributionParityTest::laplaceDistribution),
            density("lognormal logPdf", "lognormal.json", "logPdf", StatsDistributionParityTest::logNormalDistribution),
            density("logistic logPdf", "logistic.json", "logPdf", StatsDistributionParityTest::logisticDistribution),
            density("normal logPdf", "normal.json", "logPdf", StatsDistributionParityTest::normalDistribution),
            density("student-t logPdf", "student.json", "logPdf", StatsDistributionParityTest::studentsTDistribution),
            density("truncated-normal logPdf", "truncnorm.json", "logPdf", StatsDistributionParityTest::truncatedNormalDistribution),
            density("uniform logPdf", "uniform.json", "logPdf", StatsDistributionParityTest::uniformDistribution),
            mass("bernoulli logPmf", "bernoulli.json", "logPmf", StatsDistributionParityTest::bernoulliDistribution),
            mass("binomial logPmf", "binomial.json", "logPmf", StatsDistributionParityTest::binomialDistribution),
            mass("categorical logPmf", "categorical.json", "logPmf", StatsDistributionParityTest::categoricalDistribution),
            mass("discrete-uniform logPmf", "discrete_uniform.json", "logPmf", StatsDistributionParityTest::discreteUniformDistribution),
            mass("geometric logPmf", "geometric.json", "logPmf", StatsDistributionParityTest::geometricDistribution),
            mass("poisson logPmf", "poisson.json", "logPmf", StatsDistributionParityTest::poissonDistribution),
            mass("poisson large-lambda logPmf", "poisson.json", "logPmfLarge", StatsDistributionParityTest::poissonDistribution)
        );
    }

    private static Stream<SampleCase> sampleCases() {
        return Stream.of(
            doubleSample("beta sample", "beta.json", "sample", StatsDistributionParityTest::betaDistribution, ScalarReplay.NONE, SAMPLE_ABS_TOL),
            doubleSample("exponential sample", "exponential.json", "sample", StatsDistributionParityTest::exponentialDistribution, ScalarReplay.FIRST, STRICT_SAMPLE_ABS_TOL),
            doubleSample("gamma sample", "gamma.json", "sample", StatsDistributionParityTest::gammaDistribution, ScalarReplay.ALL, SAMPLE_ABS_TOL),
            doubleSample("inverse-gamma sample", "invgamma.json", "sample", StatsDistributionParityTest::inverseGammaDistribution, ScalarReplay.NONE, SAMPLE_ABS_TOL),
            doubleSample("laplace sample", "laplace.json", "sample", StatsDistributionParityTest::laplaceDistribution, ScalarReplay.FIRST, STRICT_SAMPLE_ABS_TOL),
            doubleSample("lognormal sample", "lognormal.json", "sample", StatsDistributionParityTest::logNormalDistribution, ScalarReplay.FIRST, SAMPLE_ABS_TOL),
            doubleSample("logistic sample", "logistic.json", "sample", StatsDistributionParityTest::logisticDistribution, ScalarReplay.FIRST, STRICT_SAMPLE_ABS_TOL),
            doubleSample("normal sample", "normal.json", "sample", StatsDistributionParityTest::normalDistribution, ScalarReplay.FIRST, SAMPLE_ABS_TOL),
            doubleSample("student-t sample", "student.json", "sample", StatsDistributionParityTest::studentsTDistribution, ScalarReplay.NONE, SAMPLE_ABS_TOL),
            doubleSample("truncated-normal sample", "truncnorm.json", "sample", StatsDistributionParityTest::truncatedNormalDistribution, ScalarReplay.ALL, STRICT_SAMPLE_ABS_TOL, "a", "b"),
            doubleSample("uniform sample", "uniform.json", "sample", StatsDistributionParityTest::uniformDistribution, ScalarReplay.FIRST, STRICT_SAMPLE_ABS_TOL),
            intSample("bernoulli sample", "bernoulli.json", "sample", StatsDistributionParityTest::bernoulliDistribution, ScalarReplay.NONE),
            intSample("binomial sample", "binomial.json", "sample", StatsDistributionParityTest::binomialDistribution, ScalarReplay.NONE),
            intSample("categorical sample", "categorical.json", "sample", StatsDistributionParityTest::categoricalDistribution, ScalarReplay.ALL),
            intSample("discrete-uniform sample", "discrete_uniform.json", "sample", StatsDistributionParityTest::discreteUniformDistribution, ScalarReplay.ALL),
            intSample("geometric sample", "geometric.json", "sample", StatsDistributionParityTest::geometricDistribution, ScalarReplay.NONE),
            intSample("poisson small-lambda sample", "poisson.json", "sampleSmall", StatsDistributionParityTest::poissonDistribution, ScalarReplay.NONE),
            intSample("poisson large-lambda sample", "poisson.json", "sampleLarge", StatsDistributionParityTest::poissonDistribution, ScalarReplay.NONE)
        );
    }

    private static DensityCase density(String name, String fixtureFile, String section, DistributionFactory factory) {
        return new DensityCase(name, fixtureFile, section, factory, DENSITY_ABS_TOL);
    }

    private static DensityCase mass(String name, String fixtureFile, String section, DistributionFactory factory) {
        return new DensityCase(name, fixtureFile, section, factory, MASS_ABS_TOL);
    }

    private static SampleCase doubleSample(String name, String fixtureFile, String section,
                                           DistributionFactory factory, ScalarReplay scalarReplay,
                                           double absoluteTolerance) {
        return doubleSample(name, fixtureFile, section, factory, scalarReplay, absoluteTolerance, null, null);
    }

    private static SampleCase doubleSample(String name, String fixtureFile, String section,
                                           DistributionFactory factory, ScalarReplay scalarReplay,
                                           double absoluteTolerance, String supportLowerField,
                                           String supportUpperField) {
        return new SampleCase(name, fixtureFile, section, factory, SampleValueType.DOUBLE,
            scalarReplay, absoluteTolerance, supportLowerField, supportUpperField);
    }

    private static SampleCase intSample(String name, String fixtureFile, String section,
                                        DistributionFactory factory, ScalarReplay scalarReplay) {
        return new SampleCase(name, fixtureFile, section, factory, SampleValueType.INT,
            scalarReplay, 0.0, null, null);
    }

    private static void assertSamples(SampleCase testCase, JsonNode fixture, double[] actual) {
        if (testCase.valueType == SampleValueType.INT) {
            int[] expected = asIntArray(fixture.get("sampleDraws"));
            assertThat(actual).as(testCase.name + " length").hasSize(expected.length);
            for (int index = 0; index < expected.length; index++) {
                assertSample(testCase, fixture, index, actual[index]);
            }
            return;
        }
        double[] expected = asDoubleArray(fixture.get("sampleDraws"));
        assertThat(actual).as(testCase.name + " length").hasSize(expected.length);
        for (int index = 0; index < expected.length; index++) {
            assertSample(testCase, fixture, index, actual[index]);
        }
    }

    private static void assertSample(SampleCase testCase, JsonNode fixture, int index, double actual) {
        if (testCase.valueType == SampleValueType.INT) {
            int[] expected = asIntArray(fixture.get("sampleDraws"));
            assertThat((long) actual).as(testCase.name + " draw[%d]", index).isEqualTo((long) expected[index]);
            return;
        }

        double[] expected = asDoubleArray(fixture.get("sampleDraws"));
        assertClose(testCase.name, expected[index], actual, index, testCase.absoluteTolerance);
        if (testCase.supportLowerField != null) {
            assertThat(actual).as(testCase.name + " support[%d]", index)
                .isBetween(doubleField(fixture, testCase.supportLowerField), doubleField(fixture, testCase.supportUpperField));
        }
    }

    private static double[] gridValues(JsonNode fixture) {
        JsonNode xGrid = fixture.get("xGrid");
        if (xGrid != null) return asDoubleArray(xGrid);
        JsonNode xs = fixture.get("xs");
        if (xs == null) throw new IllegalStateException("Fixture has no xGrid/xs: " + fixture.get("_fixtureName"));
        int[] integers = asIntArray(xs);
        double[] out = new double[integers.length];
        for (int index = 0; index < integers.length; index++) out[index] = integers[index];
        return out;
    }

    private static double[] densityValues(JsonNode fixture) {
        for (String field : new String[]{"expectedLogPdf", "expectedLogPdfs", "logPdf", "logPmf"}) {
            JsonNode values = fixture.get(field);
            if (values != null) return asDoubleArray(values);
        }
        throw new IllegalStateException("Fixture has no density vector: " + fixture.get("_fixtureName"));
    }

    private static void assertClose(String label, double expected, double actual, int index, double absoluteTolerance) {
        if (Double.isInfinite(expected) || Double.isInfinite(actual) || Double.isNaN(expected) || Double.isNaN(actual)) {
            assertThat(Double.doubleToRawLongBits(actual))
                .as("%s bitwise mismatch at %d: expected=%s actual=%s", label, index, expected, actual)
                .isEqualTo(Double.doubleToRawLongBits(expected));
            return;
        }
        double diff = Math.abs(actual - expected);
        double scale = Math.max(1.0, Math.abs(expected));
        assertThat(diff)
            .as("%s mismatch at %d: expected=%s actual=%s diff=%s", label, index, expected, actual, diff)
            .isLessThanOrEqualTo(Math.max(absoluteTolerance, RTOL * scale));
    }

    private static double doubleField(JsonNode fixture, String field) {
        JsonNode value = fixture.get(field);
        if (value != null) return value.asDouble();
        JsonNode config = fixture.get("_config");
        if (config != null && config.has(field)) return config.get(field).asDouble();
        throw new IllegalStateException("Missing numeric field " + field + " in " + fixture.get("_fixtureName"));
    }

    private static int intField(JsonNode fixture, String field) {
        JsonNode value = fixture.get(field);
        if (value != null) return value.asInt();
        JsonNode config = fixture.get("_config");
        if (config != null && config.has(field)) return config.get(field).asInt();
        throw new IllegalStateException("Missing integer field " + field + " in " + fixture.get("_fixtureName"));
    }

    private static long longField(JsonNode fixture, String field) {
        JsonNode value = fixture.get(field);
        if (value != null) return value.asLong();
        JsonNode config = fixture.get("_config");
        if (config != null && config.has(field)) return config.get(field).asLong();
        throw new IllegalStateException("Missing integer field " + field + " in " + fixture.get("_fixtureName"));
    }

    private static BernoulliDistribution bernoulliDistribution(JsonNode fixture) {
        return new BernoulliDistribution(doubleField(fixture, "successProbability"));
    }

    private static BetaDistribution betaDistribution(JsonNode fixture) {
        return new BetaDistribution(doubleField(fixture, "alpha"), doubleField(fixture, "beta"));
    }

    private static BinomialDistribution binomialDistribution(JsonNode fixture) {
        return new BinomialDistribution(intField(fixture, "trials"), doubleField(fixture, "successProbability"));
    }

    private static CategoricalDistribution categoricalDistribution(JsonNode fixture) {
        return new CategoricalDistribution(asDoubleArray(fixture.get("probs")));
    }

    private static DiscreteUniformDistribution discreteUniformDistribution(JsonNode fixture) {
        return new DiscreteUniformDistribution(longField(fixture, "lo"), longField(fixture, "hi"));
    }

    private static ExponentialDistribution exponentialDistribution(JsonNode fixture) {
        return new ExponentialDistribution(doubleField(fixture, "lambda"));
    }

    private static GammaDistribution gammaDistribution(JsonNode fixture) {
        return new GammaDistribution(doubleField(fixture, "shape"), doubleField(fixture, "scale"));
    }

    private static GeometricDistribution geometricDistribution(JsonNode fixture) {
        return new GeometricDistribution(doubleField(fixture, "successFraction"));
    }

    private static InverseGammaDistribution inverseGammaDistribution(JsonNode fixture) {
        return new InverseGammaDistribution(doubleField(fixture, "shape"), doubleField(fixture, "scale"));
    }

    private static LaplaceDistribution laplaceDistribution(JsonNode fixture) {
        return new LaplaceDistribution(doubleField(fixture, "location"), doubleField(fixture, "scale"));
    }

    private static LogisticDistribution logisticDistribution(JsonNode fixture) {
        return new LogisticDistribution(doubleField(fixture, "location"), doubleField(fixture, "scale"));
    }

    private static LogNormalDistribution logNormalDistribution(JsonNode fixture) {
        return new LogNormalDistribution(doubleField(fixture, "location"), doubleField(fixture, "scale"));
    }

    private static NormalDistribution normalDistribution(JsonNode fixture) {
        return new NormalDistribution(doubleField(fixture, "mean"), doubleField(fixture, "sigma"));
    }

    private static PoissonDistribution poissonDistribution(JsonNode fixture) {
        return new PoissonDistribution(doubleField(fixture, "lambda"));
    }

    private static StudentsTDistribution studentsTDistribution(JsonNode fixture) {
        return new StudentsTDistribution(doubleField(fixture, "df"));
    }

    private static TruncatedNormalDistribution truncatedNormalDistribution(JsonNode fixture) {
        return new TruncatedNormalDistribution(doubleField(fixture, "mu"), doubleField(fixture, "sigma"),
            doubleField(fixture, "a"), doubleField(fixture, "b"));
    }

    private static UniformDistribution uniformDistribution(JsonNode fixture) {
        return new UniformDistribution(doubleField(fixture, "lower"), doubleField(fixture, "upper"));
    }

    @FunctionalInterface
    private interface DistributionFactory {
        Distribution create(JsonNode fixture);
    }

    private record DensityCase(String name, String fixtureFile, String section,
                               DistributionFactory factory, double absoluteTolerance) {
        @Override
        public String toString() {
            return name;
        }
    }

    private record SampleCase(String name, String fixtureFile, String section,
                              DistributionFactory factory, SampleValueType valueType,
                              ScalarReplay scalarReplay, double absoluteTolerance,
                              String supportLowerField, String supportUpperField) {
        @Override
        public String toString() {
            return name;
        }
    }

    private enum SampleValueType {
        DOUBLE,
        INT
    }

    private enum ScalarReplay {
        NONE,
        FIRST,
        ALL
    }
}