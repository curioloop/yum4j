package com.curioloop.yum4j.fixtures;

import com.curioloop.yum4j.ssm.kalman.filter.FilterMethod;
import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherMethod;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class KalmanStatsmodelsFixtures {

    private static final String MANIFEST_RESOURCE =
        "statsmodels/tsa/statespace/tests/results/kalman_alignment_manifest.json";
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper()
        .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    private KalmanStatsmodelsFixtures() {
    }

    public static List<AlignmentCase> cases() {
        return loadManifest().cases();
    }

    public static List<SurfacePoint> surfaces(AlignmentCase fixture) {
        Objects.requireNonNull(fixture, "fixture");
        List<String[]> rows = StatsmodelsResources.readStatespaceResultsCsv(fixture.surfacesFile());
        return surfacePoints(fixture, rows);
    }

    public static List<SurfacePoint> smootherSurfaces(AlignmentCase fixture) {
        Objects.requireNonNull(fixture, "fixture");
        if (fixture.smootherSurfacesFile() == null) {
            return List.of();
        }
        List<String[]> rows = StatsmodelsResources.readStatespaceResultsCsv(fixture.smootherSurfacesFile());
        return surfacePoints(fixture, rows);
    }

    public static List<SurfacePoint> diagnosticsSurfaces(AlignmentCase fixture) {
        Objects.requireNonNull(fixture, "fixture");
        if (fixture.diagnosticsSurfacesFile() == null) {
            return List.of();
        }
        List<String[]> rows = StatsmodelsResources.readStatespaceResultsCsv(fixture.diagnosticsSurfacesFile());
        return surfacePoints(fixture, rows);
    }

    public static List<SurfacePoint> cfaSurfaces(AlignmentCase fixture) {
        Objects.requireNonNull(fixture, "fixture");
        if (fixture.cfaSurfacesFile() == null) {
            return List.of();
        }
        List<String[]> rows = StatsmodelsResources.readStatespaceResultsCsv(fixture.cfaSurfacesFile());
        return surfacePoints(fixture, rows);
    }

    public static AlignmentCase caseById(String caseId) {
        Objects.requireNonNull(caseId, "caseId");
        for (AlignmentCase fixture : cases()) {
            if (fixture.id().equals(caseId)) {
                return fixture;
            }
        }
        throw new IllegalArgumentException("Unknown Kalman statsmodels fixture: " + caseId);
    }

    private static List<SurfacePoint> surfacePoints(AlignmentCase fixture, List<String[]> rows) {
        List<SurfacePoint> points = new ArrayList<>();
        for (int rowIndex = 1; rowIndex < rows.size(); rowIndex++) {
            String[] row = rows.get(rowIndex);
            if (!fixture.id().equals(row[0])) {
                continue;
            }
            points.add(new SurfacePoint(
                row[0],
                row[1],
                row[2],
                Integer.parseInt(row[3]),
                Integer.parseInt(row[4]),
                Integer.parseInt(row[5]),
                row.length > 7 ? row[6] : "REAL",
                Double.parseDouble(row.length > 7 ? row[7] : row[6]),
                row.length > 8 && !row[8].isBlank() ? Double.parseDouble(row[8]) : 0.0));
        }
        return points;
    }

    private static Manifest loadManifest() {
        InputStream stream = KalmanStatsmodelsFixtures.class.getClassLoader().getResourceAsStream(MANIFEST_RESOURCE);
        if (stream == null) {
            throw new IllegalArgumentException("Fixture resource not found: " + MANIFEST_RESOURCE);
        }
        try (InputStream input = stream) {
            return OBJECT_MAPPER.readValue(input, Manifest.class);
        } catch (IOException exception) {
            throw new UncheckedIOException("Failed to read fixture resource: " + MANIFEST_RESOURCE, exception);
        }
    }

    public record Manifest(Map<String, String> meta, List<AlignmentCase> cases) {
    }

    public record AlignmentCase(String id,
                                String source,
                                Shape shape,
                                ModelSpec model,
                                InitialSpec initialization,
                                FilterSpec filter,
                                SmootherSpec smoother,
                                double tolerance,
                                Map<String, Double> scalars,
                                String surfacesFile,
                                String smootherSurfacesFile,
                                String diagnosticsSurfacesFile,
                                String cfaSurfacesFile,
                                String newsPreviousCaseId,
                                boolean complex) {

        public ModelFixture modelFixture() {
            int scalarWidth = complex ? 2 : 1;
            ModelFixture fixture = complex
                ? ModelFixture.complex(shape.kEndog(), shape.kStates(), shape.kPosdef(), shape.nobs())
                : new ModelFixture(shape.kEndog(), shape.kStates(), shape.kPosdef(), shape.nobs());
            double[] obsIntercept = defaulted(model.obsIntercept(), scalarWidth * shape.kEndog());
            double[] stateIntercept = defaulted(model.stateIntercept(), scalarWidth * shape.kStates());
            boolean[][] missing = model.missing();
            for (int t = 0; t < shape.nobs(); t++) {
                fixture.setDesign(block(model.designBlocks(), model.design(), t), t);
                fixture.setObsIntercept(block(model.obsInterceptBlocks(), obsIntercept, t), t);
                fixture.setObsCov(block(model.obsCovBlocks(), model.obsCov(), t), t);
                fixture.setTransition(block(model.transitionBlocks(), model.transition(), t), t);
                fixture.setStateIntercept(block(model.stateInterceptBlocks(), stateIntercept, t), t);
                fixture.setSelection(block(model.selectionBlocks(), model.selection(), t), t);
                fixture.setStateCov(block(model.stateCovBlocks(), model.stateCov(), t), t);
                fixture.setEndog(model.endog()[t], t);
                if (missing != null) {
                    fixture.setMissing(missing[t], t);
                }
            }
            return fixture;
        }

        public InitialState initialState(KalmanSSM stateSpaceModel) {
            return switch (initialization.type()) {
                case "known" -> InitialState.known(initialization.state(), initialization.cov());
                case "stationary" -> InitialState.stationary(stateSpaceModel);
                case "diffuse" -> InitialState.diffuse(shape.kStates());
                case "approximate_diffuse" -> InitialState.approximateDiffuse(shape.kStates());
                default -> throw new IllegalArgumentException("Unsupported initialization: " + initialization.type());
            };
        }

        public FilterOptions filterOptions() {
            FilterMethod method = FilterMethod.valueOf(filter.method());
            FilterOptions.Builder builder;
            if (filter.fallbackMethods() == null || filter.fallbackMethods().length == 0) {
                builder = FilterOptions.builder().method(method);
            } else {
                FilterMethod[] fallbackMethods = new FilterMethod[filter.fallbackMethods().length];
                for (int i = 0; i < fallbackMethods.length; i++) {
                    fallbackMethods[i] = FilterMethod.valueOf(filter.fallbackMethods()[i]);
                }
                builder = FilterOptions.builder().tryMethods(method, fallbackMethods);
            }
            if (filter.timing() != null) {
                builder.timing(FilterOptions.Timing.valueOf(filter.timing()));
            }
            if (filter.retainedSurfaces() != null) {
                retainExactSurfaces(builder, filter.retainedSurfaces());
            } else if (filter.retainAll()) {
                builder.retainAll();
            } else {
                builder.retainNone();
            }
            if (filter.concentratedLikelihood()) {
                builder.concentratedLikelihood(true)
                    .concentratedLikelihoodBurn(filter.concentratedLikelihoodBurn());
            }
            return builder.build();
        }

        public boolean hasSmootherSurfaces() {
            return smootherSurfacesFile != null;
        }

        public boolean hasDiagnosticsSurfaces() {
            return diagnosticsSurfacesFile != null;
        }

        public boolean hasCfaSurfaces() {
            return cfaSurfacesFile != null;
        }

        public boolean hasNewsPreviousCase() {
            return newsPreviousCaseId != null;
        }

        public boolean hasEndogOverride() {
            return model.endogOverride() != null;
        }

        public SmootherOptions smootherOptions() {
            if (smoother == null) {
                throw new IllegalArgumentException("Fixture has no smoother spec: " + id);
            }
            return SmootherOptions.builder()
                .method(SmootherMethod.valueOf(smoother.method()))
                .build();
        }

        public double scalar(String name) {
            Double value = scalars.get(name);
            if (value == null) {
                throw new IllegalArgumentException("Missing scalar " + name + " for fixture " + id);
            }
            return value;
        }

        @Override
        public String toString() {
            return id;
        }

        private static double[] defaulted(double[] values, int length) {
            return values == null ? new double[length] : values;
        }

        private static double[] block(double[][] blocks, double[] fallback, int timeIndex) {
            return blocks == null ? fallback : blocks[timeIndex];
        }

        private static void retainExactSurfaces(FilterOptions.Builder builder, String[] retainedSurfaces) {
            if (retainedSurfaces.length == 0) {
                builder.retainNone();
                return;
            }
            FilterOptions.Surface first = FilterOptions.Surface.valueOf(retainedSurfaces[0]);
            FilterOptions.Surface[] rest = new FilterOptions.Surface[retainedSurfaces.length - 1];
            for (int i = 1; i < retainedSurfaces.length; i++) {
                rest[i - 1] = FilterOptions.Surface.valueOf(retainedSurfaces[i]);
            }
            builder.retainOnly(first, rest);
        }
    }

    public record Shape(int kEndog, int kStates, int kPosdef, int nobs) {
    }

    public record ModelSpec(double[] design,
                            double[][] designBlocks,
                            double[] obsIntercept,
                            double[][] obsInterceptBlocks,
                            double[] obsCov,
                            double[][] obsCovBlocks,
                            double[] transition,
                            double[][] transitionBlocks,
                            double[] stateIntercept,
                            double[][] stateInterceptBlocks,
                            double[] selection,
                            double[][] selectionBlocks,
                            double[] stateCov,
                            double[][] stateCovBlocks,
                            double[][] endog,
                            double[][] endogOverride,
                            boolean[][] missing) {
    }

    public record InitialSpec(String type, double[] state, double[] cov) {
    }

    public record FilterSpec(String method,
                             String[] fallbackMethods,
                             String timing,
                             boolean retainAll,
                             String[] retainedSurfaces,
                             boolean concentratedLikelihood,
                             int concentratedLikelihoodBurn) {
    }

    public record SmootherSpec(String method) {
    }

    public record SurfacePoint(String caseId,
                               String route,
                               String surface,
                               int t,
                               int row,
                               int col,
                               String valueType,
                               double value,
                               double valueImag) {

        public boolean complex() {
            return "COMPLEX".equals(valueType) || valueImag != 0.0;
        }
    }
}