package com.curioloop.yum4j.fft;

import java.util.concurrent.TimeUnit;

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
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})
@Threads(1)
public class FFT2DBenchmark {
    @State(Scope.Thread)
    public static class ComplexState {
        @Param({
            "32x32", "32x128", "128x32", "48x80", "80x48",
            "64x64", "64x96", "96x64", "64x127", "127x64",
            "64x257", "257x64", "96x128", "128x96", "120x120",
            "128x128", "128x257", "257x128", "192x192", "256x256"
        })
        String shape;

        int rows;
        int columns;
        int[] fftShape;
        int[] axes;
        long[] complexStride;
        Transform.PreparedNdComplex fft;
        FftWorkspace workspace;
        double[] template;
        double[] work;

        @Setup(Level.Trial)
        public void setupTrial() {
            parseShape(shape);
            fftShape = new int[] {rows, columns};
            axes = new int[] {0, 1};
            complexStride = new long[] {2L * columns, 2L};
            fft = Transform.prepareComplex(fftShape, axes);
            template = complexGrid(rows, columns);
            work = new double[template.length];
            workspace = fft.createWorkspace(complexStride, 0L, complexStride, 0L, work, work);
        }

        @Setup(Level.Invocation)
        public void setupInvocation() {
            System.arraycopy(template, 0, work, 0, template.length);
        }

        private void parseShape(String value) {
            int separator = value.indexOf('x');
            rows = Integer.parseInt(value.substring(0, separator));
            columns = Integer.parseInt(value.substring(separator + 1));
        }
    }

    @State(Scope.Thread)
    public static class RealState {
        @Param({
            "32x32", "32x128", "128x32", "48x80", "80x48",
            "64x64", "64x96", "96x64", "64x127", "127x64",
            "64x257", "257x64", "96x128", "128x96", "120x120",
            "128x128", "128x257", "257x128", "192x192", "256x256"
        })
        String shape;

        int rows;
        int columns;
        int hermitianColumns;
        int[] fftShape;
        int[] axes;
        long[] realStride;
        long[] hermitianStride;
        Transform.PreparedNdReal fft;
        FftWorkspace workspace;
        double[] inputTemplate;
        double[] input;
        double[] spectrum;
        double[] inverseSpectrumTemplate;
        double[] inverseSpectrum;
        double[] restored;

        @Setup(Level.Trial)
        public void setupTrial() {
            parseShape(shape);
            hermitianColumns = columns / 2 + 1;
            fftShape = new int[] {rows, columns};
            axes = new int[] {0, 1};
            realStride = new long[] {columns, 1L};
            hermitianStride = new long[] {2L * hermitianColumns, 2L};
            fft = Transform.prepareReal(fftShape, axes);
            inputTemplate = realGrid(rows, columns);
            input = new double[inputTemplate.length];
            spectrum = new double[rows * 2 * hermitianColumns];
            inverseSpectrumTemplate = new double[spectrum.length];
            restored = new double[inputTemplate.length];
            workspace = fft.r2cWorkspaceRequirement(realStride, 0L, hermitianStride, 0L, input, spectrum)
                    .max(fft.c2rWorkspaceRequirement(hermitianStride, 0L, realStride, 0L,
                            inverseSpectrumTemplate, restored))
                    .allocate();
            fft.r2c(realStride, 0L, hermitianStride, 0L, true, inputTemplate, inverseSpectrumTemplate, 1.0,
                    workspace);
            inverseSpectrum = new double[inverseSpectrumTemplate.length];
        }

        @Setup(Level.Invocation)
        public void setupInvocation() {
            System.arraycopy(inputTemplate, 0, input, 0, inputTemplate.length);
            System.arraycopy(inverseSpectrumTemplate, 0, inverseSpectrum, 0, inverseSpectrumTemplate.length);
        }

        private void parseShape(String value) {
            int separator = value.indexOf('x');
            rows = Integer.parseInt(value.substring(0, separator));
            columns = Integer.parseInt(value.substring(separator + 1));
        }
    }

    @State(Scope.Thread)
    public static class BatchedComplexState {
        @Param({"2x8x64", "4x8x128", "8x16x64", "2x16x257"})
        String shape;

        int batches;
        int rows;
        int columns;
        int[] fftShape;
        int[] axes;
        long[] complexStride;
        Transform.PreparedNdComplex fft;
        FftWorkspace workspace;
        double[] template;
        double[] work;

        @Setup(Level.Trial)
        public void setupTrial() {
            parseShape(shape);
            fftShape = new int[] {batches, rows, columns};
            axes = new int[] {1, 2};
            complexStride = new long[] {2L * rows * columns, 2L * columns, 2L};
            fft = Transform.prepareComplex(fftShape, axes);
            template = complexGrid(batches * rows, columns);
            work = new double[template.length];
            workspace = fft.createWorkspace(complexStride, 0L, complexStride, 0L, work, work);
        }

        @Setup(Level.Invocation)
        public void setupInvocation() {
            System.arraycopy(template, 0, work, 0, template.length);
        }

        private void parseShape(String value) {
            String[] parts = value.split("x");
            batches = Integer.parseInt(parts[0]);
            rows = Integer.parseInt(parts[1]);
            columns = Integer.parseInt(parts[2]);
        }
    }

    @State(Scope.Thread)
    public static class BatchedRealState {
        @Param({"2x8x64", "4x8x128", "8x16x64", "2x16x257"})
        String shape;

        int batches;
        int rows;
        int columns;
        int hermitianColumns;
        int[] fftShape;
        int[] axes;
        long[] realStride;
        long[] hermitianStride;
        Transform.PreparedNdReal fft;
        FftWorkspace workspace;
        double[] inputTemplate;
        double[] input;
        double[] spectrum;
        double[] inverseSpectrumTemplate;
        double[] inverseSpectrum;
        double[] restored;

        @Setup(Level.Trial)
        public void setupTrial() {
            parseShape(shape);
            hermitianColumns = columns / 2 + 1;
            fftShape = new int[] {batches, rows, columns};
            axes = new int[] {1, 2};
            realStride = new long[] {(long) rows * columns, columns, 1L};
            hermitianStride = new long[] {(long) rows * 2 * hermitianColumns, 2L * hermitianColumns, 2L};
            fft = Transform.prepareReal(fftShape, axes);
            inputTemplate = realGrid(batches * rows, columns);
            input = new double[inputTemplate.length];
            spectrum = new double[batches * rows * 2 * hermitianColumns];
            inverseSpectrumTemplate = new double[spectrum.length];
            restored = new double[inputTemplate.length];
            workspace = fft.r2cWorkspaceRequirement(realStride, 0L, hermitianStride, 0L, input, spectrum)
                    .max(fft.c2rWorkspaceRequirement(hermitianStride, 0L, realStride, 0L,
                            inverseSpectrumTemplate, restored))
                    .allocate();
            fft.r2c(realStride, 0L, hermitianStride, 0L, true, inputTemplate, inverseSpectrumTemplate, 1.0,
                    workspace);
            inverseSpectrum = new double[inverseSpectrumTemplate.length];
        }

        @Setup(Level.Invocation)
        public void setupInvocation() {
            System.arraycopy(inputTemplate, 0, input, 0, inputTemplate.length);
            System.arraycopy(inverseSpectrumTemplate, 0, inverseSpectrum, 0, inverseSpectrumTemplate.length);
        }

        private void parseShape(String value) {
            String[] parts = value.split("x");
            batches = Integer.parseInt(parts[0]);
            rows = Integer.parseInt(parts[1]);
            columns = Integer.parseInt(parts[2]);
        }
    }

    @Benchmark
    public void fft2dC2cForward(ComplexState state, Blackhole blackhole) {
        state.fft.c2c(state.complexStride, 0L, state.complexStride, 0L, true, state.work, state.work, 1.0,
                state.workspace);
        blackhole.consume(state.work);
    }

    @Benchmark
    public void fft2dR2cForward(RealState state, Blackhole blackhole) {
        state.fft.r2c(state.realStride, 0L, state.hermitianStride, 0L, true, state.input, state.spectrum,
                1.0, state.workspace);
        blackhole.consume(state.spectrum);
    }

    @Benchmark
    public void fft2dC2rInverse(RealState state, Blackhole blackhole) {
        state.fft.c2r(state.hermitianStride, 0L, state.realStride, 0L, false, state.inverseSpectrum,
                state.restored, 1.0 / (state.rows * state.columns), state.workspace);
        blackhole.consume(state.restored);
    }

    @Benchmark
    public void fftBatched2dC2cForward(BatchedComplexState state, Blackhole blackhole) {
        state.fft.c2c(state.complexStride, 0L, state.complexStride, 0L, true, state.work, state.work, 1.0,
                state.workspace);
        blackhole.consume(state.work);
    }

    @Benchmark
    public void fftBatched2dR2cForward(BatchedRealState state, Blackhole blackhole) {
        state.fft.r2c(state.realStride, 0L, state.hermitianStride, 0L, true, state.input, state.spectrum,
                1.0, state.workspace);
        blackhole.consume(state.spectrum);
    }

    @Benchmark
    public void fftBatched2dC2rInverse(BatchedRealState state, Blackhole blackhole) {
        state.fft.c2r(state.hermitianStride, 0L, state.realStride, 0L, false, state.inverseSpectrum,
                state.restored, 1.0 / (state.rows * state.columns), state.workspace);
        blackhole.consume(state.restored);
    }

    private static double[] complexGrid(int rows, int columns) {
        double[] data = new double[2 * rows * columns];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                int offset = 2 * (row * columns + column);
                data[offset] = 0.25 + 0.5 * row + 0.125 * ((7 * column + row) % 11);
                data[offset + 1] = -0.375 + 0.0625 * ((5 * row + 3 * column) % 13);
            }
        }
        return data;
    }

    private static double[] realGrid(int rows, int columns) {
        double[] data = new double[rows * columns];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                data[row * columns + column] = 0.25 + 0.5 * row - 0.125 * column
                        + 0.03125 * ((row + 3 * column) % 5);
            }
        }
        return data;
    }
}