package com.curioloop.yum4j.fft;

import java.util.Arrays;
import java.util.Objects;

/**
 * Source-level Java port target for PocketFFT ND execution helpers.
 *
 * <p>Traceability map to {@code pocketfft_hdronly.h}: {@code c2cPrepared},
 * {@code r2rFftpackPrepared}, {@code dcstPrepared}, and {@code separableHartleyPrepared} are the
 * scalar Java expansion of {@code general_nd} with {@code ExecC2C}, {@code ExecR2R},
 * {@code ExecDcst}, {@code ExecHartley}, and {@code ExecFHT}; {@code transformRealToComplexAxis}
 * and {@code transformComplexToRealAxis} expand {@code general_r2c} and {@code general_c2r}.
 * Java boundary adaptations are limited to primitive {@code double[]} slots, explicit workspace
 * ownership, and prebuilt DCT/DST plans that validate {@code type}/{@code cosine} at execution.</p>
 */
final class FftNd {
    // Keep this in sync with the workspace sizing helpers below. A batch of four matches the
    // JTransforms 2D column strategy and was the fastest stable choice in the local JMH matrix.
    private static final int FAST_COLUMN_BATCH = 4;

    private FftNd() {
    }

    static void c2c(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                    boolean forward, double[] input, double[] output, double factor) {
        // This entry point knows the concrete layout, so it can allocate the compact requirement
        // instead of the layout-agnostic prepared-plan upper bound.
        FftWorkspace workspace = c2cWorkspaceRequirement(shape, strideIn, offsetIn, strideOut, offsetOut, axes,
                input, output).allocate();
        workspace.reset();
        c2cPrepared(shape, strideIn, offsetIn, strideOut, offsetOut, axes, null, forward, input, output, factor,
                workspace);
    }

    static void c2c(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                    boolean forward, double[] input, double[] output, double factor, FftWorkspace workspace) {
        validateCommon(shape, strideIn, offsetIn, strideOut, offsetOut, axes, input, output, 2);
        c2cPrepared(shape, strideIn, offsetIn, strideOut, offsetOut, axes, null, forward, input, output, factor,
                workspace);
    }

    static void c2cPrepared(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                            FftComplexTransform[] transforms, boolean forward, double[] input, double[] output,
                            double factor, FftWorkspace workspace) {
        if (hasZeroExtent(shape) || axes.length == 0) {
            return;
        }
        // Fast path: standard in-place 2D complex planes. The predicate deliberately accepts
        // rank > 2; dimensions outside the two axes are independent batch dimensions handled by
        // fastBatched2dC2c without increasing the workspace peak.
        if (axes.length == 2 && input == output && offsetIn == offsetOut
                && java.util.Arrays.equals(strideIn, strideOut)
                && canUseFast2dC2c(shape, strideOut, axes)) {
            fastBatched2dC2c(shape, strideOut, (int) offsetOut, axes, transforms, forward, output, factor,
                    workspace);
            return;
        }
        // Fast path: single non-contiguous axis in a 2D layout, e.g. the complex column pass
        // reached by real-to-complex or complex-to-real ND transforms.
        if (shape.length == 2 && axes.length == 1 && input == output && offsetIn == offsetOut
                && java.util.Arrays.equals(strideIn, strideOut)
                && canUseFastColumnPass(shape, strideOut, axes[0])) {
            fastColumnPass(shape, strideOut, (int) offsetOut, axes[0], transforms != null ? transforms[0] : null,
                forward, output, factor, workspace);
            return;
        }
        double currentFactor = factor;
        for (int axisIndex = 0; axisIndex < axes.length; axisIndex++) {
            double[] source = axisIndex == 0 ? input : output;
            long[] sourceStride = axisIndex == 0 ? strideIn : strideOut;
            long sourceOffset = axisIndex == 0 ? offsetIn : offsetOut;
            FftComplexTransform transform = transforms == null
                    ? FftPlanCache.complexTransform(shape[axes[axisIndex]])
                    : transforms[axisIndex];
            transformComplexAxis(shape, sourceStride, sourceOffset, strideOut, offsetOut, axes[axisIndex], forward,
                    currentFactor, source, output, transform, workspace);
            currentFactor = 1.0;
        }
    }

    /**
        * Check if the fast 2D C2C path can be used on the selected axes.
        * Requires standard row-major interleaved complex planes:
        * stride[lastAxis]=2, stride[firstAxis]=2*shape[lastAxis]. Other dimensions, if present,
        * are batch dimensions and are intentionally ignored here.
     */
    private static boolean canUseFast2dC2c(int[] shape, long[] stride, int[] axes) {
        int lastAxis = axes[axes.length - 1];
        int firstAxis = axes[0];
        if (stride[lastAxis] != 2L) return false;
        if (stride[firstAxis] != 2L * shape[lastAxis]) return false;
        return true;
    }

    /**
     * Check if the fast single-axis column pass can be used.
     * Requires: 2D shape, the axis is non-contiguous, the other axis has stride=2.
     */
    private static boolean canUseFastColumnPass(int[] shape, long[] stride, int axis) {
        if (stride[axis] <= 2L) return false;
        // The other axis must be contiguous (stride=2 for interleaved complex)
        int otherAxis = 1 - axis;
        if (stride[otherAxis] != 2L) return false;
        return true;
    }

    private static boolean canUseFast2dR2c(int[] shapeIn, int[] shapeOut, long[] strideIn, long[] strideOut,
                                           int[] axes) {
        if (axes.length != 2) {
            return false;
        }
        int contiguousAxis = axes[axes.length - 1];
        int nonContiguousAxis = axes[0];
        // Only the selected plane layout matters; any remaining dimensions are processed as a
        // sequential batch by fastBatched2dR2c and do not require additional scratch.
        return strideIn[contiguousAxis] == 1L
                && strideIn[nonContiguousAxis] == shapeIn[contiguousAxis]
                && strideOut[contiguousAxis] == 2L
                && strideOut[nonContiguousAxis] == 2L * shapeOut[contiguousAxis];
    }

    private static boolean canUseFast2dC2r(int[] shapeOut, int[] shapeIn, long[] strideIn, long[] strideOut,
                                           int[] axes) {
        if (axes.length != 2) {
            return false;
        }
        int contiguousAxis = axes[axes.length - 1];
        int nonContiguousAxis = axes[0];
        // The Hermitian input and real output must each be standard row-major within the selected
        // plane. Outer dimensions are treated as independent planes with separate base offsets.
        return strideIn[contiguousAxis] == 2L
                && strideIn[nonContiguousAxis] == 2L * shapeIn[contiguousAxis]
                && strideOut[contiguousAxis] == 1L
                && strideOut[nonContiguousAxis] == shapeOut[contiguousAxis];
    }

    /**
     * Fast single-axis column pass for 2D complex data.
     * Processes 4 columns at a time with batched gather/scatter. The temporary storage lives in
     * the caller workspace so prepared transforms can reuse the same arena across invocations.
     */
    private static void fastColumnPass(int[] shape, long[] stride, int offset, int axis,
                                       FftComplexTransform transform, boolean forward,
                                       double[] data, double factor, FftWorkspace workspace) {
        int rows = shape[axis];
        int otherAxis = 1 - axis;
        int cols = shape[otherAxis];
        int rowStride = (int) stride[axis];
        fastColumnPass(rows, cols, rowStride, offset, transform, forward, data, factor, workspace);
    }

    private static void fastColumnPass(int rows, int cols, int rowStride, int offset,
                                       FftComplexTransform transform, boolean forward,
                                       double[] data, double factor, FftWorkspace workspace) {

        FftComplexTransform rowTransform = transform != null ? transform
                : FftPlanCache.complexTransform(rows);

        int baseMark = workspace.mark();
        double[] t = workspace.doubles();
        int tempOffset = workspace.alloc(fastColumnBufferDoubles(rows, cols));
        int transformMark = workspace.mark();
        int col = 0;
        for (; col + FAST_COLUMN_BATCH <= cols; col += FAST_COLUMN_BATCH) {
            int colBase = offset + 2 * col;
            for (int r = 0; r < rows; r++) {
                int src = colBase + r * rowStride;
                int t0 = tempOffset + 2 * r;
                int t1 = tempOffset + 2 * rows + 2 * r;
                int t2 = tempOffset + 4 * rows + 2 * r;
                int t3 = tempOffset + 6 * rows + 2 * r;
                t[t0]     = data[src];
                t[t0 + 1] = data[src + 1];
                t[t1]     = data[src + 2];
                t[t1 + 1] = data[src + 3];
                t[t2]     = data[src + 4];
                t[t2 + 1] = data[src + 5];
                t[t3]     = data[src + 6];
                t[t3 + 1] = data[src + 7];
            }
            workspace.rewind(transformMark);
            rowTransform.exec(t, tempOffset, factor, forward, workspace);
            workspace.rewind(transformMark);
            rowTransform.exec(t, tempOffset + 2 * rows, factor, forward, workspace);
            workspace.rewind(transformMark);
            rowTransform.exec(t, tempOffset + 4 * rows, factor, forward, workspace);
            workspace.rewind(transformMark);
            rowTransform.exec(t, tempOffset + 6 * rows, factor, forward, workspace);
            for (int r = 0; r < rows; r++) {
                int dst = colBase + r * rowStride;
                int t0 = tempOffset + 2 * r;
                int t1 = tempOffset + 2 * rows + 2 * r;
                int t2 = tempOffset + 4 * rows + 2 * r;
                int t3 = tempOffset + 6 * rows + 2 * r;
                data[dst]     = t[t0];
                data[dst + 1] = t[t0 + 1];
                data[dst + 2] = t[t1];
                data[dst + 3] = t[t1 + 1];
                data[dst + 4] = t[t2];
                data[dst + 5] = t[t2 + 1];
                data[dst + 6] = t[t3];
                data[dst + 7] = t[t3 + 1];
            }
        }
        for (; col < cols; col++) {
            int colBase = offset + 2 * col;
            for (int r = 0; r < rows; r++) {
                int src = colBase + r * rowStride;
                int ti = tempOffset + 2 * r;
                t[ti]     = data[src];
                t[ti + 1] = data[src + 1];
            }
            workspace.rewind(transformMark);
            rowTransform.exec(t, tempOffset, factor, forward, workspace);
            for (int r = 0; r < rows; r++) {
                int dst = colBase + r * rowStride;
                int ti = tempOffset + 2 * r;
                data[dst]     = t[ti];
                data[dst + 1] = t[ti + 1];
            }
        }
        workspace.rewind(baseMark);
    }

    /**
     * Batched 2D fast paths reuse the single-plane kernels and advance only base offsets between
     * planes. Rank-3 standard layouts stay on scalar loops to avoid iterator/array allocation;
     * higher-rank layouts fall back to {@link PlaneIter} while keeping the same single-plane
     * workspace peak.
     */
    private static void fastBatched2dC2c(int[] shape, long[] stride, int offset, int[] axes,
                                         FftComplexTransform[] transforms, boolean forward, double[] data,
                                         double factor, FftWorkspace workspace) {
        int nonContiguousAxis = axes[0];
        int contiguousAxis = axes[1];
        int rows = shape[nonContiguousAxis];
        int cols = shape[contiguousAxis];
        int rowStride = Math.toIntExact(stride[nonContiguousAxis]);
        if (shape.length == 2) {
            fast2dC2c(rows, cols, rowStride, offset, transforms, forward, data, factor, workspace);
            return;
        }
        if (shape.length == 3) {
            // Common hot layout: {batch, rows, cols}. Keep this branch allocation-free instead of
            // constructing a PlaneIter for every prepared-transform invocation.
            int batchAxis = 3 - nonContiguousAxis - contiguousAxis;
            int planeOffset = offset;
            int step = Math.toIntExact(stride[batchAxis]);
            for (int batch = 0; batch < shape[batchAxis]; batch++) {
                fast2dC2c(rows, cols, rowStride, planeOffset, transforms, forward, data, factor, workspace);
                planeOffset += step;
            }
            return;
        }
        int[] planeShape = {rows, cols};
        long[] planeStride = {rowStride, 2L};
        int[] planeAxes = {0, 1};
        // Less common higher-rank layouts still share the same single-plane kernel; PlaneIter only
        // advances base offsets in dimensions outside the selected 2D plane.
        PlaneIter planes = new PlaneIter(shape, shape, stride, offset, stride, offset, nonContiguousAxis,
                contiguousAxis);
        while (planes.remaining() > 0) {
            planes.advance();
            fast2dC2c(planeShape, planeStride, planes.outputBase(), planeAxes, transforms, forward, data, factor,
                    workspace);
        }
    }

    private static void fastBatched2dR2c(int[] shapeIn, int[] shapeOut, long[] strideIn, int offsetIn,
                                         long[] strideOut, int offsetOut, int[] axes,
                                         FftRealTransform realTransform, FftComplexTransform[] complexTransforms,
                                         boolean forward, double[] input, double[] output, double factor,
                                         FftWorkspace workspace) {
        int nonContiguousAxis = axes[0];
        int contiguousAxis = axes[1];
        int rows = shapeIn[nonContiguousAxis];
        int cols = shapeIn[contiguousAxis];
        int hermitianColumns = shapeOut[contiguousAxis];
        int realRowStride = Math.toIntExact(strideIn[nonContiguousAxis]);
        int complexRowStride = Math.toIntExact(strideOut[nonContiguousAxis]);
        if (shapeIn.length == 2) {
            fast2dR2c(rows, cols, hermitianColumns, realRowStride, complexRowStride, offsetIn, offsetOut,
                    realTransform, complexTransforms, forward, input, output, factor, workspace);
            return;
        }
        if (shapeIn.length == 3) {
            // Rank-3 batched real transforms are common in simulations and time-series pipelines;
            // keep the loop scalar so the fast path remains allocation-free.
            int batchAxis = 3 - nonContiguousAxis - contiguousAxis;
            int inputBase = offsetIn;
            int outputBase = offsetOut;
            int inputStep = Math.toIntExact(strideIn[batchAxis]);
            int outputStep = Math.toIntExact(strideOut[batchAxis]);
            for (int batch = 0; batch < shapeIn[batchAxis]; batch++) {
                fast2dR2c(rows, cols, hermitianColumns, realRowStride, complexRowStride, inputBase, outputBase,
                        realTransform, complexTransforms, forward, input, output, factor, workspace);
                inputBase += inputStep;
                outputBase += outputStep;
            }
            return;
        }
        int[] planeShapeIn = {rows, cols};
        int[] planeShapeOut = {rows, hermitianColumns};
        long[] planeStrideIn = {realRowStride, 1L};
        long[] planeStrideOut = {complexRowStride, 2L};
        int[] planeAxes = {0, 1};
        // Higher-rank real batches reuse the same compact R2C staging per plane; the iterator does
        // not own scratch, so workspace sizing remains the single-plane peak.
        PlaneIter planes = new PlaneIter(shapeIn, shapeOut, strideIn, offsetIn, strideOut, offsetOut,
                nonContiguousAxis, contiguousAxis);
        while (planes.remaining() > 0) {
            planes.advance();
            fast2dR2c(planeShapeIn, planeShapeOut, planeStrideIn, planes.inputBase(), planeStrideOut,
                    planes.outputBase(), planeAxes, realTransform, complexTransforms, forward, input, output, factor,
                    workspace);
        }
    }

    private static void fastBatched2dC2r(int[] shapeOut, int[] shapeIn, long[] strideIn, int offsetIn,
                                         long[] strideOut, int offsetOut, int[] axes,
                                         FftRealTransform realTransform, FftComplexTransform[] complexTransforms,
                                         boolean forward, double[] input, double[] output, double factor,
                                         FftWorkspace workspace) {
        int nonContiguousAxis = axes[0];
        int contiguousAxis = axes[1];
        int rows = shapeOut[nonContiguousAxis];
        int realLength = shapeOut[contiguousAxis];
        int hermitianColumns = shapeIn[contiguousAxis];
        int complexRowStride = Math.toIntExact(strideIn[nonContiguousAxis]);
        int realRowStride = Math.toIntExact(strideOut[nonContiguousAxis]);
        if (shapeOut.length == 2) {
            fast2dC2r(rows, realLength, hermitianColumns, complexRowStride, realRowStride, offsetIn, offsetOut,
                    realTransform, complexTransforms, forward, input, output, factor, workspace);
            return;
        }
        if (shapeOut.length == 3) {
            // Mirror the R2C rank-3 path: each Hermitian plane is converted back to its real plane
            // before advancing the two base offsets by their respective batch strides.
            int batchAxis = 3 - nonContiguousAxis - contiguousAxis;
            int inputBase = offsetIn;
            int outputBase = offsetOut;
            int inputStep = Math.toIntExact(strideIn[batchAxis]);
            int outputStep = Math.toIntExact(strideOut[batchAxis]);
            for (int batch = 0; batch < shapeOut[batchAxis]; batch++) {
                fast2dC2r(rows, realLength, hermitianColumns, complexRowStride, realRowStride, inputBase,
                        outputBase, realTransform, complexTransforms, forward, input, output, factor, workspace);
                inputBase += inputStep;
                outputBase += outputStep;
            }
            return;
        }
        int[] planeShapeOut = {rows, realLength};
        int[] planeShapeIn = {rows, hermitianColumns};
        long[] planeStrideIn = {complexRowStride, 2L};
        long[] planeStrideOut = {realRowStride, 1L};
        int[] planeAxes = {0, 1};
        // The input and output shapes differ on the Hermitian axis, so PlaneIter tracks both base
        // offsets while skipping the two axes that make up the active plane.
        PlaneIter planes = new PlaneIter(shapeIn, shapeOut, strideIn, offsetIn, strideOut, offsetOut,
                nonContiguousAxis, contiguousAxis);
        while (planes.remaining() > 0) {
            planes.advance();
            fast2dC2r(planeShapeOut, planeShapeIn, planeStrideIn, planes.inputBase(), planeStrideOut,
                    planes.outputBase(), planeAxes, realTransform, complexTransforms, forward, input, output, factor,
                    workspace);
        }
    }

    /**
     * JTransforms-style fast 2D complex FFT for standard row-major interleaved layout.
     * <ul>
     *   <li>Contiguous axis: direct in-place FFT per row (zero copy)</li>
     *   <li>Non-contiguous axis: 4-column batched gather → FFT × 4 → scatter</li>
     * </ul>
     * <p>The two phases reuse the same workspace arena sequentially. Peak memory is the larger of
     * the contiguous-axis transform scratch and the column batch buffer plus row-transform scratch.</p>
     */
    private static void fast2dC2c(int[] shape, long[] stride, int offset, int[] axes,
                                  FftComplexTransform[] transforms, boolean forward, double[] data, double factor,
                                  FftWorkspace workspace) {
        int contiguousAxis = axes[axes.length - 1];
        int nonContiguousAxis = axes[0];
        int cols = shape[contiguousAxis];   // FFT length along contiguous axis
        int rows = shape[nonContiguousAxis]; // FFT length along non-contiguous axis
        int rowStride = (int) stride[nonContiguousAxis]; // in doubles

        FftComplexTransform colTransform = transforms != null ? transforms[axes.length - 1]
                : FftPlanCache.complexTransform(cols);
        FftComplexTransform rowTransform = transforms != null ? transforms[0]
                : FftPlanCache.complexTransform(rows);

        // Phase 1: contiguous axis — direct in-place FFT per row, apply factor here
        int phaseMark = workspace.mark();
        for (int r = 0; r < rows; r++) {
            workspace.rewind(phaseMark);
            colTransform.exec(data, offset + r * rowStride, factor, forward, workspace);
        }
        workspace.rewind(phaseMark);

        // Phase 2: non-contiguous axis — 4-column batched transpose
        int baseMark = workspace.mark();
        double[] t = workspace.doubles();
        int tempOffset = workspace.alloc(fastColumnBufferDoubles(rows, cols));
        int transformMark = workspace.mark();
        int col = 0;
        for (; col + FAST_COLUMN_BATCH <= cols; col += FAST_COLUMN_BATCH) {
            // Gather 4 columns
            int colBase = offset + 2 * col;
            for (int r = 0; r < rows; r++) {
                int src = colBase + r * rowStride;
                int t0 = tempOffset + 2 * r;
                int t1 = tempOffset + 2 * rows + 2 * r;
                int t2 = tempOffset + 4 * rows + 2 * r;
                int t3 = tempOffset + 6 * rows + 2 * r;
                t[t0]     = data[src];
                t[t0 + 1] = data[src + 1];
                t[t1]     = data[src + 2];
                t[t1 + 1] = data[src + 3];
                t[t2]     = data[src + 4];
                t[t2 + 1] = data[src + 5];
                t[t3]     = data[src + 6];
                t[t3 + 1] = data[src + 7];
            }
            // 4 FFTs
            workspace.rewind(transformMark);
            rowTransform.exec(t, tempOffset, 1.0, forward, workspace);
            workspace.rewind(transformMark);
            rowTransform.exec(t, tempOffset + 2 * rows, 1.0, forward, workspace);
            workspace.rewind(transformMark);
            rowTransform.exec(t, tempOffset + 4 * rows, 1.0, forward, workspace);
            workspace.rewind(transformMark);
            rowTransform.exec(t, tempOffset + 6 * rows, 1.0, forward, workspace);
            // Scatter 4 columns back
            for (int r = 0; r < rows; r++) {
                int dst = colBase + r * rowStride;
                int t0 = tempOffset + 2 * r;
                int t1 = tempOffset + 2 * rows + 2 * r;
                int t2 = tempOffset + 4 * rows + 2 * r;
                int t3 = tempOffset + 6 * rows + 2 * r;
                data[dst]     = t[t0];
                data[dst + 1] = t[t0 + 1];
                data[dst + 2] = t[t1];
                data[dst + 3] = t[t1 + 1];
                data[dst + 4] = t[t2];
                data[dst + 5] = t[t2 + 1];
                data[dst + 6] = t[t3];
                data[dst + 7] = t[t3 + 1];
            }
        }
        // Remaining columns (< 4)
        for (; col < cols; col++) {
            int colBase = offset + 2 * col;
            for (int r = 0; r < rows; r++) {
                int src = colBase + r * rowStride;
                int ti = tempOffset + 2 * r;
                t[ti]     = data[src];
                t[ti + 1] = data[src + 1];
            }
            workspace.rewind(transformMark);
            rowTransform.exec(t, tempOffset, 1.0, forward, workspace);
            for (int r = 0; r < rows; r++) {
                int dst = colBase + r * rowStride;
                int ti = tempOffset + 2 * r;
                data[dst]     = t[ti];
                data[dst + 1] = t[ti + 1];
            }
        }
        workspace.rewind(baseMark);
    }

    private static void fast2dC2c(int rows, int cols, int rowStride, int offset,
                                  FftComplexTransform[] transforms, boolean forward, double[] data, double factor,
                                  FftWorkspace workspace) {
        FftComplexTransform colTransform = transforms != null ? transforms[1]
                : FftPlanCache.complexTransform(cols);
        FftComplexTransform rowTransform = transforms != null ? transforms[0]
                : FftPlanCache.complexTransform(rows);

        int phaseMark = workspace.mark();
        for (int row = 0; row < rows; row++) {
            workspace.rewind(phaseMark);
            colTransform.exec(data, offset + row * rowStride, factor, forward, workspace);
        }
        workspace.rewind(phaseMark);
        fastColumnPass(rows, cols, rowStride, offset, rowTransform, forward, data, 1.0, workspace);
    }

    private static void fast2dR2c(int[] shapeIn, int[] shapeOut, long[] strideIn, int offsetIn, long[] strideOut,
                                  int offsetOut, int[] axes, FftRealTransform realTransform,
                                  FftComplexTransform[] complexTransforms, boolean forward, double[] input,
                                  double[] output, double factor, FftWorkspace workspace) {
        int contiguousAxis = axes[axes.length - 1];
        int nonContiguousAxis = axes[0];
        int rows = shapeIn[nonContiguousAxis];
        int cols = shapeIn[contiguousAxis];
        int realRowStride = (int) strideIn[nonContiguousAxis];
        int complexRowStride = (int) strideOut[nonContiguousAxis];

        // Stage the real-axis transform directly into the Hermitian output row. Split-radix rows
        // can write interleaved complex bins immediately; other plans fall back to FFTPACK
        // half-complex output and expand in the same row.
        int phaseMark = workspace.mark();
        for (int row = 0; row < rows; row++) {
            workspace.rewind(phaseMark);
            int source = offsetIn + row * realRowStride;
            int target = offsetOut + row * complexRowStride;
            System.arraycopy(input, source, output, target, cols);
            if (!realTransform.tryRealToHermitianInPlace(output, target, factor, forward, workspace)) {
                realTransform.exec(output, target, factor, true, workspace);
                expandHalfComplexToComplexInPlace(output, target, cols, forward);
            }
        }
        workspace.rewind(phaseMark);

        int[] complexAxes = Arrays.copyOf(axes, axes.length - 1);
        c2cPrepared(shapeOut, strideOut, offsetOut, strideOut, offsetOut, complexAxes, complexTransforms,
                forward, output, output, 1.0, workspace);
    }

    private static void fast2dR2c(int rows, int cols, int hermitianColumns, int realRowStride,
                                  int complexRowStride, int offsetIn, int offsetOut,
                                  FftRealTransform realTransform, FftComplexTransform[] complexTransforms,
                                  boolean forward, double[] input, double[] output, double factor,
                                  FftWorkspace workspace) {
        int phaseMark = workspace.mark();
        for (int row = 0; row < rows; row++) {
            workspace.rewind(phaseMark);
            int source = offsetIn + row * realRowStride;
            int target = offsetOut + row * complexRowStride;
            System.arraycopy(input, source, output, target, cols);
            if (!realTransform.tryRealToHermitianInPlace(output, target, factor, forward, workspace)) {
                realTransform.exec(output, target, factor, true, workspace);
                expandHalfComplexToComplexInPlace(output, target, cols, forward);
            }
        }
        workspace.rewind(phaseMark);

        FftComplexTransform rowTransform = complexTransforms != null && complexTransforms.length > 0
                ? complexTransforms[0]
                : FftPlanCache.complexTransform(rows);
        fastColumnPass(rows, hermitianColumns, complexRowStride, offsetOut, rowTransform, forward, output, 1.0,
                workspace);
    }

    private static void fast2dC2r(int[] shapeOut, int[] shapeIn, long[] strideIn, int offsetIn, long[] strideOut,
                                  int offsetOut, int[] axes, FftRealTransform realTransform,
                                  FftComplexTransform[] complexTransforms, boolean forward, double[] input,
                                  double[] output, double factor, FftWorkspace workspace) {
        int contiguousAxis = axes[axes.length - 1];
        int nonContiguousAxis = axes[0];
        int rows = shapeOut[nonContiguousAxis];
        int hermitianColumns = shapeIn[contiguousAxis];
        int realLength = shapeOut[contiguousAxis];
        int complexRowStride = (int) strideIn[nonContiguousAxis];
        int realRowStride = (int) strideOut[nonContiguousAxis];
        fast2dC2r(rows, realLength, hermitianColumns, complexRowStride, realRowStride, offsetIn, offsetOut,
                realTransform, complexTransforms, forward, input, output, factor, workspace);
    }

    private static void fast2dC2r(int rows, int realLength, int hermitianColumns, int complexRowStride,
                                  int realRowStride, int offsetIn, int offsetOut,
                                  FftRealTransform realTransform, FftComplexTransform[] complexTransforms,
                                  boolean forward, double[] input, double[] output, double factor,
                                  FftWorkspace workspace) {
        FftComplexTransform rowTransform = complexTransforms != null && complexTransforms.length > 0
                ? complexTransforms[0]
                : FftPlanCache.complexTransform(rows);

        int baseMark = workspace.mark();
        double[] t = workspace.doubles();
        int tempOffset = workspace.alloc(fastColumnBufferDoubles(rows, hermitianColumns));
        int transformMark = workspace.mark();
        int column = 0;
        // Transform Hermitian columns in B4 chunks, then pack each chunk directly into FFTPACK
        // half-complex rows for the final real-axis inverse. This avoids the full complex temp.
        for (; column + FAST_COLUMN_BATCH <= hermitianColumns; column += FAST_COLUMN_BATCH) {
            int columnBase = offsetIn + 2 * column;
            for (int row = 0; row < rows; row++) {
                int source = columnBase + row * complexRowStride;
                int t0 = tempOffset + 2 * row;
                int t1 = tempOffset + 2 * rows + 2 * row;
                int t2 = tempOffset + 4 * rows + 2 * row;
                int t3 = tempOffset + 6 * rows + 2 * row;
                t[t0]     = input[source];
                t[t0 + 1] = input[source + 1];
                t[t1]     = input[source + 2];
                t[t1 + 1] = input[source + 3];
                t[t2]     = input[source + 4];
                t[t2 + 1] = input[source + 5];
                t[t3]     = input[source + 6];
                t[t3 + 1] = input[source + 7];
            }
            workspace.rewind(transformMark);
            rowTransform.exec(t, tempOffset, 1.0, forward, workspace);
            workspace.rewind(transformMark);
            rowTransform.exec(t, tempOffset + 2 * rows, 1.0, forward, workspace);
            workspace.rewind(transformMark);
            rowTransform.exec(t, tempOffset + 4 * rows, 1.0, forward, workspace);
            workspace.rewind(transformMark);
            rowTransform.exec(t, tempOffset + 6 * rows, 1.0, forward, workspace);
            scatterHermitianColumnBatch4ToHalfComplex(t, tempOffset, rows, column, realLength, output, offsetOut,
                    realRowStride, forward);
        }
        for (; column < hermitianColumns; column++) {
            int columnBase = offsetIn + 2 * column;
            for (int row = 0; row < rows; row++) {
                int source = columnBase + row * complexRowStride;
                int ti = tempOffset + 2 * row;
                t[ti]     = input[source];
                t[ti + 1] = input[source + 1];
            }
            workspace.rewind(transformMark);
            rowTransform.exec(t, tempOffset, 1.0, forward, workspace);
            scatterHermitianColumnToHalfComplex(t, tempOffset, rows, column, realLength, output, offsetOut,
                    realRowStride, forward);
        }
        workspace.rewind(baseMark);

        int phaseMark = workspace.mark();
        for (int row = 0; row < rows; row++) {
            workspace.rewind(phaseMark);
            realTransform.exec(output, offsetOut + row * realRowStride, factor, false, workspace);
        }
        workspace.rewind(phaseMark);
    }

    private static void expandHalfComplexToComplexInPlace(double[] data, int offset, int length, boolean forward) {
        double imagSign = forward ? 1.0 : -1.0;
        int hermitianColumns = length / 2 + 1;
        for (int column = hermitianColumns - 1; column >= 1; column--) {
            int target = offset + 2 * column;
            if (2 * column == length) {
                data[target] = data[offset + length - 1];
                data[target + 1] = 0.0;
            } else {
                int source = offset + 2 * column - 1;
                double real = data[source];
                double imaginary = data[source + 1];
                data[target] = real;
                data[target + 1] = imagSign * imaginary;
            }
        }
        data[offset + 1] = 0.0;
    }

    private static void scatterHermitianColumnToHalfComplex(double[] source, int sourceOffset, int rows, int column,
                                                            int realLength, double[] output, int outputOffset,
                                                            int rowStride, boolean forward) {
        double imagSign = forward ? -1.0 : 1.0;
        for (int row = 0; row < rows; row++) {
            int sourceIndex = sourceOffset + 2 * row;
            int rowOffset = outputOffset + row * rowStride;
            if (column == 0) {
                output[rowOffset] = source[sourceIndex];
            } else if (2 * column == realLength) {
                output[rowOffset + realLength - 1] = source[sourceIndex];
            } else {
                int target = rowOffset + 2 * column - 1;
                output[target] = source[sourceIndex];
                output[target + 1] = imagSign * source[sourceIndex + 1];
            }
        }
    }

    private static void scatterHermitianColumnBatch4ToHalfComplex(double[] source, int sourceOffset, int rows,
                                                                  int column, int realLength, double[] output,
                                                                  int outputOffset, int rowStride,
                                                                  boolean forward) {
        double imagSign = forward ? -1.0 : 1.0;
        int column1 = column + 1;
        int column2 = column + 2;
        int column3 = column + 3;
        for (int row = 0; row < rows; row++) {
            int rowOffset = outputOffset + row * rowStride;
            scatterHermitianValueToHalfComplex(source, sourceOffset + 2 * row, column, realLength, output,
                    rowOffset, imagSign);
            scatterHermitianValueToHalfComplex(source, sourceOffset + 2 * rows + 2 * row, column1, realLength,
                    output, rowOffset, imagSign);
            scatterHermitianValueToHalfComplex(source, sourceOffset + 4 * rows + 2 * row, column2, realLength,
                    output, rowOffset, imagSign);
            scatterHermitianValueToHalfComplex(source, sourceOffset + 6 * rows + 2 * row, column3, realLength,
                    output, rowOffset, imagSign);
        }
    }

    private static void scatterHermitianValueToHalfComplex(double[] source, int sourceIndex, int column,
                                                           int realLength, double[] output, int rowOffset,
                                                           double imagSign) {
        if (column == 0) {
            output[rowOffset] = source[sourceIndex];
        } else if (2 * column == realLength) {
            output[rowOffset + realLength - 1] = source[sourceIndex];
        } else {
            int target = rowOffset + 2 * column - 1;
            output[target] = source[sourceIndex];
            output[target + 1] = imagSign * source[sourceIndex + 1];
        }
    }

    private static int fastColumnBufferDoubles(int rows, int cols) {
        int batch = Math.min(FAST_COLUMN_BATCH, cols);
        return Math.multiplyExact(Math.multiplyExact(2, batch), rows);
    }

    static void r2rFftpack(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                           boolean realToHermitian, boolean forward, double[] input, double[] output, double factor) {
        FftWorkspace workspace = realAxisWorkspaceRequirement(shape, strideOut, axes).allocate();
        workspace.reset();
        r2rFftpack(shape, strideIn, offsetIn, strideOut, offsetOut, axes, realToHermitian, forward, input, output,
                factor, workspace);
    }

    static void r2rFftpack(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                           boolean realToHermitian, boolean forward, double[] input, double[] output, double factor,
                           FftWorkspace workspace) {
        validateCommon(shape, strideIn, offsetIn, strideOut, offsetOut, axes, input, output, 1);
        r2rFftpackPrepared(shape, strideIn, offsetIn, strideOut, offsetOut, axes, null, realToHermitian, forward,
                input, output, factor, workspace);
    }

    static void r2rFftpackPrepared(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut,
                                   int[] axes, FftRealTransform[] transforms, boolean realToHermitian,
                                   boolean forward, double[] input, double[] output, double factor,
                                   FftWorkspace workspace) {
        if (hasZeroExtent(shape) || axes.length == 0) {
            return;
        }
        double currentFactor = factor;
        for (int axisIndex = 0; axisIndex < axes.length; axisIndex++) {
            double[] source = axisIndex == 0 ? input : output;
            long[] sourceStride = axisIndex == 0 ? strideIn : strideOut;
            long sourceOffset = axisIndex == 0 ? offsetIn : offsetOut;
            FftRealTransform transform = transforms == null
                    ? FftPlanCache.realTransform(shape[axes[axisIndex]])
                    : transforms[axisIndex];
            transformRealAxis(shape, sourceStride, sourceOffset, strideOut, offsetOut, axes[axisIndex], realToHermitian,
                    forward, currentFactor, source, output, transform, workspace);
            currentFactor = 1.0;
        }
    }

    static void dcst(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                     int type, boolean cosine, boolean ortho, double[] input, double[] output, double factor) {
        FftWorkspace workspace = dcstWorkspaceRequirement(shape, strideOut, axes, type, cosine).allocate();
        workspace.reset();
        dcst(shape, strideIn, offsetIn, strideOut, offsetOut, axes, type, cosine, ortho, input, output, factor,
                workspace);
    }

    static void dcst(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                     int type, boolean cosine, boolean ortho, double[] input, double[] output, double factor,
                     FftWorkspace workspace) {
        if (type < 1 || type > 4) {
            throw new IllegalArgumentException("invalid DCT/DST type");
        }
        validateCommon(shape, strideIn, offsetIn, strideOut, offsetOut, axes, input, output, 1);
        dcstPrepared(shape, strideIn, offsetIn, strideOut, offsetOut, axes, null, type, cosine, ortho, input, output,
                factor, workspace);
    }

    static void dcstPrepared(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                             FftDcstPlan[] plans, int type, boolean cosine, boolean ortho, double[] input,
                             double[] output, double factor, FftWorkspace workspace) {
        if (hasZeroExtent(shape) || axes.length == 0) {
            return;
        }
        double currentFactor = factor;
        for (int axisIndex = 0; axisIndex < axes.length; axisIndex++) {
            double[] source = axisIndex == 0 ? input : output;
            long[] sourceStride = axisIndex == 0 ? strideIn : strideOut;
            long sourceOffset = axisIndex == 0 ? offsetIn : offsetOut;
            FftDcstPlan plan = plans == null ? FftPlanCache.dcstPlan(shape[axes[axisIndex]], type, cosine)
                    : plans[axisIndex];
            transformDcstAxis(shape, sourceStride, sourceOffset, strideOut, offsetOut, axes[axisIndex], type, cosine,
                    ortho, currentFactor, source, output, plan, workspace);
            currentFactor = 1.0;
        }
    }

    static void separableHartley(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut,
                                 int[] axes, boolean fht, double[] input, double[] output, double factor) {
        FftWorkspace workspace = separableHartleyWorkspaceRequirement(shape, strideOut, axes).allocate();
        workspace.reset();
        separableHartley(shape, strideIn, offsetIn, strideOut, offsetOut, axes, fht, input, output, factor, workspace);
    }

    static void separableHartley(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut,
                                 int[] axes, boolean fht, double[] input, double[] output, double factor,
                                 FftWorkspace workspace) {
        validateCommon(shape, strideIn, offsetIn, strideOut, offsetOut, axes, input, output, 1);
        separableHartleyPrepared(shape, strideIn, offsetIn, strideOut, offsetOut, axes, null, fht, input, output,
                factor, workspace);
    }

    static void separableHartleyPrepared(int[] shape, long[] strideIn, long offsetIn, long[] strideOut,
                                         long offsetOut, int[] axes, FftRealTransform[] transforms, boolean fht,
                                         double[] input, double[] output, double factor, FftWorkspace workspace) {
        if (hasZeroExtent(shape) || axes.length == 0) {
            return;
        }
        double currentFactor = factor;
        for (int axisIndex = 0; axisIndex < axes.length; axisIndex++) {
            double[] source = axisIndex == 0 ? input : output;
            long[] sourceStride = axisIndex == 0 ? strideIn : strideOut;
            long sourceOffset = axisIndex == 0 ? offsetIn : offsetOut;
            FftRealTransform transform = transforms == null
                    ? FftPlanCache.realTransform(shape[axes[axisIndex]])
                    : transforms[axisIndex];
            transformHartleyAxis(shape, sourceStride, sourceOffset, strideOut, offsetOut, axes[axisIndex], fht,
                    currentFactor, source, output, transform, workspace);
            currentFactor = 1.0;
        }
    }

    static void genuineHartley(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut,
                               int[] axes, boolean fht, double[] input, double[] output, double factor) {
        FftWorkspace workspace = genuineHartleyWorkspaceRequirement(shape, strideOut, axes).allocate();
        workspace.reset();
        genuineHartley(shape, strideIn, offsetIn, strideOut, offsetOut, axes, fht, input, output, factor, workspace);
    }

    static void genuineHartley(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut,
                               int[] axes, boolean fht, double[] input, double[] output, double factor,
                               FftWorkspace workspace) {
        validateCommon(shape, strideIn, offsetIn, strideOut, offsetOut, axes, input, output, 1);
        int lastAxis = axes.length == 0 ? -1 : axes[axes.length - 1];
        int[] tempShape = axes.length <= 1 ? shape : hermitianShape(shape, lastAxis);
        long[] tempStride = axes.length <= 1 ? strideOut : contiguousComplexStride(tempShape);
        int[] complexAxes = axes.length <= 1 ? new int[0] : Arrays.copyOf(axes, axes.length - 1);
        FftComplexTransform[] complexTransforms = axes.length <= 1 ? new FftComplexTransform[0]
                : complexTransforms(tempShape, complexAxes);
        genuineHartleyPrepared(shape, tempShape, tempStride, Math.toIntExact(2L * product(tempShape)), strideIn,
                offsetIn, strideOut, offsetOut, axes, complexAxes, realTransforms(shape, axes),
                axes.length == 0 ? null : FftPlanCache.realTransform(shape[lastAxis]), complexTransforms, fht, input,
                output, factor, workspace);
    }

    static void genuineHartleyPrepared(int[] shape, int[] tempShape, long[] tempStride, int tempSize,
                                       long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                                       int[] complexAxes, FftRealTransform[] realTransforms,
                                       FftRealTransform lastRealTransform, FftComplexTransform[] complexTransforms,
                                       boolean fht, double[] input, double[] output, double factor,
                                       FftWorkspace workspace) {
        if (hasZeroExtent(shape) || axes.length == 0) {
            return;
        }
        if (axes.length == 1) {
            separableHartleyPrepared(shape, strideIn, offsetIn, strideOut, offsetOut, axes, realTransforms, fht,
                    input, output, factor, workspace);
            return;
        }
        double[] temp = workspace.doubles();
        int tempOffset = workspace.alloc(tempSize);
        int childMark = workspace.mark();
        r2cPrepared(shape, tempShape, strideIn, offsetIn, tempStride, tempOffset, axes, complexAxes, lastRealTransform,
                complexTransforms, true, input, temp, factor, workspace);
        workspace.rewind(childMark);
        if (fht) {
            reconstructGenuineFht(shape, tempShape, tempStride, tempOffset, strideOut, offsetOut, axes, temp, output);
        } else {
            reconstructGenuineHartley(shape, tempShape, tempStride, tempOffset, strideOut, offsetOut, axes, temp,
                    output);
        }
    }

    static void r2c(int[] shapeIn, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                    boolean forward, double[] input, double[] output, double factor) {
        FftWorkspace workspace = r2cWorkspaceRequirement(shapeIn, strideIn, offsetIn, strideOut, offsetOut, axes,
                input, output).allocate();
        workspace.reset();
        r2c(shapeIn, strideIn, offsetIn, strideOut, offsetOut, axes, forward, input, output, factor, workspace);
    }

    static void r2c(int[] shapeIn, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                    boolean forward, double[] input, double[] output, double factor, FftWorkspace workspace) {
        validateRealComplexCommon(shapeIn, strideIn, offsetIn, strideOut, offsetOut, axes, input, output,
                true);
        int lastAxis = axes[axes.length - 1];
        int[] shapeOut = hermitianShape(shapeIn, lastAxis);
        int[] complexAxes = Arrays.copyOf(axes, axes.length - 1);
        FftComplexTransform[] complexTransforms = complexTransforms(shapeOut, complexAxes);
        r2cPrepared(shapeIn, shapeOut, strideIn, offsetIn, strideOut, offsetOut, axes, complexAxes,
                FftPlanCache.realTransform(shapeIn[lastAxis]), complexTransforms, forward, input, output, factor,
                workspace);
    }

    static void r2cPrepared(int[] shapeIn, int[] shapeOut, long[] strideIn, long offsetIn, long[] strideOut,
                            long offsetOut, int[] axes, int[] complexAxes, FftRealTransform realTransform,
                            FftComplexTransform[] complexTransforms, boolean forward, double[] input,
                            double[] output, double factor, FftWorkspace workspace) {
        if (hasZeroExtent(shapeIn)) {
            return;
        }
        int lastAxis = axes[axes.length - 1];
        if (canUseFast2dR2c(shapeIn, shapeOut, strideIn, strideOut, axes)) {
            fastBatched2dR2c(shapeIn, shapeOut, strideIn, (int) offsetIn, strideOut, (int) offsetOut, axes,
                    realTransform, complexTransforms, forward, input, output, factor, workspace);
            return;
        }
        transformRealToComplexAxis(shapeIn, strideIn, offsetIn, strideOut, offsetOut, lastAxis, forward, factor,
                input, output, realTransform, workspace);
        if (complexAxes.length > 0) {
            c2cPrepared(shapeOut, strideOut, offsetOut, strideOut, offsetOut, complexAxes, complexTransforms,
                    forward, output, output, 1.0, workspace);
        }
    }

    static void c2r(int[] shapeOut, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                    boolean forward, double[] input, double[] output, double factor) {
        FftWorkspace workspace = c2rWorkspaceRequirement(shapeOut, strideIn, offsetIn, strideOut, offsetOut, axes,
                input, output).allocate();
        workspace.reset();
        c2r(shapeOut, strideIn, offsetIn, strideOut, offsetOut, axes, forward, input, output, factor, workspace);
    }

    static void c2r(int[] shapeOut, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                    boolean forward, double[] input, double[] output, double factor, FftWorkspace workspace) {
        validateRealComplexCommon(shapeOut, strideOut, offsetOut, strideIn, offsetIn, axes, output, input,
                false);
        int lastAxis = axes[axes.length - 1];
        int[] shapeIn = hermitianShape(shapeOut, lastAxis);
        int[] complexAxes = Arrays.copyOf(axes, axes.length - 1);
        long[] tempStride = contiguousComplexStride(shapeIn);
        FftComplexTransform[] complexTransforms = complexTransforms(shapeIn, complexAxes);
        c2rPrepared(shapeOut, shapeIn, tempStride, Math.toIntExact(2L * product(shapeIn)), strideIn, offsetIn,
                strideOut, offsetOut, axes, complexAxes, FftPlanCache.realTransform(shapeOut[lastAxis]),
                complexTransforms, forward, input, output, factor, workspace);
    }

    static void c2rPrepared(int[] shapeOut, int[] shapeIn, long[] tempStride, int tempSize, long[] strideIn,
                            long offsetIn, long[] strideOut, long offsetOut, int[] axes, int[] complexAxes,
                            FftRealTransform realTransform, FftComplexTransform[] complexTransforms,
                            boolean forward, double[] input, double[] output, double factor,
                            FftWorkspace workspace) {
        if (hasZeroExtent(shapeOut)) {
            return;
        }
        int lastAxis = axes[axes.length - 1];
        if (complexAxes.length > 0 && canUseFast2dC2r(shapeOut, shapeIn, strideIn, strideOut, axes)) {
            fastBatched2dC2r(shapeOut, shapeIn, strideIn, (int) offsetIn, strideOut, (int) offsetOut, axes,
                    realTransform, complexTransforms, forward, input, output, factor, workspace);
            return;
        }
        double[] source = input;
        long[] sourceStride = strideIn;
        long sourceOffset = offsetIn;
        int childMark = workspace.mark();
        if (complexAxes.length > 0) {
            double[] temp = workspace.doubles();
            int tempOffset = workspace.alloc(tempSize);
            childMark = workspace.mark();
            c2cPrepared(shapeIn, strideIn, offsetIn, tempStride, tempOffset, complexAxes, complexTransforms, forward,
                input, temp, 1.0, workspace);
            source = temp;
            sourceStride = tempStride;
            sourceOffset = tempOffset;
        }
        workspace.rewind(childMark);
        transformComplexToRealAxis(shapeOut, sourceStride, sourceOffset, strideOut, offsetOut, lastAxis, forward,
                factor, source, output, realTransform, workspace);
    }

    static FftWorkspace.Requirement c2cWorkspaceRequirement(int[] shape, int[] axes) {
        validateRequirementShapeAxes(shape, axes, false);
        // This layout-agnostic requirement backs PreparedNdComplex.createWorkspace(). It must stay
        // safe for arbitrary legal strides, so the standard-layout fast path can only be added as
        // another possible peak, not used to shrink the generic bound.
        FftWorkspace.Requirement requirement = c2cGenericWorkspaceRequirement(shape, null, axes);
        if (shape.length == 2 && axes.length == 1) {
            int axis = axes[0];
            int otherAxis = 1 - axis;
            requirement = requirement.max(fastColumnPassWorkspaceRequirement(shape[axis], shape[otherAxis]));
        } else if (axes.length == 2) {
            requirement = requirement.max(fast2dC2cWorkspaceRequirement(shape, axes));
        }
        return requirement;
    }

    static FftWorkspace.Requirement c2cWorkspaceRequirement(int[] shape, long[] strideIn, long offsetIn,
                                                            long[] strideOut, long offsetOut, int[] axes,
                                                            double[] input, double[] output) {
        validateCommon(shape, strideIn, offsetIn, strideOut, offsetOut, axes, input, output, 2);
        if (hasZeroExtent(shape) || axes.length == 0) {
            return FftWorkspace.Requirement.empty();
        }
        // Concrete-layout sizing is allowed to select the compact in-place 2D requirement because
        // callers pass the same layout to both the requirement method and the execution method.
        // Batched layouts still need only the single-plane peak because planes are processed one at
        // a time against the caller-provided workspace.
        if (axes.length == 2 && input == output && offsetIn == offsetOut
                && java.util.Arrays.equals(strideIn, strideOut)
                && canUseFast2dC2c(shape, strideOut, axes)) {
            return fast2dC2cWorkspaceRequirement(shape, axes);
        }
        if (shape.length == 2 && axes.length == 1 && input == output && offsetIn == offsetOut
                && java.util.Arrays.equals(strideIn, strideOut)
                && canUseFastColumnPass(shape, strideOut, axes[0])) {
            int axis = axes[0];
            return fastColumnPassWorkspaceRequirement(shape[axis], shape[1 - axis]);
        }
        return c2cGenericWorkspaceRequirement(shape, strideOut, axes);
    }

    private static FftWorkspace.Requirement c2cGenericWorkspaceRequirement(int[] shape, long[] strideOut,
                                                                          int[] axes) {
        FftWorkspace.Requirement requirement = FftWorkspace.Requirement.empty();
        for (int axis : axes) {
            int length = shape[axis];
            // When the output axis is contiguous, execC2c can use the final output slice as its
            // line buffer. Strided output still needs a temporary contiguous line in the arena.
            int lineBuffer = strideOut != null && strideOut[axis] == 2L ? 0 : Math.multiplyExact(2, length);
            FftWorkspace.Requirement transformRequirement = FftPlanCache.complexTransform(length)
                .workspaceRequirement();
            requirement = requirement.max(FftWorkspace.Requirement.doubles(lineBuffer).plus(transformRequirement));
        }
        return requirement;
    }

    private static FftWorkspace.Requirement fastColumnPassWorkspaceRequirement(int rows, int cols) {
        // The B4 gather buffer and the child transform scratch are live at the same time.
        FftWorkspace.Requirement transformRequirement = FftPlanCache.complexTransform(rows).workspaceRequirement();
        return FftWorkspace.Requirement.doubles(fastColumnBufferDoubles(rows, cols)).plus(transformRequirement);
    }

    private static FftWorkspace.Requirement fast2dC2cWorkspaceRequirement(int[] shape, int[] axes) {
        int contiguousAxis = axes[axes.length - 1];
        int nonContiguousAxis = axes[0];
        // The row phase and column phase are sequential, so the arena only needs the larger peak.
        FftWorkspace.Requirement contiguousTransform = FftPlanCache.complexTransform(shape[contiguousAxis])
            .workspaceRequirement();
        return contiguousTransform.max(fastColumnPassWorkspaceRequirement(shape[nonContiguousAxis],
                shape[contiguousAxis]));
    }

    static FftWorkspace.Requirement realAxisWorkspaceRequirement(int[] shape, int[] axes) {
        validateRequirementShapeAxes(shape, axes, false);
        FftWorkspace.Requirement requirement = FftWorkspace.Requirement.empty();
        for (int axis : axes) {
            int length = shape[axis];
            requirement = requirement.max(FftWorkspace.Requirement.doubles(length)
                    .plus(FftPlanCache.realTransform(length).workspaceRequirement()));
        }
        return requirement;
    }

    static FftWorkspace.Requirement realAxisWorkspaceRequirement(int[] shape, long[] strideOut, int[] axes) {
        validateRequirementShapeAxes(shape, axes, false);
        Objects.requireNonNull(strideOut, "strideOut");
        if (strideOut.length != shape.length) {
            throw new IllegalArgumentException("stride dimension mismatch");
        }
        FftWorkspace.Requirement requirement = FftWorkspace.Requirement.empty();
        for (int axis : axes) {
            int length = shape[axis];
            FftWorkspace.Requirement transformRequirement = FftPlanCache.realTransform(length).workspaceRequirement();
            FftWorkspace.Requirement axisRequirement = strideOut[axis] == 1L
                    ? transformRequirement
                    : FftWorkspace.Requirement.doubles(length).plus(transformRequirement);
            requirement = requirement.max(axisRequirement);
        }
        return requirement;
    }

    static FftWorkspace.Requirement separableHartleyWorkspaceRequirement(int[] shape, long[] strideOut,
                                                                         int[] axes) {
        validateRequirementShapeAxes(shape, axes, false);
        Objects.requireNonNull(strideOut, "strideOut");
        if (strideOut.length != shape.length) {
            throw new IllegalArgumentException("stride dimension mismatch");
        }
        FftWorkspace.Requirement requirement = FftWorkspace.Requirement.empty();
        for (int axis : axes) {
            int length = shape[axis];
            FftWorkspace.Requirement transformRequirement = FftPlanCache.realTransform(length).workspaceRequirement();
            FftWorkspace.Requirement axisRequirement = strideOut[axis] == 1L
                    ? transformRequirement.max(FftWorkspace.Requirement.doubles(length / 2))
                    : FftWorkspace.Requirement.doubles(length).plus(transformRequirement);
            requirement = requirement.max(axisRequirement);
        }
        return requirement;
    }

    static FftWorkspace.Requirement dcstWorkspaceRequirement(int[] shape, int[] axes, int type, boolean cosine) {
        validateRequirementShapeAxes(shape, axes, false);
        FftWorkspace.Requirement requirement = FftWorkspace.Requirement.empty();
        for (int axis : axes) {
            int length = shape[axis];
            requirement = requirement.max(FftWorkspace.Requirement.doubles(length)
                    .plus(FftPlanCache.dcstPlan(length, type, cosine).workspaceRequirement()));
        }
        return requirement;
    }

    static FftWorkspace.Requirement dcstWorkspaceRequirement(int[] shape, long[] strideOut, int[] axes, int type,
                                                             boolean cosine) {
        validateRequirementShapeAxes(shape, axes, false);
        Objects.requireNonNull(strideOut, "strideOut");
        if (strideOut.length != shape.length) {
            throw new IllegalArgumentException("stride dimension mismatch");
        }
        FftWorkspace.Requirement requirement = FftWorkspace.Requirement.empty();
        for (int axis : axes) {
            int length = shape[axis];
            FftWorkspace.Requirement planRequirement = FftPlanCache.dcstPlan(length, type, cosine)
                    .workspaceRequirement();
            if (strideOut[axis] == 1L) {
                requirement = requirement.max(planRequirement);
            } else {
                requirement = requirement.max(FftWorkspace.Requirement.doubles(length).plus(planRequirement));
            }
        }
        return requirement;
    }

    static FftWorkspace.Requirement r2cWorkspaceRequirement(int[] shape, int[] axes) {
        validateRequirementShapeAxes(shape, axes, true);
        int lastAxis = axes[axes.length - 1];
        FftWorkspace.Requirement requirement = FftWorkspace.Requirement.doubles(shape[lastAxis])
            .plus(FftPlanCache.realTransform(shape[lastAxis]).workspaceRequirement());
        if (axes.length > 1) {
            int[] shapeOut = hermitianShape(shape, lastAxis);
            int[] complexAxes = Arrays.copyOf(axes, axes.length - 1);
            requirement = requirement.max(c2cWorkspaceRequirement(shapeOut, complexAxes));
            if (axes.length == 2) {
                requirement = requirement.max(fast2dR2cWorkspaceRequirement(shape, shapeOut, axes));
            }
        }
        return requirement;
    }

    static FftWorkspace.Requirement r2cWorkspaceRequirement(int[] shape, long[] strideIn, long offsetIn,
                                                            long[] strideOut, long offsetOut, int[] axes,
                                                            double[] input, double[] output) {
        validateRealComplexCommon(shape, strideIn, offsetIn, strideOut, offsetOut, axes, input, output, true);
        if (hasZeroExtent(shape)) {
            return FftWorkspace.Requirement.empty();
        }
        int lastAxis = axes[axes.length - 1];
        int[] shapeOut = hermitianShape(shape, lastAxis);
        if (canUseFast2dR2c(shape, shapeOut, strideIn, strideOut, axes)) {
            // Standard batched planes reuse the same direct Hermitian row staging and B4 column
            // scratch for every plane, so the layout-specific requirement is batch-independent.
            return fast2dR2cWorkspaceRequirement(shape, shapeOut, axes);
        }
        return r2cWorkspaceRequirement(shape, axes);
    }

    static FftWorkspace.Requirement c2rWorkspaceRequirement(int[] shape, int[] axes) {
        validateRequirementShapeAxes(shape, axes, true);
        int lastAxis = axes[axes.length - 1];
        FftWorkspace.Requirement finalAxis = FftWorkspace.Requirement.doubles(shape[lastAxis])
            .plus(FftPlanCache.realTransform(shape[lastAxis]).workspaceRequirement());
        if (axes.length == 1) {
            return finalAxis;
        }
        int[] shapeIn = hermitianShape(shape, lastAxis);
        int[] complexAxes = Arrays.copyOf(axes, axes.length - 1);
        FftWorkspace.Requirement child = c2cWorkspaceRequirement(shapeIn, complexAxes).max(finalAxis);
        FftWorkspace.Requirement requirement = FftWorkspace.Requirement.doubles(Math.toIntExact(2L * product(shapeIn)))
            .plus(child);
        if (axes.length == 2) {
            requirement = requirement.max(fast2dC2rWorkspaceRequirement(shape, shapeIn, axes));
        }
        return requirement;
    }

    static FftWorkspace.Requirement c2rWorkspaceRequirement(int[] shape, long[] strideIn, long offsetIn,
                                                            long[] strideOut, long offsetOut, int[] axes,
                                                            double[] input, double[] output) {
        int lastAxis = axes.length == 0 ? -1 : axes[axes.length - 1];
        int[] shapeIn = axes.length == 0 ? shape : hermitianShape(shape, lastAxis);
        validateRealComplexCommon(shape, shapeIn, strideOut, offsetOut, strideIn, offsetIn, axes, output, input,
                false);
        if (hasZeroExtent(shape)) {
            return FftWorkspace.Requirement.empty();
        }
        if (canUseFast2dC2r(shape, shapeIn, strideIn, strideOut, axes)) {
            // The fast inverse path avoids the full Hermitian temporary used by the generic route;
            // only the current plane's column buffer and child transform scratch are live.
            return fast2dC2rWorkspaceRequirement(shape, shapeIn, axes);
        }
        return c2rWorkspaceRequirement(shape, axes);
    }

    private static FftWorkspace.Requirement fast2dR2cWorkspaceRequirement(int[] shapeIn, int[] shapeOut, int[] axes) {
        int contiguousAxis = axes[axes.length - 1];
        int nonContiguousAxis = axes[0];
        FftWorkspace.Requirement finalAxis = FftPlanCache.realTransform(shapeIn[contiguousAxis])
                .workspaceRequirement();
        FftWorkspace.Requirement columnPass = fastColumnPassWorkspaceRequirement(shapeIn[nonContiguousAxis],
                shapeOut[contiguousAxis]);
        return finalAxis.max(columnPass);
    }

    private static FftWorkspace.Requirement fast2dC2rWorkspaceRequirement(int[] shapeOut, int[] shapeIn, int[] axes) {
        int contiguousAxis = axes[axes.length - 1];
        int nonContiguousAxis = axes[0];
        FftWorkspace.Requirement columnPass = fastColumnPassWorkspaceRequirement(shapeOut[nonContiguousAxis],
                shapeIn[contiguousAxis]);
        FftWorkspace.Requirement finalAxis = FftPlanCache.realTransform(shapeOut[contiguousAxis])
                .workspaceRequirement();
        return columnPass.max(finalAxis);
    }

    static FftWorkspace.Requirement genuineHartleyWorkspaceRequirement(int[] shape, int[] axes) {
        validateRequirementShapeAxes(shape, axes, true);
        if (axes.length == 1) {
            return realAxisWorkspaceRequirement(shape, axes);
        }
        int lastAxis = axes[axes.length - 1];
        int[] tempShape = hermitianShape(shape, lastAxis);
        return FftWorkspace.Requirement.doubles(Math.toIntExact(2L * product(tempShape)))
                .plus(r2cWorkspaceRequirement(shape, axes))
                .withIntSlots(shape.length);
    }

    static FftWorkspace.Requirement genuineHartleyWorkspaceRequirement(int[] shape, long[] strideOut, int[] axes) {
        validateRequirementShapeAxes(shape, axes, true);
        if (axes.length == 1) {
            return separableHartleyWorkspaceRequirement(shape, strideOut, axes);
        }
        return genuineHartleyWorkspaceRequirement(shape, axes);
    }

    static FftComplexTransform[] complexTransforms(int[] shape, int[] axes) {
        FftComplexTransform[] transforms = new FftComplexTransform[axes.length];
        for (int index = 0; index < axes.length; index++) {
            transforms[index] = FftPlanCache.complexTransform(shape[axes[index]]);
        }
        return transforms;
    }

    static FftRealTransform[] realTransforms(int[] shape, int[] axes) {
        FftRealTransform[] transforms = new FftRealTransform[axes.length];
        for (int index = 0; index < axes.length; index++) {
            transforms[index] = FftPlanCache.realTransform(shape[axes[index]]);
        }
        return transforms;
    }

    static FftDcstPlan[] dcstPlans(int[] shape, int[] axes, int type, boolean cosine) {
        FftDcstPlan[] plans = new FftDcstPlan[axes.length];
        for (int index = 0; index < axes.length; index++) {
            plans[index] = FftPlanCache.dcstPlan(shape[axes[index]], type, cosine);
        }
        return plans;
    }

    private static void validateRequirementShapeAxes(int[] shape, int[] axes, boolean requireAxes) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(axes, "axes");
        if (shape.length < 1) {
            throw new IllegalArgumentException("ndim must be >= 1");
        }
        if (requireAxes && axes.length == 0) {
            throw new IllegalArgumentException("at least one axis is required");
        }
        for (int extent : shape) {
            if (extent < 0) {
                throw new IllegalArgumentException("shape entries must be >= 0");
            }
        }
        validateAxes(shape.length, axes);
    }

    private static void transformComplexAxis(int[] shape, long[] strideIn, long offsetIn, long[] strideOut,
                                             long offsetOut, int axis, boolean forward, double factor,
                                             double[] input, double[] output, FftComplexTransform transform,
                                             FftWorkspace workspace) {
        if (forward) {
            transformComplexAxisForward(shape, strideIn, offsetIn, strideOut, offsetOut, axis, factor, input, output,
                    transform, workspace);
        } else {
            transformComplexAxisBackward(shape, strideIn, offsetIn, strideOut, offsetOut, axis, factor, input, output,
                    transform, workspace);
        }
    }

    private static void transformComplexAxisForward(int[] shape, long[] strideIn, long offsetIn, long[] strideOut,
                                                    long offsetOut, int axis, double factor, double[] input,
                                                    double[] output, FftComplexTransform transform,
                                                    FftWorkspace workspace) {
        int length = shape[axis];
        int baseMark = workspace.mark();
        boolean allowOutputBuffer = strideOut[axis] == 2L;
        // A contiguous output line is already a valid transform buffer. Only strided output needs
        // an arena-backed contiguous copy before scattering results to their final locations.
        double[] storage = output;
        int storageOffset = -1;
        if (!allowOutputBuffer) {
            storage = workspace.doubles();
            storageOffset = workspace.alloc(2 * length);
        }
        int transformMark = workspace.mark();
        MultiIter lines = new MultiIter(shape, strideIn, offsetIn, strideOut, offsetOut, axis);
        while (lines.remaining() > 0) {
            workspace.rewind(transformMark);
            lines.advance(1);
            int lineOffset = allowOutputBuffer ? lines.oofs(0) : storageOffset;
            execC2cForward(lines, input, output, storage, lineOffset, transform, factor, workspace);
        }
        workspace.rewind(baseMark);
    }

    private static void transformComplexAxisBackward(int[] shape, long[] strideIn, long offsetIn, long[] strideOut,
                                                     long offsetOut, int axis, double factor, double[] input,
                                                     double[] output, FftComplexTransform transform,
                                                     FftWorkspace workspace) {
        int length = shape[axis];
        int baseMark = workspace.mark();
        boolean allowOutputBuffer = strideOut[axis] == 2L;
        // Keep the backward path symmetric with the forward path: reuse contiguous output slices
        // directly and reserve an arena line buffer only for strided output layouts.
        double[] storage = output;
        int storageOffset = -1;
        if (!allowOutputBuffer) {
            storage = workspace.doubles();
            storageOffset = workspace.alloc(2 * length);
        }
        int transformMark = workspace.mark();
        MultiIter lines = new MultiIter(shape, strideIn, offsetIn, strideOut, offsetOut, axis);
        while (lines.remaining() > 0) {
            workspace.rewind(transformMark);
            lines.advance(1);
            int lineOffset = allowOutputBuffer ? lines.oofs(0) : storageOffset;
            execC2cBackward(lines, input, output, storage, lineOffset, transform, factor, workspace);
        }
        workspace.rewind(baseMark);
    }

    private static void transformRealAxis(int[] shape, long[] strideIn, long offsetIn, long[] strideOut,
                                          long offsetOut, int axis, boolean realToHermitian, boolean forward,
                                          double factor, double[] input, double[] output, FftRealTransform transform,
                                          FftWorkspace workspace) {
        if (realToHermitian) {
            if (forward) {
                transformRealAxisR2hForward(shape, strideIn, offsetIn, strideOut, offsetOut, axis, factor, input,
                        output, transform, workspace);
            } else {
                transformRealAxisR2hBackward(shape, strideIn, offsetIn, strideOut, offsetOut, axis, factor, input,
                        output, transform, workspace);
            }
        } else {
            if (forward) {
                transformRealAxisH2rForward(shape, strideIn, offsetIn, strideOut, offsetOut, axis, factor, input,
                        output, transform, workspace);
            } else {
                transformRealAxisH2rBackward(shape, strideIn, offsetIn, strideOut, offsetOut, axis, factor, input,
                        output, transform, workspace);
            }
        }
    }

    private static void transformRealAxisR2hForward(int[] shape, long[] strideIn, long offsetIn, long[] strideOut,
                                                    long offsetOut, int axis, double factor, double[] input,
                                                    double[] output, FftRealTransform transform,
                                                    FftWorkspace workspace) {
        int length = shape[axis];
        int baseMark = workspace.mark();
        boolean allowOutputBuffer = strideOut[axis] == 1L;
        double[] line = output;
        int workspaceLineOffset = -1;
        if (!allowOutputBuffer) {
            line = workspace.doubles();
            workspaceLineOffset = workspace.alloc(length);
        }
        int transformMark = workspace.mark();
        MultiIter lines = new MultiIter(shape, strideIn, offsetIn, strideOut, offsetOut, axis);
        while (lines.remaining() > 0) {
            workspace.rewind(transformMark);
            lines.advance(1);
            int lineOffset = allowOutputBuffer ? lines.oofs(0) : workspaceLineOffset;
            execR2hForward(lines, input, output, line, lineOffset, transform, factor, workspace);
        }
        workspace.rewind(baseMark);
    }

    private static void transformRealAxisR2hBackward(int[] shape, long[] strideIn, long offsetIn, long[] strideOut,
                                                     long offsetOut, int axis, double factor, double[] input,
                                                     double[] output, FftRealTransform transform,
                                                     FftWorkspace workspace) {
        int length = shape[axis];
        int baseMark = workspace.mark();
        boolean allowOutputBuffer = strideOut[axis] == 1L;
        double[] line = output;
        int workspaceLineOffset = -1;
        if (!allowOutputBuffer) {
            line = workspace.doubles();
            workspaceLineOffset = workspace.alloc(length);
        }
        int transformMark = workspace.mark();
        MultiIter lines = new MultiIter(shape, strideIn, offsetIn, strideOut, offsetOut, axis);
        while (lines.remaining() > 0) {
            workspace.rewind(transformMark);
            lines.advance(1);
            int lineOffset = allowOutputBuffer ? lines.oofs(0) : workspaceLineOffset;
            execR2hBackward(lines, input, output, line, lineOffset, transform, factor, workspace);
        }
        workspace.rewind(baseMark);
    }

    private static void transformRealAxisH2rForward(int[] shape, long[] strideIn, long offsetIn, long[] strideOut,
                                                    long offsetOut, int axis, double factor, double[] input,
                                                    double[] output, FftRealTransform transform,
                                                    FftWorkspace workspace) {
        int length = shape[axis];
        int baseMark = workspace.mark();
        boolean allowOutputBuffer = strideOut[axis] == 1L;
        double[] line = output;
        int workspaceLineOffset = -1;
        if (!allowOutputBuffer) {
            line = workspace.doubles();
            workspaceLineOffset = workspace.alloc(length);
        }
        int transformMark = workspace.mark();
        MultiIter lines = new MultiIter(shape, strideIn, offsetIn, strideOut, offsetOut, axis);
        while (lines.remaining() > 0) {
            workspace.rewind(transformMark);
            lines.advance(1);
            int lineOffset = allowOutputBuffer ? lines.oofs(0) : workspaceLineOffset;
            execH2rForward(lines, input, output, line, lineOffset, transform, factor, workspace);
        }
        workspace.rewind(baseMark);
    }

    private static void transformRealAxisH2rBackward(int[] shape, long[] strideIn, long offsetIn, long[] strideOut,
                                                     long offsetOut, int axis, double factor, double[] input,
                                                     double[] output, FftRealTransform transform,
                                                     FftWorkspace workspace) {
        int length = shape[axis];
        int baseMark = workspace.mark();
        boolean allowOutputBuffer = strideOut[axis] == 1L;
        double[] line = output;
        int workspaceLineOffset = -1;
        if (!allowOutputBuffer) {
            line = workspace.doubles();
            workspaceLineOffset = workspace.alloc(length);
        }
        int transformMark = workspace.mark();
        MultiIter lines = new MultiIter(shape, strideIn, offsetIn, strideOut, offsetOut, axis);
        while (lines.remaining() > 0) {
            workspace.rewind(transformMark);
            lines.advance(1);
            int lineOffset = allowOutputBuffer ? lines.oofs(0) : workspaceLineOffset;
            execH2rBackward(lines, input, output, line, lineOffset, transform, factor, workspace);
        }
        workspace.rewind(baseMark);
    }

    private static void transformDcstAxis(int[] shape, long[] strideIn, long offsetIn, long[] strideOut,
                                          long offsetOut, int axis, int type, boolean cosine, boolean ortho,
                                          double factor, double[] input, double[] output, FftDcstPlan plan,
                                          FftWorkspace workspace) {
        int length = shape[axis];
        int baseMark = workspace.mark();
        boolean allowOutputBuffer = strideOut[axis] == 1L;
        double[] line = output;
        int workspaceLineOffset = -1;
        if (!allowOutputBuffer) {
            line = workspace.doubles();
            workspaceLineOffset = workspace.alloc(length);
        }
        int transformMark = workspace.mark();
        MultiIter lines = new MultiIter(shape, strideIn, offsetIn, strideOut, offsetOut, axis);
        while (lines.remaining() > 0) {
            workspace.rewind(transformMark);
            lines.advance(1);
            int lineOffset = allowOutputBuffer ? lines.oofs(0) : workspaceLineOffset;
            execDcst(lines, input, output, line, lineOffset, plan, factor, ortho, type, cosine, workspace);
        }
        workspace.rewind(baseMark);
    }

    private static void transformHartleyAxis(int[] shape, long[] strideIn, long offsetIn, long[] strideOut,
                                             long offsetOut, int axis, boolean fht, double factor, double[] input,
                                             double[] output, FftRealTransform transform, FftWorkspace workspace) {
        if (fht) {
            transformFhtAxis(shape, strideIn, offsetIn, strideOut, offsetOut, axis, factor, input, output, transform,
                    workspace);
        } else {
            transformHartleyAxis(shape, strideIn, offsetIn, strideOut, offsetOut, axis, factor, input, output,
                    transform, workspace);
        }
    }

    private static void transformHartleyAxis(int[] shape, long[] strideIn, long offsetIn, long[] strideOut,
                                             long offsetOut, int axis, double factor, double[] input,
                                             double[] output, FftRealTransform transform, FftWorkspace workspace) {
        int length = shape[axis];
        int baseMark = workspace.mark();
        boolean allowOutputBuffer = strideOut[axis] == 1L;
        double[] line = output;
        int lineOffset = -1;
        if (!allowOutputBuffer) {
            line = workspace.doubles();
            lineOffset = workspace.alloc(length);
        }
        int transformMark = workspace.mark();
        MultiIter lines = new MultiIter(shape, strideIn, offsetIn, strideOut, offsetOut, axis);
        while (lines.remaining() > 0) {
            workspace.rewind(transformMark);
            lines.advance(1);
            int currentLineOffset = allowOutputBuffer ? lines.oofs(0) : lineOffset;
            if (allowOutputBuffer) {
                execHartleyInPlace(lines, input, line, currentLineOffset, transform, factor, workspace,
                        transformMark);
            } else {
                execHartley(lines, input, output, line, currentLineOffset, transform, factor, workspace);
            }
        }
        workspace.rewind(baseMark);
    }

    private static void transformFhtAxis(int[] shape, long[] strideIn, long offsetIn, long[] strideOut,
                                         long offsetOut, int axis, double factor, double[] input, double[] output,
                                         FftRealTransform transform, FftWorkspace workspace) {
        int length = shape[axis];
        int baseMark = workspace.mark();
        boolean allowOutputBuffer = strideOut[axis] == 1L;
        double[] line = output;
        int lineOffset = -1;
        if (!allowOutputBuffer) {
            line = workspace.doubles();
            lineOffset = workspace.alloc(length);
        }
        int transformMark = workspace.mark();
        MultiIter lines = new MultiIter(shape, strideIn, offsetIn, strideOut, offsetOut, axis);
        while (lines.remaining() > 0) {
            workspace.rewind(transformMark);
            lines.advance(1);
            int currentLineOffset = allowOutputBuffer ? lines.oofs(0) : lineOffset;
            if (allowOutputBuffer) {
                execFhtInPlace(lines, input, line, currentLineOffset, transform, factor, workspace, transformMark);
            } else {
                execFht(lines, input, output, line, currentLineOffset, transform, factor, workspace);
            }
        }
        workspace.rewind(baseMark);
    }

    private static void transformRealToComplexAxis(int[] shapeIn, long[] strideIn, long offsetIn, long[] strideOut,
                                                   long offsetOut, int axis, boolean forward, double factor,
                                                   double[] input, double[] output, FftRealTransform transform,
                                                   FftWorkspace workspace) {
        if (forward) {
            transformRealToComplexAxisForward(shapeIn, strideIn, offsetIn, strideOut, offsetOut, axis, factor, input,
                    output, transform, workspace);
        } else {
            transformRealToComplexAxisBackward(shapeIn, strideIn, offsetIn, strideOut, offsetOut, axis, factor, input,
                    output, transform, workspace);
        }
    }

    private static void transformRealToComplexAxisForward(int[] shapeIn, long[] strideIn, long offsetIn,
                                                          long[] strideOut, long offsetOut, int axis, double factor,
                                                          double[] input, double[] output,
                                                          FftRealTransform transform, FftWorkspace workspace) {
        int length = shapeIn[axis];
        int[] shapeOut = hermitianShape(shapeIn, axis);
        int baseMark = workspace.mark();
        double[] halfComplex = workspace.doubles();
        int halfComplexOffset = workspace.alloc(length);
        int transformMark = workspace.mark();
        MultiIter lines = new MultiIter(shapeIn, shapeOut, strideIn, offsetIn, strideOut, offsetOut, axis);
        while (lines.remaining() > 0) {
            workspace.rewind(transformMark);
            lines.advance(1);
            copyInputReal(lines, input, halfComplex, halfComplexOffset);
            transform.exec(halfComplex, halfComplexOffset, factor, true, workspace);
            copyR2cForwardOutput(lines, halfComplex, halfComplexOffset, output);
        }
        workspace.rewind(baseMark);
    }

    private static void transformRealToComplexAxisBackward(int[] shapeIn, long[] strideIn, long offsetIn,
                                                           long[] strideOut, long offsetOut, int axis, double factor,
                                                           double[] input, double[] output,
                                                           FftRealTransform transform, FftWorkspace workspace) {
        int length = shapeIn[axis];
        int[] shapeOut = hermitianShape(shapeIn, axis);
        int baseMark = workspace.mark();
        double[] halfComplex = workspace.doubles();
        int halfComplexOffset = workspace.alloc(length);
        int transformMark = workspace.mark();
        MultiIter lines = new MultiIter(shapeIn, shapeOut, strideIn, offsetIn, strideOut, offsetOut, axis);
        while (lines.remaining() > 0) {
            workspace.rewind(transformMark);
            lines.advance(1);
            copyInputReal(lines, input, halfComplex, halfComplexOffset);
            transform.exec(halfComplex, halfComplexOffset, factor, true, workspace);
            copyR2cBackwardOutput(lines, halfComplex, halfComplexOffset, output);
        }
        workspace.rewind(baseMark);
    }

    private static void transformComplexToRealAxis(int[] shapeOut, long[] strideIn, long offsetIn, long[] strideOut,
                                                   long offsetOut, int axis, boolean forward, double factor,
                                                   double[] input, double[] output, FftRealTransform transform,
                                                   FftWorkspace workspace) {
        if (forward) {
            transformComplexToRealAxisForward(shapeOut, strideIn, offsetIn, strideOut, offsetOut, axis, factor, input,
                    output, transform, workspace);
        } else {
            transformComplexToRealAxisBackward(shapeOut, strideIn, offsetIn, strideOut, offsetOut, axis, factor, input,
                    output, transform, workspace);
        }
    }

    private static void transformComplexToRealAxisForward(int[] shapeOut, long[] strideIn, long offsetIn,
                                                          long[] strideOut, long offsetOut, int axis, double factor,
                                                          double[] input, double[] output,
                                                          FftRealTransform transform, FftWorkspace workspace) {
        int length = shapeOut[axis];
        int[] shapeIn = hermitianShape(shapeOut, axis);
        int baseMark = workspace.mark();
        double[] halfComplex = workspace.doubles();
        int halfComplexOffset = workspace.alloc(length);
        int transformMark = workspace.mark();
        MultiIter lines = new MultiIter(shapeIn, shapeOut, strideIn, offsetIn, strideOut, offsetOut, axis);
        while (lines.remaining() > 0) {
            workspace.rewind(transformMark);
            lines.advance(1);
            copyC2rForwardInput(lines, input, halfComplex, halfComplexOffset);
            transform.exec(halfComplex, halfComplexOffset, factor, false, workspace);
            copyOutputReal(lines, halfComplex, halfComplexOffset, output);
        }
        workspace.rewind(baseMark);
    }

    private static void transformComplexToRealAxisBackward(int[] shapeOut, long[] strideIn, long offsetIn,
                                                           long[] strideOut, long offsetOut, int axis, double factor,
                                                           double[] input, double[] output,
                                                           FftRealTransform transform, FftWorkspace workspace) {
        int length = shapeOut[axis];
        int[] shapeIn = hermitianShape(shapeOut, axis);
        int baseMark = workspace.mark();
        double[] halfComplex = workspace.doubles();
        int halfComplexOffset = workspace.alloc(length);
        int transformMark = workspace.mark();
        MultiIter lines = new MultiIter(shapeIn, shapeOut, strideIn, offsetIn, strideOut, offsetOut, axis);
        while (lines.remaining() > 0) {
            workspace.rewind(transformMark);
            lines.advance(1);
            copyC2rBackwardInput(lines, input, halfComplex, halfComplexOffset);
            transform.exec(halfComplex, halfComplexOffset, factor, false, workspace);
            copyOutputReal(lines, halfComplex, halfComplexOffset, output);
        }
        workspace.rewind(baseMark);
    }

    static void validateCommon(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut,
                               int[] axes, double[] input, double[] output, int elementWidth) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(strideIn, "strideIn");
        Objects.requireNonNull(strideOut, "strideOut");
        Objects.requireNonNull(axes, "axes");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        if (shape.length < 1) {
            throw new IllegalArgumentException("ndim must be >= 1");
        }
        if (strideIn.length != shape.length || strideOut.length != shape.length) {
            throw new IllegalArgumentException("stride dimension mismatch");
        }
        for (int extent : shape) {
            if (extent < 0) {
                throw new IllegalArgumentException("shape entries must be >= 0");
            }
        }
        if (input == output && (offsetIn != offsetOut || !Arrays.equals(strideIn, strideOut))) {
            throw new IllegalArgumentException("stride mismatch");
        }
        validateAxes(shape.length, axes);
        if (!hasZeroExtent(shape)) {
            checkBounds(input, shape, strideIn, offsetIn, elementWidth, "input");
            checkBounds(output, shape, strideOut, offsetOut, elementWidth, "output");
        }
    }

    static void validateRealComplexCommon(int[] realShape, long[] realStride, long realOffset,
                                          long[] complexStride, long complexOffset, int[] axes,
                                          double[] realData, double[] complexData, boolean realToComplex) {
        Objects.requireNonNull(realShape, "shape");
        Objects.requireNonNull(axes, "axes");
        int lastAxis = axes.length == 0 ? -1 : axes[axes.length - 1];
        int[] complexShape = axes.length == 0 ? realShape : hermitianShape(realShape, lastAxis);
        validateRealComplexCommon(realShape, complexShape, realStride, realOffset, complexStride, complexOffset, axes,
            realData, complexData, realToComplex);
    }

    static void validateRealComplexCommon(int[] realShape, int[] complexShape, long[] realStride, long realOffset,
                                          long[] complexStride, long complexOffset, int[] axes,
                                          double[] realData, double[] complexData, boolean realToComplex) {
        Objects.requireNonNull(realShape, "shape");
        Objects.requireNonNull(complexShape, "complexShape");
        Objects.requireNonNull(realStride, "realStride");
        Objects.requireNonNull(complexStride, "complexStride");
        Objects.requireNonNull(axes, "axes");
        Objects.requireNonNull(realData, "realData");
        Objects.requireNonNull(complexData, "complexData");
        if (realShape.length < 1) {
            throw new IllegalArgumentException("ndim must be >= 1");
        }
        if (complexShape.length != realShape.length || realStride.length != realShape.length
                || complexStride.length != realShape.length) {
            throw new IllegalArgumentException("stride dimension mismatch");
        }
        if (axes.length == 0) {
            throw new IllegalArgumentException("at least one axis is required");
        }
        for (int extent : realShape) {
            if (extent < 0) {
                throw new IllegalArgumentException("shape entries must be >= 0");
            }
        }
        validateAxes(realShape.length, axes);
        if (realData == complexData) {
            throw new IllegalArgumentException("real and complex arrays must be distinct");
        }
        if (!hasZeroExtent(realShape)) {
            checkBounds(realData, realShape, realStride, realOffset, 1, realToComplex ? "input" : "output");
            checkBounds(complexData, complexShape, complexStride, complexOffset, 2, realToComplex ? "output" : "input");
        }
    }

    private static void validateAxes(int dimensions, int[] axes) {
        for (int index = 0; index < axes.length; index++) {
            int axis = axes[index];
            if (axis < 0 || axis >= dimensions) {
                throw new IllegalArgumentException("bad axis number");
            }
            for (int previous = 0; previous < index; previous++) {
                if (axes[previous] == axis) {
                    throw new IllegalArgumentException("axis specified repeatedly");
                }
            }
        }
    }

    private static boolean hasZeroExtent(int[] shape) {
        for (int extent : shape) {
            if (extent == 0) {
                return true;
            }
        }
        return false;
    }

    private static void checkBounds(double[] data, int[] shape, long[] stride, long offset, int elementWidth, String name) {
        long minimum = offset;
        long maximum = offset;
        for (int dimension = 0; dimension < shape.length; dimension++) {
            long span = (long) (shape[dimension] - 1) * stride[dimension];
            if (span >= 0) {
                maximum += span;
            } else {
                minimum += span;
            }
        }
        if (minimum < 0 || maximum > (long) data.length - elementWidth) {
            throw new IllegalArgumentException(name + " strides and offset address data outside the array");
        }
    }

    private static long lineCount(int[] shape, int axis) {
        long result = 1;
        for (int dimension = 0; dimension < shape.length; dimension++) {
            if (dimension != axis) {
                result = Math.multiplyExact(result, shape[dimension]);
            }
        }
        return result;
    }

    private static long planeCount(int[] shape, int firstAxis, int secondAxis) {
        long result = 1;
        for (int dimension = 0; dimension < shape.length; dimension++) {
            if (dimension != firstAxis && dimension != secondAxis) {
                result = Math.multiplyExact(result, shape[dimension]);
            }
        }
        return result;
    }

    static long product(int[] shape) {
        long result = 1;
        for (int extent : shape) {
            result = Math.multiplyExact(result, extent);
        }
        return result;
    }

    static int[] hermitianShape(int[] realShape, int axis) {
        int[] complexShape = realShape.clone();
        complexShape[axis] = realShape[axis] / 2 + 1;
        return complexShape;
    }

    static long[] contiguousComplexStride(int[] shape) {
        long[] stride = new long[shape.length];
        stride[shape.length - 1] = 2L;
        for (int dimension = shape.length - 2; dimension >= 0; dimension--) {
            stride[dimension] = Math.multiplyExact(stride[dimension + 1], shape[dimension + 1]);
        }
        return stride;
    }

    private static final class MultiIter {
        private final int[] shapeIn;
        private final int[] shapeOut;
        private final long[] inputStride;
        private final long[] outputStride;
        private final int axis;
        private final int[] position;
        private int inputBase0;
        private int outputBase0;
        private final int strideIn;
        private final int strideOut;
        private int nextInputBase;
        private int nextOutputBase;
        private int remaining;

        MultiIter(int[] shape, long[] inputStride, long inputOffset, long[] outputStride, long outputOffset,
                  int axis) {
            this(shape, shape, inputStride, inputOffset, outputStride, outputOffset, axis);
        }

        MultiIter(int[] shapeIn, int[] shapeOut, long[] inputStride, long inputOffset, long[] outputStride,
                  long outputOffset, int axis) {
            this.shapeIn = shapeIn;
            this.shapeOut = shapeOut;
            this.inputStride = inputStride;
            this.outputStride = outputStride;
            this.axis = axis;
            this.position = new int[shapeIn.length];
            this.strideIn = (int) inputStride[axis];
            this.strideOut = (int) outputStride[axis];
            this.nextInputBase = (int) inputOffset;
            this.nextOutputBase = (int) outputOffset;
            this.remaining = (int) lineCount(shapeIn, axis);
        }

        void advance(int count) {
            for (int lane = 0; lane < count; lane++) {
                inputBase0 = nextInputBase;
                outputBase0 = nextOutputBase;
                advanceInputOutputBases();
            }
            remaining -= count;
        }

        private void advanceInputOutputBases() {
            for (int dimension = shapeIn.length - 1; dimension >= 0; dimension--) {
                if (dimension == axis) {
                    continue;
                }
                nextInputBase += (int) inputStride[dimension];
                nextOutputBase += (int) outputStride[dimension];
                if (++position[dimension] < shapeIn[dimension]) {
                    return;
                }
                position[dimension] = 0;
                nextInputBase -= (int) inputStride[dimension] * shapeIn[dimension];
                nextOutputBase -= (int) outputStride[dimension] * shapeOut[dimension];
            }
        }

        int iofs(int element) {
            return inputBase0 + element * strideIn;
        }

        int oofs(int element) {
            return outputBase0 + element * strideOut;
        }

        int lengthIn() {
            return shapeIn[axis];
        }

        int lengthOut() {
            return shapeOut[axis];
        }

        int strideIn() {
            return strideIn;
        }

        int strideOut() {
            return strideOut;
        }

        int remaining() {
            return remaining;
        }
    }

    private static final class PlaneIter {
        private final int[] shapeIn;
        private final int[] shapeOut;
        private final long[] inputStride;
        private final long[] outputStride;
        private final int firstAxis;
        private final int secondAxis;
        private final int[] position;
        private int inputBase0;
        private int outputBase0;
        private int nextInputBase;
        private int nextOutputBase;
        private int remaining;

        PlaneIter(int[] shapeIn, int[] shapeOut, long[] inputStride, long inputOffset, long[] outputStride,
                  long outputOffset, int firstAxis, int secondAxis) {
            this.shapeIn = shapeIn;
            this.shapeOut = shapeOut;
            this.inputStride = inputStride;
            this.outputStride = outputStride;
            this.firstAxis = firstAxis;
            this.secondAxis = secondAxis;
            this.position = new int[shapeIn.length];
            this.nextInputBase = (int) inputOffset;
            this.nextOutputBase = (int) outputOffset;
            this.remaining = Math.toIntExact(planeCount(shapeIn, firstAxis, secondAxis));
        }

        void advance() {
            inputBase0 = nextInputBase;
            outputBase0 = nextOutputBase;
            advanceInputOutputBases();
            --remaining;
        }

        private void advanceInputOutputBases() {
            for (int dimension = shapeIn.length - 1; dimension >= 0; dimension--) {
                if (dimension == firstAxis || dimension == secondAxis) {
                    continue;
                }
                nextInputBase += (int) inputStride[dimension];
                nextOutputBase += (int) outputStride[dimension];
                if (++position[dimension] < shapeIn[dimension]) {
                    return;
                }
                position[dimension] = 0;
                nextInputBase -= (int) inputStride[dimension] * shapeIn[dimension];
                nextOutputBase -= (int) outputStride[dimension] * shapeOut[dimension];
            }
        }

        int inputBase() {
            return inputBase0;
        }

        int outputBase() {
            return outputBase0;
        }

        int remaining() {
            return remaining;
        }
    }

    private static final class SimpleIter {
        private final int[] shape;
        private final long[] stride;
        private final int[] position;
        private int offset;
        private int remaining;

        SimpleIter(int[] shape, long[] stride, long offset) {
            this.shape = shape;
            this.stride = stride;
            this.position = new int[shape.length];
            this.offset = (int) offset;
            this.remaining = (int) product(shape);
        }

        void advance() {
            --remaining;
            for (int dimension = shape.length - 1; dimension >= 0; dimension--) {
                offset += (int) stride[dimension];
                if (++position[dimension] < shape[dimension]) {
                    return;
                }
                position[dimension] = 0;
                offset -= (int) stride[dimension] * shape[dimension];
            }
        }

        int offset() {
            return offset;
        }

        int remaining() {
            return remaining;
        }
    }

    private static final class ReverseIter {
        private final int[] shape;
        private final long[] stride;
        private final int[] position;
        private final boolean[] reverseAxis;
        private final boolean[] reverseJump;
        private final int[] iterationShape;
        private int offset;
        private int reverseOffset;
        private int remaining;

        ReverseIter(int[] shape, long[] stride, long offset, int[] axes) {
            this.shape = shape;
            this.stride = stride;
            this.position = new int[shape.length];
            this.reverseAxis = new boolean[shape.length];
            this.reverseJump = new boolean[shape.length];
            Arrays.fill(reverseJump, true);
            for (int axis : axes) {
                reverseAxis[axis] = true;
            }
            int lastAxis = axes[axes.length - 1];
            this.iterationShape = shape.clone();
            this.iterationShape[lastAxis] = shape[lastAxis] / 2 + 1;
            this.offset = (int) offset;
            this.reverseOffset = (int) offset;
            this.remaining = (int) product(iterationShape);
        }

        void advance() {
            --remaining;
            for (int dimension = shape.length - 1; dimension >= 0; dimension--) {
                int s = (int) stride[dimension];
                offset += s;
                if (!reverseAxis[dimension]) {
                    reverseOffset += s;
                } else {
                    reverseOffset -= s;
                    if (reverseJump[dimension]) {
                        reverseOffset += s * shape[dimension];
                        reverseJump[dimension] = false;
                    }
                }
                if (++position[dimension] < iterationShape[dimension]) {
                    return;
                }
                position[dimension] = 0;
                offset -= s * iterationShape[dimension];
                if (reverseAxis[dimension]) {
                    reverseOffset -= s * (shape[dimension] - iterationShape[dimension]);
                    reverseJump[dimension] = true;
                } else {
                    reverseOffset -= s * iterationShape[dimension];
                }
            }
        }

        int offset() {
            return offset;
        }

        int reverseOffset() {
            return reverseOffset;
        }

        int remaining() {
            return remaining;
        }
    }

    private static void execC2cForward(MultiIter line, double[] input, double[] output, double[] buffer,
                                       int bufferOffset, FftComplexTransform transform, double factor,
                                       FftWorkspace workspace) {
        copyInputComplex(line, input, buffer, bufferOffset);
        transform.exec(buffer, bufferOffset, factor, true, workspace);
        copyOutputComplex(line, buffer, bufferOffset, output);
    }

    private static void execC2cBackward(MultiIter line, double[] input, double[] output, double[] buffer,
                                        int bufferOffset, FftComplexTransform transform, double factor,
                                        FftWorkspace workspace) {
        copyInputComplex(line, input, buffer, bufferOffset);
        transform.exec(buffer, bufferOffset, factor, false, workspace);
        copyOutputComplex(line, buffer, bufferOffset, output);
    }

    private static void execR2hForward(MultiIter line, double[] input, double[] output, double[] buffer,
                                       int bufferOffset, FftRealTransform transform, double factor,
                                       FftWorkspace workspace) {
        copyInputReal(line, input, buffer, bufferOffset);
        transform.exec(buffer, bufferOffset, factor, true, workspace);
        copyOutputReal(line, buffer, bufferOffset, output);
    }

    private static void execR2hBackward(MultiIter line, double[] input, double[] output, double[] buffer,
                                        int bufferOffset, FftRealTransform transform, double factor,
                                        FftWorkspace workspace) {
        copyInputReal(line, input, buffer, bufferOffset);
        transform.exec(buffer, bufferOffset, factor, true, workspace);
        negateHalfComplexImaginary(buffer, bufferOffset, line.lengthOut());
        copyOutputReal(line, buffer, bufferOffset, output);
    }

    private static void execH2rForward(MultiIter line, double[] input, double[] output, double[] buffer,
                                       int bufferOffset, FftRealTransform transform, double factor,
                                       FftWorkspace workspace) {
        copyInputReal(line, input, buffer, bufferOffset);
        negateHalfComplexImaginary(buffer, bufferOffset, line.lengthOut());
        transform.exec(buffer, bufferOffset, factor, false, workspace);
        copyOutputReal(line, buffer, bufferOffset, output);
    }

    private static void execH2rBackward(MultiIter line, double[] input, double[] output, double[] buffer,
                                        int bufferOffset, FftRealTransform transform, double factor,
                                        FftWorkspace workspace) {
        copyInputReal(line, input, buffer, bufferOffset);
        transform.exec(buffer, bufferOffset, factor, false, workspace);
        copyOutputReal(line, buffer, bufferOffset, output);
    }

    private static void execHartley(MultiIter line, double[] input, double[] output, double[] buffer,
                                    int bufferOffset, FftRealTransform transform, double factor,
                                    FftWorkspace workspace) {
        copyInputReal(line, input, buffer, bufferOffset);
        transform.exec(buffer, bufferOffset, factor, true, workspace);
        copyHartleyLine(line, buffer, bufferOffset, output);
    }

    private static void execHartleyInPlace(MultiIter line, double[] input, double[] buffer, int bufferOffset,
                                           FftRealTransform transform, double factor, FftWorkspace workspace,
                                           int transformMark) {
        copyInputReal(line, input, buffer, bufferOffset);
        transform.exec(buffer, bufferOffset, factor, true, workspace);
        workspace.rewind(transformMark);
        halfComplexToHartleyInPlace(buffer, bufferOffset, line.lengthOut(), false, workspace);
    }

    private static void execFht(MultiIter line, double[] input, double[] output, double[] buffer,
                                int bufferOffset, FftRealTransform transform, double factor,
                                FftWorkspace workspace) {
        copyInputReal(line, input, buffer, bufferOffset);
        transform.exec(buffer, bufferOffset, factor, true, workspace);
        copyFhtLine(line, buffer, bufferOffset, output);
    }

    private static void execFhtInPlace(MultiIter line, double[] input, double[] buffer, int bufferOffset,
                                       FftRealTransform transform, double factor, FftWorkspace workspace,
                                       int transformMark) {
        copyInputReal(line, input, buffer, bufferOffset);
        transform.exec(buffer, bufferOffset, factor, true, workspace);
        workspace.rewind(transformMark);
        halfComplexToHartleyInPlace(buffer, bufferOffset, line.lengthOut(), true, workspace);
    }

    private static void execDcst(MultiIter line, double[] input, double[] output, double[] buffer,
                                 int bufferOffset, FftDcstPlan plan, double factor, boolean ortho, int type,
                                 boolean cosine, FftWorkspace workspace) {
        copyInputReal(line, input, buffer, bufferOffset);
        plan.exec(buffer, bufferOffset, factor, ortho, type, cosine, workspace);
        copyOutputReal(line, buffer, bufferOffset, output);
    }

    private static void copyInputComplex(MultiIter line, double[] input, double[] buffer, int bufferOffset) {
        if (input == buffer && line.iofs(0) == bufferOffset && line.strideIn() == 2) {
            return;
        }
        if (line.strideIn() == 2) {
            System.arraycopy(input, line.iofs(0), buffer, bufferOffset, 2 * line.lengthIn());
            return;
        }
        for (int i = 0; i < line.lengthIn(); i++) {
            int source = line.iofs(i);
            buffer[bufferOffset + 2 * i] = input[source];
            buffer[bufferOffset + 2 * i + 1] = input[source + 1];
        }
    }

    private static void copyInputReal(MultiIter line, double[] input, double[] buffer, int bufferOffset) {
        if (input == buffer && line.iofs(0) == bufferOffset && line.strideIn() == 1) {
            return;
        }
        if (line.strideIn() == 1) {
            System.arraycopy(input, line.iofs(0), buffer, bufferOffset, line.lengthIn());
            return;
        }
        for (int i = 0; i < line.lengthIn(); i++) {
            buffer[bufferOffset + i] = input[line.iofs(i)];
        }
    }

    private static void copyOutputComplex(MultiIter line, double[] buffer, int bufferOffset, double[] output) {
        if (buffer == output && bufferOffset == line.oofs(0) && line.strideOut() == 2) {
            return;
        }
        if (line.strideOut() == 2) {
            System.arraycopy(buffer, bufferOffset, output, line.oofs(0), 2 * line.lengthOut());
            return;
        }
        for (int i = 0; i < line.lengthOut(); i++) {
            int target = line.oofs(i);
            output[target] = buffer[bufferOffset + 2 * i];
            output[target + 1] = buffer[bufferOffset + 2 * i + 1];
        }
    }

    private static void copyOutputReal(MultiIter line, double[] buffer, int bufferOffset, double[] output) {
        if (buffer == output && bufferOffset == line.oofs(0) && line.strideOut() == 1) {
            return;
        }
        if (line.strideOut() == 1) {
            System.arraycopy(buffer, bufferOffset, output, line.oofs(0), line.lengthOut());
            return;
        }
        for (int i = 0; i < line.lengthOut(); i++) {
            output[line.oofs(i)] = buffer[bufferOffset + i];
        }
    }

    private static void copyR2cForwardOutput(MultiIter line, double[] tdata, int dataOffset, double[] output) {
        int target = line.oofs(0);
        output[target] = tdata[dataOffset];
        output[target + 1] = 0.0;
        int i = 1;
        int ii = 1;
        for (; i < line.lengthIn() - 1; i += 2, ii++) {
            target = line.oofs(ii);
            output[target] = tdata[dataOffset + i];
            output[target + 1] = tdata[dataOffset + i + 1];
        }
        if (i < line.lengthIn()) {
            target = line.oofs(ii);
            output[target] = tdata[dataOffset + i];
            output[target + 1] = 0.0;
        }
    }

    private static void copyR2cBackwardOutput(MultiIter line, double[] tdata, int dataOffset, double[] output) {
        int target = line.oofs(0);
        output[target] = tdata[dataOffset];
        output[target + 1] = 0.0;
        int i = 1;
        int ii = 1;
        for (; i < line.lengthIn() - 1; i += 2, ii++) {
            target = line.oofs(ii);
            output[target] = tdata[dataOffset + i];
            output[target + 1] = -tdata[dataOffset + i + 1];
        }
        if (i < line.lengthIn()) {
            target = line.oofs(ii);
            output[target] = tdata[dataOffset + i];
            output[target + 1] = 0.0;
        }
    }

    private static void copyC2rForwardInput(MultiIter line, double[] input, double[] tdata, int dataOffset) {
        tdata[dataOffset] = input[line.iofs(0)];
        int i = 1;
        int ii = 1;
        for (; i < line.lengthOut() - 1; i += 2, ii++) {
            int source = line.iofs(ii);
            tdata[dataOffset + i] = input[source];
            tdata[dataOffset + i + 1] = -input[source + 1];
        }
        if (i < line.lengthOut()) {
            tdata[dataOffset + i] = input[line.iofs(ii)];
        }
    }

    private static void copyC2rBackwardInput(MultiIter line, double[] input, double[] tdata, int dataOffset) {
        tdata[dataOffset] = input[line.iofs(0)];
        int i = 1;
        int ii = 1;
        for (; i < line.lengthOut() - 1; i += 2, ii++) {
            int source = line.iofs(ii);
            tdata[dataOffset + i] = input[source];
            tdata[dataOffset + i + 1] = input[source + 1];
        }
        if (i < line.lengthOut()) {
            tdata[dataOffset + i] = input[line.iofs(ii)];
        }
    }

    private static void copyHartleyLine(MultiIter line, double[] source, int sourceOffset, double[] output) {
        output[line.oofs(0)] = source[sourceOffset];
        int i = 1;
        int i1 = 1;
        int i2 = line.lengthOut() - 1;
        for (; i < line.lengthOut() - 1; i += 2, i1++, i2--) {
            output[line.oofs(i1)] = source[sourceOffset + i] + source[sourceOffset + i + 1];
            output[line.oofs(i2)] = source[sourceOffset + i] - source[sourceOffset + i + 1];
        }
        if (i < line.lengthOut()) {
            output[line.oofs(i1)] = source[sourceOffset + i];
        }
    }

    private static void copyFhtLine(MultiIter line, double[] source, int sourceOffset, double[] output) {
        output[line.oofs(0)] = source[sourceOffset];
        int i = 1;
        int i1 = 1;
        int i2 = line.lengthOut() - 1;
        for (; i < line.lengthOut() - 1; i += 2, i1++, i2--) {
            output[line.oofs(i1)] = source[sourceOffset + i] - source[sourceOffset + i + 1];
            output[line.oofs(i2)] = source[sourceOffset + i] + source[sourceOffset + i + 1];
        }
        if (i < line.lengthOut()) {
            output[line.oofs(i1)] = source[sourceOffset + i];
        }
    }

    private static void halfComplexToHartleyInPlace(double[] data, int dataOffset, int length, boolean fht,
                                                    FftWorkspace workspace) {
        int pairCount = length / 2;
        if (pairCount == 0) {
            return;
        }
        double[] highValues = workspace.doubles();
        int highOffset = workspace.alloc(pairCount);
        int highIndex = 0;
        int frequency = 1;
        for (; 2 * frequency < length; frequency++) {
            double real = data[dataOffset + 2 * frequency - 1];
            double imaginary = data[dataOffset + 2 * frequency];
            data[dataOffset + frequency] = fht ? real - imaginary : real + imaginary;
            highValues[highOffset + highIndex++] = fht ? real + imaginary : real - imaginary;
        }
        if (2 * frequency == length) {
            data[dataOffset + frequency] = data[dataOffset + length - 1];
        }
        for (int index = 0; index < highIndex; index++) {
            data[dataOffset + length - 1 - index] = highValues[highOffset + index];
        }
    }

    private static void reconstructGenuineHartley(int[] outputShape, int[] tempShape, long[] tempStride,
                                                  long tempOffset, long[] outputStride, long outputOffset, int[] axes,
                                                  double[] temp, double[] output) {
        SimpleIter input = new SimpleIter(tempShape, tempStride, tempOffset);
        ReverseIter reverseOutput = new ReverseIter(outputShape, outputStride, outputOffset, axes);
        if (input.remaining() != reverseOutput.remaining()) {
            throw new IllegalStateException("iterator size mismatch");
        }
        while (input.remaining() > 0) {
            int source = input.offset();
            double real = temp[source];
            double imag = temp[source + 1];
            output[reverseOutput.offset()] = real + imag;
            output[reverseOutput.reverseOffset()] = real - imag;
            input.advance();
            reverseOutput.advance();
        }
    }

    private static void reconstructGenuineFht(int[] outputShape, int[] tempShape, long[] tempStride,
                                              long tempOffset, long[] outputStride, long outputOffset, int[] axes,
                                              double[] temp, double[] output) {
        SimpleIter input = new SimpleIter(tempShape, tempStride, tempOffset);
        ReverseIter reverseOutput = new ReverseIter(outputShape, outputStride, outputOffset, axes);
        if (input.remaining() != reverseOutput.remaining()) {
            throw new IllegalStateException("iterator size mismatch");
        }
        while (input.remaining() > 0) {
            int source = input.offset();
            double real = temp[source];
            double imag = temp[source + 1];
            output[reverseOutput.offset()] = real - imag;
            output[reverseOutput.reverseOffset()] = real + imag;
            input.advance();
            reverseOutput.advance();
        }
    }

    private static void negateHalfComplexImaginary(double[] halfComplex, int halfComplexOffset, int length) {
        for (int index = 2; index < length; index += 2) {
            halfComplex[halfComplexOffset + index] = -halfComplex[halfComplexOffset + index];
        }
    }
}