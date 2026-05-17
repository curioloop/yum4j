package com.curioloop.yum4j.fft;

import com.curioloop.yum4j.math.VectorOps;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Objects;

/**
 * Source-level Java port of PocketFFT {@code cfftp<T0>}.
 *
 * <p>Upstream mapping: {@code reference/pocketfft-upstream/pocketfft_hdronly.h},
 * class {@code cfftp<T0>}: {@code fctdata}, {@code factorize}, {@code twsize},
 * {@code comp_twiddle}, {@code pass_all}, {@code pass2/3/4/5/7/8/11}, and {@code passg}.
 * Complex values use the public Java interleaved {@code double[]} layout, but factor order,
 * twiddle layout, buffer swapping, generic-pass behavior, and scale placement follow upstream.
 */
final class FftComplexPlan {
    private static final double HSQT2 = 0.707106781186547524400844362104849;

    /**
     * @param twOffset  offset into shared mem array for regular twiddles
     * @param twsOffset offset into shared mem array for generic twiddles (-1 if none)
     */
    private value record FactorData(int factor, int twOffset, int twsOffset) {}
    
    private final int length;
    private final double[] mem;     // single contiguous twiddle storage
    private final FactorData[] factors;
    private final int genericWorkspaceLength;
    private final FftWorkspace.Requirement workspaceRequirement;
    
    FftComplexPlan(int length) {
        if (length < 1) {
            throw new IllegalArgumentException("zero-length FFT requested");
        }
        this.length = length;
        if (length == 1) {
            this.factors = new FactorData[0];
            this.mem = null;
            this.genericWorkspaceLength = 0;
            this.workspaceRequirement = FftWorkspace.Requirement.empty();
            return;
        }
        
        this.factors = factorize();
        this.mem = new double[computeTwiddleSize()];
        computeTwiddles();
        this.genericWorkspaceLength = computeGenericWorkspaceLength();
        this.workspaceRequirement = genericWorkspaceLength == 0 ? FftWorkspace.Requirement.doubles(2 * length)
                : FftWorkspace.Requirement.doubles(2 * length, genericWorkspaceLength);
    }
    
    /**
     * Execute complex-to-complex FFT.
     * @param data interleaved real/imaginary data (length = 2 * this.length)
     * @param factor scaling factor
     * @param forward true for forward transform, false for inverse
     */
    void exec(double[] data, double factor, boolean forward) {
        FftWorkspace workspace = workspaceRequirement().allocate();
        workspace.reset();
        exec(data, factor, forward, workspace);
    }

    void exec(double[] data, double factor, boolean forward, FftWorkspace workspace) {
        exec(data, 0, factor, forward, workspace);
    }

    void exec(double[] data, int dataOffset, double factor, boolean forward, FftWorkspace workspace) {
        Objects.requireNonNull(workspace, "workspace");
        if (dataOffset < 0 || data.length - dataOffset < 2 * length) {
            throw new IllegalArgumentException("data length is smaller than transform length");
        }
        if (length == 1) {
            data[dataOffset] *= factor;
            data[dataOffset + 1] *= factor;
            return;
        }

        double[] arena = workspace.doubles();
        int workOffset = workspace.alloc(2 * length);
        int genericWorkOffset = genericWorkspaceLength == 0 ? -1 : workspace.alloc(genericWorkspaceLength);
        double[] input = data;
        int inputOffset = dataOffset;
        double[] output = arena;
        int outputOffset = workOffset;
        
        int l1 = 1;
        for (FactorData factorData : factors) {
            int ip = factorData.factor;
            int l2 = ip * l1;
            int ido = length / l2;
            int twOff = factorData.twOffset;
            
            switch (ip) {
                case 4:
                    if (forward) {
                        pass4Forward(ido, l1, input, inputOffset, output, outputOffset, mem, twOff);
                    } else {
                        pass4Backward(ido, l1, input, inputOffset, output, outputOffset, mem, twOff);
                    }
                    break;
                case 8:
                    if (forward) {
                        pass8Forward(ido, l1, input, inputOffset, output, outputOffset, mem, twOff);
                    } else {
                        pass8Backward(ido, l1, input, inputOffset, output, outputOffset, mem, twOff);
                    }
                    break;
                case 2:
                    if (forward) {
                        pass2Forward(ido, l1, input, inputOffset, output, outputOffset, mem, twOff);
                    } else {
                        pass2Backward(ido, l1, input, inputOffset, output, outputOffset, mem, twOff);
                    }
                    break;
                case 3:
                    if (forward) {
                        pass3Forward(ido, l1, input, inputOffset, output, outputOffset, mem, twOff);
                    } else {
                        pass3Backward(ido, l1, input, inputOffset, output, outputOffset, mem, twOff);
                    }
                    break;
                case 5:
                    if (forward) {
                        pass5Forward(ido, l1, input, inputOffset, output, outputOffset, mem, twOff);
                    } else {
                        pass5Backward(ido, l1, input, inputOffset, output, outputOffset, mem, twOff);
                    }
                    break;
                case 7:
                    if (forward) {
                        pass7Forward(ido, l1, input, inputOffset, output, outputOffset, mem, twOff);
                    } else {
                        pass7Backward(ido, l1, input, inputOffset, output, outputOffset, mem, twOff);
                    }
                    break;
                case 11:
                    if (forward) {
                        pass11Forward(ido, l1, input, inputOffset, output, outputOffset, mem, twOff);
                    } else {
                        pass11Backward(ido, l1, input, inputOffset, output, outputOffset, mem, twOff);
                    }
                    break;
                default:
                    if (forward) {
                        passgForward(ido, ip, l1, input, inputOffset, output, outputOffset, mem, twOff,
                                mem, factorData.twsOffset, arena, genericWorkOffset);
                    } else {
                        passgBackward(ido, ip, l1, input, inputOffset, output, outputOffset, mem, twOff,
                                mem, factorData.twsOffset, arena, genericWorkOffset);
                    }
                    // Generic pass swaps buffers - undo it
                    double[] temp = input;
                    input = output;
                    output = temp;
                    int tempOffset = inputOffset;
                    inputOffset = outputOffset;
                    outputOffset = tempOffset;
                    break;
            }
            
            // Swap input and output buffers
            double[] temp = input;
            input = output;
            output = temp;
            int tempOffset = inputOffset;
            inputOffset = outputOffset;
            outputOffset = tempOffset;
            l1 = l2;
        }
        
        VectorOps.scalTo(data, dataOffset, factor, input, inputOffset, 2 * length);
    }

    FftWorkspace.Requirement workspaceRequirement() {
        return workspaceRequirement;
    }

    private int computeGenericWorkspaceLength() {
        int result = 0;
        for (FactorData factorData : factors) {
            if (factorData.factor > 11) {
                result = Math.max(result, 2 * factorData.factor);
            }
        }
        return result;
    }
    
    private FactorData[] factorize() {
        ArrayList<FactorData> result = new ArrayList<>();
        int len = length;
        
        // Factor out powers of 8
        while ((len & 7) == 0) {
            addFactor(result, 8);
            len >>>= 3;
        }
        
        // Factor out powers of 4
        while ((len & 3) == 0) {
            addFactor(result, 4);
            len >>>= 2;
        }
        
        // Factor out one power of 2 if present, but add it to the front
        if ((len & 1) == 0) {
            len >>>= 1;
            addFactor(result, 2);
            Collections.swap(result, 0, result.size() - 1);
        }
        
        // Factor out odd factors starting from 3
        for (int divisor = 3; divisor * divisor <= len; divisor += 2) {
            while (len % divisor == 0) {
                addFactor(result, divisor);
                len /= divisor;
            }
        }
        
        // If len > 1, it's a prime factor
        if (len > 1) {
            addFactor(result, len);
        }
        return result.toArray(FactorData[]::new);
    }
    
    private static void addFactor(ArrayList<FactorData> factors, int factor) {
        factors.add(new FactorData(factor, -1, -1));
    }
    
    /** Compute total twiddle memory size in doubles (matching pocketfft twsize). */
    private int computeTwiddleSize() {
        int twsize = 0;
        int l1 = 1;
        for (FactorData factor : factors) {
            int ip = factor.factor;
            int ido = length / (l1 * ip);
            twsize += 2 * (ip - 1) * (ido - 1);
            if (ip > 11) {
                twsize += 2 * ip;
            }
            l1 *= ip;
        }
        return twsize;
    }
    
    private void computeTwiddles() {
        if (factors.length == 0) return;
        
        FftSincos twiddle = new FftSincos(length);
        int l1 = 1;
        int memOffset = 0;
        
        for (int k = 0; k < factors.length; k++) {
            int ip = factors[k].factor;
            int ido = length / (l1 * ip);
            
            // Regular twiddle factors
            int twOffset = memOffset;
            for (int j = 1; j < ip; j++) {
                for (int i = 1; i < ido; i++) {
                    long twiddleIndex = (long) j * l1 * i;
                    twiddle.fill(twiddleIndex, mem, memOffset);
                    memOffset += 2;
                }
            }
            
            // Special twiddle factors for generic passes (ip > 11)
            int twsOffset = -1;
            if (ip > 11) {
                twsOffset = memOffset;
                for (int j = 0; j < ip; j++) {
                    long twiddleIndex = (long) j * l1 * ido;
                    twiddle.fill(twiddleIndex, mem, memOffset);
                    memOffset += 2;
                }
            }
            
            factors[k] = new FactorData(ip, twOffset, twsOffset);
            l1 *= ip;
        }
    }
    
    // Complex index helpers for interleaved arrays
    private static int cc(int a, int b, int c, int ido, int ip) {
        return 2 * (a + ido * (b + ip * c));
    }

    private static int cc(int base, int a, int b, int c, int ido, int ip) {
        return base + cc(a, b, c, ido, ip);
    }
    
    private static int ch(int a, int b, int c, int ido, int l1) {
        return 2 * (a + ido * (b + l1 * c));
    }

    private static int ch(int base, int a, int b, int c, int ido, int l1) {
        return base + ch(a, b, c, ido, l1);
    }
    
    private static int wa(int x, int i, int ido) {
        return 2 * (i - 1 + x * (ido - 1));
    }
    
    // Pass 2 implementation
    private void pass2Forward(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        pass2ForwardImpl(ido, l1, cc, ccBase, ch, chBase, wa, waBase);
    }
    
    private void pass2Backward(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        pass2BackwardImpl(ido, l1, cc, ccBase, ch, chBase, wa, waBase);
    }

    private void pass2ForwardImpl(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase,
                       double[] wa, int waBase) {
        int ccKStride = 4 * ido;
        int chKStride = 2 * ido;
        int chU1Offset = 2 * ido * l1;
        for (int k = 0; k < l1; k++) {
            int cc0 = ccBase + k * ccKStride;
            int cc1 = cc0 + 2 * ido;
            int ch0 = chBase + k * chKStride;
            int ch1 = ch0 + chU1Offset;
            double sumReal = cc[cc0] + cc[cc1];
            double sumImag = cc[cc0 + 1] + cc[cc1 + 1];
            double diffReal = cc[cc0] - cc[cc1];
            double diffImag = cc[cc0 + 1] - cc[cc1 + 1];
            ch[ch0] = sumReal;
            ch[ch0 + 1] = sumImag;
            ch[ch1] = diffReal;
            ch[ch1 + 1] = diffImag;
            for (int i = 1; i < ido; i++) {
                int ccI0 = cc0 + 2 * i;
                int ccI1 = cc1 + 2 * i;
                int chI0 = ch0 + 2 * i;
                int chI1 = ch1 + 2 * i;
                sumReal = cc[ccI0] + cc[ccI1];
                sumImag = cc[ccI0 + 1] + cc[ccI1 + 1];
                diffReal = cc[ccI0] - cc[ccI1];
                diffImag = cc[ccI0 + 1] - cc[ccI1 + 1];
                ch[chI0] = sumReal;
                ch[chI0 + 1] = sumImag;
                specialMulForwardTo(diffReal, diffImag, wa, waBase + wa(0, i, ido), ch, chI1);
            }
        }
    }

    private void pass2BackwardImpl(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase,
                       double[] wa, int waBase) {
        int ccKStride = 4 * ido;
        int chKStride = 2 * ido;
        int chU1Offset = 2 * ido * l1;
        for (int k = 0; k < l1; k++) {
            int cc0 = ccBase + k * ccKStride;
            int cc1 = cc0 + 2 * ido;
            int ch0 = chBase + k * chKStride;
            int ch1 = ch0 + chU1Offset;
            double sumReal = cc[cc0] + cc[cc1];
            double sumImag = cc[cc0 + 1] + cc[cc1 + 1];
            double diffReal = cc[cc0] - cc[cc1];
            double diffImag = cc[cc0 + 1] - cc[cc1 + 1];
            ch[ch0] = sumReal;
            ch[ch0 + 1] = sumImag;
            ch[ch1] = diffReal;
            ch[ch1 + 1] = diffImag;
            for (int i = 1; i < ido; i++) {
                int ccI0 = cc0 + 2 * i;
                int ccI1 = cc1 + 2 * i;
                int chI0 = ch0 + 2 * i;
                int chI1 = ch1 + 2 * i;
                sumReal = cc[ccI0] + cc[ccI1];
                sumImag = cc[ccI0 + 1] + cc[ccI1 + 1];
                diffReal = cc[ccI0] - cc[ccI1];
                diffImag = cc[ccI0 + 1] - cc[ccI1 + 1];
                ch[chI0] = sumReal;
                ch[chI0 + 1] = sumImag;
                specialMulBackwardTo(diffReal, diffImag, wa, waBase + wa(0, i, ido), ch, chI1);
            }
        }
    }
    
    // Pass 3 implementation 
    private void pass3Forward(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        pass3ForwardImpl(ido, l1, cc, ccBase, ch, chBase, wa, waBase);
    }
    
    private void pass3Backward(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        pass3BackwardImpl(ido, l1, cc, ccBase, ch, chBase, wa, waBase);
    }

    private void pass3ForwardImpl(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase,
                       double[] wa, int waBase) {
        final double tw1r = -0.5;
        final double tw1i = -0.8660254037844386467637231707529362;
        int ccKStride = 6 * ido;
        int chKStride = 2 * ido;
        int chU1Offset = 2 * ido * l1;
        int chU2Offset = 2 * chU1Offset;
        for (int k = 0; k < l1; k++) {
            int cc0 = ccBase + k * ccKStride;
            int cc1 = cc0 + 2 * ido;
            int cc2 = cc1 + 2 * ido;
            int ch0 = chBase + k * chKStride;
            int ch1 = ch0 + chU1Offset;
            int ch2 = ch0 + chU2Offset;
            double t0r = cc[cc0];
            double t0i = cc[cc0 + 1];
            double t1r = cc[cc1] + cc[cc2];
            double t1i = cc[cc1 + 1] + cc[cc2 + 1];
            double t2r = cc[cc1] - cc[cc2];
            double t2i = cc[cc1 + 1] - cc[cc2 + 1];
            setPair(ch, ch0, t0r + t1r, t0i + t1i);
            double car = t0r + t1r * tw1r;
            double cai = t0i + t1i * tw1r;
            double cbr = -t2i * tw1i;
            double cbi = t2r * tw1i;
            setPair(ch, ch1, car + cbr, cai + cbi);
            setPair(ch, ch2, car - cbr, cai - cbi);
            for (int i = 1; i < ido; i++) {
                int iOffset = 2 * i;
                int ccI0 = cc0 + iOffset;
                int ccI1 = cc1 + iOffset;
                int ccI2 = cc2 + iOffset;
                int chI0 = ch0 + iOffset;
                int chI1 = ch1 + iOffset;
                int chI2 = ch2 + iOffset;
                t0r = cc[ccI0];
                t0i = cc[ccI0 + 1];
                t1r = cc[ccI1] + cc[ccI2];
                t1i = cc[ccI1 + 1] + cc[ccI2 + 1];
                t2r = cc[ccI1] - cc[ccI2];
                t2i = cc[ccI1 + 1] - cc[ccI2 + 1];
                setPair(ch, chI0, t0r + t1r, t0i + t1i);
                car = t0r + t1r * tw1r;
                cai = t0i + t1i * tw1r;
                cbr = -t2i * tw1i;
                cbi = t2r * tw1i;
                specialMulForwardTo(car + cbr, cai + cbi, wa, waBase + wa(0, i, ido), ch, chI1);
                specialMulForwardTo(car - cbr, cai - cbi, wa, waBase + wa(1, i, ido), ch, chI2);
            }
        }
    }

    private void pass3BackwardImpl(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase,
                       double[] wa, int waBase) {
        final double tw1r = -0.5;
        final double tw1i = 0.8660254037844386467637231707529362;
        int ccKStride = 6 * ido;
        int chKStride = 2 * ido;
        int chU1Offset = 2 * ido * l1;
        int chU2Offset = 2 * chU1Offset;
        for (int k = 0; k < l1; k++) {
            int cc0 = ccBase + k * ccKStride;
            int cc1 = cc0 + 2 * ido;
            int cc2 = cc1 + 2 * ido;
            int ch0 = chBase + k * chKStride;
            int ch1 = ch0 + chU1Offset;
            int ch2 = ch0 + chU2Offset;
            double t0r = cc[cc0];
            double t0i = cc[cc0 + 1];
            double t1r = cc[cc1] + cc[cc2];
            double t1i = cc[cc1 + 1] + cc[cc2 + 1];
            double t2r = cc[cc1] - cc[cc2];
            double t2i = cc[cc1 + 1] - cc[cc2 + 1];
            setPair(ch, ch0, t0r + t1r, t0i + t1i);
            double car = t0r + t1r * tw1r;
            double cai = t0i + t1i * tw1r;
            double cbr = -t2i * tw1i;
            double cbi = t2r * tw1i;
            setPair(ch, ch1, car + cbr, cai + cbi);
            setPair(ch, ch2, car - cbr, cai - cbi);
            for (int i = 1; i < ido; i++) {
                int iOffset = 2 * i;
                int ccI0 = cc0 + iOffset;
                int ccI1 = cc1 + iOffset;
                int ccI2 = cc2 + iOffset;
                int chI0 = ch0 + iOffset;
                int chI1 = ch1 + iOffset;
                int chI2 = ch2 + iOffset;
                t0r = cc[ccI0];
                t0i = cc[ccI0 + 1];
                t1r = cc[ccI1] + cc[ccI2];
                t1i = cc[ccI1 + 1] + cc[ccI2 + 1];
                t2r = cc[ccI1] - cc[ccI2];
                t2i = cc[ccI1 + 1] - cc[ccI2 + 1];
                setPair(ch, chI0, t0r + t1r, t0i + t1i);
                car = t0r + t1r * tw1r;
                cai = t0i + t1i * tw1r;
                cbr = -t2i * tw1i;
                cbi = t2r * tw1i;
                specialMulBackwardTo(car + cbr, cai + cbi, wa, waBase + wa(0, i, ido), ch, chI1);
                specialMulBackwardTo(car - cbr, cai - cbi, wa, waBase + wa(1, i, ido), ch, chI2);
            }
        }
    }
    
    private void pass4Forward(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        pass4ForwardImpl(ido, l1, cc, ccBase, ch, chBase, wa, waBase);
    }
    
    private void pass4Backward(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        pass4BackwardImpl(ido, l1, cc, ccBase, ch, chBase, wa, waBase);
    }
    
    private void pass5Forward(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        pass5ForwardImpl(ido, l1, cc, ccBase, ch, chBase, wa, waBase);
    }
    
    private void pass5Backward(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        pass5BackwardImpl(ido, l1, cc, ccBase, ch, chBase, wa, waBase);
    }

    private void pass4ForwardImpl(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        int ccKStride = 8 * ido;
        int chKStride = 2 * ido;
        int chStride = 2 * ido * l1;
        for (int k = 0; k < l1; k++) {
            int cc0 = ccBase + k * ccKStride;
            int cc1 = cc0 + 2 * ido;
            int cc2 = cc1 + 2 * ido;
            int cc3 = cc2 + 2 * ido;
            int ch0 = chBase + k * chKStride;
            double t2r = cc[cc0] + cc[cc2];
            double t2i = cc[cc0 + 1] + cc[cc2 + 1];
            double t1r = cc[cc0] - cc[cc2];
            double t1i = cc[cc0 + 1] - cc[cc2 + 1];
            double t3r = cc[cc1] + cc[cc3];
            double t3i = cc[cc1 + 1] + cc[cc3 + 1];
            double t4r = cc[cc1] - cc[cc3];
            double t4i = cc[cc1 + 1] - cc[cc3 + 1];
            double rot4r = t4i;
            double rot4i = -t4r;
            setPair(ch, ch0, t2r + t3r, t2i + t3i);
            setPair(ch, ch0 + 2 * chStride, t2r - t3r, t2i - t3i);
            setPair(ch, ch0 + chStride, t1r + rot4r, t1i + rot4i);
            setPair(ch, ch0 + 3 * chStride, t1r - rot4r, t1i - rot4i);

            for (int i = 1; i < ido; i++) {
                int iOffset = 2 * i;
                cc0 = ccBase + k * ccKStride + iOffset;
                cc1 = cc0 + 2 * ido;
                cc2 = cc1 + 2 * ido;
                cc3 = cc2 + 2 * ido;
                ch0 = chBase + k * chKStride + iOffset;
                t2r = cc[cc0] + cc[cc2];
                t2i = cc[cc0 + 1] + cc[cc2 + 1];
                t1r = cc[cc0] - cc[cc2];
                t1i = cc[cc0 + 1] - cc[cc2 + 1];
                t3r = cc[cc1] + cc[cc3];
                t3i = cc[cc1 + 1] + cc[cc3 + 1];
                t4r = cc[cc1] - cc[cc3];
                t4i = cc[cc1 + 1] - cc[cc3 + 1];
                rot4r = t4i;
                rot4i = -t4r;
                setPair(ch, ch0, t2r + t3r, t2i + t3i);
                specialMulForwardTo(t1r + rot4r, t1i + rot4i, wa, waBase + wa(0, i, ido), ch, ch0 + chStride);
                specialMulForwardTo(t2r - t3r, t2i - t3i, wa, waBase + wa(1, i, ido), ch, ch0 + 2 * chStride);
                specialMulForwardTo(t1r - rot4r, t1i - rot4i, wa, waBase + wa(2, i, ido), ch, ch0 + 3 * chStride);
            }
        }
    }

    private void pass4BackwardImpl(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        int ccKStride = 8 * ido;
        int chKStride = 2 * ido;
        int chStride = 2 * ido * l1;
        for (int k = 0; k < l1; k++) {
            int cc0 = ccBase + k * ccKStride;
            int cc1 = cc0 + 2 * ido;
            int cc2 = cc1 + 2 * ido;
            int cc3 = cc2 + 2 * ido;
            int ch0 = chBase + k * chKStride;
            double t2r = cc[cc0] + cc[cc2];
            double t2i = cc[cc0 + 1] + cc[cc2 + 1];
            double t1r = cc[cc0] - cc[cc2];
            double t1i = cc[cc0 + 1] - cc[cc2 + 1];
            double t3r = cc[cc1] + cc[cc3];
            double t3i = cc[cc1 + 1] + cc[cc3 + 1];
            double t4r = cc[cc1] - cc[cc3];
            double t4i = cc[cc1 + 1] - cc[cc3 + 1];
            double rot4r = -t4i;
            double rot4i = t4r;
            setPair(ch, ch0, t2r + t3r, t2i + t3i);
            setPair(ch, ch0 + 2 * chStride, t2r - t3r, t2i - t3i);
            setPair(ch, ch0 + chStride, t1r + rot4r, t1i + rot4i);
            setPair(ch, ch0 + 3 * chStride, t1r - rot4r, t1i - rot4i);

            for (int i = 1; i < ido; i++) {
                int iOffset = 2 * i;
                cc0 = ccBase + k * ccKStride + iOffset;
                cc1 = cc0 + 2 * ido;
                cc2 = cc1 + 2 * ido;
                cc3 = cc2 + 2 * ido;
                ch0 = chBase + k * chKStride + iOffset;
                t2r = cc[cc0] + cc[cc2];
                t2i = cc[cc0 + 1] + cc[cc2 + 1];
                t1r = cc[cc0] - cc[cc2];
                t1i = cc[cc0 + 1] - cc[cc2 + 1];
                t3r = cc[cc1] + cc[cc3];
                t3i = cc[cc1 + 1] + cc[cc3 + 1];
                t4r = cc[cc1] - cc[cc3];
                t4i = cc[cc1 + 1] - cc[cc3 + 1];
                rot4r = -t4i;
                rot4i = t4r;
                setPair(ch, ch0, t2r + t3r, t2i + t3i);
                specialMulBackwardTo(t1r + rot4r, t1i + rot4i, wa, waBase + wa(0, i, ido), ch, ch0 + chStride);
                specialMulBackwardTo(t2r - t3r, t2i - t3i, wa, waBase + wa(1, i, ido), ch, ch0 + 2 * chStride);
                specialMulBackwardTo(t1r - rot4r, t1i - rot4i, wa, waBase + wa(2, i, ido), ch, ch0 + 3 * chStride);
            }
        }
    }

    private void pass5ForwardImpl(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        double tw1r = 0.3090169943749474241022934171828191;
        double tw1i = -0.9510565162951535721164393333793821;
        double tw2r = -0.8090169943749474241022934171828191;
        double tw2i = -0.5877852522924731291687059546390728;
        int ccKStride = 10 * ido;
        int chKStride = 2 * ido;
        int chStride = 2 * ido * l1;
        for (int k = 0; k < l1; k++) {
            int cc0 = ccBase + k * ccKStride;
            int cc1 = cc0 + 2 * ido;
            int cc2 = cc1 + 2 * ido;
            int cc3 = cc2 + 2 * ido;
            int cc4 = cc3 + 2 * ido;
            int ch0 = chBase + k * chKStride;
            double t0r = cc[cc0];
            double t0i = cc[cc0 + 1];
            double t1r = cc[cc1] + cc[cc4];
            double t1i = cc[cc1 + 1] + cc[cc4 + 1];
            double t4r = cc[cc1] - cc[cc4];
            double t4i = cc[cc1 + 1] - cc[cc4 + 1];
            double t2r = cc[cc2] + cc[cc3];
            double t2i = cc[cc2 + 1] + cc[cc3 + 1];
            double t3r = cc[cc2] - cc[cc3];
            double t3i = cc[cc2 + 1] - cc[cc3 + 1];
            setPair(ch, ch0, t0r + t1r + t2r, t0i + t1i + t2i);
            double car = t0r + tw1r * t1r + tw2r * t2r;
            double cai = t0i + tw1r * t1i + tw2r * t2i;
            double cbr = -(tw1i * t4i + tw2i * t3i);
            double cbi = tw1i * t4r + tw2i * t3r;
            setPair(ch, ch0 + chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 4 * chStride, car - cbr, cai - cbi);
            car = t0r + tw2r * t1r + tw1r * t2r;
            cai = t0i + tw2r * t1i + tw1r * t2i;
            cbr = -(tw2i * t4i - tw1i * t3i);
            cbi = tw2i * t4r - tw1i * t3r;
            setPair(ch, ch0 + 2 * chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 3 * chStride, car - cbr, cai - cbi);

            for (int i = 1; i < ido; i++) {
                int iOffset = 2 * i;
                cc0 = ccBase + k * ccKStride + iOffset;
                cc1 = cc0 + 2 * ido;
                cc2 = cc1 + 2 * ido;
                cc3 = cc2 + 2 * ido;
                cc4 = cc3 + 2 * ido;
                ch0 = chBase + k * chKStride + iOffset;
                t0r = cc[cc0];
                t0i = cc[cc0 + 1];
                t1r = cc[cc1] + cc[cc4];
                t1i = cc[cc1 + 1] + cc[cc4 + 1];
                t4r = cc[cc1] - cc[cc4];
                t4i = cc[cc1 + 1] - cc[cc4 + 1];
                t2r = cc[cc2] + cc[cc3];
                t2i = cc[cc2 + 1] + cc[cc3 + 1];
                t3r = cc[cc2] - cc[cc3];
                t3i = cc[cc2 + 1] - cc[cc3 + 1];
                setPair(ch, ch0, t0r + t1r + t2r, t0i + t1i + t2i);
                car = t0r + tw1r * t1r + tw2r * t2r;
                cai = t0i + tw1r * t1i + tw2r * t2i;
                cbr = -(tw1i * t4i + tw2i * t3i);
                cbi = tw1i * t4r + tw2i * t3r;
                specialMulForwardTo(car + cbr, cai + cbi, wa, waBase + wa(0, i, ido), ch, ch0 + chStride);
                specialMulForwardTo(car - cbr, cai - cbi, wa, waBase + wa(3, i, ido), ch, ch0 + 4 * chStride);
                car = t0r + tw2r * t1r + tw1r * t2r;
                cai = t0i + tw2r * t1i + tw1r * t2i;
                cbr = -(tw2i * t4i - tw1i * t3i);
                cbi = tw2i * t4r - tw1i * t3r;
                specialMulForwardTo(car + cbr, cai + cbi, wa, waBase + wa(1, i, ido), ch, ch0 + 2 * chStride);
                specialMulForwardTo(car - cbr, cai - cbi, wa, waBase + wa(2, i, ido), ch, ch0 + 3 * chStride);
            }
        }
    }

    private void pass5BackwardImpl(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        double tw1r = 0.3090169943749474241022934171828191;
        double tw1i = 0.9510565162951535721164393333793821;
        double tw2r = -0.8090169943749474241022934171828191;
        double tw2i = 0.5877852522924731291687059546390728;
        int ccKStride = 10 * ido;
        int chKStride = 2 * ido;
        int chStride = 2 * ido * l1;
        for (int k = 0; k < l1; k++) {
            int cc0 = ccBase + k * ccKStride;
            int cc1 = cc0 + 2 * ido;
            int cc2 = cc1 + 2 * ido;
            int cc3 = cc2 + 2 * ido;
            int cc4 = cc3 + 2 * ido;
            int ch0 = chBase + k * chKStride;
            double t0r = cc[cc0];
            double t0i = cc[cc0 + 1];
            double t1r = cc[cc1] + cc[cc4];
            double t1i = cc[cc1 + 1] + cc[cc4 + 1];
            double t4r = cc[cc1] - cc[cc4];
            double t4i = cc[cc1 + 1] - cc[cc4 + 1];
            double t2r = cc[cc2] + cc[cc3];
            double t2i = cc[cc2 + 1] + cc[cc3 + 1];
            double t3r = cc[cc2] - cc[cc3];
            double t3i = cc[cc2 + 1] - cc[cc3 + 1];
            setPair(ch, ch0, t0r + t1r + t2r, t0i + t1i + t2i);
            double car = t0r + tw1r * t1r + tw2r * t2r;
            double cai = t0i + tw1r * t1i + tw2r * t2i;
            double cbr = -(tw1i * t4i + tw2i * t3i);
            double cbi = tw1i * t4r + tw2i * t3r;
            setPair(ch, ch0 + chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 4 * chStride, car - cbr, cai - cbi);
            car = t0r + tw2r * t1r + tw1r * t2r;
            cai = t0i + tw2r * t1i + tw1r * t2i;
            cbr = -(tw2i * t4i - tw1i * t3i);
            cbi = tw2i * t4r - tw1i * t3r;
            setPair(ch, ch0 + 2 * chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 3 * chStride, car - cbr, cai - cbi);

            for (int i = 1; i < ido; i++) {
                int iOffset = 2 * i;
                cc0 = ccBase + k * ccKStride + iOffset;
                cc1 = cc0 + 2 * ido;
                cc2 = cc1 + 2 * ido;
                cc3 = cc2 + 2 * ido;
                cc4 = cc3 + 2 * ido;
                ch0 = chBase + k * chKStride + iOffset;
                t0r = cc[cc0];
                t0i = cc[cc0 + 1];
                t1r = cc[cc1] + cc[cc4];
                t1i = cc[cc1 + 1] + cc[cc4 + 1];
                t4r = cc[cc1] - cc[cc4];
                t4i = cc[cc1 + 1] - cc[cc4 + 1];
                t2r = cc[cc2] + cc[cc3];
                t2i = cc[cc2 + 1] + cc[cc3 + 1];
                t3r = cc[cc2] - cc[cc3];
                t3i = cc[cc2 + 1] - cc[cc3 + 1];
                setPair(ch, ch0, t0r + t1r + t2r, t0i + t1i + t2i);
                car = t0r + tw1r * t1r + tw2r * t2r;
                cai = t0i + tw1r * t1i + tw2r * t2i;
                cbr = -(tw1i * t4i + tw2i * t3i);
                cbi = tw1i * t4r + tw2i * t3r;
                specialMulBackwardTo(car + cbr, cai + cbi, wa, waBase + wa(0, i, ido), ch, ch0 + chStride);
                specialMulBackwardTo(car - cbr, cai - cbi, wa, waBase + wa(3, i, ido), ch, ch0 + 4 * chStride);
                car = t0r + tw2r * t1r + tw1r * t2r;
                cai = t0i + tw2r * t1i + tw1r * t2i;
                cbr = -(tw2i * t4i - tw1i * t3i);
                cbi = tw2i * t4r - tw1i * t3r;
                specialMulBackwardTo(car + cbr, cai + cbi, wa, waBase + wa(1, i, ido), ch, ch0 + 2 * chStride);
                specialMulBackwardTo(car - cbr, cai - cbi, wa, waBase + wa(2, i, ido), ch, ch0 + 3 * chStride);
            }
        }
    }

    private void pass7Forward(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        pass7ForwardImpl(ido, l1, cc, ccBase, ch, chBase, wa, waBase);
    }
    
    private void pass7Backward(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        pass7BackwardImpl(ido, l1, cc, ccBase, ch, chBase, wa, waBase);
    }

    private void pass7ForwardImpl(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        double tw1r = 0.6234898018587335305250048840042398;
        double tw1i = -0.7818314824680298087084445266740578;
        double tw2r = -0.2225209339563144042889025644967948;
        double tw2i = -0.9749279121818236070181316829939312;
        double tw3r = -0.9009688679024191262361023195074451;
        double tw3i = -0.433883739117558120475768332848359;
        int ccKStride = 14 * ido;
        int chKStride = 2 * ido;
        int chStride = 2 * ido * l1;
        for (int k = 0; k < l1; k++) {
            int cc0 = ccBase + k * ccKStride;
            int cc1 = cc0 + 2 * ido;
            int cc2 = cc1 + 2 * ido;
            int cc3 = cc2 + 2 * ido;
            int cc4 = cc3 + 2 * ido;
            int cc5 = cc4 + 2 * ido;
            int cc6 = cc5 + 2 * ido;
            int ch0 = chBase + k * chKStride;
            double t1r = cc[cc0];
            double t1i = cc[cc0 + 1];
            double t2r = cc[cc1] + cc[cc6];
            double t2i = cc[cc1 + 1] + cc[cc6 + 1];
            double t7r = cc[cc1] - cc[cc6];
            double t7i = cc[cc1 + 1] - cc[cc6 + 1];
            double t3r = cc[cc2] + cc[cc5];
            double t3i = cc[cc2 + 1] + cc[cc5 + 1];
            double t6r = cc[cc2] - cc[cc5];
            double t6i = cc[cc2 + 1] - cc[cc5 + 1];
            double t4r = cc[cc3] + cc[cc4];
            double t4i = cc[cc3 + 1] + cc[cc4 + 1];
            double t5r = cc[cc3] - cc[cc4];
            double t5i = cc[cc3 + 1] - cc[cc4 + 1];
            setPair(ch, ch0, t1r + t2r + t3r + t4r, t1i + t2i + t3i + t4i);
            double car = t1r + tw1r * t2r + tw2r * t3r + tw3r * t4r;
            double cai = t1i + tw1r * t2i + tw2r * t3i + tw3r * t4i;
            double cbr = -(tw1i * t7i + tw2i * t6i + tw3i * t5i);
            double cbi = tw1i * t7r + tw2i * t6r + tw3i * t5r;
            setPair(ch, ch0 + chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 6 * chStride, car - cbr, cai - cbi);
            car = t1r + tw2r * t2r + tw3r * t3r + tw1r * t4r;
            cai = t1i + tw2r * t2i + tw3r * t3i + tw1r * t4i;
            cbr = -(tw2i * t7i - tw3i * t6i - tw1i * t5i);
            cbi = tw2i * t7r - tw3i * t6r - tw1i * t5r;
            setPair(ch, ch0 + 2 * chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 5 * chStride, car - cbr, cai - cbi);
            car = t1r + tw3r * t2r + tw1r * t3r + tw2r * t4r;
            cai = t1i + tw3r * t2i + tw1r * t3i + tw2r * t4i;
            cbr = -(tw3i * t7i - tw1i * t6i + tw2i * t5i);
            cbi = tw3i * t7r - tw1i * t6r + tw2i * t5r;
            setPair(ch, ch0 + 3 * chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 4 * chStride, car - cbr, cai - cbi);

            for (int i = 1; i < ido; i++) {
                int iOffset = 2 * i;
                cc0 = ccBase + k * ccKStride + iOffset;
                cc1 = cc0 + 2 * ido;
                cc2 = cc1 + 2 * ido;
                cc3 = cc2 + 2 * ido;
                cc4 = cc3 + 2 * ido;
                cc5 = cc4 + 2 * ido;
                cc6 = cc5 + 2 * ido;
                ch0 = chBase + k * chKStride + iOffset;
                t1r = cc[cc0];
                t1i = cc[cc0 + 1];
                t2r = cc[cc1] + cc[cc6];
                t2i = cc[cc1 + 1] + cc[cc6 + 1];
                t7r = cc[cc1] - cc[cc6];
                t7i = cc[cc1 + 1] - cc[cc6 + 1];
                t3r = cc[cc2] + cc[cc5];
                t3i = cc[cc2 + 1] + cc[cc5 + 1];
                t6r = cc[cc2] - cc[cc5];
                t6i = cc[cc2 + 1] - cc[cc5 + 1];
                t4r = cc[cc3] + cc[cc4];
                t4i = cc[cc3 + 1] + cc[cc4 + 1];
                t5r = cc[cc3] - cc[cc4];
                t5i = cc[cc3 + 1] - cc[cc4 + 1];
                setPair(ch, ch0, t1r + t2r + t3r + t4r, t1i + t2i + t3i + t4i);
                car = t1r + tw1r * t2r + tw2r * t3r + tw3r * t4r;
                cai = t1i + tw1r * t2i + tw2r * t3i + tw3r * t4i;
                cbr = -(tw1i * t7i + tw2i * t6i + tw3i * t5i);
                cbi = tw1i * t7r + tw2i * t6r + tw3i * t5r;
                specialMulForwardTo(car + cbr, cai + cbi, wa, waBase + wa(0, i, ido), ch, ch0 + chStride);
                specialMulForwardTo(car - cbr, cai - cbi, wa, waBase + wa(5, i, ido), ch, ch0 + 6 * chStride);
                car = t1r + tw2r * t2r + tw3r * t3r + tw1r * t4r;
                cai = t1i + tw2r * t2i + tw3r * t3i + tw1r * t4i;
                cbr = -(tw2i * t7i - tw3i * t6i - tw1i * t5i);
                cbi = tw2i * t7r - tw3i * t6r - tw1i * t5r;
                specialMulForwardTo(car + cbr, cai + cbi, wa, waBase + wa(1, i, ido), ch, ch0 + 2 * chStride);
                specialMulForwardTo(car - cbr, cai - cbi, wa, waBase + wa(4, i, ido), ch, ch0 + 5 * chStride);
                car = t1r + tw3r * t2r + tw1r * t3r + tw2r * t4r;
                cai = t1i + tw3r * t2i + tw1r * t3i + tw2r * t4i;
                cbr = -(tw3i * t7i - tw1i * t6i + tw2i * t5i);
                cbi = tw3i * t7r - tw1i * t6r + tw2i * t5r;
                specialMulForwardTo(car + cbr, cai + cbi, wa, waBase + wa(2, i, ido), ch, ch0 + 3 * chStride);
                specialMulForwardTo(car - cbr, cai - cbi, wa, waBase + wa(3, i, ido), ch, ch0 + 4 * chStride);
            }
        }
    }

    private void pass7BackwardImpl(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        double tw1r = 0.6234898018587335305250048840042398;
        double tw1i = 0.7818314824680298087084445266740578;
        double tw2r = -0.2225209339563144042889025644967948;
        double tw2i = 0.9749279121818236070181316829939312;
        double tw3r = -0.9009688679024191262361023195074451;
        double tw3i = 0.433883739117558120475768332848359;
        int ccKStride = 14 * ido;
        int chKStride = 2 * ido;
        int chStride = 2 * ido * l1;
        for (int k = 0; k < l1; k++) {
            int cc0 = ccBase + k * ccKStride;
            int cc1 = cc0 + 2 * ido;
            int cc2 = cc1 + 2 * ido;
            int cc3 = cc2 + 2 * ido;
            int cc4 = cc3 + 2 * ido;
            int cc5 = cc4 + 2 * ido;
            int cc6 = cc5 + 2 * ido;
            int ch0 = chBase + k * chKStride;
            double t1r = cc[cc0];
            double t1i = cc[cc0 + 1];
            double t2r = cc[cc1] + cc[cc6];
            double t2i = cc[cc1 + 1] + cc[cc6 + 1];
            double t7r = cc[cc1] - cc[cc6];
            double t7i = cc[cc1 + 1] - cc[cc6 + 1];
            double t3r = cc[cc2] + cc[cc5];
            double t3i = cc[cc2 + 1] + cc[cc5 + 1];
            double t6r = cc[cc2] - cc[cc5];
            double t6i = cc[cc2 + 1] - cc[cc5 + 1];
            double t4r = cc[cc3] + cc[cc4];
            double t4i = cc[cc3 + 1] + cc[cc4 + 1];
            double t5r = cc[cc3] - cc[cc4];
            double t5i = cc[cc3 + 1] - cc[cc4 + 1];
            setPair(ch, ch0, t1r + t2r + t3r + t4r, t1i + t2i + t3i + t4i);
            double car = t1r + tw1r * t2r + tw2r * t3r + tw3r * t4r;
            double cai = t1i + tw1r * t2i + tw2r * t3i + tw3r * t4i;
            double cbr = -(tw1i * t7i + tw2i * t6i + tw3i * t5i);
            double cbi = tw1i * t7r + tw2i * t6r + tw3i * t5r;
            setPair(ch, ch0 + chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 6 * chStride, car - cbr, cai - cbi);
            car = t1r + tw2r * t2r + tw3r * t3r + tw1r * t4r;
            cai = t1i + tw2r * t2i + tw3r * t3i + tw1r * t4i;
            cbr = -(tw2i * t7i - tw3i * t6i - tw1i * t5i);
            cbi = tw2i * t7r - tw3i * t6r - tw1i * t5r;
            setPair(ch, ch0 + 2 * chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 5 * chStride, car - cbr, cai - cbi);
            car = t1r + tw3r * t2r + tw1r * t3r + tw2r * t4r;
            cai = t1i + tw3r * t2i + tw1r * t3i + tw2r * t4i;
            cbr = -(tw3i * t7i - tw1i * t6i + tw2i * t5i);
            cbi = tw3i * t7r - tw1i * t6r + tw2i * t5r;
            setPair(ch, ch0 + 3 * chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 4 * chStride, car - cbr, cai - cbi);

            for (int i = 1; i < ido; i++) {
                int iOffset = 2 * i;
                cc0 = ccBase + k * ccKStride + iOffset;
                cc1 = cc0 + 2 * ido;
                cc2 = cc1 + 2 * ido;
                cc3 = cc2 + 2 * ido;
                cc4 = cc3 + 2 * ido;
                cc5 = cc4 + 2 * ido;
                cc6 = cc5 + 2 * ido;
                ch0 = chBase + k * chKStride + iOffset;
                t1r = cc[cc0];
                t1i = cc[cc0 + 1];
                t2r = cc[cc1] + cc[cc6];
                t2i = cc[cc1 + 1] + cc[cc6 + 1];
                t7r = cc[cc1] - cc[cc6];
                t7i = cc[cc1 + 1] - cc[cc6 + 1];
                t3r = cc[cc2] + cc[cc5];
                t3i = cc[cc2 + 1] + cc[cc5 + 1];
                t6r = cc[cc2] - cc[cc5];
                t6i = cc[cc2 + 1] - cc[cc5 + 1];
                t4r = cc[cc3] + cc[cc4];
                t4i = cc[cc3 + 1] + cc[cc4 + 1];
                t5r = cc[cc3] - cc[cc4];
                t5i = cc[cc3 + 1] - cc[cc4 + 1];
                setPair(ch, ch0, t1r + t2r + t3r + t4r, t1i + t2i + t3i + t4i);
                car = t1r + tw1r * t2r + tw2r * t3r + tw3r * t4r;
                cai = t1i + tw1r * t2i + tw2r * t3i + tw3r * t4i;
                cbr = -(tw1i * t7i + tw2i * t6i + tw3i * t5i);
                cbi = tw1i * t7r + tw2i * t6r + tw3i * t5r;
                specialMulBackwardTo(car + cbr, cai + cbi, wa, waBase + wa(0, i, ido), ch, ch0 + chStride);
                specialMulBackwardTo(car - cbr, cai - cbi, wa, waBase + wa(5, i, ido), ch, ch0 + 6 * chStride);
                car = t1r + tw2r * t2r + tw3r * t3r + tw1r * t4r;
                cai = t1i + tw2r * t2i + tw3r * t3i + tw1r * t4i;
                cbr = -(tw2i * t7i - tw3i * t6i - tw1i * t5i);
                cbi = tw2i * t7r - tw3i * t6r - tw1i * t5r;
                specialMulBackwardTo(car + cbr, cai + cbi, wa, waBase + wa(1, i, ido), ch, ch0 + 2 * chStride);
                specialMulBackwardTo(car - cbr, cai - cbi, wa, waBase + wa(4, i, ido), ch, ch0 + 5 * chStride);
                car = t1r + tw3r * t2r + tw1r * t3r + tw2r * t4r;
                cai = t1i + tw3r * t2i + tw1r * t3i + tw2r * t4i;
                cbr = -(tw3i * t7i - tw1i * t6i + tw2i * t5i);
                cbi = tw3i * t7r - tw1i * t6r + tw2i * t5r;
                specialMulBackwardTo(car + cbr, cai + cbi, wa, waBase + wa(2, i, ido), ch, ch0 + 3 * chStride);
                specialMulBackwardTo(car - cbr, cai - cbi, wa, waBase + wa(3, i, ido), ch, ch0 + 4 * chStride);
            }
        }
    }
    
    private void pass8Forward(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        pass8ForwardImpl(ido, l1, cc, ccBase, ch, chBase, wa, waBase);
    }
    
    private void pass8Backward(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        pass8BackwardImpl(ido, l1, cc, ccBase, ch, chBase, wa, waBase);
    }

    private void pass8ForwardImpl(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        for (int k = 0; k < l1; k++) {
            int ccBaseK = ccBase + 16 * ido * k;
            int chBaseK = chBase + 2 * ido * k;
            int chStride = 2 * ido * l1;

            int cc0 = ccBaseK;
            int cc1 = cc0 + 2 * ido;
            int cc2 = cc1 + 2 * ido;
            int cc3 = cc2 + 2 * ido;
            int cc4 = cc3 + 2 * ido;
            int cc5 = cc4 + 2 * ido;
            int cc6 = cc5 + 2 * ido;
            int cc7 = cc6 + 2 * ido;
            double a1r = cc[cc1] + cc[cc5];
            double a1i = cc[cc1 + 1] + cc[cc5 + 1];
            double a5r = cc[cc1] - cc[cc5];
            double a5i = cc[cc1 + 1] - cc[cc5 + 1];
            double a3r = cc[cc3] + cc[cc7];
            double a3i = cc[cc3 + 1] + cc[cc7 + 1];
            double a7r = cc[cc3] - cc[cc7];
            double a7i = cc[cc3 + 1] - cc[cc7 + 1];
            double oldR = a1r;
            double oldI = a1i;
            a1r += a3r;
            a1i += a3i;
            a3r = oldR - a3r;
            a3i = oldI - a3i;
            double rotR = rotx90ForwardReal(a3r, a3i);
            double rotI = rotx90ForwardImag(a3r, a3i);
            a3r = rotR;
            a3i = rotI;
            rotR = rotx90ForwardReal(a7r, a7i);
            rotI = rotx90ForwardImag(a7r, a7i);
            a7r = rotR;
            a7i = rotI;
            oldR = a5r;
            oldI = a5i;
            a5r += a7r;
            a5i += a7i;
            a7r = oldR - a7r;
            a7i = oldI - a7i;
            rotR = rotx45ForwardReal(a5r, a5i);
            rotI = rotx45ForwardImag(a5r, a5i);
            a5r = rotR;
            a5i = rotI;
            rotR = rotx135ForwardReal(a7r, a7i);
            rotI = rotx135ForwardImag(a7r, a7i);
            a7r = rotR;
            a7i = rotI;
            double a0r = cc[cc0] + cc[cc4];
            double a0i = cc[cc0 + 1] + cc[cc4 + 1];
            double a4r = cc[cc0] - cc[cc4];
            double a4i = cc[cc0 + 1] - cc[cc4 + 1];
            double a2r = cc[cc2] + cc[cc6];
            double a2i = cc[cc2 + 1] + cc[cc6 + 1];
            double a6r = cc[cc2] - cc[cc6];
            double a6i = cc[cc2 + 1] - cc[cc6 + 1];
            setPair(ch, chBaseK, a0r + a2r + a1r, a0i + a2i + a1i);
            setPair(ch, chBaseK + 4 * chStride, a0r + a2r - a1r, a0i + a2i - a1i);
            setPair(ch, chBaseK + 2 * chStride, a0r - a2r + a3r, a0i - a2i + a3i);
            setPair(ch, chBaseK + 6 * chStride, a0r - a2r - a3r, a0i - a2i - a3i);
            rotR = rotx90ForwardReal(a6r, a6i);
            rotI = rotx90ForwardImag(a6r, a6i);
            a6r = rotR;
            a6i = rotI;
            setPair(ch, chBaseK + chStride, a4r + a6r + a5r, a4i + a6i + a5i);
            setPair(ch, chBaseK + 5 * chStride, a4r + a6r - a5r, a4i + a6i - a5i);
            setPair(ch, chBaseK + 3 * chStride, a4r - a6r + a7r, a4i - a6i + a7i);
            setPair(ch, chBaseK + 7 * chStride, a4r - a6r - a7r, a4i - a6i - a7i);

            for (int i = 1; i < ido; i++) {
                int iOffset = 2 * i;
                cc0 = ccBaseK + iOffset;
                cc1 = cc0 + 2 * ido;
                cc2 = cc1 + 2 * ido;
                cc3 = cc2 + 2 * ido;
                cc4 = cc3 + 2 * ido;
                cc5 = cc4 + 2 * ido;
                cc6 = cc5 + 2 * ido;
                cc7 = cc6 + 2 * ido;
                a1r = cc[cc1] + cc[cc5];
                a1i = cc[cc1 + 1] + cc[cc5 + 1];
                a5r = cc[cc1] - cc[cc5];
                a5i = cc[cc1 + 1] - cc[cc5 + 1];
                a3r = cc[cc3] + cc[cc7];
                a3i = cc[cc3 + 1] + cc[cc7 + 1];
                a7r = cc[cc3] - cc[cc7];
                a7i = cc[cc3 + 1] - cc[cc7 + 1];
                rotR = rotx90ForwardReal(a7r, a7i);
                rotI = rotx90ForwardImag(a7r, a7i);
                a7r = rotR;
                a7i = rotI;
                oldR = a1r;
                oldI = a1i;
                a1r += a3r;
                a1i += a3i;
                a3r = oldR - a3r;
                a3i = oldI - a3i;
                rotR = rotx90ForwardReal(a3r, a3i);
                rotI = rotx90ForwardImag(a3r, a3i);
                a3r = rotR;
                a3i = rotI;
                oldR = a5r;
                oldI = a5i;
                a5r += a7r;
                a5i += a7i;
                a7r = oldR - a7r;
                a7i = oldI - a7i;
                rotR = rotx45ForwardReal(a5r, a5i);
                rotI = rotx45ForwardImag(a5r, a5i);
                a5r = rotR;
                a5i = rotI;
                rotR = rotx135ForwardReal(a7r, a7i);
                rotI = rotx135ForwardImag(a7r, a7i);
                a7r = rotR;
                a7i = rotI;
                a0r = cc[cc0] + cc[cc4];
                a0i = cc[cc0 + 1] + cc[cc4 + 1];
                a4r = cc[cc0] - cc[cc4];
                a4i = cc[cc0 + 1] - cc[cc4 + 1];
                a2r = cc[cc2] + cc[cc6];
                a2i = cc[cc2 + 1] + cc[cc6 + 1];
                a6r = cc[cc2] - cc[cc6];
                a6i = cc[cc2 + 1] - cc[cc6 + 1];
                oldR = a0r;
                oldI = a0i;
                a0r += a2r;
                a0i += a2i;
                a2r = oldR - a2r;
                a2i = oldI - a2i;
                int ch0 = chBaseK + iOffset;
                setPair(ch, ch0, a0r + a1r, a0i + a1i);
                specialMulForwardTo(a0r - a1r, a0i - a1i, wa, waBase + wa(3, i, ido), ch,
                        ch0 + 4 * chStride);
                specialMulForwardTo(a2r + a3r, a2i + a3i, wa, waBase + wa(1, i, ido), ch,
                        ch0 + 2 * chStride);
                specialMulForwardTo(a2r - a3r, a2i - a3i, wa, waBase + wa(5, i, ido), ch,
                        ch0 + 6 * chStride);
                rotR = rotx90ForwardReal(a6r, a6i);
                rotI = rotx90ForwardImag(a6r, a6i);
                a6r = rotR;
                a6i = rotI;
                oldR = a4r;
                oldI = a4i;
                a4r += a6r;
                a4i += a6i;
                a6r = oldR - a6r;
                a6i = oldI - a6i;
                specialMulForwardTo(a4r + a5r, a4i + a5i, wa, waBase + wa(0, i, ido), ch, ch0 + chStride);
                specialMulForwardTo(a4r - a5r, a4i - a5i, wa, waBase + wa(4, i, ido), ch,
                        ch0 + 5 * chStride);
                specialMulForwardTo(a6r + a7r, a6i + a7i, wa, waBase + wa(2, i, ido), ch,
                        ch0 + 3 * chStride);
                specialMulForwardTo(a6r - a7r, a6i - a7i, wa, waBase + wa(6, i, ido), ch,
                        ch0 + 7 * chStride);
            }
        }
    }

    private void pass8BackwardImpl(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        for (int k = 0; k < l1; k++) {
            int ccBaseK = ccBase + 16 * ido * k;
            int chBaseK = chBase + 2 * ido * k;
            int chStride = 2 * ido * l1;

            int cc0 = ccBaseK;
            int cc1 = cc0 + 2 * ido;
            int cc2 = cc1 + 2 * ido;
            int cc3 = cc2 + 2 * ido;
            int cc4 = cc3 + 2 * ido;
            int cc5 = cc4 + 2 * ido;
            int cc6 = cc5 + 2 * ido;
            int cc7 = cc6 + 2 * ido;
            double a1r = cc[cc1] + cc[cc5];
            double a1i = cc[cc1 + 1] + cc[cc5 + 1];
            double a5r = cc[cc1] - cc[cc5];
            double a5i = cc[cc1 + 1] - cc[cc5 + 1];
            double a3r = cc[cc3] + cc[cc7];
            double a3i = cc[cc3 + 1] + cc[cc7 + 1];
            double a7r = cc[cc3] - cc[cc7];
            double a7i = cc[cc3 + 1] - cc[cc7 + 1];
            double oldR = a1r;
            double oldI = a1i;
            a1r += a3r;
            a1i += a3i;
            a3r = oldR - a3r;
            a3i = oldI - a3i;
            double rotR = rotx90BackwardReal(a3r, a3i);
            double rotI = rotx90BackwardImag(a3r, a3i);
            a3r = rotR;
            a3i = rotI;
            rotR = rotx90BackwardReal(a7r, a7i);
            rotI = rotx90BackwardImag(a7r, a7i);
            a7r = rotR;
            a7i = rotI;
            oldR = a5r;
            oldI = a5i;
            a5r += a7r;
            a5i += a7i;
            a7r = oldR - a7r;
            a7i = oldI - a7i;
            rotR = rotx45BackwardReal(a5r, a5i);
            rotI = rotx45BackwardImag(a5r, a5i);
            a5r = rotR;
            a5i = rotI;
            rotR = rotx135BackwardReal(a7r, a7i);
            rotI = rotx135BackwardImag(a7r, a7i);
            a7r = rotR;
            a7i = rotI;
            double a0r = cc[cc0] + cc[cc4];
            double a0i = cc[cc0 + 1] + cc[cc4 + 1];
            double a4r = cc[cc0] - cc[cc4];
            double a4i = cc[cc0 + 1] - cc[cc4 + 1];
            double a2r = cc[cc2] + cc[cc6];
            double a2i = cc[cc2 + 1] + cc[cc6 + 1];
            double a6r = cc[cc2] - cc[cc6];
            double a6i = cc[cc2 + 1] - cc[cc6 + 1];
            setPair(ch, chBaseK, a0r + a2r + a1r, a0i + a2i + a1i);
            setPair(ch, chBaseK + 4 * chStride, a0r + a2r - a1r, a0i + a2i - a1i);
            setPair(ch, chBaseK + 2 * chStride, a0r - a2r + a3r, a0i - a2i + a3i);
            setPair(ch, chBaseK + 6 * chStride, a0r - a2r - a3r, a0i - a2i - a3i);
            rotR = rotx90BackwardReal(a6r, a6i);
            rotI = rotx90BackwardImag(a6r, a6i);
            a6r = rotR;
            a6i = rotI;
            setPair(ch, chBaseK + chStride, a4r + a6r + a5r, a4i + a6i + a5i);
            setPair(ch, chBaseK + 5 * chStride, a4r + a6r - a5r, a4i + a6i - a5i);
            setPair(ch, chBaseK + 3 * chStride, a4r - a6r + a7r, a4i - a6i + a7i);
            setPair(ch, chBaseK + 7 * chStride, a4r - a6r - a7r, a4i - a6i - a7i);

            for (int i = 1; i < ido; i++) {
                int iOffset = 2 * i;
                cc0 = ccBaseK + iOffset;
                cc1 = cc0 + 2 * ido;
                cc2 = cc1 + 2 * ido;
                cc3 = cc2 + 2 * ido;
                cc4 = cc3 + 2 * ido;
                cc5 = cc4 + 2 * ido;
                cc6 = cc5 + 2 * ido;
                cc7 = cc6 + 2 * ido;
                a1r = cc[cc1] + cc[cc5];
                a1i = cc[cc1 + 1] + cc[cc5 + 1];
                a5r = cc[cc1] - cc[cc5];
                a5i = cc[cc1 + 1] - cc[cc5 + 1];
                a3r = cc[cc3] + cc[cc7];
                a3i = cc[cc3 + 1] + cc[cc7 + 1];
                a7r = cc[cc3] - cc[cc7];
                a7i = cc[cc3 + 1] - cc[cc7 + 1];
                rotR = rotx90BackwardReal(a7r, a7i);
                rotI = rotx90BackwardImag(a7r, a7i);
                a7r = rotR;
                a7i = rotI;
                oldR = a1r;
                oldI = a1i;
                a1r += a3r;
                a1i += a3i;
                a3r = oldR - a3r;
                a3i = oldI - a3i;
                rotR = rotx90BackwardReal(a3r, a3i);
                rotI = rotx90BackwardImag(a3r, a3i);
                a3r = rotR;
                a3i = rotI;
                oldR = a5r;
                oldI = a5i;
                a5r += a7r;
                a5i += a7i;
                a7r = oldR - a7r;
                a7i = oldI - a7i;
                rotR = rotx45BackwardReal(a5r, a5i);
                rotI = rotx45BackwardImag(a5r, a5i);
                a5r = rotR;
                a5i = rotI;
                rotR = rotx135BackwardReal(a7r, a7i);
                rotI = rotx135BackwardImag(a7r, a7i);
                a7r = rotR;
                a7i = rotI;
                a0r = cc[cc0] + cc[cc4];
                a0i = cc[cc0 + 1] + cc[cc4 + 1];
                a4r = cc[cc0] - cc[cc4];
                a4i = cc[cc0 + 1] - cc[cc4 + 1];
                a2r = cc[cc2] + cc[cc6];
                a2i = cc[cc2 + 1] + cc[cc6 + 1];
                a6r = cc[cc2] - cc[cc6];
                a6i = cc[cc2 + 1] - cc[cc6 + 1];
                oldR = a0r;
                oldI = a0i;
                a0r += a2r;
                a0i += a2i;
                a2r = oldR - a2r;
                a2i = oldI - a2i;
                int ch0 = chBaseK + iOffset;
                setPair(ch, ch0, a0r + a1r, a0i + a1i);
                specialMulBackwardTo(a0r - a1r, a0i - a1i, wa, waBase + wa(3, i, ido), ch,
                        ch0 + 4 * chStride);
                specialMulBackwardTo(a2r + a3r, a2i + a3i, wa, waBase + wa(1, i, ido), ch,
                        ch0 + 2 * chStride);
                specialMulBackwardTo(a2r - a3r, a2i - a3i, wa, waBase + wa(5, i, ido), ch,
                        ch0 + 6 * chStride);
                rotR = rotx90BackwardReal(a6r, a6i);
                rotI = rotx90BackwardImag(a6r, a6i);
                a6r = rotR;
                a6i = rotI;
                oldR = a4r;
                oldI = a4i;
                a4r += a6r;
                a4i += a6i;
                a6r = oldR - a6r;
                a6i = oldI - a6i;
                specialMulBackwardTo(a4r + a5r, a4i + a5i, wa, waBase + wa(0, i, ido), ch, ch0 + chStride);
                specialMulBackwardTo(a4r - a5r, a4i - a5i, wa, waBase + wa(4, i, ido), ch,
                        ch0 + 5 * chStride);
                specialMulBackwardTo(a6r + a7r, a6i + a7i, wa, waBase + wa(2, i, ido), ch,
                        ch0 + 3 * chStride);
                specialMulBackwardTo(a6r - a7r, a6i - a7i, wa, waBase + wa(6, i, ido), ch,
                        ch0 + 7 * chStride);
            }
        }
    }

    private void pass11Forward(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        pass11ForwardImpl(ido, l1, cc, ccBase, ch, chBase, wa, waBase);
    }

    private void pass11Backward(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        pass11BackwardImpl(ido, l1, cc, ccBase, ch, chBase, wa, waBase);
    }

    private void pass11ForwardImpl(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        double tw1r = 0.8412535328311811688618116489193677;
        double tw1i = -0.5406408174555975821076359543186917;
        double tw2r = 0.4154150130018864255292741492296232;
        double tw2i = -0.9096319953545183714117153830790285;
        double tw3r = -0.1423148382732851404437926686163697;
        double tw3i = -0.9898214418809327323760920377767188;
        double tw4r = -0.6548607339452850640569250724662936;
        double tw4i = -0.7557495743542582837740358439723444;
        double tw5r = -0.9594929736144973898903680570663277;
        double tw5i = -0.2817325568414296977114179153466169;
        int ccKStride = 22 * ido;
        int chKStride = 2 * ido;
        int chStride = 2 * ido * l1;
        for (int k = 0; k < l1; k++) {
            int cc0 = ccBase + k * ccKStride;
            int cc1 = cc0 + 2 * ido;
            int cc2 = cc1 + 2 * ido;
            int cc3 = cc2 + 2 * ido;
            int cc4 = cc3 + 2 * ido;
            int cc5 = cc4 + 2 * ido;
            int cc6 = cc5 + 2 * ido;
            int cc7 = cc6 + 2 * ido;
            int cc8 = cc7 + 2 * ido;
            int cc9 = cc8 + 2 * ido;
            int cc10 = cc9 + 2 * ido;
            int ch0 = chBase + k * chKStride;
            double t1r = cc[cc0];
            double t1i = cc[cc0 + 1];
            double t2r = cc[cc1] + cc[cc10];
            double t2i = cc[cc1 + 1] + cc[cc10 + 1];
            double t11r = cc[cc1] - cc[cc10];
            double t11i = cc[cc1 + 1] - cc[cc10 + 1];
            double t3r = cc[cc2] + cc[cc9];
            double t3i = cc[cc2 + 1] + cc[cc9 + 1];
            double t10r = cc[cc2] - cc[cc9];
            double t10i = cc[cc2 + 1] - cc[cc9 + 1];
            double t4r = cc[cc3] + cc[cc8];
            double t4i = cc[cc3 + 1] + cc[cc8 + 1];
            double t9r = cc[cc3] - cc[cc8];
            double t9i = cc[cc3 + 1] - cc[cc8 + 1];
            double t5r = cc[cc4] + cc[cc7];
            double t5i = cc[cc4 + 1] + cc[cc7 + 1];
            double t8r = cc[cc4] - cc[cc7];
            double t8i = cc[cc4 + 1] - cc[cc7 + 1];
            double t6r = cc[cc5] + cc[cc6];
            double t6i = cc[cc5 + 1] + cc[cc6 + 1];
            double t7r = cc[cc5] - cc[cc6];
            double t7i = cc[cc5 + 1] - cc[cc6 + 1];
            setPair(ch, ch0, t1r + t2r + t3r + t4r + t5r + t6r,
                    t1i + t2i + t3i + t4i + t5i + t6i);
            double car = t1r + tw1r * t2r + tw2r * t3r + tw3r * t4r + tw4r * t5r + tw5r * t6r;
            double cai = t1i + tw1r * t2i + tw2r * t3i + tw3r * t4i + tw4r * t5i + tw5r * t6i;
            double cbr = -(tw1i * t11i + tw2i * t10i + tw3i * t9i + tw4i * t8i + tw5i * t7i);
            double cbi = tw1i * t11r + tw2i * t10r + tw3i * t9r + tw4i * t8r + tw5i * t7r;
            setPair(ch, ch0 + chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 10 * chStride, car - cbr, cai - cbi);
            car = t1r + tw2r * t2r + tw4r * t3r + tw5r * t4r + tw3r * t5r + tw1r * t6r;
            cai = t1i + tw2r * t2i + tw4r * t3i + tw5r * t4i + tw3r * t5i + tw1r * t6i;
            cbr = -(tw2i * t11i + tw4i * t10i - tw5i * t9i - tw3i * t8i - tw1i * t7i);
            cbi = tw2i * t11r + tw4i * t10r - tw5i * t9r - tw3i * t8r - tw1i * t7r;
            setPair(ch, ch0 + 2 * chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 9 * chStride, car - cbr, cai - cbi);
            car = t1r + tw3r * t2r + tw5r * t3r + tw2r * t4r + tw1r * t5r + tw4r * t6r;
            cai = t1i + tw3r * t2i + tw5r * t3i + tw2r * t4i + tw1r * t5i + tw4r * t6i;
            cbr = -(tw3i * t11i - tw5i * t10i - tw2i * t9i + tw1i * t8i + tw4i * t7i);
            cbi = tw3i * t11r - tw5i * t10r - tw2i * t9r + tw1i * t8r + tw4i * t7r;
            setPair(ch, ch0 + 3 * chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 8 * chStride, car - cbr, cai - cbi);
            car = t1r + tw4r * t2r + tw3r * t3r + tw1r * t4r + tw5r * t5r + tw2r * t6r;
            cai = t1i + tw4r * t2i + tw3r * t3i + tw1r * t4i + tw5r * t5i + tw2r * t6i;
            cbr = -(tw4i * t11i - tw3i * t10i + tw1i * t9i + tw5i * t8i - tw2i * t7i);
            cbi = tw4i * t11r - tw3i * t10r + tw1i * t9r + tw5i * t8r - tw2i * t7r;
            setPair(ch, ch0 + 4 * chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 7 * chStride, car - cbr, cai - cbi);
            car = t1r + tw5r * t2r + tw1r * t3r + tw4r * t4r + tw2r * t5r + tw3r * t6r;
            cai = t1i + tw5r * t2i + tw1r * t3i + tw4r * t4i + tw2r * t5i + tw3r * t6i;
            cbr = -(tw5i * t11i - tw1i * t10i + tw4i * t9i - tw2i * t8i + tw3i * t7i);
            cbi = tw5i * t11r - tw1i * t10r + tw4i * t9r - tw2i * t8r + tw3i * t7r;
            setPair(ch, ch0 + 5 * chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 6 * chStride, car - cbr, cai - cbi);

            for (int i = 1; i < ido; i++) {
                int iOffset = 2 * i;
                cc0 = ccBase + k * ccKStride + iOffset;
                cc1 = cc0 + 2 * ido;
                cc2 = cc1 + 2 * ido;
                cc3 = cc2 + 2 * ido;
                cc4 = cc3 + 2 * ido;
                cc5 = cc4 + 2 * ido;
                cc6 = cc5 + 2 * ido;
                cc7 = cc6 + 2 * ido;
                cc8 = cc7 + 2 * ido;
                cc9 = cc8 + 2 * ido;
                cc10 = cc9 + 2 * ido;
                ch0 = chBase + k * chKStride + iOffset;
                t1r = cc[cc0];
                t1i = cc[cc0 + 1];
                t2r = cc[cc1] + cc[cc10];
                t2i = cc[cc1 + 1] + cc[cc10 + 1];
                t11r = cc[cc1] - cc[cc10];
                t11i = cc[cc1 + 1] - cc[cc10 + 1];
                t3r = cc[cc2] + cc[cc9];
                t3i = cc[cc2 + 1] + cc[cc9 + 1];
                t10r = cc[cc2] - cc[cc9];
                t10i = cc[cc2 + 1] - cc[cc9 + 1];
                t4r = cc[cc3] + cc[cc8];
                t4i = cc[cc3 + 1] + cc[cc8 + 1];
                t9r = cc[cc3] - cc[cc8];
                t9i = cc[cc3 + 1] - cc[cc8 + 1];
                t5r = cc[cc4] + cc[cc7];
                t5i = cc[cc4 + 1] + cc[cc7 + 1];
                t8r = cc[cc4] - cc[cc7];
                t8i = cc[cc4 + 1] - cc[cc7 + 1];
                t6r = cc[cc5] + cc[cc6];
                t6i = cc[cc5 + 1] + cc[cc6 + 1];
                t7r = cc[cc5] - cc[cc6];
                t7i = cc[cc5 + 1] - cc[cc6 + 1];
                setPair(ch, ch0, t1r + t2r + t3r + t4r + t5r + t6r,
                        t1i + t2i + t3i + t4i + t5i + t6i);
                car = t1r + tw1r * t2r + tw2r * t3r + tw3r * t4r + tw4r * t5r + tw5r * t6r;
                cai = t1i + tw1r * t2i + tw2r * t3i + tw3r * t4i + tw4r * t5i + tw5r * t6i;
                cbr = -(tw1i * t11i + tw2i * t10i + tw3i * t9i + tw4i * t8i + tw5i * t7i);
                cbi = tw1i * t11r + tw2i * t10r + tw3i * t9r + tw4i * t8r + tw5i * t7r;
                specialMulForwardTo(car + cbr, cai + cbi, wa, waBase + wa(0, i, ido), ch, ch0 + chStride);
                specialMulForwardTo(car - cbr, cai - cbi, wa, waBase + wa(9, i, ido), ch, ch0 + 10 * chStride);
                car = t1r + tw2r * t2r + tw4r * t3r + tw5r * t4r + tw3r * t5r + tw1r * t6r;
                cai = t1i + tw2r * t2i + tw4r * t3i + tw5r * t4i + tw3r * t5i + tw1r * t6i;
                cbr = -(tw2i * t11i + tw4i * t10i - tw5i * t9i - tw3i * t8i - tw1i * t7i);
                cbi = tw2i * t11r + tw4i * t10r - tw5i * t9r - tw3i * t8r - tw1i * t7r;
                specialMulForwardTo(car + cbr, cai + cbi, wa, waBase + wa(1, i, ido), ch, ch0 + 2 * chStride);
                specialMulForwardTo(car - cbr, cai - cbi, wa, waBase + wa(8, i, ido), ch, ch0 + 9 * chStride);
                car = t1r + tw3r * t2r + tw5r * t3r + tw2r * t4r + tw1r * t5r + tw4r * t6r;
                cai = t1i + tw3r * t2i + tw5r * t3i + tw2r * t4i + tw1r * t5i + tw4r * t6i;
                cbr = -(tw3i * t11i - tw5i * t10i - tw2i * t9i + tw1i * t8i + tw4i * t7i);
                cbi = tw3i * t11r - tw5i * t10r - tw2i * t9r + tw1i * t8r + tw4i * t7r;
                specialMulForwardTo(car + cbr, cai + cbi, wa, waBase + wa(2, i, ido), ch, ch0 + 3 * chStride);
                specialMulForwardTo(car - cbr, cai - cbi, wa, waBase + wa(7, i, ido), ch, ch0 + 8 * chStride);
                car = t1r + tw4r * t2r + tw3r * t3r + tw1r * t4r + tw5r * t5r + tw2r * t6r;
                cai = t1i + tw4r * t2i + tw3r * t3i + tw1r * t4i + tw5r * t5i + tw2r * t6i;
                cbr = -(tw4i * t11i - tw3i * t10i + tw1i * t9i + tw5i * t8i - tw2i * t7i);
                cbi = tw4i * t11r - tw3i * t10r + tw1i * t9r + tw5i * t8r - tw2i * t7r;
                specialMulForwardTo(car + cbr, cai + cbi, wa, waBase + wa(3, i, ido), ch, ch0 + 4 * chStride);
                specialMulForwardTo(car - cbr, cai - cbi, wa, waBase + wa(6, i, ido), ch, ch0 + 7 * chStride);
                car = t1r + tw5r * t2r + tw1r * t3r + tw4r * t4r + tw2r * t5r + tw3r * t6r;
                cai = t1i + tw5r * t2i + tw1r * t3i + tw4r * t4i + tw2r * t5i + tw3r * t6i;
                cbr = -(tw5i * t11i - tw1i * t10i + tw4i * t9i - tw2i * t8i + tw3i * t7i);
                cbi = tw5i * t11r - tw1i * t10r + tw4i * t9r - tw2i * t8r + tw3i * t7r;
                specialMulForwardTo(car + cbr, cai + cbi, wa, waBase + wa(4, i, ido), ch, ch0 + 5 * chStride);
                specialMulForwardTo(car - cbr, cai - cbi, wa, waBase + wa(5, i, ido), ch, ch0 + 6 * chStride);
            }
        }
    }

    private void pass11BackwardImpl(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        double tw1r = 0.8412535328311811688618116489193677;
        double tw1i = 0.5406408174555975821076359543186917;
        double tw2r = 0.4154150130018864255292741492296232;
        double tw2i = 0.9096319953545183714117153830790285;
        double tw3r = -0.1423148382732851404437926686163697;
        double tw3i = 0.9898214418809327323760920377767188;
        double tw4r = -0.6548607339452850640569250724662936;
        double tw4i = 0.7557495743542582837740358439723444;
        double tw5r = -0.9594929736144973898903680570663277;
        double tw5i = 0.2817325568414296977114179153466169;
        int ccKStride = 22 * ido;
        int chKStride = 2 * ido;
        int chStride = 2 * ido * l1;
        for (int k = 0; k < l1; k++) {
            int cc0 = ccBase + k * ccKStride;
            int cc1 = cc0 + 2 * ido;
            int cc2 = cc1 + 2 * ido;
            int cc3 = cc2 + 2 * ido;
            int cc4 = cc3 + 2 * ido;
            int cc5 = cc4 + 2 * ido;
            int cc6 = cc5 + 2 * ido;
            int cc7 = cc6 + 2 * ido;
            int cc8 = cc7 + 2 * ido;
            int cc9 = cc8 + 2 * ido;
            int cc10 = cc9 + 2 * ido;
            int ch0 = chBase + k * chKStride;
            double t1r = cc[cc0];
            double t1i = cc[cc0 + 1];
            double t2r = cc[cc1] + cc[cc10];
            double t2i = cc[cc1 + 1] + cc[cc10 + 1];
            double t11r = cc[cc1] - cc[cc10];
            double t11i = cc[cc1 + 1] - cc[cc10 + 1];
            double t3r = cc[cc2] + cc[cc9];
            double t3i = cc[cc2 + 1] + cc[cc9 + 1];
            double t10r = cc[cc2] - cc[cc9];
            double t10i = cc[cc2 + 1] - cc[cc9 + 1];
            double t4r = cc[cc3] + cc[cc8];
            double t4i = cc[cc3 + 1] + cc[cc8 + 1];
            double t9r = cc[cc3] - cc[cc8];
            double t9i = cc[cc3 + 1] - cc[cc8 + 1];
            double t5r = cc[cc4] + cc[cc7];
            double t5i = cc[cc4 + 1] + cc[cc7 + 1];
            double t8r = cc[cc4] - cc[cc7];
            double t8i = cc[cc4 + 1] - cc[cc7 + 1];
            double t6r = cc[cc5] + cc[cc6];
            double t6i = cc[cc5 + 1] + cc[cc6 + 1];
            double t7r = cc[cc5] - cc[cc6];
            double t7i = cc[cc5 + 1] - cc[cc6 + 1];
            setPair(ch, ch0, t1r + t2r + t3r + t4r + t5r + t6r,
                    t1i + t2i + t3i + t4i + t5i + t6i);
            double car = t1r + tw1r * t2r + tw2r * t3r + tw3r * t4r + tw4r * t5r + tw5r * t6r;
            double cai = t1i + tw1r * t2i + tw2r * t3i + tw3r * t4i + tw4r * t5i + tw5r * t6i;
            double cbr = -(tw1i * t11i + tw2i * t10i + tw3i * t9i + tw4i * t8i + tw5i * t7i);
            double cbi = tw1i * t11r + tw2i * t10r + tw3i * t9r + tw4i * t8r + tw5i * t7r;
            setPair(ch, ch0 + chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 10 * chStride, car - cbr, cai - cbi);
            car = t1r + tw2r * t2r + tw4r * t3r + tw5r * t4r + tw3r * t5r + tw1r * t6r;
            cai = t1i + tw2r * t2i + tw4r * t3i + tw5r * t4i + tw3r * t5i + tw1r * t6i;
            cbr = -(tw2i * t11i + tw4i * t10i - tw5i * t9i - tw3i * t8i - tw1i * t7i);
            cbi = tw2i * t11r + tw4i * t10r - tw5i * t9r - tw3i * t8r - tw1i * t7r;
            setPair(ch, ch0 + 2 * chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 9 * chStride, car - cbr, cai - cbi);
            car = t1r + tw3r * t2r + tw5r * t3r + tw2r * t4r + tw1r * t5r + tw4r * t6r;
            cai = t1i + tw3r * t2i + tw5r * t3i + tw2r * t4i + tw1r * t5i + tw4r * t6i;
            cbr = -(tw3i * t11i - tw5i * t10i - tw2i * t9i + tw1i * t8i + tw4i * t7i);
            cbi = tw3i * t11r - tw5i * t10r - tw2i * t9r + tw1i * t8r + tw4i * t7r;
            setPair(ch, ch0 + 3 * chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 8 * chStride, car - cbr, cai - cbi);
            car = t1r + tw4r * t2r + tw3r * t3r + tw1r * t4r + tw5r * t5r + tw2r * t6r;
            cai = t1i + tw4r * t2i + tw3r * t3i + tw1r * t4i + tw5r * t5i + tw2r * t6i;
            cbr = -(tw4i * t11i - tw3i * t10i + tw1i * t9i + tw5i * t8i - tw2i * t7i);
            cbi = tw4i * t11r - tw3i * t10r + tw1i * t9r + tw5i * t8r - tw2i * t7r;
            setPair(ch, ch0 + 4 * chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 7 * chStride, car - cbr, cai - cbi);
            car = t1r + tw5r * t2r + tw1r * t3r + tw4r * t4r + tw2r * t5r + tw3r * t6r;
            cai = t1i + tw5r * t2i + tw1r * t3i + tw4r * t4i + tw2r * t5i + tw3r * t6i;
            cbr = -(tw5i * t11i - tw1i * t10i + tw4i * t9i - tw2i * t8i + tw3i * t7i);
            cbi = tw5i * t11r - tw1i * t10r + tw4i * t9r - tw2i * t8r + tw3i * t7r;
            setPair(ch, ch0 + 5 * chStride, car + cbr, cai + cbi);
            setPair(ch, ch0 + 6 * chStride, car - cbr, cai - cbi);

            for (int i = 1; i < ido; i++) {
                int iOffset = 2 * i;
                cc0 = ccBase + k * ccKStride + iOffset;
                cc1 = cc0 + 2 * ido;
                cc2 = cc1 + 2 * ido;
                cc3 = cc2 + 2 * ido;
                cc4 = cc3 + 2 * ido;
                cc5 = cc4 + 2 * ido;
                cc6 = cc5 + 2 * ido;
                cc7 = cc6 + 2 * ido;
                cc8 = cc7 + 2 * ido;
                cc9 = cc8 + 2 * ido;
                cc10 = cc9 + 2 * ido;
                ch0 = chBase + k * chKStride + iOffset;
                t1r = cc[cc0];
                t1i = cc[cc0 + 1];
                t2r = cc[cc1] + cc[cc10];
                t2i = cc[cc1 + 1] + cc[cc10 + 1];
                t11r = cc[cc1] - cc[cc10];
                t11i = cc[cc1 + 1] - cc[cc10 + 1];
                t3r = cc[cc2] + cc[cc9];
                t3i = cc[cc2 + 1] + cc[cc9 + 1];
                t10r = cc[cc2] - cc[cc9];
                t10i = cc[cc2 + 1] - cc[cc9 + 1];
                t4r = cc[cc3] + cc[cc8];
                t4i = cc[cc3 + 1] + cc[cc8 + 1];
                t9r = cc[cc3] - cc[cc8];
                t9i = cc[cc3 + 1] - cc[cc8 + 1];
                t5r = cc[cc4] + cc[cc7];
                t5i = cc[cc4 + 1] + cc[cc7 + 1];
                t8r = cc[cc4] - cc[cc7];
                t8i = cc[cc4 + 1] - cc[cc7 + 1];
                t6r = cc[cc5] + cc[cc6];
                t6i = cc[cc5 + 1] + cc[cc6 + 1];
                t7r = cc[cc5] - cc[cc6];
                t7i = cc[cc5 + 1] - cc[cc6 + 1];
                setPair(ch, ch0, t1r + t2r + t3r + t4r + t5r + t6r,
                        t1i + t2i + t3i + t4i + t5i + t6i);
                car = t1r + tw1r * t2r + tw2r * t3r + tw3r * t4r + tw4r * t5r + tw5r * t6r;
                cai = t1i + tw1r * t2i + tw2r * t3i + tw3r * t4i + tw4r * t5i + tw5r * t6i;
                cbr = -(tw1i * t11i + tw2i * t10i + tw3i * t9i + tw4i * t8i + tw5i * t7i);
                cbi = tw1i * t11r + tw2i * t10r + tw3i * t9r + tw4i * t8r + tw5i * t7r;
                specialMulBackwardTo(car + cbr, cai + cbi, wa, waBase + wa(0, i, ido), ch, ch0 + chStride);
                specialMulBackwardTo(car - cbr, cai - cbi, wa, waBase + wa(9, i, ido), ch, ch0 + 10 * chStride);
                car = t1r + tw2r * t2r + tw4r * t3r + tw5r * t4r + tw3r * t5r + tw1r * t6r;
                cai = t1i + tw2r * t2i + tw4r * t3i + tw5r * t4i + tw3r * t5i + tw1r * t6i;
                cbr = -(tw2i * t11i + tw4i * t10i - tw5i * t9i - tw3i * t8i - tw1i * t7i);
                cbi = tw2i * t11r + tw4i * t10r - tw5i * t9r - tw3i * t8r - tw1i * t7r;
                specialMulBackwardTo(car + cbr, cai + cbi, wa, waBase + wa(1, i, ido), ch, ch0 + 2 * chStride);
                specialMulBackwardTo(car - cbr, cai - cbi, wa, waBase + wa(8, i, ido), ch, ch0 + 9 * chStride);
                car = t1r + tw3r * t2r + tw5r * t3r + tw2r * t4r + tw1r * t5r + tw4r * t6r;
                cai = t1i + tw3r * t2i + tw5r * t3i + tw2r * t4i + tw1r * t5i + tw4r * t6i;
                cbr = -(tw3i * t11i - tw5i * t10i - tw2i * t9i + tw1i * t8i + tw4i * t7i);
                cbi = tw3i * t11r - tw5i * t10r - tw2i * t9r + tw1i * t8r + tw4i * t7r;
                specialMulBackwardTo(car + cbr, cai + cbi, wa, waBase + wa(2, i, ido), ch, ch0 + 3 * chStride);
                specialMulBackwardTo(car - cbr, cai - cbi, wa, waBase + wa(7, i, ido), ch, ch0 + 8 * chStride);
                car = t1r + tw4r * t2r + tw3r * t3r + tw1r * t4r + tw5r * t5r + tw2r * t6r;
                cai = t1i + tw4r * t2i + tw3r * t3i + tw1r * t4i + tw5r * t5i + tw2r * t6i;
                cbr = -(tw4i * t11i - tw3i * t10i + tw1i * t9i + tw5i * t8i - tw2i * t7i);
                cbi = tw4i * t11r - tw3i * t10r + tw1i * t9r + tw5i * t8r - tw2i * t7r;
                specialMulBackwardTo(car + cbr, cai + cbi, wa, waBase + wa(3, i, ido), ch, ch0 + 4 * chStride);
                specialMulBackwardTo(car - cbr, cai - cbi, wa, waBase + wa(6, i, ido), ch, ch0 + 7 * chStride);
                car = t1r + tw5r * t2r + tw1r * t3r + tw4r * t4r + tw2r * t5r + tw3r * t6r;
                cai = t1i + tw5r * t2i + tw1r * t3i + tw4r * t4i + tw2r * t5i + tw3r * t6i;
                cbr = -(tw5i * t11i - tw1i * t10i + tw4i * t9i - tw2i * t8i + tw3i * t7i);
                cbi = tw5i * t11r - tw1i * t10r + tw4i * t9r - tw2i * t8r + tw3i * t7r;
                specialMulBackwardTo(car + cbr, cai + cbi, wa, waBase + wa(4, i, ido), ch, ch0 + 5 * chStride);
                specialMulBackwardTo(car - cbr, cai - cbi, wa, waBase + wa(5, i, ido), ch, ch0 + 6 * chStride);
            }
        }
    }
    
    private void passgForward(int ido, int ip, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase, double[] tws, int twsBase,
                                    double[] wal, int walBase) {
        passgForwardImpl(ido, ip, l1, cc, ccBase, ch, chBase, wa, waBase, tws, twsBase, wal, walBase);
    }
    
    private void passgBackward(int ido, int ip, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase, double[] tws, int twsBase,
                                     double[] wal, int walBase) {
        passgBackwardImpl(ido, ip, l1, cc, ccBase, ch, chBase, wa, waBase, tws, twsBase, wal, walBase);
    }

    private void passgForwardImpl(int ido, int ip, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase,
                             double[] tws, int twsBase, double[] wal, int walBase) {
        int ipph = (ip + 1) / 2;
        int idl1 = ido * l1;
        wal[walBase] = 1.0;
        wal[walBase + 1] = 0.0;
        for (int i = 1; i < ip; i++) {
            wal[walBase + 2 * i] = tws[twsBase + 2 * i];
            wal[walBase + 2 * i + 1] = -tws[twsBase + 2 * i + 1];
        }

        for (int k = 0; k < l1; k++) {
            for (int i = 0; i < ido; i++) {
                copyPair(cc, cc(ccBase, i, 0, k, ido, ip), ch, ch(chBase, i, k, 0, ido, l1));
            }
        }
        for (int j = 1, jc = ip - 1; j < ipph; j++, jc--) {
            for (int k = 0; k < l1; k++) {
                for (int i = 0; i < ido; i++) {
                    int left = cc(ccBase, i, j, k, ido, ip);
                    int right = cc(ccBase, i, jc, k, ido, ip);
                    setPair(ch, ch(chBase, i, k, j, ido, l1), cc[left] + cc[right], cc[left + 1] + cc[right + 1]);
                    setPair(ch, ch(chBase, i, k, jc, ido, l1), cc[left] - cc[right], cc[left + 1] - cc[right + 1]);
                }
            }
        }
        for (int k = 0; k < l1; k++) {
            for (int i = 0; i < ido; i++) {
                int tmp = ch(chBase, i, k, 0, ido, l1);
                double sumR = ch[tmp];
                double sumI = ch[tmp + 1];
                for (int j = 1; j < ipph; j++) {
                    int term = ch(chBase, i, k, j, ido, l1);
                    sumR += ch[term];
                    sumI += ch[term + 1];
                }
                setPair(cc, cx(ccBase, i, k, 0, ido, l1), sumR, sumI);
            }
        }

        for (int l = 1, lc = ip - 1; l < ipph; l++, lc--) {
            for (int ik = 0; ik < idl1; ik++) {
                int outL = cx2(ccBase, ik, l, idl1);
                int outLc = cx2(ccBase, ik, lc, idl1);
                int ch0 = ch2(chBase, ik, 0, idl1);
                int ch1 = ch2(chBase, ik, 1, idl1);
                int ch2v = ch2(chBase, ik, 2, idl1);
                int chIp1 = ch2(chBase, ik, ip - 1, idl1);
                int chIp2 = ch2(chBase, ik, ip - 2, idl1);
                setPair(cc, outL,
                        ch[ch0] + wal[walBase + 2 * l] * ch[ch1] + wal[walBase + 4 * l] * ch[ch2v],
                        ch[ch0 + 1] + wal[walBase + 2 * l] * ch[ch1 + 1] + wal[walBase + 4 * l] * ch[ch2v + 1]);
                setPair(cc, outLc,
                        -wal[walBase + 2 * l + 1] * ch[chIp1 + 1] - wal[walBase + 4 * l + 1] * ch[chIp2 + 1],
                        wal[walBase + 2 * l + 1] * ch[chIp1] + wal[walBase + 4 * l + 1] * ch[chIp2]);
            }
            int iwal = 2 * l;
            int j = 3;
            int jc = ip - 3;
            for (; j < ipph - 1; j += 2, jc -= 2) {
                iwal += l;
                if (iwal > ip) {
                    iwal -= ip;
                }
                int xwal = 2 * iwal;
                iwal += l;
                if (iwal > ip) {
                    iwal -= ip;
                }
                int xwal2 = 2 * iwal;
                for (int ik = 0; ik < idl1; ik++) {
                    int outL = cx2(ccBase, ik, l, idl1);
                    int outLc = cx2(ccBase, ik, lc, idl1);
                    int chJ = ch2(chBase, ik, j, idl1);
                    int chJ1 = ch2(chBase, ik, j + 1, idl1);
                    int chJc = ch2(chBase, ik, jc, idl1);
                    int chJc1 = ch2(chBase, ik, jc - 1, idl1);
                    cc[outL] += ch[chJ] * wal[walBase + xwal] + ch[chJ1] * wal[walBase + xwal2];
                    cc[outL + 1] += ch[chJ + 1] * wal[walBase + xwal] + ch[chJ1 + 1] * wal[walBase + xwal2];
                    cc[outLc] -= ch[chJc + 1] * wal[walBase + xwal + 1] + ch[chJc1 + 1] * wal[walBase + xwal2 + 1];
                    cc[outLc + 1] += ch[chJc] * wal[walBase + xwal + 1] + ch[chJc1] * wal[walBase + xwal2 + 1];
                }
            }
            for (; j < ipph; j++, jc--) {
                iwal += l;
                if (iwal > ip) {
                    iwal -= ip;
                }
                int xwal = 2 * iwal;
                for (int ik = 0; ik < idl1; ik++) {
                    int outL = cx2(ccBase, ik, l, idl1);
                    int outLc = cx2(ccBase, ik, lc, idl1);
                    int chJ = ch2(chBase, ik, j, idl1);
                    int chJc = ch2(chBase, ik, jc, idl1);
                    cc[outL] += ch[chJ] * wal[walBase + xwal];
                    cc[outL + 1] += ch[chJ + 1] * wal[walBase + xwal];
                    cc[outLc] -= ch[chJc + 1] * wal[walBase + xwal + 1];
                    cc[outLc + 1] += ch[chJc] * wal[walBase + xwal + 1];
                }
            }
        }

        if (ido == 1) {
            for (int j = 1, jc = ip - 1; j < ipph; j++, jc--) {
                for (int ik = 0; ik < idl1; ik++) {
                    int left = cx2(ccBase, ik, j, idl1);
                    int right = cx2(ccBase, ik, jc, idl1);
                    double t1r = cc[left];
                    double t1i = cc[left + 1];
                    double t2r = cc[right];
                    double t2i = cc[right + 1];
                    setPair(cc, left, t1r + t2r, t1i + t2i);
                    setPair(cc, right, t1r - t2r, t1i - t2i);
                }
            }
        } else {
            for (int j = 1, jc = ip - 1; j < ipph; j++, jc--) {
                for (int k = 0; k < l1; k++) {
                    int left0 = cx(ccBase, 0, k, j, ido, l1);
                    int right0 = cx(ccBase, 0, k, jc, ido, l1);
                    double t1r = cc[left0];
                    double t1i = cc[left0 + 1];
                    double t2r = cc[right0];
                    double t2i = cc[right0 + 1];
                    setPair(cc, left0, t1r + t2r, t1i + t2i);
                    setPair(cc, right0, t1r - t2r, t1i - t2i);
                    for (int i = 1; i < ido; i++) {
                        int left = cx(ccBase, i, k, j, ido, l1);
                        int right = cx(ccBase, i, k, jc, ido, l1);
                        double x1r = cc[left] + cc[right];
                        double x1i = cc[left + 1] + cc[right + 1];
                        double x2r = cc[left] - cc[right];
                        double x2i = cc[left + 1] - cc[right + 1];
                        specialMulForwardTo(x1r, x1i, wa, waBase + 2 * ((j - 1) * (ido - 1) + i - 1), cc, left);
                        specialMulForwardTo(x2r, x2i, wa, waBase + 2 * ((jc - 1) * (ido - 1) + i - 1), cc, right);
                    }
                }
            }
        }
    }

    private void passgBackwardImpl(int ido, int ip, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase,
                             double[] tws, int twsBase, double[] wal, int walBase) {
        int ipph = (ip + 1) / 2;
        int idl1 = ido * l1;
        wal[walBase] = 1.0;
        wal[walBase + 1] = 0.0;
        for (int i = 1; i < ip; i++) {
            wal[walBase + 2 * i] = tws[twsBase + 2 * i];
            wal[walBase + 2 * i + 1] = tws[twsBase + 2 * i + 1];
        }

        for (int k = 0; k < l1; k++) {
            for (int i = 0; i < ido; i++) {
                copyPair(cc, cc(ccBase, i, 0, k, ido, ip), ch, ch(chBase, i, k, 0, ido, l1));
            }
        }
        for (int j = 1, jc = ip - 1; j < ipph; j++, jc--) {
            for (int k = 0; k < l1; k++) {
                for (int i = 0; i < ido; i++) {
                    int left = cc(ccBase, i, j, k, ido, ip);
                    int right = cc(ccBase, i, jc, k, ido, ip);
                    setPair(ch, ch(chBase, i, k, j, ido, l1), cc[left] + cc[right], cc[left + 1] + cc[right + 1]);
                    setPair(ch, ch(chBase, i, k, jc, ido, l1), cc[left] - cc[right], cc[left + 1] - cc[right + 1]);
                }
            }
        }
        for (int k = 0; k < l1; k++) {
            for (int i = 0; i < ido; i++) {
                int tmp = ch(chBase, i, k, 0, ido, l1);
                double sumR = ch[tmp];
                double sumI = ch[tmp + 1];
                for (int j = 1; j < ipph; j++) {
                    int term = ch(chBase, i, k, j, ido, l1);
                    sumR += ch[term];
                    sumI += ch[term + 1];
                }
                setPair(cc, cx(ccBase, i, k, 0, ido, l1), sumR, sumI);
            }
        }

        for (int l = 1, lc = ip - 1; l < ipph; l++, lc--) {
            for (int ik = 0; ik < idl1; ik++) {
                int outL = cx2(ccBase, ik, l, idl1);
                int outLc = cx2(ccBase, ik, lc, idl1);
                int ch0 = ch2(chBase, ik, 0, idl1);
                int ch1 = ch2(chBase, ik, 1, idl1);
                int ch2v = ch2(chBase, ik, 2, idl1);
                int chIp1 = ch2(chBase, ik, ip - 1, idl1);
                int chIp2 = ch2(chBase, ik, ip - 2, idl1);
                setPair(cc, outL,
                        ch[ch0] + wal[walBase + 2 * l] * ch[ch1] + wal[walBase + 4 * l] * ch[ch2v],
                        ch[ch0 + 1] + wal[walBase + 2 * l] * ch[ch1 + 1] + wal[walBase + 4 * l] * ch[ch2v + 1]);
                setPair(cc, outLc,
                        -wal[walBase + 2 * l + 1] * ch[chIp1 + 1] - wal[walBase + 4 * l + 1] * ch[chIp2 + 1],
                        wal[walBase + 2 * l + 1] * ch[chIp1] + wal[walBase + 4 * l + 1] * ch[chIp2]);
            }
            int iwal = 2 * l;
            int j = 3;
            int jc = ip - 3;
            for (; j < ipph - 1; j += 2, jc -= 2) {
                iwal += l;
                if (iwal > ip) {
                    iwal -= ip;
                }
                int xwal = 2 * iwal;
                iwal += l;
                if (iwal > ip) {
                    iwal -= ip;
                }
                int xwal2 = 2 * iwal;
                for (int ik = 0; ik < idl1; ik++) {
                    int outL = cx2(ccBase, ik, l, idl1);
                    int outLc = cx2(ccBase, ik, lc, idl1);
                    int chJ = ch2(chBase, ik, j, idl1);
                    int chJ1 = ch2(chBase, ik, j + 1, idl1);
                    int chJc = ch2(chBase, ik, jc, idl1);
                    int chJc1 = ch2(chBase, ik, jc - 1, idl1);
                    cc[outL] += ch[chJ] * wal[walBase + xwal] + ch[chJ1] * wal[walBase + xwal2];
                    cc[outL + 1] += ch[chJ + 1] * wal[walBase + xwal] + ch[chJ1 + 1] * wal[walBase + xwal2];
                    cc[outLc] -= ch[chJc + 1] * wal[walBase + xwal + 1] + ch[chJc1 + 1] * wal[walBase + xwal2 + 1];
                    cc[outLc + 1] += ch[chJc] * wal[walBase + xwal + 1] + ch[chJc1] * wal[walBase + xwal2 + 1];
                }
            }
            for (; j < ipph; j++, jc--) {
                iwal += l;
                if (iwal > ip) {
                    iwal -= ip;
                }
                int xwal = 2 * iwal;
                for (int ik = 0; ik < idl1; ik++) {
                    int outL = cx2(ccBase, ik, l, idl1);
                    int outLc = cx2(ccBase, ik, lc, idl1);
                    int chJ = ch2(chBase, ik, j, idl1);
                    int chJc = ch2(chBase, ik, jc, idl1);
                    cc[outL] += ch[chJ] * wal[walBase + xwal];
                    cc[outL + 1] += ch[chJ + 1] * wal[walBase + xwal];
                    cc[outLc] -= ch[chJc + 1] * wal[walBase + xwal + 1];
                    cc[outLc + 1] += ch[chJc] * wal[walBase + xwal + 1];
                }
            }
        }

        if (ido == 1) {
            for (int j = 1, jc = ip - 1; j < ipph; j++, jc--) {
                for (int ik = 0; ik < idl1; ik++) {
                    int left = cx2(ccBase, ik, j, idl1);
                    int right = cx2(ccBase, ik, jc, idl1);
                    double t1r = cc[left];
                    double t1i = cc[left + 1];
                    double t2r = cc[right];
                    double t2i = cc[right + 1];
                    setPair(cc, left, t1r + t2r, t1i + t2i);
                    setPair(cc, right, t1r - t2r, t1i - t2i);
                }
            }
        } else {
            for (int j = 1, jc = ip - 1; j < ipph; j++, jc--) {
                for (int k = 0; k < l1; k++) {
                    int left0 = cx(ccBase, 0, k, j, ido, l1);
                    int right0 = cx(ccBase, 0, k, jc, ido, l1);
                    double t1r = cc[left0];
                    double t1i = cc[left0 + 1];
                    double t2r = cc[right0];
                    double t2i = cc[right0 + 1];
                    setPair(cc, left0, t1r + t2r, t1i + t2i);
                    setPair(cc, right0, t1r - t2r, t1i - t2i);
                    for (int i = 1; i < ido; i++) {
                        int left = cx(ccBase, i, k, j, ido, l1);
                        int right = cx(ccBase, i, k, jc, ido, l1);
                        double x1r = cc[left] + cc[right];
                        double x1i = cc[left + 1] + cc[right + 1];
                        double x2r = cc[left] - cc[right];
                        double x2i = cc[left + 1] - cc[right + 1];
                        specialMulBackwardTo(x1r, x1i, wa, waBase + 2 * ((j - 1) * (ido - 1) + i - 1), cc, left);
                        specialMulBackwardTo(x2r, x2i, wa, waBase + 2 * ((jc - 1) * (ido - 1) + i - 1), cc, right);
                    }
                }
            }
        }
    }

    private static void setPair(double[] out, int offset, double real, double imag) {
        out[offset] = real;
        out[offset + 1] = imag;
    }

    private static void copyPair(double[] src, int srcOffset, double[] dst, int dstOffset) {
        dst[dstOffset] = src[srcOffset];
        dst[dstOffset + 1] = src[srcOffset + 1];
    }

    private static int cx(int a, int b, int c, int ido, int l1) {
        return 2 * (a + ido * (b + l1 * c));
    }

    private static int cx(int base, int a, int b, int c, int ido, int l1) {
        return base + cx(a, b, c, ido, l1);
    }

    private static int cx2(int a, int b, int idl1) {
        return 2 * (a + idl1 * b);
    }

    private static int cx2(int base, int a, int b, int idl1) {
        return base + cx2(a, b, idl1);
    }

    private static int ch2(int a, int b, int idl1) {
        return 2 * (a + idl1 * b);
    }

    private static int ch2(int base, int a, int b, int idl1) {
        return base + ch2(a, b, idl1);
    }

    static void specialMulForwardTo(double real, double imag, double[] wa, int waOffset, double[] out, int outOffset) {
        double wr = wa[waOffset];
        double wi = wa[waOffset + 1];
        out[outOffset] = Math.fma(real, wr, imag * wi);
        out[outOffset + 1] = Math.fma(imag, wr, -real * wi);
    }

    static void specialMulBackwardTo(double real, double imag, double[] wa, int waOffset, double[] out, int outOffset) {
        double wr = wa[waOffset];
        double wi = wa[waOffset + 1];
        out[outOffset] = Math.fma(real, wr, -imag * wi);
        out[outOffset + 1] = Math.fma(real, wi, imag * wr);
    }

    static double rotx90ForwardReal(double real, double imag) {
        return imag;
    }

    static double rotx90ForwardImag(double real, double imag) {
        return -real;
    }

    static double rotx90BackwardReal(double real, double imag) {
        return -imag;
    }

    static double rotx90BackwardImag(double real, double imag) {
        return real;
    }

    static double rotx45ForwardReal(double real, double imag) {
        return HSQT2 * (real + imag);
    }

    static double rotx45ForwardImag(double real, double imag) {
        return HSQT2 * (imag - real);
    }

    static double rotx45BackwardReal(double real, double imag) {
        return HSQT2 * (real - imag);
    }

    static double rotx45BackwardImag(double real, double imag) {
        return HSQT2 * (imag + real);
    }

    static double rotx135ForwardReal(double real, double imag) {
        return HSQT2 * (imag - real);
    }

    static double rotx135ForwardImag(double real, double imag) {
        return HSQT2 * (-real - imag);
    }

    static double rotx135BackwardReal(double real, double imag) {
        return HSQT2 * (-real - imag);
    }

    static double rotx135BackwardImag(double real, double imag) {
        return HSQT2 * (real - imag);
    }
    
    // Getters for testing
    int getLength() {
        return length;
    }
    
    int factorCount() {
        return factors.length;
    }

    int factorAt(int factorIndex) {
        return factors[factorIndex].factor;
    }

    int twiddlePairCountAt(int factorIndex) {
        FactorData fd = factors[factorIndex];
        int ip = fd.factor;
        int l1 = 1;
        for (int i = 0; i < factorIndex; i++) l1 *= factors[i].factor;
        int ido = length / (l1 * ip);
        return (ip - 1) * (ido - 1);
    }

    double twiddleRealAt(int factorIndex, int pairIndex) {
        return mem[factors[factorIndex].twOffset + 2 * pairIndex];
    }

    double twiddleImagAt(int factorIndex, int pairIndex) {
        return mem[factors[factorIndex].twOffset + 2 * pairIndex + 1];
    }

    int genericTwiddlePairCountAt(int factorIndex) {
        int twsOffset = factors[factorIndex].twsOffset;
        return twsOffset < 0 ? 0 : factors[factorIndex].factor;
    }

    double genericTwiddleRealAt(int factorIndex, int pairIndex) {
        return mem[factors[factorIndex].twsOffset + 2 * pairIndex];
    }

    double genericTwiddleImagAt(int factorIndex, int pairIndex) {
        return mem[factors[factorIndex].twsOffset + 2 * pairIndex + 1];
    }
}
