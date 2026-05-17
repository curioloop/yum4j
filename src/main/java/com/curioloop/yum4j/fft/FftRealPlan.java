package com.curioloop.yum4j.fft;

import com.curioloop.yum4j.math.VectorOps;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Objects;

/**
 * Source-level Java port of PocketFFT {@code rfftp<T0>}.
 *
 * <p>Upstream mapping: {@code reference/pocketfft-upstream/pocketfft_hdronly.h},
 * class {@code rfftp<T0>}: {@code factorize}, {@code twsize}, {@code comp_twiddle},
 * forward passes {@code radf2/3/4/5/radfg}, backward passes {@code radb2/3/4/5/radbg},
 * pass order, half-complex layout, and scale placement.</p>
 */
final class FftRealPlan {
    /**
     * @param twOffset  offset into shared mem array for regular twiddles (-1 if none)
     * @param twsOffset offset into shared mem array for generic twiddles (-1 if none)
     */
    private value record FactorData(int factor, int twOffset, int twsOffset) {}

    private final int length;
    private final double[] mem;     // single contiguous twiddle storage
    private final FactorData[] factors;
    private final FftWorkspace.Requirement workspaceRequirement;

    FftRealPlan(int length) {
        if (length < 1) {
            throw new IllegalArgumentException("zero-length FFT requested");
        }
        this.length = length;
        if (length == 1) {
            this.factors = new FactorData[0];
            this.mem = null;
            this.workspaceRequirement = FftWorkspace.Requirement.empty();
            return;
        }
        this.factors = factorize();
        this.mem = new double[computeTwiddleSize()];
        computeTwiddles();
        this.workspaceRequirement = FftWorkspace.Requirement.doubles(length);
    }

    void exec(double[] data, double factor, boolean realToHalfComplex) {
        FftWorkspace workspace = workspaceRequirement().allocate();
        workspace.reset();
        exec(data, factor, realToHalfComplex, workspace);
    }

    void exec(double[] data, double factor, boolean realToHalfComplex, FftWorkspace workspace) {
        exec(data, 0, factor, realToHalfComplex, workspace);
    }

    void exec(double[] data, int dataOffset, double factor, boolean realToHalfComplex, FftWorkspace workspace) {
        Objects.requireNonNull(workspace, "workspace");
        if (dataOffset < 0 || data.length - dataOffset < length) {
            throw new IllegalArgumentException("data length is smaller than transform length");
        }
        if (length == 1) {
            data[dataOffset] *= factor;
            return;
        }

        double[] arena = workspace.doubles();
        int workOffset = workspace.alloc(length);
        double[] input = data;
        int inputOffset = dataOffset;
        double[] output = arena;
        int outputOffset = workOffset;
        int nf = factors.length;

        if (realToHalfComplex) {
            int l1 = length;
            for (int factorIndex = 0; factorIndex < nf; factorIndex++) {
                int index = nf - factorIndex - 1;
                FactorData factorData = factors[index];
                int ip = factorData.factor;
                int ido = length / l1;
                l1 /= ip;
                switch (ip) {
                    case 2 -> radf2(ido, l1, input, inputOffset, output, outputOffset, mem, factorData.twOffset);
                    case 3 -> radf3(ido, l1, input, inputOffset, output, outputOffset, mem, factorData.twOffset);
                    case 4 -> radf4(ido, l1, input, inputOffset, output, outputOffset, mem, factorData.twOffset);
                    case 5 -> radf5(ido, l1, input, inputOffset, output, outputOffset, mem, factorData.twOffset);
                    default -> {
                        radfg(ido, ip, l1, input, inputOffset, output, outputOffset, mem, factorData.twOffset, mem, factorData.twsOffset);
                        double[] temporary = input;
                        input = output;
                        output = temporary;
                        int temporaryOffset = inputOffset;
                        inputOffset = outputOffset;
                        outputOffset = temporaryOffset;
                    }
                }
                double[] temporary = input;
                input = output;
                output = temporary;
                int temporaryOffset = inputOffset;
                inputOffset = outputOffset;
                outputOffset = temporaryOffset;
            }
        } else {
            int l1 = 1;
            for (FactorData factorData : factors) {
                int ip = factorData.factor;
                int ido = length / (ip * l1);
                switch (ip) {
                    case 2 -> radb2(ido, l1, input, inputOffset, output, outputOffset, mem, factorData.twOffset);
                    case 3 -> radb3(ido, l1, input, inputOffset, output, outputOffset, mem, factorData.twOffset);
                    case 4 -> radb4(ido, l1, input, inputOffset, output, outputOffset, mem, factorData.twOffset);
                    case 5 -> radb5(ido, l1, input, inputOffset, output, outputOffset, mem, factorData.twOffset);
                    default -> radbg(ido, ip, l1, input, inputOffset, output, outputOffset, mem, factorData.twOffset, mem, factorData.twsOffset);
                }
                double[] temporary = input;
                input = output;
                output = temporary;
                int temporaryOffset = inputOffset;
                inputOffset = outputOffset;
                outputOffset = temporaryOffset;
                l1 *= ip;
            }
        }

        VectorOps.scalTo(data, dataOffset, factor, input, inputOffset, length);
    }

    FftWorkspace.Requirement workspaceRequirement() {
        return workspaceRequirement;
    }

    int getLength() {
        return length;
    }

    private FactorData[] factorize() {
        ArrayList<FactorData> result = new ArrayList<>();
        int len = length;
        while ((len % 4) == 0) {
            result.add(new FactorData(4, -1, -1));
            len >>= 2;
        }
        if ((len % 2) == 0) {
            len >>= 1;
            result.add(new FactorData(2, -1, -1));
            Collections.swap(result, 0, result.size() - 1);
        }
        for (int divisor = 3; divisor <= len / divisor; divisor += 2) {
            while (len % divisor == 0) {
                result.add(new FactorData(divisor, -1, -1));
                len /= divisor;
            }
        }
        if (len > 1) {
            result.add(new FactorData(len, -1, -1));
        }
        return result.toArray(FactorData[]::new);
    }

    /** Compute total twiddle memory size in doubles. */
    private int computeTwiddleSize() {
        int twsize = 0;
        int l1 = 1;
        for (FactorData factor : factors) {
            int ip = factor.factor;
            int ido = length / (l1 * ip);
            // Regular twiddles: only for non-last factors
            // We compute for all factors here; computeTwiddles will skip the last one
            twsize += (ip - 1) * (ido - 1);
            if (ip > 5) {
                twsize += 2 * ip;
            }
            l1 *= ip;
        }
        return twsize;
    }

    private void computeTwiddles() {
        FftSincos twiddle = new FftSincos(length);
        int l1 = 1;
        int memOffset = 0;
        for (int factorIndex = 0; factorIndex < factors.length; factorIndex++) {
            FactorData factorData = factors[factorIndex];
            int ip = factorData.factor;
            int ido = length / (l1 * ip);

            // Regular twiddle factors (only for non-last factors)
            int twOffset = -1;
            if (factorIndex < factors.length - 1) {
                twOffset = memOffset;
                for (int j = 1; j < ip; j++) {
                    for (int i = 1; i <= (ido - 1) / 2; i++) {
                        twiddle.fill((long) j * l1 * i, mem, memOffset + (j - 1) * (ido - 1) + 2 * i - 2);
                    }
                }
                memOffset += (ip - 1) * (ido - 1);
            }

            // Special twiddle factors for generic passes (ip > 5)
            int twsOffset = -1;
            if (ip > 5) {
                twsOffset = memOffset;
                mem[memOffset] = 1.0;
                mem[memOffset + 1] = 0.0;
                for (int i = 2, ic = 2 * ip - 2; i <= ic; i += 2, ic -= 2) {
                    twiddle.fill((long) (i / 2) * (length / ip), mem, memOffset + i);
                    mem[memOffset + ic] = mem[memOffset + i];
                    mem[memOffset + ic + 1] = -mem[memOffset + i + 1];
                }
                memOffset += 2 * ip;
            }

            factors[factorIndex] = new FactorData(ip, twOffset, twsOffset);
            l1 *= ip;
        }
    }

    private void radf2(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        for (int k = 0; k < l1; k++) {
            double left = cc[ccF(ccBase, 0, k, 0, ido, l1)];
            double right = cc[ccF(ccBase, 0, k, 1, ido, l1)];
            ch[chF(chBase, 0, 0, k, ido, 2)] = left + right;
            ch[chF(chBase, ido - 1, 1, k, ido, 2)] = left - right;
        }
        if ((ido & 1) == 0) {
            for (int k = 0; k < l1; k++) {
                ch[chF(chBase, 0, 1, k, ido, 2)] = -cc[ccF(ccBase, ido - 1, k, 1, ido, l1)];
                ch[chF(chBase, ido - 1, 0, k, ido, 2)] = cc[ccF(ccBase, ido - 1, k, 0, ido, l1)];
            }
        }
        if (ido <= 2) {
            return;
        }
        for (int k = 0; k < l1; k++) {
            for (int i = 2; i < ido; i += 2) {
                int ic = ido - i;
                double tr2 = mulPmFirst(wa(wa, waBase, 0, i - 2, ido), wa(wa, waBase, 0, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 1, ido, l1)], cc[ccF(ccBase, i, k, 1, ido, l1)]);
                double ti2 = mulPmSecond(wa(wa, waBase, 0, i - 2, ido), wa(wa, waBase, 0, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 1, ido, l1)], cc[ccF(ccBase, i, k, 1, ido, l1)]);
                double baseReal = cc[ccF(ccBase, i - 1, k, 0, ido, l1)];
                double baseImag = cc[ccF(ccBase, i, k, 0, ido, l1)];
                ch[chF(chBase, i - 1, 0, k, ido, 2)] = baseReal + tr2;
                ch[chF(chBase, ic - 1, 1, k, ido, 2)] = baseReal - tr2;
                ch[chF(chBase, i, 0, k, ido, 2)] = ti2 + baseImag;
                ch[chF(chBase, ic, 1, k, ido, 2)] = ti2 - baseImag;
            }
        }
    }

    private void radf3(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        double taur = -0.5;
        double taui = 0.8660254037844386467637231707529362;
        for (int k = 0; k < l1; k++) {
            double cr2 = cc[ccF(ccBase, 0, k, 1, ido, l1)] + cc[ccF(ccBase, 0, k, 2, ido, l1)];
            ch[chF(chBase, 0, 0, k, ido, 3)] = cc[ccF(ccBase, 0, k, 0, ido, l1)] + cr2;
            ch[chF(chBase, 0, 2, k, ido, 3)] = taui * (cc[ccF(ccBase, 0, k, 2, ido, l1)] - cc[ccF(ccBase, 0, k, 1, ido, l1)]);
            ch[chF(chBase, ido - 1, 1, k, ido, 3)] = cc[ccF(ccBase, 0, k, 0, ido, l1)] + taur * cr2;
        }
        if (ido == 1) {
            return;
        }
        for (int k = 0; k < l1; k++) {
            for (int i = 2; i < ido; i += 2) {
                int ic = ido - i;
                double dr2 = mulPmFirst(wa(wa, waBase, 0, i - 2, ido), wa(wa, waBase, 0, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 1, ido, l1)], cc[ccF(ccBase, i, k, 1, ido, l1)]);
                double di2 = mulPmSecond(wa(wa, waBase, 0, i - 2, ido), wa(wa, waBase, 0, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 1, ido, l1)], cc[ccF(ccBase, i, k, 1, ido, l1)]);
                double dr3 = mulPmFirst(wa(wa, waBase, 1, i - 2, ido), wa(wa, waBase, 1, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 2, ido, l1)], cc[ccF(ccBase, i, k, 2, ido, l1)]);
                double di3 = mulPmSecond(wa(wa, waBase, 1, i - 2, ido), wa(wa, waBase, 1, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 2, ido, l1)], cc[ccF(ccBase, i, k, 2, ido, l1)]);

                double t1 = dr2 + dr3;
                double t2 = dr3 - dr2;
                double t3 = di2 + di3;
                double t4 = di2 - di3;
                dr2 = t1;
                di2 = t3;
                dr3 = t4;
                di3 = t2;

                double baseReal = cc[ccF(ccBase, i - 1, k, 0, ido, l1)];
                double baseImag = cc[ccF(ccBase, i, k, 0, ido, l1)];
                ch[chF(chBase, i - 1, 0, k, ido, 3)] = baseReal + dr2;
                ch[chF(chBase, i, 0, k, ido, 3)] = baseImag + di2;
                double tr2 = baseReal + taur * dr2;
                double ti2 = baseImag + taur * di2;
                double tr3 = taui * dr3;
                double ti3 = taui * di3;
                ch[chF(chBase, i - 1, 2, k, ido, 3)] = tr2 + tr3;
                ch[chF(chBase, ic - 1, 1, k, ido, 3)] = tr2 - tr3;
                ch[chF(chBase, i, 2, k, ido, 3)] = ti3 + ti2;
                ch[chF(chBase, ic, 1, k, ido, 3)] = ti3 - ti2;
            }
        }
    }

    private void radf4(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        double hsqt2 = 0.707106781186547524400844362104849;
        for (int k = 0; k < l1; k++) {
            double tr1 = cc[ccF(ccBase, 0, k, 3, ido, l1)] + cc[ccF(ccBase, 0, k, 1, ido, l1)];
            ch[chF(chBase, 0, 2, k, ido, 4)] = cc[ccF(ccBase, 0, k, 3, ido, l1)] - cc[ccF(ccBase, 0, k, 1, ido, l1)];
            double tr2 = cc[ccF(ccBase, 0, k, 0, ido, l1)] + cc[ccF(ccBase, 0, k, 2, ido, l1)];
            ch[chF(chBase, ido - 1, 1, k, ido, 4)] = cc[ccF(ccBase, 0, k, 0, ido, l1)] - cc[ccF(ccBase, 0, k, 2, ido, l1)];
            ch[chF(chBase, 0, 0, k, ido, 4)] = tr2 + tr1;
            ch[chF(chBase, ido - 1, 3, k, ido, 4)] = tr2 - tr1;
        }
        if ((ido & 1) == 0) {
            for (int k = 0; k < l1; k++) {
                double ti1 = -hsqt2 * (cc[ccF(ccBase, ido - 1, k, 1, ido, l1)] + cc[ccF(ccBase, ido - 1, k, 3, ido, l1)]);
                double tr1 = hsqt2 * (cc[ccF(ccBase, ido - 1, k, 1, ido, l1)] - cc[ccF(ccBase, ido - 1, k, 3, ido, l1)]);
                ch[chF(chBase, ido - 1, 0, k, ido, 4)] = cc[ccF(ccBase, ido - 1, k, 0, ido, l1)] + tr1;
                ch[chF(chBase, ido - 1, 2, k, ido, 4)] = cc[ccF(ccBase, ido - 1, k, 0, ido, l1)] - tr1;
                ch[chF(chBase, 0, 3, k, ido, 4)] = ti1 + cc[ccF(ccBase, ido - 1, k, 2, ido, l1)];
                ch[chF(chBase, 0, 1, k, ido, 4)] = ti1 - cc[ccF(ccBase, ido - 1, k, 2, ido, l1)];
            }
        }
        if (ido <= 2) {
            return;
        }
        for (int k = 0; k < l1; k++) {
            for (int i = 2; i < ido; i += 2) {
                int ic = ido - i;
                double cr2 = mulPmFirst(wa(wa, waBase, 0, i - 2, ido), wa(wa, waBase, 0, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 1, ido, l1)], cc[ccF(ccBase, i, k, 1, ido, l1)]);
                double ci2 = mulPmSecond(wa(wa, waBase, 0, i - 2, ido), wa(wa, waBase, 0, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 1, ido, l1)], cc[ccF(ccBase, i, k, 1, ido, l1)]);
                double cr3 = mulPmFirst(wa(wa, waBase, 1, i - 2, ido), wa(wa, waBase, 1, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 2, ido, l1)], cc[ccF(ccBase, i, k, 2, ido, l1)]);
                double ci3 = mulPmSecond(wa(wa, waBase, 1, i - 2, ido), wa(wa, waBase, 1, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 2, ido, l1)], cc[ccF(ccBase, i, k, 2, ido, l1)]);
                double cr4 = mulPmFirst(wa(wa, waBase, 2, i - 2, ido), wa(wa, waBase, 2, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 3, ido, l1)], cc[ccF(ccBase, i, k, 3, ido, l1)]);
                double ci4 = mulPmSecond(wa(wa, waBase, 2, i - 2, ido), wa(wa, waBase, 2, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 3, ido, l1)], cc[ccF(ccBase, i, k, 3, ido, l1)]);
                double tr1 = cr4 + cr2;
                double tr4 = cr4 - cr2;
                double ti1 = ci2 + ci4;
                double ti4 = ci2 - ci4;
                double tr2 = cc[ccF(ccBase, i - 1, k, 0, ido, l1)] + cr3;
                double tr3 = cc[ccF(ccBase, i - 1, k, 0, ido, l1)] - cr3;
                double ti2 = cc[ccF(ccBase, i, k, 0, ido, l1)] + ci3;
                double ti3 = cc[ccF(ccBase, i, k, 0, ido, l1)] - ci3;
                ch[chF(chBase, i - 1, 0, k, ido, 4)] = tr2 + tr1;
                ch[chF(chBase, ic - 1, 3, k, ido, 4)] = tr2 - tr1;
                ch[chF(chBase, i, 0, k, ido, 4)] = ti1 + ti2;
                ch[chF(chBase, ic, 3, k, ido, 4)] = ti1 - ti2;
                ch[chF(chBase, i - 1, 2, k, ido, 4)] = tr3 + ti4;
                ch[chF(chBase, ic - 1, 1, k, ido, 4)] = tr3 - ti4;
                ch[chF(chBase, i, 2, k, ido, 4)] = tr4 + ti3;
                ch[chF(chBase, ic, 1, k, ido, 4)] = tr4 - ti3;
            }
        }
    }

    private void radf5(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        double tr11 = 0.3090169943749474241022934171828191;
        double ti11 = 0.9510565162951535721164393333793821;
        double tr12 = -0.8090169943749474241022934171828191;
        double ti12 = 0.5877852522924731291687059546390728;
        for (int k = 0; k < l1; k++) {
            double cr2 = cc[ccF(ccBase, 0, k, 4, ido, l1)] + cc[ccF(ccBase, 0, k, 1, ido, l1)];
            double ci5 = cc[ccF(ccBase, 0, k, 4, ido, l1)] - cc[ccF(ccBase, 0, k, 1, ido, l1)];
            double cr3 = cc[ccF(ccBase, 0, k, 3, ido, l1)] + cc[ccF(ccBase, 0, k, 2, ido, l1)];
            double ci4 = cc[ccF(ccBase, 0, k, 3, ido, l1)] - cc[ccF(ccBase, 0, k, 2, ido, l1)];
            ch[chF(chBase, 0, 0, k, ido, 5)] = cc[ccF(ccBase, 0, k, 0, ido, l1)] + cr2 + cr3;
            ch[chF(chBase, ido - 1, 1, k, ido, 5)] = cc[ccF(ccBase, 0, k, 0, ido, l1)] + tr11 * cr2 + tr12 * cr3;
            ch[chF(chBase, 0, 2, k, ido, 5)] = ti11 * ci5 + ti12 * ci4;
            ch[chF(chBase, ido - 1, 3, k, ido, 5)] = cc[ccF(ccBase, 0, k, 0, ido, l1)] + tr12 * cr2 + tr11 * cr3;
            ch[chF(chBase, 0, 4, k, ido, 5)] = ti12 * ci5 - ti11 * ci4;
        }
        if (ido == 1) {
            return;
        }
        for (int k = 0; k < l1; k++) {
            for (int i = 2, ic = ido - 2; i < ido; i += 2, ic -= 2) {
                double dr2 = mulPmFirst(wa(wa, waBase, 0, i - 2, ido), wa(wa, waBase, 0, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 1, ido, l1)], cc[ccF(ccBase, i, k, 1, ido, l1)]);
                double di2 = mulPmSecond(wa(wa, waBase, 0, i - 2, ido), wa(wa, waBase, 0, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 1, ido, l1)], cc[ccF(ccBase, i, k, 1, ido, l1)]);
                double dr3 = mulPmFirst(wa(wa, waBase, 1, i - 2, ido), wa(wa, waBase, 1, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 2, ido, l1)], cc[ccF(ccBase, i, k, 2, ido, l1)]);
                double di3 = mulPmSecond(wa(wa, waBase, 1, i - 2, ido), wa(wa, waBase, 1, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 2, ido, l1)], cc[ccF(ccBase, i, k, 2, ido, l1)]);
                double dr4 = mulPmFirst(wa(wa, waBase, 2, i - 2, ido), wa(wa, waBase, 2, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 3, ido, l1)], cc[ccF(ccBase, i, k, 3, ido, l1)]);
                double di4 = mulPmSecond(wa(wa, waBase, 2, i - 2, ido), wa(wa, waBase, 2, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 3, ido, l1)], cc[ccF(ccBase, i, k, 3, ido, l1)]);
                double dr5 = mulPmFirst(wa(wa, waBase, 3, i - 2, ido), wa(wa, waBase, 3, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 4, ido, l1)], cc[ccF(ccBase, i, k, 4, ido, l1)]);
                double di5 = mulPmSecond(wa(wa, waBase, 3, i - 2, ido), wa(wa, waBase, 3, i - 1, ido),
                        cc[ccF(ccBase, i - 1, k, 4, ido, l1)], cc[ccF(ccBase, i, k, 4, ido, l1)]);

                double t1 = dr2 + dr5;
                double t2 = dr5 - dr2;
                double t3 = di2 + di5;
                double t4 = di2 - di5;
                dr2 = t1;
                di2 = t3;
                dr5 = t4;
                di5 = t2;
                t1 = dr3 + dr4;
                t2 = dr4 - dr3;
                t3 = di3 + di4;
                t4 = di3 - di4;
                dr3 = t1;
                di3 = t3;
                dr4 = t4;
                di4 = t2;

                double baseReal = cc[ccF(ccBase, i - 1, k, 0, ido, l1)];
                double baseImag = cc[ccF(ccBase, i, k, 0, ido, l1)];
                ch[chF(chBase, i - 1, 0, k, ido, 5)] = baseReal + dr2 + dr3;
                ch[chF(chBase, i, 0, k, ido, 5)] = baseImag + di2 + di3;
                double tr2 = baseReal + tr11 * dr2 + tr12 * dr3;
                double ti2 = baseImag + tr11 * di2 + tr12 * di3;
                double tr3 = baseReal + tr12 * dr2 + tr11 * dr3;
                double ti3 = baseImag + tr12 * di2 + tr11 * di3;
                double tr5 = ti11 * dr5 + ti12 * dr4;
                double ti5 = ti11 * di5 + ti12 * di4;
                double tr4 = ti12 * dr5 - ti11 * dr4;
                double ti4 = ti12 * di5 - ti11 * di4;
                ch[chF(chBase, i - 1, 2, k, ido, 5)] = tr2 + tr5;
                ch[chF(chBase, ic - 1, 1, k, ido, 5)] = tr2 - tr5;
                ch[chF(chBase, i, 2, k, ido, 5)] = ti5 + ti2;
                ch[chF(chBase, ic, 1, k, ido, 5)] = ti5 - ti2;
                ch[chF(chBase, i - 1, 4, k, ido, 5)] = tr3 + tr4;
                ch[chF(chBase, ic - 1, 3, k, ido, 5)] = tr3 - tr4;
                ch[chF(chBase, i, 4, k, ido, 5)] = ti4 + ti3;
                ch[chF(chBase, ic, 3, k, ido, 5)] = ti4 - ti3;
            }
        }
    }

    private void radfg(int ido, int ip, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase, double[] csarr, int csarrBase) {
        int ipph = (ip + 1) / 2;
        int idl1 = ido * l1;

        if (ido > 1) {
            for (int j = 1, jc = ip - 1; j < ipph; j++, jc--) {
                int is = waBase + (j - 1) * (ido - 1);
                int is2 = waBase + (jc - 1) * (ido - 1);
                for (int k = 0; k < l1; k++) {
                    int idij = is;
                    int idij2 = is2;
                    for (int i = 1; i <= ido - 2; i += 2) {
                        double t1 = cc[c1(ccBase, i, k, j, ido, l1)];
                        double t2 = cc[c1(ccBase, i + 1, k, j, ido, l1)];
                        double t3 = cc[c1(ccBase, i, k, jc, ido, l1)];
                        double t4 = cc[c1(ccBase, i + 1, k, jc, ido, l1)];
                        double x1 = wa[idij] * t1 + wa[idij + 1] * t2;
                        double x2 = wa[idij] * t2 - wa[idij + 1] * t1;
                        double x3 = wa[idij2] * t3 + wa[idij2 + 1] * t4;
                        double x4 = wa[idij2] * t4 - wa[idij2 + 1] * t3;
                        cc[c1(ccBase, i, k, j, ido, l1)] = x3 + x1;
                        cc[c1(ccBase, i + 1, k, jc, ido, l1)] = x3 - x1;
                        cc[c1(ccBase, i + 1, k, j, ido, l1)] = x2 + x4;
                        cc[c1(ccBase, i, k, jc, ido, l1)] = x2 - x4;
                        idij += 2;
                        idij2 += 2;
                    }
                }
            }
        }

        for (int j = 1, jc = ip - 1; j < ipph; j++, jc--) {
            for (int k = 0; k < l1; k++) {
                int left = c1(ccBase, 0, k, jc, ido, l1);
                int right = c1(ccBase, 0, k, j, ido, l1);
                double old = cc[left];
                cc[left] = old - cc[right];
                cc[right] = old + cc[right];
            }
        }

        for (int l = 1, lc = ip - 1; l < ipph; l++, lc--) {
            for (int ik = 0; ik < idl1; ik++) {
                ch[c2(chBase, ik, l, idl1)] = cc[c2(ccBase, ik, 0, idl1)] + csarr[csarrBase + 2 * l] * cc[c2(ccBase, ik, 1, idl1)]
                        + csarr[csarrBase + 4 * l] * cc[c2(ccBase, ik, 2, idl1)];
                ch[c2(chBase, ik, lc, idl1)] = csarr[csarrBase + 2 * l + 1] * cc[c2(ccBase, ik, ip - 1, idl1)]
                        + csarr[csarrBase + 4 * l + 1] * cc[c2(ccBase, ik, ip - 2, idl1)];
            }
            int iang = 2 * l;
            int j = 3;
            int jc = ip - 3;
            for (; j < ipph - 3; j += 4, jc -= 4) {
                iang += l;
                if (iang >= ip) {
                    iang -= ip;
                }
                double ar1 = csarr[csarrBase + 2 * iang];
                double ai1 = csarr[csarrBase + 2 * iang + 1];
                iang += l;
                if (iang >= ip) {
                    iang -= ip;
                }
                double ar2 = csarr[csarrBase + 2 * iang];
                double ai2 = csarr[csarrBase + 2 * iang + 1];
                iang += l;
                if (iang >= ip) {
                    iang -= ip;
                }
                double ar3 = csarr[csarrBase + 2 * iang];
                double ai3 = csarr[csarrBase + 2 * iang + 1];
                iang += l;
                if (iang >= ip) {
                    iang -= ip;
                }
                double ar4 = csarr[csarrBase + 2 * iang];
                double ai4 = csarr[csarrBase + 2 * iang + 1];
                for (int ik = 0; ik < idl1; ik++) {
                    ch[c2(chBase, ik, l, idl1)] += ar1 * cc[c2(ccBase, ik, j, idl1)] + ar2 * cc[c2(ccBase, ik, j + 1, idl1)]
                            + ar3 * cc[c2(ccBase, ik, j + 2, idl1)] + ar4 * cc[c2(ccBase, ik, j + 3, idl1)];
                    ch[c2(chBase, ik, lc, idl1)] += ai1 * cc[c2(ccBase, ik, jc, idl1)] + ai2 * cc[c2(ccBase, ik, jc - 1, idl1)]
                            + ai3 * cc[c2(ccBase, ik, jc - 2, idl1)] + ai4 * cc[c2(ccBase, ik, jc - 3, idl1)];
                }
            }
            for (; j < ipph - 1; j += 2, jc -= 2) {
                iang += l;
                if (iang >= ip) {
                    iang -= ip;
                }
                double ar1 = csarr[csarrBase + 2 * iang];
                double ai1 = csarr[csarrBase + 2 * iang + 1];
                iang += l;
                if (iang >= ip) {
                    iang -= ip;
                }
                double ar2 = csarr[csarrBase + 2 * iang];
                double ai2 = csarr[csarrBase + 2 * iang + 1];
                for (int ik = 0; ik < idl1; ik++) {
                    ch[c2(chBase, ik, l, idl1)] += ar1 * cc[c2(ccBase, ik, j, idl1)] + ar2 * cc[c2(ccBase, ik, j + 1, idl1)];
                    ch[c2(chBase, ik, lc, idl1)] += ai1 * cc[c2(ccBase, ik, jc, idl1)] + ai2 * cc[c2(ccBase, ik, jc - 1, idl1)];
                }
            }
            for (; j < ipph; j++, jc--) {
                iang += l;
                if (iang >= ip) {
                    iang -= ip;
                }
                double ar = csarr[csarrBase + 2 * iang];
                double ai = csarr[csarrBase + 2 * iang + 1];
                for (int ik = 0; ik < idl1; ik++) {
                    ch[c2(chBase, ik, l, idl1)] += ar * cc[c2(ccBase, ik, j, idl1)];
                    ch[c2(chBase, ik, lc, idl1)] += ai * cc[c2(ccBase, ik, jc, idl1)];
                }
            }
        }
        for (int ik = 0; ik < idl1; ik++) {
            ch[c2(chBase, ik, 0, idl1)] = cc[c2(ccBase, ik, 0, idl1)];
        }
        for (int j = 1; j < ipph; j++) {
            for (int ik = 0; ik < idl1; ik++) {
                ch[c2(chBase, ik, 0, idl1)] += cc[c2(ccBase, ik, j, idl1)];
            }
        }

        for (int k = 0; k < l1; k++) {
            for (int i = 0; i < ido; i++) {
                cc[ccG(ccBase, i, 0, k, ido, ip)] = ch[c1(chBase, i, k, 0, ido, l1)];
            }
        }
        for (int j = 1, jc = ip - 1; j < ipph; j++, jc--) {
            int j2 = 2 * j - 1;
            for (int k = 0; k < l1; k++) {
                cc[ccG(ccBase, ido - 1, j2, k, ido, ip)] = ch[c1(chBase, 0, k, j, ido, l1)];
                cc[ccG(ccBase, 0, j2 + 1, k, ido, ip)] = ch[c1(chBase, 0, k, jc, ido, l1)];
            }
        }
        if (ido == 1) {
            return;
        }
        for (int j = 1, jc = ip - 1; j < ipph; j++, jc--) {
            int j2 = 2 * j - 1;
            for (int k = 0; k < l1; k++) {
                for (int i = 1, ic = ido - i - 2; i <= ido - 2; i += 2, ic -= 2) {
                    cc[ccG(ccBase, i, j2 + 1, k, ido, ip)] = ch[c1(chBase, i, k, j, ido, l1)] + ch[c1(chBase, i, k, jc, ido, l1)];
                    cc[ccG(ccBase, ic, j2, k, ido, ip)] = ch[c1(chBase, i, k, j, ido, l1)] - ch[c1(chBase, i, k, jc, ido, l1)];
                    cc[ccG(ccBase, i + 1, j2 + 1, k, ido, ip)] = ch[c1(chBase, i + 1, k, j, ido, l1)] + ch[c1(chBase, i + 1, k, jc, ido, l1)];
                    cc[ccG(ccBase, ic + 1, j2, k, ido, ip)] = ch[c1(chBase, i + 1, k, jc, ido, l1)] - ch[c1(chBase, i + 1, k, j, ido, l1)];
                }
            }
        }
    }

    private void radb2(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        for (int k = 0; k < l1; k++) {
            double left = cc[ccB(ccBase, 0, 0, k, ido, 2)];
            double right = cc[ccB(ccBase, ido - 1, 1, k, ido, 2)];
            ch[chB(chBase, 0, k, 0, ido, l1)] = left + right;
            ch[chB(chBase, 0, k, 1, ido, l1)] = left - right;
        }
        if ((ido & 1) == 0) {
            for (int k = 0; k < l1; k++) {
                ch[chB(chBase, ido - 1, k, 0, ido, l1)] = 2.0 * cc[ccB(ccBase, ido - 1, 0, k, ido, 2)];
                ch[chB(chBase, ido - 1, k, 1, ido, l1)] = -2.0 * cc[ccB(ccBase, 0, 1, k, ido, 2)];
            }
        }
        if (ido <= 2) {
            return;
        }
        for (int k = 0; k < l1; k++) {
            for (int i = 2; i < ido; i += 2) {
                int ic = ido - i;
                double tr2 = cc[ccB(ccBase, i - 1, 0, k, ido, 2)] - cc[ccB(ccBase, ic - 1, 1, k, ido, 2)];
                ch[chB(chBase, i - 1, k, 0, ido, l1)] = cc[ccB(ccBase, i - 1, 0, k, ido, 2)] + cc[ccB(ccBase, ic - 1, 1, k, ido, 2)];
                double ti2 = cc[ccB(ccBase, i, 0, k, ido, 2)] + cc[ccB(ccBase, ic, 1, k, ido, 2)];
                ch[chB(chBase, i, k, 0, ido, l1)] = cc[ccB(ccBase, i, 0, k, ido, 2)] - cc[ccB(ccBase, ic, 1, k, ido, 2)];
                ch[chB(chBase, i, k, 1, ido, l1)] = mulPmFirst(wa(wa, waBase, 0, i - 2, ido), wa(wa, waBase, 0, i - 1, ido), ti2, tr2);
                ch[chB(chBase, i - 1, k, 1, ido, l1)] = mulPmSecond(wa(wa, waBase, 0, i - 2, ido), wa(wa, waBase, 0, i - 1, ido), ti2, tr2);
            }
        }
    }

    private void radb3(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        double taur = -0.5;
        double taui = 0.8660254037844386467637231707529362;
        for (int k = 0; k < l1; k++) {
            double tr2 = 2.0 * cc[ccB(ccBase, ido - 1, 1, k, ido, 3)];
            double cr2 = cc[ccB(ccBase, 0, 0, k, ido, 3)] + taur * tr2;
            ch[chB(chBase, 0, k, 0, ido, l1)] = cc[ccB(ccBase, 0, 0, k, ido, 3)] + tr2;
            double ci3 = 2.0 * taui * cc[ccB(ccBase, 0, 2, k, ido, 3)];
            ch[chB(chBase, 0, k, 2, ido, l1)] = cr2 + ci3;
            ch[chB(chBase, 0, k, 1, ido, l1)] = cr2 - ci3;
        }
        if (ido == 1) {
            return;
        }
        for (int k = 0; k < l1; k++) {
            for (int i = 2, ic = ido - 2; i < ido; i += 2, ic -= 2) {
                double tr2 = cc[ccB(ccBase, i - 1, 2, k, ido, 3)] + cc[ccB(ccBase, ic - 1, 1, k, ido, 3)];
                double ti2 = cc[ccB(ccBase, i, 2, k, ido, 3)] - cc[ccB(ccBase, ic, 1, k, ido, 3)];
                double cr2 = cc[ccB(ccBase, i - 1, 0, k, ido, 3)] + taur * tr2;
                double ci2 = cc[ccB(ccBase, i, 0, k, ido, 3)] + taur * ti2;
                ch[chB(chBase, i - 1, k, 0, ido, l1)] = cc[ccB(ccBase, i - 1, 0, k, ido, 3)] + tr2;
                ch[chB(chBase, i, k, 0, ido, l1)] = cc[ccB(ccBase, i, 0, k, ido, 3)] + ti2;
                double cr3 = taui * (cc[ccB(ccBase, i - 1, 2, k, ido, 3)] - cc[ccB(ccBase, ic - 1, 1, k, ido, 3)]);
                double ci3 = taui * (cc[ccB(ccBase, i, 2, k, ido, 3)] + cc[ccB(ccBase, ic, 1, k, ido, 3)]);
                double dr3 = cr2 + ci3;
                double dr2 = cr2 - ci3;
                double di2 = ci2 + cr3;
                double di3 = ci2 - cr3;
                ch[chB(chBase, i, k, 1, ido, l1)] = mulPmFirst(wa(wa, waBase, 0, i - 2, ido), wa(wa, waBase, 0, i - 1, ido), di2, dr2);
                ch[chB(chBase, i - 1, k, 1, ido, l1)] = mulPmSecond(wa(wa, waBase, 0, i - 2, ido), wa(wa, waBase, 0, i - 1, ido), di2, dr2);
                ch[chB(chBase, i, k, 2, ido, l1)] = mulPmFirst(wa(wa, waBase, 1, i - 2, ido), wa(wa, waBase, 1, i - 1, ido), di3, dr3);
                ch[chB(chBase, i - 1, k, 2, ido, l1)] = mulPmSecond(wa(wa, waBase, 1, i - 2, ido), wa(wa, waBase, 1, i - 1, ido), di3, dr3);
            }
        }
    }

    private void radb4(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        double sqrt2 = 1.414213562373095048801688724209698;
        for (int k = 0; k < l1; k++) {
            double tr2 = cc[ccB(ccBase, 0, 0, k, ido, 4)] + cc[ccB(ccBase, ido - 1, 3, k, ido, 4)];
            double tr1 = cc[ccB(ccBase, 0, 0, k, ido, 4)] - cc[ccB(ccBase, ido - 1, 3, k, ido, 4)];
            double tr3 = 2.0 * cc[ccB(ccBase, ido - 1, 1, k, ido, 4)];
            double tr4 = 2.0 * cc[ccB(ccBase, 0, 2, k, ido, 4)];
            ch[chB(chBase, 0, k, 0, ido, l1)] = tr2 + tr3;
            ch[chB(chBase, 0, k, 2, ido, l1)] = tr2 - tr3;
            ch[chB(chBase, 0, k, 3, ido, l1)] = tr1 + tr4;
            ch[chB(chBase, 0, k, 1, ido, l1)] = tr1 - tr4;
        }
        if ((ido & 1) == 0) {
            for (int k = 0; k < l1; k++) {
                double ti1 = cc[ccB(ccBase, 0, 3, k, ido, 4)] + cc[ccB(ccBase, 0, 1, k, ido, 4)];
                double ti2 = cc[ccB(ccBase, 0, 3, k, ido, 4)] - cc[ccB(ccBase, 0, 1, k, ido, 4)];
                double tr2 = cc[ccB(ccBase, ido - 1, 0, k, ido, 4)] + cc[ccB(ccBase, ido - 1, 2, k, ido, 4)];
                double tr1 = cc[ccB(ccBase, ido - 1, 0, k, ido, 4)] - cc[ccB(ccBase, ido - 1, 2, k, ido, 4)];
                ch[chB(chBase, ido - 1, k, 0, ido, l1)] = tr2 + tr2;
                ch[chB(chBase, ido - 1, k, 1, ido, l1)] = sqrt2 * (tr1 - ti1);
                ch[chB(chBase, ido - 1, k, 2, ido, l1)] = ti2 + ti2;
                ch[chB(chBase, ido - 1, k, 3, ido, l1)] = -sqrt2 * (tr1 + ti1);
            }
        }
        if (ido <= 2) {
            return;
        }
        for (int k = 0; k < l1; k++) {
            for (int i = 2; i < ido; i += 2) {
                int ic = ido - i;
                double tr2 = cc[ccB(ccBase, i - 1, 0, k, ido, 4)] + cc[ccB(ccBase, ic - 1, 3, k, ido, 4)];
                double tr1 = cc[ccB(ccBase, i - 1, 0, k, ido, 4)] - cc[ccB(ccBase, ic - 1, 3, k, ido, 4)];
                double ti1 = cc[ccB(ccBase, i, 0, k, ido, 4)] + cc[ccB(ccBase, ic, 3, k, ido, 4)];
                double ti2 = cc[ccB(ccBase, i, 0, k, ido, 4)] - cc[ccB(ccBase, ic, 3, k, ido, 4)];
                double tr4 = cc[ccB(ccBase, i, 2, k, ido, 4)] + cc[ccB(ccBase, ic, 1, k, ido, 4)];
                double ti3 = cc[ccB(ccBase, i, 2, k, ido, 4)] - cc[ccB(ccBase, ic, 1, k, ido, 4)];
                double tr3 = cc[ccB(ccBase, i - 1, 2, k, ido, 4)] + cc[ccB(ccBase, ic - 1, 1, k, ido, 4)];
                double ti4 = cc[ccB(ccBase, i - 1, 2, k, ido, 4)] - cc[ccB(ccBase, ic - 1, 1, k, ido, 4)];
                ch[chB(chBase, i - 1, k, 0, ido, l1)] = tr2 + tr3;
                ch[chB(chBase, i, k, 0, ido, l1)] = ti2 + ti3;
                double cr3 = tr2 - tr3;
                double ci3 = ti2 - ti3;
                double cr4 = tr1 + tr4;
                double cr2 = tr1 - tr4;
                double ci2 = ti1 + ti4;
                double ci4 = ti1 - ti4;
                ch[chB(chBase, i, k, 1, ido, l1)] = mulPmFirst(wa(wa, waBase, 0, i - 2, ido), wa(wa, waBase, 0, i - 1, ido), ci2, cr2);
                ch[chB(chBase, i - 1, k, 1, ido, l1)] = mulPmSecond(wa(wa, waBase, 0, i - 2, ido), wa(wa, waBase, 0, i - 1, ido), ci2, cr2);
                ch[chB(chBase, i, k, 2, ido, l1)] = mulPmFirst(wa(wa, waBase, 1, i - 2, ido), wa(wa, waBase, 1, i - 1, ido), ci3, cr3);
                ch[chB(chBase, i - 1, k, 2, ido, l1)] = mulPmSecond(wa(wa, waBase, 1, i - 2, ido), wa(wa, waBase, 1, i - 1, ido), ci3, cr3);
                ch[chB(chBase, i, k, 3, ido, l1)] = mulPmFirst(wa(wa, waBase, 2, i - 2, ido), wa(wa, waBase, 2, i - 1, ido), ci4, cr4);
                ch[chB(chBase, i - 1, k, 3, ido, l1)] = mulPmSecond(wa(wa, waBase, 2, i - 2, ido), wa(wa, waBase, 2, i - 1, ido), ci4, cr4);
            }
        }
    }

    private void radb5(int ido, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase) {
        double tr11 = 0.3090169943749474241022934171828191;
        double ti11 = 0.9510565162951535721164393333793821;
        double tr12 = -0.8090169943749474241022934171828191;
        double ti12 = 0.5877852522924731291687059546390728;
        for (int k = 0; k < l1; k++) {
            double ti5 = 2.0 * cc[ccB(ccBase, 0, 2, k, ido, 5)];
            double ti4 = 2.0 * cc[ccB(ccBase, 0, 4, k, ido, 5)];
            double tr2 = 2.0 * cc[ccB(ccBase, ido - 1, 1, k, ido, 5)];
            double tr3 = 2.0 * cc[ccB(ccBase, ido - 1, 3, k, ido, 5)];
            ch[chB(chBase, 0, k, 0, ido, l1)] = cc[ccB(ccBase, 0, 0, k, ido, 5)] + tr2 + tr3;
            double cr2 = cc[ccB(ccBase, 0, 0, k, ido, 5)] + tr11 * tr2 + tr12 * tr3;
            double cr3 = cc[ccB(ccBase, 0, 0, k, ido, 5)] + tr12 * tr2 + tr11 * tr3;
            double ci5 = ti5 * ti11 + ti4 * ti12;
            double ci4 = ti5 * ti12 - ti4 * ti11;
            ch[chB(chBase, 0, k, 4, ido, l1)] = cr2 + ci5;
            ch[chB(chBase, 0, k, 1, ido, l1)] = cr2 - ci5;
            ch[chB(chBase, 0, k, 3, ido, l1)] = cr3 + ci4;
            ch[chB(chBase, 0, k, 2, ido, l1)] = cr3 - ci4;
        }
        if (ido == 1) {
            return;
        }
        for (int k = 0; k < l1; k++) {
            for (int i = 2, ic = ido - 2; i < ido; i += 2, ic -= 2) {
                double tr2 = cc[ccB(ccBase, i - 1, 2, k, ido, 5)] + cc[ccB(ccBase, ic - 1, 1, k, ido, 5)];
                double tr5 = cc[ccB(ccBase, i - 1, 2, k, ido, 5)] - cc[ccB(ccBase, ic - 1, 1, k, ido, 5)];
                double ti5 = cc[ccB(ccBase, i, 2, k, ido, 5)] + cc[ccB(ccBase, ic, 1, k, ido, 5)];
                double ti2 = cc[ccB(ccBase, i, 2, k, ido, 5)] - cc[ccB(ccBase, ic, 1, k, ido, 5)];
                double tr3 = cc[ccB(ccBase, i - 1, 4, k, ido, 5)] + cc[ccB(ccBase, ic - 1, 3, k, ido, 5)];
                double tr4 = cc[ccB(ccBase, i - 1, 4, k, ido, 5)] - cc[ccB(ccBase, ic - 1, 3, k, ido, 5)];
                double ti4 = cc[ccB(ccBase, i, 4, k, ido, 5)] + cc[ccB(ccBase, ic, 3, k, ido, 5)];
                double ti3 = cc[ccB(ccBase, i, 4, k, ido, 5)] - cc[ccB(ccBase, ic, 3, k, ido, 5)];
                ch[chB(chBase, i - 1, k, 0, ido, l1)] = cc[ccB(ccBase, i - 1, 0, k, ido, 5)] + tr2 + tr3;
                ch[chB(chBase, i, k, 0, ido, l1)] = cc[ccB(ccBase, i, 0, k, ido, 5)] + ti2 + ti3;
                double cr2 = cc[ccB(ccBase, i - 1, 0, k, ido, 5)] + tr11 * tr2 + tr12 * tr3;
                double ci2 = cc[ccB(ccBase, i, 0, k, ido, 5)] + tr11 * ti2 + tr12 * ti3;
                double cr3 = cc[ccB(ccBase, i - 1, 0, k, ido, 5)] + tr12 * tr2 + tr11 * tr3;
                double ci3 = cc[ccB(ccBase, i, 0, k, ido, 5)] + tr12 * ti2 + tr11 * ti3;
                double cr5 = tr5 * ti11 + tr4 * ti12;
                double cr4 = tr5 * ti12 - tr4 * ti11;
                double ci5 = ti5 * ti11 + ti4 * ti12;
                double ci4 = ti5 * ti12 - ti4 * ti11;
                double dr4 = cr3 + ci4;
                double dr3 = cr3 - ci4;
                double di3 = ci3 + cr4;
                double di4 = ci3 - cr4;
                double dr5 = cr2 + ci5;
                double dr2 = cr2 - ci5;
                double di2 = ci2 + cr5;
                double di5 = ci2 - cr5;
                ch[chB(chBase, i, k, 1, ido, l1)] = mulPmFirst(wa(wa, waBase, 0, i - 2, ido), wa(wa, waBase, 0, i - 1, ido), di2, dr2);
                ch[chB(chBase, i - 1, k, 1, ido, l1)] = mulPmSecond(wa(wa, waBase, 0, i - 2, ido), wa(wa, waBase, 0, i - 1, ido), di2, dr2);
                ch[chB(chBase, i, k, 2, ido, l1)] = mulPmFirst(wa(wa, waBase, 1, i - 2, ido), wa(wa, waBase, 1, i - 1, ido), di3, dr3);
                ch[chB(chBase, i - 1, k, 2, ido, l1)] = mulPmSecond(wa(wa, waBase, 1, i - 2, ido), wa(wa, waBase, 1, i - 1, ido), di3, dr3);
                ch[chB(chBase, i, k, 3, ido, l1)] = mulPmFirst(wa(wa, waBase, 2, i - 2, ido), wa(wa, waBase, 2, i - 1, ido), di4, dr4);
                ch[chB(chBase, i - 1, k, 3, ido, l1)] = mulPmSecond(wa(wa, waBase, 2, i - 2, ido), wa(wa, waBase, 2, i - 1, ido), di4, dr4);
                ch[chB(chBase, i, k, 4, ido, l1)] = mulPmFirst(wa(wa, waBase, 3, i - 2, ido), wa(wa, waBase, 3, i - 1, ido), di5, dr5);
                ch[chB(chBase, i - 1, k, 4, ido, l1)] = mulPmSecond(wa(wa, waBase, 3, i - 2, ido), wa(wa, waBase, 3, i - 1, ido), di5, dr5);
            }
        }
    }

    private void radbg(int ido, int ip, int l1, double[] cc, int ccBase, double[] ch, int chBase, double[] wa, int waBase, double[] csarr, int csarrBase) {
        int ipph = (ip + 1) / 2;
        int idl1 = ido * l1;

        for (int k = 0; k < l1; k++) {
            for (int i = 0; i < ido; i++) {
                ch[c1(chBase, i, k, 0, ido, l1)] = cc[ccG(ccBase, i, 0, k, ido, ip)];
            }
        }
        for (int j = 1, jc = ip - 1; j < ipph; j++, jc--) {
            int j2 = 2 * j - 1;
            for (int k = 0; k < l1; k++) {
                ch[c1(chBase, 0, k, j, ido, l1)] = 2.0 * cc[ccG(ccBase, ido - 1, j2, k, ido, ip)];
                ch[c1(chBase, 0, k, jc, ido, l1)] = 2.0 * cc[ccG(ccBase, 0, j2 + 1, k, ido, ip)];
            }
        }
        if (ido != 1) {
            for (int j = 1, jc = ip - 1; j < ipph; j++, jc--) {
                int j2 = 2 * j - 1;
                for (int k = 0; k < l1; k++) {
                    for (int i = 1, ic = ido - i - 2; i <= ido - 2; i += 2, ic -= 2) {
                        ch[c1(chBase, i, k, j, ido, l1)] = cc[ccG(ccBase, i, j2 + 1, k, ido, ip)] + cc[ccG(ccBase, ic, j2, k, ido, ip)];
                        ch[c1(chBase, i, k, jc, ido, l1)] = cc[ccG(ccBase, i, j2 + 1, k, ido, ip)] - cc[ccG(ccBase, ic, j2, k, ido, ip)];
                        ch[c1(chBase, i + 1, k, j, ido, l1)] = cc[ccG(ccBase, i + 1, j2 + 1, k, ido, ip)] - cc[ccG(ccBase, ic + 1, j2, k, ido, ip)];
                        ch[c1(chBase, i + 1, k, jc, ido, l1)] = cc[ccG(ccBase, i + 1, j2 + 1, k, ido, ip)] + cc[ccG(ccBase, ic + 1, j2, k, ido, ip)];
                    }
                }
            }
        }

        for (int l = 1, lc = ip - 1; l < ipph; l++, lc--) {
            for (int ik = 0; ik < idl1; ik++) {
                cc[c2(ccBase, ik, l, idl1)] = ch[c2(chBase, ik, 0, idl1)] + csarr[csarrBase + 2 * l] * ch[c2(chBase, ik, 1, idl1)]
                        + csarr[csarrBase + 4 * l] * ch[c2(chBase, ik, 2, idl1)];
                cc[c2(ccBase, ik, lc, idl1)] = csarr[csarrBase + 2 * l + 1] * ch[c2(chBase, ik, ip - 1, idl1)]
                        + csarr[csarrBase + 4 * l + 1] * ch[c2(chBase, ik, ip - 2, idl1)];
            }
            int iang = 2 * l;
            int j = 3;
            int jc = ip - 3;
            for (; j < ipph - 3; j += 4, jc -= 4) {
                iang += l;
                if (iang > ip) {
                    iang -= ip;
                }
                double ar1 = csarr[csarrBase + 2 * iang];
                double ai1 = csarr[csarrBase + 2 * iang + 1];
                iang += l;
                if (iang > ip) {
                    iang -= ip;
                }
                double ar2 = csarr[csarrBase + 2 * iang];
                double ai2 = csarr[csarrBase + 2 * iang + 1];
                iang += l;
                if (iang > ip) {
                    iang -= ip;
                }
                double ar3 = csarr[csarrBase + 2 * iang];
                double ai3 = csarr[csarrBase + 2 * iang + 1];
                iang += l;
                if (iang > ip) {
                    iang -= ip;
                }
                double ar4 = csarr[csarrBase + 2 * iang];
                double ai4 = csarr[csarrBase + 2 * iang + 1];
                for (int ik = 0; ik < idl1; ik++) {
                    cc[c2(ccBase, ik, l, idl1)] += ar1 * ch[c2(chBase, ik, j, idl1)] + ar2 * ch[c2(chBase, ik, j + 1, idl1)]
                            + ar3 * ch[c2(chBase, ik, j + 2, idl1)] + ar4 * ch[c2(chBase, ik, j + 3, idl1)];
                    cc[c2(ccBase, ik, lc, idl1)] += ai1 * ch[c2(chBase, ik, jc, idl1)] + ai2 * ch[c2(chBase, ik, jc - 1, idl1)]
                            + ai3 * ch[c2(chBase, ik, jc - 2, idl1)] + ai4 * ch[c2(chBase, ik, jc - 3, idl1)];
                }
            }
            for (; j < ipph - 1; j += 2, jc -= 2) {
                iang += l;
                if (iang > ip) {
                    iang -= ip;
                }
                double ar1 = csarr[csarrBase + 2 * iang];
                double ai1 = csarr[csarrBase + 2 * iang + 1];
                iang += l;
                if (iang > ip) {
                    iang -= ip;
                }
                double ar2 = csarr[csarrBase + 2 * iang];
                double ai2 = csarr[csarrBase + 2 * iang + 1];
                for (int ik = 0; ik < idl1; ik++) {
                    cc[c2(ccBase, ik, l, idl1)] += ar1 * ch[c2(chBase, ik, j, idl1)] + ar2 * ch[c2(chBase, ik, j + 1, idl1)];
                    cc[c2(ccBase, ik, lc, idl1)] += ai1 * ch[c2(chBase, ik, jc, idl1)] + ai2 * ch[c2(chBase, ik, jc - 1, idl1)];
                }
            }
            for (; j < ipph; j++, jc--) {
                iang += l;
                if (iang > ip) {
                    iang -= ip;
                }
                double ar = csarr[csarrBase + 2 * iang];
                double ai = csarr[csarrBase + 2 * iang + 1];
                for (int ik = 0; ik < idl1; ik++) {
                    cc[c2(ccBase, ik, l, idl1)] += ar * ch[c2(chBase, ik, j, idl1)];
                    cc[c2(ccBase, ik, lc, idl1)] += ai * ch[c2(chBase, ik, jc, idl1)];
                }
            }
        }
        for (int j = 1; j < ipph; j++) {
            for (int ik = 0; ik < idl1; ik++) {
                ch[c2(chBase, ik, 0, idl1)] += ch[c2(chBase, ik, j, idl1)];
            }
        }
        for (int j = 1, jc = ip - 1; j < ipph; j++, jc--) {
            for (int k = 0; k < l1; k++) {
                int leftCc = c1(ccBase, 0, k, jc, ido, l1);
                int rightCc = c1(ccBase, 0, k, j, ido, l1);
                int leftCh = c1(chBase, 0, k, jc, ido, l1);
                int rightCh = c1(chBase, 0, k, j, ido, l1);
                double c1Left = cc[leftCc];
                double c1Right = cc[rightCc];
                ch[leftCh] = c1Left + c1Right;
                ch[rightCh] = c1Right - c1Left;
            }
        }

        if (ido == 1) {
            return;
        }

        for (int j = 1, jc = ip - 1; j < ipph; j++, jc--) {
            for (int k = 0; k < l1; k++) {
                for (int i = 1; i <= ido - 2; i += 2) {
                    ch[c1(chBase, i, k, j, ido, l1)] = cc[c1(ccBase, i, k, j, ido, l1)] - cc[c1(ccBase, i + 1, k, jc, ido, l1)];
                    ch[c1(chBase, i, k, jc, ido, l1)] = cc[c1(ccBase, i, k, j, ido, l1)] + cc[c1(ccBase, i + 1, k, jc, ido, l1)];
                    ch[c1(chBase, i + 1, k, j, ido, l1)] = cc[c1(ccBase, i + 1, k, j, ido, l1)] + cc[c1(ccBase, i, k, jc, ido, l1)];
                    ch[c1(chBase, i + 1, k, jc, ido, l1)] = cc[c1(ccBase, i + 1, k, j, ido, l1)] - cc[c1(ccBase, i, k, jc, ido, l1)];
                }
            }
        }

        for (int j = 1; j < ip; j++) {
            int is = waBase + (j - 1) * (ido - 1);
            for (int k = 0; k < l1; k++) {
                int idij = is;
                for (int i = 1; i <= ido - 2; i += 2) {
                    double t1 = ch[c1(chBase, i, k, j, ido, l1)];
                    double t2 = ch[c1(chBase, i + 1, k, j, ido, l1)];
                    ch[c1(chBase, i, k, j, ido, l1)] = wa[idij] * t1 - wa[idij + 1] * t2;
                    ch[c1(chBase, i + 1, k, j, ido, l1)] = wa[idij] * t2 + wa[idij + 1] * t1;
                    idij += 2;
                }
            }
        }
    }

    private static double mulPmFirst(double c, double d, double e, double f) {
        return c * e + d * f;
    }

    private static double mulPmSecond(double c, double d, double e, double f) {
        return c * f - d * e;
    }

    private static double wa(double[] wa, int waBase, int x, int i, int ido) {
        return wa[waBase + i + x * (ido - 1)];
    }

    private static int ccF(int a, int b, int c, int ido, int l1) {
        return a + ido * (b + l1 * c);
    }

    private static int ccF(int base, int a, int b, int c, int ido, int l1) {
        return base + ccF(a, b, c, ido, l1);
    }

    private static int chF(int a, int b, int c, int ido, int ip) {
        return a + ido * (b + ip * c);
    }

    private static int chF(int base, int a, int b, int c, int ido, int ip) {
        return base + chF(a, b, c, ido, ip);
    }

    private static int ccB(int a, int b, int c, int ido, int ip) {
        return a + ido * (b + ip * c);
    }

    private static int ccB(int base, int a, int b, int c, int ido, int ip) {
        return base + ccB(a, b, c, ido, ip);
    }

    private static int chB(int a, int b, int c, int ido, int l1) {
        return a + ido * (b + l1 * c);
    }

    private static int chB(int base, int a, int b, int c, int ido, int l1) {
        return base + chB(a, b, c, ido, l1);
    }

    private static int c1(int a, int b, int c, int ido, int l1) {
        return a + ido * (b + l1 * c);
    }

    private static int c1(int base, int a, int b, int c, int ido, int l1) {
        return base + c1(a, b, c, ido, l1);
    }

    private static int c2(int a, int b, int idl1) {
        return a + idl1 * b;
    }

    private static int c2(int base, int a, int b, int idl1) {
        return base + c2(a, b, idl1);
    }

    private static int ccG(int a, int b, int c, int ido, int ip) {
        return a + ido * (b + ip * c);
    }

    private static int ccG(int base, int a, int b, int c, int ido, int ip) {
        return base + ccG(a, b, c, ido, ip);
    }
}