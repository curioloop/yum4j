package com.curioloop.yum4j.fft;

final class FftDcstPlan {
    private static final double SQRT2 = 1.414213562373095048801688724209698;

    private final int length;
    private final int type;
    private final boolean cosine;
    private final FftRealTransform realPlan;
    private final FftComplexTransform complexPlan;
    private final double[] twiddle;
    private final double[] c2;

    FftDcstPlan(int length, int type, boolean cosine) {
        if (length < 1) {
            throw new IllegalArgumentException("zero-length transform requested");
        }
        if (type < 1 || type > 4) {
            throw new IllegalArgumentException("invalid DCT/DST type");
        }
        if (cosine && type == 1 && length < 2) {
            throw new IllegalArgumentException("DCT-I requires length >= 2");
        }
        this.length = length;
        this.type = type;
        this.cosine = cosine;
        if (type == 1) {
            this.realPlan = FftPlanCache.realTransform(cosine ? 2 * (length - 1) : 2 * (length + 1));
            this.complexPlan = null;
            this.twiddle = null;
            this.c2 = null;
        } else if (type == 2 || type == 3) {
            this.realPlan = FftPlanCache.realTransform(length);
            this.complexPlan = null;
            this.twiddle = new double[length];
            FftSincos sincos = new FftSincos(4L * length);
            for (int index = 0; index < length; index++) {
                twiddle[index] = sincos.real(index + 1L);
            }
            this.c2 = null;
        } else {
            if ((length & 1) == 0) {
                this.realPlan = null;
                this.complexPlan = FftPlanCache.complexTransform(length / 2);
                this.c2 = new double[length];
                FftSincos sincos = new FftSincos(16L * length);
                for (int index = 0; index < length / 2; index++) {
                    double[] value = sincos.value(8L * index + 1L);
                    c2[2 * index] = value[0];
                    c2[2 * index + 1] = -value[1];
                }
            } else {
                this.realPlan = FftPlanCache.realTransform(length);
                this.complexPlan = null;
                this.c2 = null;
            }
            this.twiddle = null;
        }
    }

    void exec(double[] data, double factor, boolean ortho) {
        FftWorkspace workspace = workspaceRequirement().allocate();
        workspace.reset();
        exec(data, factor, ortho, workspace);
    }

    void exec(double[] data, double factor, boolean ortho, FftWorkspace workspace) {
        exec(data, 0, factor, ortho, workspace);
    }

    void exec(double[] data, int dataOffset, double factor, boolean ortho, FftWorkspace workspace) {
        if (dataOffset < 0 || data.length - dataOffset < length) {
            throw new IllegalArgumentException("data length is smaller than transform length");
        }
        switch (type) {
            case 1 -> {
                if (cosine) {
                    execDct1(data, dataOffset, factor, ortho, workspace);
                } else {
                    execDst1(data, dataOffset, factor, workspace);
                }
            }
            case 2 -> execDcst23(data, dataOffset, factor, ortho, true, workspace);
            case 3 -> execDcst23(data, dataOffset, factor, ortho, false, workspace);
            case 4 -> execDcst4(data, dataOffset, factor, workspace);
            default -> throw new IllegalStateException("unexpected DCT/DST type");
        }
    }

    void exec(double[] data, int dataOffset, double factor, boolean ortho, int type, boolean cosine,
              FftWorkspace workspace) {
        if (type != this.type || cosine != this.cosine) {
            throw new IllegalArgumentException("DCT/DST plan type mismatch");
        }
        exec(data, dataOffset, factor, ortho, workspace);
    }

    FftWorkspace.Requirement workspaceRequirement() {
        if (type == 1) {
            int transformLength = realPlan.length();
            return FftWorkspace.Requirement.doubles(transformLength).plus(realPlan.workspaceRequirement());
        }
        if (type == 2 || type == 3) {
            return realPlan.workspaceRequirement();
        }
        if ((length & 1) != 0) {
            return FftWorkspace.Requirement.doubles(length).plus(realPlan.workspaceRequirement());
        }
        return FftWorkspace.Requirement.doubles(length).plus(complexPlan.workspaceRequirement());
    }

    int length() {
        return length;
    }

    private void execDct1(double[] data, int dataOffset, double factor, boolean ortho, FftWorkspace workspace) {
        int transformLength = realPlan.length();
        int n = transformLength / 2 + 1;
        if (ortho) {
            data[dataOffset] *= SQRT2;
            data[dataOffset + n - 1] *= SQRT2;
        }
        double[] tmp = workspace.doubles();
        int tmpOffset = workspace.alloc(transformLength);
        tmp[tmpOffset] = data[dataOffset];
        for (int index = 1; index < n; index++) {
            tmp[tmpOffset + index] = data[dataOffset + index];
            tmp[tmpOffset + transformLength - index] = data[dataOffset + index];
        }
        realPlan.exec(tmp, tmpOffset, factor, true, workspace);
        data[dataOffset] = tmp[tmpOffset];
        for (int index = 1; index < n; index++) {
            data[dataOffset + index] = tmp[tmpOffset + 2 * index - 1];
        }
        if (ortho) {
            data[dataOffset] *= SQRT2 * 0.5;
            data[dataOffset + n - 1] *= SQRT2 * 0.5;
        }
    }

    private void execDst1(double[] data, int dataOffset, double factor, FftWorkspace workspace) {
        int transformLength = realPlan.length();
        int n = transformLength / 2 - 1;
        double[] tmp = workspace.doubles();
        int tmpOffset = workspace.alloc(transformLength);
        tmp[tmpOffset] = 0.0;
        tmp[tmpOffset + n + 1] = 0.0;
        for (int index = 0; index < n; index++) {
            tmp[tmpOffset + index + 1] = data[dataOffset + index];
            tmp[tmpOffset + transformLength - 1 - index] = -data[dataOffset + index];
        }
        realPlan.exec(tmp, tmpOffset, factor, true, workspace);
        for (int index = 0; index < n; index++) {
            data[dataOffset + index] = -tmp[tmpOffset + 2 * index + 2];
        }
    }

    private void execDcst23(double[] data, int dataOffset, double factor, boolean ortho, boolean type2,
                            FftWorkspace workspace) {
        int n = length;
        int ns2 = (n + 1) / 2;
        if (type2) {
            if (!cosine) {
                for (int k = 1; k < n; k += 2) {
                    data[dataOffset + k] = -data[dataOffset + k];
                }
            }
            data[dataOffset] *= 2.0;
            if ((n & 1) == 0) {
                data[dataOffset + n - 1] *= 2.0;
            }
            for (int k = 1; k < n - 1; k += 2) {
                mpInPlace(data, dataOffset + k + 1, dataOffset + k);
            }
            realPlan.exec(data, dataOffset, factor, false, workspace);
            for (int k = 1, kc = n - 1; k < ns2; k++, kc--) {
                double t1 = twiddle[k - 1] * data[dataOffset + kc] + twiddle[kc - 1] * data[dataOffset + k];
                double t2 = twiddle[k - 1] * data[dataOffset + k] - twiddle[kc - 1] * data[dataOffset + kc];
                data[dataOffset + k] = 0.5 * (t1 + t2);
                data[dataOffset + kc] = 0.5 * (t1 - t2);
            }
            if ((n & 1) == 0) {
                data[dataOffset + ns2] *= twiddle[ns2 - 1];
            }
            if (!cosine) {
                reverse(data, dataOffset, n);
            }
            if (ortho) {
                if (cosine) {
                    data[dataOffset] *= SQRT2 * 0.5;
                } else {
                    data[dataOffset + n - 1] *= SQRT2 * 0.5;
                }
            }
        } else {
            if (ortho) {
                if (cosine) {
                    data[dataOffset] *= SQRT2;
                } else {
                    data[dataOffset + n - 1] *= SQRT2;
                }
            }
            if (!cosine) {
                reverseLowerHalf(data, dataOffset, n, ns2);
            }
            for (int k = 1, kc = n - 1; k < ns2; k++, kc--) {
                double t1 = data[dataOffset + k] + data[dataOffset + kc];
                double t2 = data[dataOffset + k] - data[dataOffset + kc];
                data[dataOffset + k] = twiddle[k - 1] * t2 + twiddle[kc - 1] * t1;
                data[dataOffset + kc] = twiddle[k - 1] * t1 - twiddle[kc - 1] * t2;
            }
            if ((n & 1) == 0) {
                data[dataOffset + ns2] *= 2.0 * twiddle[ns2 - 1];
            }
            realPlan.exec(data, dataOffset, factor, true, workspace);
            for (int k = 1; k < n - 1; k += 2) {
                mpInPlace(data, dataOffset + k, dataOffset + k + 1);
            }
            if (!cosine) {
                for (int k = 1; k < n; k += 2) {
                    data[dataOffset + k] = -data[dataOffset + k];
                }
            }
        }
    }

    private void execDcst4(double[] data, int dataOffset, double factor, FftWorkspace workspace) {
        int n2 = length / 2;
        if (!cosine) {
            for (int k = 0, kc = length - 1; k < n2; k++, kc--) {
                swap(data, dataOffset + k, dataOffset + kc);
            }
        }
        if ((length & 1) != 0) {
            execDcst4Odd(data, dataOffset, factor, n2, workspace);
        } else {
            execDcst4Even(data, dataOffset, factor, n2, workspace);
        }
        if (!cosine) {
            for (int k = 1; k < length; k += 2) {
                data[dataOffset + k] = -data[dataOffset + k];
            }
        }
    }

    private void execDcst4Odd(double[] data, int dataOffset, double factor, int n2, FftWorkspace workspace) {
        double[] y = workspace.doubles();
        int yOffset = workspace.alloc(length);
        int index = 0;
        int m = n2;
        for (; m < length; index++, m += 4) {
            y[yOffset + index] = data[dataOffset + m];
        }
        for (; m < 2 * length; index++, m += 4) {
            y[yOffset + index] = -data[dataOffset + 2 * length - m - 1];
        }
        for (; m < 3 * length; index++, m += 4) {
            y[yOffset + index] = -data[dataOffset + m - 2 * length];
        }
        for (; m < 4 * length; index++, m += 4) {
            y[yOffset + index] = data[dataOffset + 4 * length - m - 1];
        }
        for (; index < length; index++, m += 4) {
            y[yOffset + index] = data[dataOffset + m - 4 * length];
        }
        realPlan.exec(y, yOffset, factor, true, workspace);

        data[dataOffset + n2] = y[yOffset] * sign(n2 + 1);
        int i = 0;
        int i1 = 1;
        int k = 1;
        for (; k < n2; i++, i1++, k += 2) {
            data[dataOffset + i] = y[yOffset + 2 * k - 1] * sign(i1) + y[yOffset + 2 * k] * sign(i);
            data[dataOffset + length - i1] = y[yOffset + 2 * k - 1] * sign(length - i) - y[yOffset + 2 * k] * sign(length - i1);
            data[dataOffset + n2 - i1] = y[yOffset + 2 * k + 1] * sign(n2 - i) - y[yOffset + 2 * k + 2] * sign(n2 - i1);
            data[dataOffset + n2 + i1] = y[yOffset + 2 * k + 1] * sign(n2 + i + 2) + y[yOffset + 2 * k + 2] * sign(n2 + i1);
        }
        if (k == n2) {
            data[dataOffset + i] = y[yOffset + 2 * k - 1] * sign(i + 1) + y[yOffset + 2 * k] * sign(i);
            data[dataOffset + length - i1] = y[yOffset + 2 * k - 1] * sign(i + 2) + y[yOffset + 2 * k] * sign(i1);
        }
    }

    private void execDcst4Even(double[] data, int dataOffset, double factor, int n2, FftWorkspace workspace) {
        double[] y = workspace.doubles();
        int yOffset = workspace.alloc(2 * n2);
        for (int index = 0; index < n2; index++) {
            double real = data[dataOffset + 2 * index];
            double imag = data[dataOffset + length - 1 - 2 * index];
            double c2Real = c2[2 * index];
            double c2Imag = c2[2 * index + 1];
            y[yOffset + 2 * index] = real * c2Real - imag * c2Imag;
            y[yOffset + 2 * index + 1] = real * c2Imag + imag * c2Real;
        }
        complexPlan.exec(y, yOffset, factor, true, workspace);
        for (int index = 0, mirror = n2 - 1; index < n2; index++, mirror--) {
            double c2Real = c2[2 * index];
            double c2Imag = c2[2 * index + 1];
            data[dataOffset + 2 * index] = 2.0 * (y[yOffset + 2 * index] * c2Real - y[yOffset + 2 * index + 1] * c2Imag);

            double c2MirrorReal = c2[2 * mirror];
            double c2MirrorImag = c2[2 * mirror + 1];
            data[dataOffset + 2 * index + 1] = -2.0 * (y[yOffset + 2 * mirror + 1] * c2MirrorReal + y[yOffset + 2 * mirror] * c2MirrorImag);
        }
    }

    private static void mpInPlace(double[] data, int a, int b) {
        double oldA = data[a];
        data[a] -= data[b];
        data[b] = oldA + data[b];
    }

    private static void reverse(double[] data, int offset, int length) {
        for (int left = 0, right = length - 1; left < right; left++, right--) {
            swap(data, offset + left, offset + right);
        }
    }

    private static void reverseLowerHalf(double[] data, int offset, int length, int upperExclusive) {
        for (int left = 0, right = length - 1; left < upperExclusive; left++, right--) {
            swap(data, offset + left, offset + right);
        }
    }

    private static void swap(double[] data, int first, int second) {
        double temporary = data[first];
        data[first] = data[second];
        data[second] = temporary;
    }

    private static double sign(int index) {
        return (index & 2) != 0 ? -SQRT2 : SQRT2;
    }
}