package com.curioloop.yum4j.fft;

import java.util.Arrays;

final class FftUtil {
    private static final long MAX_SIZE = Long.MAX_VALUE;
    private static final double LARGE_FACTOR_PENALTY = 1.1;

    private FftUtil() {
    }

    static long largestPrimeFactor(long n) {
        if (n < 1) {
            throw new IllegalArgumentException("n must be >= 1");
        }
        long result = 1;
        while ((n & 1L) == 0L) {
            result = 2;
            n >>= 1;
        }
        for (long divisor = 3; divisor <= n / divisor; divisor += 2) {
            while (n % divisor == 0) {
                result = divisor;
                n /= divisor;
            }
        }
        if (n > 1) {
            result = n;
        }
        return result;
    }

    static double costGuess(long n) {
        if (n < 1) {
            throw new IllegalArgumentException("n must be >= 1");
        }
        long ni = n;
        double result = 0.0;
        while ((n & 1L) == 0L) {
            result += 2.0;
            n >>= 1;
        }
        for (long divisor = 3; divisor <= n / divisor; divisor += 2) {
            while (n % divisor == 0) {
                result += divisor <= 5 ? (double) divisor : LARGE_FACTOR_PENALTY * (double) divisor;
                n /= divisor;
            }
        }
        if (n > 1) {
            result += n <= 5 ? (double) n : LARGE_FACTOR_PENALTY * (double) n;
        }
        return result * (double) ni;
    }

    static long goodSizeComplex(long n) {
        if (n < 1) {
            throw new IllegalArgumentException("n must be >= 1");
        }
        if (n <= 12) {
            return n;
        }
        if (n > MAX_SIZE / 11L / 2L) {
            throw new IllegalArgumentException("FFT size is too large");
        }
        long bestFactor = Math.multiplyExact(2L, n);
        for (long factor11 = 1; factor11 < bestFactor; factor11 = multiplyOrStop(factor11, 11, bestFactor)) {
            for (long factor117 = factor11; factor117 < bestFactor; factor117 = multiplyOrStop(factor117, 7, bestFactor)) {
                for (long factor1175 = factor117; factor1175 < bestFactor; factor1175 = multiplyOrStop(factor1175, 5, bestFactor)) {
                    long value = factor1175;
                    while (value < n) {
                        value = Math.multiplyExact(value, 2L);
                    }
                    for (;;) {
                        if (value < n) {
                            value = Math.multiplyExact(value, 3L);
                        } else if (value > n) {
                            if (value < bestFactor) {
                                bestFactor = value;
                            }
                            if ((value & 1L) != 0L) {
                                break;
                            }
                            value >>= 1;
                        } else {
                            return n;
                        }
                    }
                }
            }
        }
        return bestFactor;
    }

    static long goodSizeComplex(long n, long requiredFactor) {
        if (requiredFactor < 1) {
            throw new IllegalArgumentException("required factor must not be 0");
        }
        long scaled = Math.addExact(n, requiredFactor - 1) / requiredFactor;
        return Math.multiplyExact(goodSizeComplex(scaled), requiredFactor);
    }

    static long goodSizeReal(long n) {
        if (n < 1) {
            throw new IllegalArgumentException("n must be >= 1");
        }
        if (n <= 6) {
            return n;
        }
        if (n > MAX_SIZE / 5L / 2L) {
            throw new IllegalArgumentException("FFT size is too large");
        }
        long bestFactor = Math.multiplyExact(2L, n);
        for (long factor5 = 1; factor5 < bestFactor; factor5 = multiplyOrStop(factor5, 5, bestFactor)) {
            long value = factor5;
            while (value < n) {
                value = Math.multiplyExact(value, 2L);
            }
            for (;;) {
                if (value < n) {
                    value = Math.multiplyExact(value, 3L);
                } else if (value > n) {
                    if (value < bestFactor) {
                        bestFactor = value;
                    }
                    if ((value & 1L) != 0L) {
                        break;
                    }
                    value >>= 1;
                } else {
                    return n;
                }
            }
        }
        return bestFactor;
    }

    static long goodSizeReal(long n, long requiredFactor) {
        if (requiredFactor < 1) {
            throw new IllegalArgumentException("required factor must not be 0");
        }
        long scaled = Math.addExact(n, requiredFactor - 1) / requiredFactor;
        return Math.multiplyExact(goodSizeReal(scaled), requiredFactor);
    }

    static long prevGoodSizeComplex(long n) {
        if (n < 1) {
            throw new IllegalArgumentException("n must be >= 1");
        }
        if (n <= 12) {
            return n;
        }
        if (n > MAX_SIZE / 11L) {
            throw new IllegalArgumentException("FFT size is too large");
        }
        long bestFound = 1;
        for (long factor11 = 1; factor11 <= n; factor11 = multiplyOrStopInclusive(factor11, 11, n)) {
            for (long factor117 = factor11; factor117 <= n; factor117 = multiplyOrStopInclusive(factor117, 7, n)) {
                for (long factor1175 = factor117; factor1175 <= n; factor1175 = multiplyOrStopInclusive(factor1175, 5, n)) {
                    long value = factor1175;
                    while (value <= n / 2L) {
                        value *= 2L;
                    }
                    if (value > bestFound) {
                        bestFound = value;
                    }
                    for (;;) {
                        if (value <= n / 3L) {
                            value *= 3L;
                        } else if (value % 2L == 0L) {
                            value /= 2L;
                        } else {
                            break;
                        }
                        if (value > bestFound) {
                            bestFound = value;
                        }
                    }
                }
            }
        }
        return bestFound;
    }

    static long prevGoodSizeReal(long n) {
        if (n < 1) {
            throw new IllegalArgumentException("n must be >= 1");
        }
        if (n <= 6) {
            return n;
        }
        if (n > MAX_SIZE / 5L) {
            throw new IllegalArgumentException("FFT size is too large");
        }
        long bestFound = 1;
        for (long factor5 = 1; factor5 <= n; factor5 = multiplyOrStopInclusive(factor5, 5, n)) {
            long value = factor5;
            while (value <= n / 2L) {
                value *= 2L;
            }
            if (value > bestFound) {
                bestFound = value;
            }
            for (;;) {
                if (value <= n / 3L) {
                    value *= 3L;
                } else if (value % 2L == 0L) {
                    value /= 2L;
                } else {
                    break;
                }
                if (value > bestFound) {
                    bestFound = value;
                }
            }
        }
        return bestFound;
    }

    static long product(int[] shape) {
        long product = 1;
        for (int extent : shape) {
            if (extent < 1) {
                throw new IllegalArgumentException("shape entries must be >= 1");
            }
            product = Math.multiplyExact(product, extent);
        }
        return product;
    }

    static void sanityCheck(int[] shape, long[] strideIn, long[] strideOut, boolean inPlace) {
        if (shape.length < 1) {
            throw new IllegalArgumentException("ndim must be >= 1");
        }
        if (strideIn.length != shape.length || strideOut.length != shape.length) {
            throw new IllegalArgumentException("stride dimension mismatch");
        }
        product(shape);
        if (inPlace && !Arrays.equals(strideIn, strideOut)) {
            throw new IllegalArgumentException("stride mismatch");
        }
    }

    static void sanityCheck(int[] shape, long[] strideIn, long[] strideOut, boolean inPlace, int[] axes) {
        sanityCheck(shape, strideIn, strideOut, inPlace);
        int[] seen = new int[shape.length];
        for (int axis : axes) {
            if (axis < 0 || axis >= shape.length) {
                throw new IllegalArgumentException("bad axis number");
            }
            if (++seen[axis] > 1) {
                throw new IllegalArgumentException("axis specified repeatedly");
            }
        }
    }

    static void sanityCheck(int[] shape, long[] strideIn, long[] strideOut, boolean inPlace, int axis) {
        sanityCheck(shape, strideIn, strideOut, inPlace);
        if (axis < 0 || axis >= shape.length) {
            throw new IllegalArgumentException("bad axis number");
        }
    }

    private static long multiplyOrStop(long value, long factor, long upperExclusive) {
        if (value > Long.MAX_VALUE / factor) {
            return upperExclusive;
        }
        long next = value * factor;
        return next <= value ? upperExclusive : next;
    }

    private static long multiplyOrStopInclusive(long value, long factor, long upperInclusive) {
        if (value > Long.MAX_VALUE / factor) {
            return upperInclusive + 1;
        }
        long next = value * factor;
        return next <= value ? upperInclusive + 1 : next;
    }
}