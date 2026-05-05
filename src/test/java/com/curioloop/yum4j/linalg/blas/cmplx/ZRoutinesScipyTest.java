package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.Dlamch;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ZRoutinesScipyTest {
    private static final double TOL = 1e-10;
    private static final int N = 4;

    static double[] makePosDef(int n) {
        double[] A = new double[n * n * 2];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                double re1 = Math.cos((i+1)*(k+1)*0.7), im1 = Math.sin((i+1)*(k+1)*0.3);
                double re2 = Math.cos((j+1)*(k+1)*0.7), im2 = -Math.sin((j+1)*(k+1)*0.3);
                sum += re1*re2 - im1*im2;
            }
            A[(i*n+j)*2] = sum + (i==j ? n : 0);
            A[(i*n+j)*2+1] = 0;
        }
        return A;
    }

    static double[] makeRandom(int rows, int cols, long seed) {
        java.util.Random r = new java.util.Random(seed);
        double[] A = new double[rows*cols*2];
        for (int i = 0; i < rows*cols; i++) {
            A[i*2] = r.nextGaussian();
            A[i*2+1] = r.nextGaussian();
        }
        return A;
    }

    static void copyMatrixToOffset(double[] source, int rows, int cols,
                                   double[] target, int targetOff, int targetLd) {
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) {
            int src = (i * cols + j) * 2;
            int dst = (targetOff + i * targetLd + j) * 2;
            target[dst] = source[src];
            target[dst + 1] = source[src + 1];
        }
    }

    static void assertTriangularSubmatrixEquals(BLAS.Uplo uplo, int n,
                                                double[] expected, int expectedOff, int expectedLd,
                                                double[] actual, int actualOff, int actualLd,
                                                double tol) {
        boolean upper = uplo == BLAS.Uplo.Upper;
        for (int i = 0; i < n; i++) {
            int start = upper ? i : 0;
            int end = upper ? n : i + 1;
            for (int j = start; j < end; j++) {
                int ep = (expectedOff + i * expectedLd + j) * 2;
                int ap = (actualOff + i * actualLd + j) * 2;
                assertEquals(expected[ep], actual[ap], tol);
                assertEquals(expected[ep + 1], actual[ap + 1], tol);
            }
        }
    }

    @Test void testZherk() {
        double[] A = makeRandom(N, 3, 42);
        double[] C = new double[N*N*2];
        Zherk.zherk(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, N, 3, 1.0, A, 0, 3, 0.0, C, 0, N);
        for (int i = 0; i < N; i++) for (int j = i; j < N; j++) {
            double sumRe = 0, sumIm = 0;
            for (int k = 0; k < 3; k++) {
                double aRe = A[(i*3+k)*2], aIm = A[(i*3+k)*2+1];
                double bRe = A[(j*3+k)*2], bIm = A[(j*3+k)*2+1];
                sumRe += aRe*bRe + aIm*bIm;
                sumIm += aIm*bRe - aRe*bIm;
            }
            assertEquals(sumRe, C[(i*N+j)*2], TOL);
            if (i == j) assertEquals(0, C[(i*N+j)*2+1], TOL);
            else assertEquals(sumIm, C[(i*N+j)*2+1], TOL);
        }
    }

    @Test void testZtrmm() {
        double[] A = makeRandom(N, N, 43);
        for (int i = 0; i < N; i++) for (int j = 0; j < i; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        double[] B = makeRandom(N, N, 44), B2 = B.clone();
        Ztrmm.ztrmm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.NoTrans,
                BLAS.Diag.NonUnit, N, N, 1.0, 0.0, A, 0, N, B2, 0, N);
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            double sr = 0, si = 0;
            for (int k = i; k < N; k++) {
                double ar = A[(i*N+k)*2], ai = A[(i*N+k)*2+1];
                double br = B[(k*N+j)*2], bi = B[(k*N+j)*2+1];
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            assertEquals(sr, B2[(i*N+j)*2], TOL);
            assertEquals(si, B2[(i*N+j)*2+1], TOL);
        }
    }

    @Test void testZtrtri() {
        double[] A = makePosDef(N);
        Zpotr.zpotrf(BLAS.Uplo.Upper, N, A, 0, N);
        double[] Ai = A.clone();
        assertEquals(0, Ztrtri.ztrtri(BLAS.Uplo.Upper, BLAS.Diag.NonUnit, N, Ai, 0, N));
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            double sr = 0, si = 0;
            for (int k = 0; k < N; k++) {
                double ar = (k >= i) ? A[(i*N+k)*2] : 0;
                double ai = (k >= i) ? A[(i*N+k)*2+1] : 0;
                double br = (k <= j) ? Ai[(k*N+j)*2] : 0;
                double bi = (k <= j) ? Ai[(k*N+j)*2+1] : 0;
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            if (i == j) { assertEquals(1, sr, TOL); assertEquals(0, si, TOL); }
            else { assertEquals(0, sr, TOL); assertEquals(0, si, TOL); }
        }
    }

    @Test void testZtrtri4x4() {
        int n = 4;
        double[] A = makePosDef(n);
        int info = Zpotr.zpotrf(BLAS.Uplo.Upper, n, A, 0, n);
        assertEquals(0, info);
        double[] U = A.clone();
        info = Ztrtri.ztrtri(BLAS.Uplo.Upper, BLAS.Diag.NonUnit, n, A, 0, n);
        assertEquals(0, info);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            double sr = 0, si = 0;
            for (int k = 0; k < n; k++) {
                double ur = U[(i*n+k)*2], ui = (k >= i) ? U[(i*n+k)*2+1] : 0;
                if (k < i) { ur = 0; ui = 0; }
                double vr = A[(k*n+j)*2], vi = A[(k*n+j)*2+1];
                if (k > j) { vr = 0; vi = 0; }
                sr += ur * vr - ui * vi;
                si += ur * vi + ui * vr;
            }
            if (i == j) { assertEquals(1, sr, TOL); assertEquals(0, si, TOL); }
            else { assertEquals(0, sr, TOL); assertEquals(0, si, TOL); }
        }
    }

    @Test void testZlauum() {
        double[] L = makePosDef(N);
        Zpotr.zpotrf(BLAS.Uplo.Lower, N, L, 0, N);
        double[] Lc = L.clone();
        Zlauum.zlauum(BLAS.Uplo.Lower, N, L, 0, N);
        for (int i = 0; i < N; i++) for (int j = 0; j <= i; j++) {
            double sr = 0;
            for (int k = i; k < N; k++) {
                double lr = Lc[(k*N+i)*2], li = Lc[(k*N+i)*2+1];
                double rr = Lc[(k*N+j)*2], ri = -Lc[(k*N+j)*2+1];
                sr += lr*rr - li*ri;
            }
            assertEquals(sr, L[(i*N+j)*2], TOL);
            assertEquals(0, L[(i*N+j)*2+1], TOL);
        }
    }

    @Test void testZpotrfBlocked() {
        double[] A = makePosDef(N), Ac = A.clone();
        assertEquals(0, Zpotr.zpotrf(BLAS.Uplo.Upper, N, A, 0, N));
        for (int i = 0; i < N; i++) for (int j = i; j < N; j++) {
            double sr = 0, si = 0;
            for (int k = 0; k <= i; k++) {
                double ar = A[(k*N+i)*2], ai = -A[(k*N+i)*2+1];
                double br = A[(k*N+j)*2], bi = A[(k*N+j)*2+1];
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            assertEquals(Ac[(i*N+j)*2], sr, TOL);
            assertEquals(Ac[(i*N+j)*2+1], si, TOL);
        }
    }

    @Test void testZpotrs() {
        double[] A = makePosDef(N);
        double[] B = new double[N*N*2];
        java.util.Random r = new java.util.Random(45);
        for (int i = 0; i < N; i++) { B[(i*N)*2] = r.nextGaussian(); B[(i*N)*2+1] = r.nextGaussian(); }
        double[] Bc = B.clone();
        Zpotr.zpotrf(BLAS.Uplo.Upper, N, A, 0, N);
        Zpotr.zpotrs(BLAS.Uplo.Upper, N, 1, A, 0, N, B, 0, N);
        double[] Ao = makePosDef(N);
        for (int i = 0; i < N; i++) {
            double sr = 0, si = 0;
            for (int j = 0; j < N; j++) {
                double ar = Ao[(i*N+j)*2], ai = Ao[(i*N+j)*2+1];
                double br = B[j*N*2], bi = B[j*N*2+1];
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            assertEquals(Bc[i*N*2], sr, TOL);
            assertEquals(Bc[i*N*2+1], si, TOL);
        }
    }

    @Test void testZpotri() {
        double[] A = makePosDef(N);
        Zpotr.zpotrf(BLAS.Uplo.Upper, N, A, 0, N);
        assertTrue(Zpotr.zpotri(BLAS.Uplo.Upper, N, A, 0, N));
        double[] Ao = makePosDef(N);
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            double sr = 0, si = 0;
            for (int k = 0; k < N; k++) {
                double ar = (k >= i) ? Ao[(i*N+k)*2] : Ao[(k*N+i)*2];
                double ai = (k >= i) ? Ao[(i*N+k)*2+1] : -Ao[(k*N+i)*2+1];
                double br = (k <= j) ? A[(k*N+j)*2] : A[(j*N+k)*2];
                double bi = (k <= j) ? A[(k*N+j)*2+1] : -A[(j*N+k)*2+1];
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            if (i == j) { assertEquals(1, sr, TOL); assertEquals(0, si, TOL); }
            else { assertEquals(0, sr, TOL); assertEquals(0, si, TOL); }
        }
    }

    @Test void testZgetf2() {
        double[] A = makeRandom(N, N, 46);
        int[] ipiv = new int[N];
        assertTrue(Zgetr.zgetf2(N, N, A, 0, N, ipiv, 0) >= 0);
    }

    @Test void testZgetri() {
        double[] A = makeRandom(N, N, 47), Ao = A.clone();
        int[] ipiv = new int[N];
        Zgetr.zgetrf(N, N, A, 0, N, ipiv);
        assertTrue(Zgetr.zgetri(N, A, 0, N, ipiv, new double[N*N*4], 0, N*N*4));
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            double sr = 0, si = 0;
            for (int k = 0; k < N; k++) {
                double ar = Ao[(i*N+k)*2], ai = Ao[(i*N+k)*2+1];
                double br = A[(k*N+j)*2], bi = A[(k*N+j)*2+1];
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            if (i == j) { assertEquals(1, sr, TOL); assertEquals(0, si, TOL); }
            else { assertEquals(0, sr, TOL); assertEquals(0, si, TOL); }
        }
    }

    @Test void testZgecon() {
        double[] A = makeRandom(N, N, 48);
        double anorm = 0;
        for (int j = 0; j < N; j++) {
            double cs = 0;
            for (int i = 0; i < N; i++) {
                double re = A[(i*N+j)*2], im = A[(i*N+j)*2+1];
                cs += Math.sqrt(re*re + im*im);
            }
            anorm = Math.max(anorm, cs);
        }
        int[] ipiv = new int[N];
        Zgetr.zgetrf(N, N, A, 0, N, ipiv);
        double rc = Zgetr.zgecon('1', N, A, 0, N, anorm, new double[N*8], new int[N*2]);
        assertTrue(rc > 0 && rc <= 1.0);
    }

    @Test void testZpocon() {
        double[] A = makePosDef(N);
        double anorm = 0;
        for (int j = 0; j < N; j++) {
            double cs = 0;
            for (int i = 0; i < N; i++) {
                double re = A[(i*N+j)*2], im = A[(i*N+j)*2+1];
                cs += Math.sqrt(re*re + im*im);
            }
            anorm = Math.max(anorm, cs);
        }
        Zpotr.zpotrf(BLAS.Uplo.Upper, N, A, 0, N);
        double rc = Zpotr.zpocon(BLAS.Uplo.Upper, N, A, 0, N, anorm, new double[N*8], new int[N*2]);
        assertTrue(rc > 0 && rc <= 1.0);
    }

    @Test void testZlatrs() {
        double[] A = makeRandom(N, N, 49);
        for (int i = 0; i < N; i++) for (int j = 0; j < i; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        double[] x = makeRandom(N, 1, 50), xc = x.clone();
        Zlatrs.zlatrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                false, N, A, 0, N, x, 0, new double[N], 0);
        Ztrsv.ztrsv(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, N, A, 0, N, xc, 0, 1);
        for (int i = 0; i < N; i++) {
            assertEquals(xc[i*2], x[i*2], TOL);
            assertEquals(xc[i*2+1], x[i*2+1], TOL);
        }
    }

    @Test void testZdrscl() {
        double[] x = makeRandom(5, 1, 51), xc = x.clone();
        double a = 3.7;
        Zdrscl.zdrscl(5, a, x, 0, 1);
        for (int i = 0; i < 5; i++) {
            assertEquals(xc[i*2]/a, x[i*2], TOL);
            assertEquals(xc[i*2+1]/a, x[i*2+1], TOL);
        }
    }

    @Test void testZlacn2() {
        double[] v = new double[16], x = new double[8];
        int[] isgn = new int[16];
        for (int i = 0; i < 4; i++) { x[i*2] = 1; x[i*2+1] = 0; }
        double est = 0;
        int[] kase = {0}, isave = new int[3];
        while (true) {
            est = Zlacn2.zlacn2(4, v, 0, x, 0, isgn, 0, est, kase, 0, isave, 0);
            if (kase[0] == 0) break;
        }
        assertTrue(est > 0);
    }

    @Test void testZtrcon() {
        double[] A = makePosDef(N);
        Zpotr.zpotrf(BLAS.Uplo.Upper, N, A, 0, N);
        double rc = Ztrtri.ztrcon('1', BLAS.Uplo.Upper, BLAS.Diag.NonUnit, N, A, 0, N,
                new double[N * 4], new int[N * 2]);
        assertTrue(rc > 0 && rc <= 1.0);
    }

    @Test void testZpotf2() {
        double[] A = makePosDef(N), Ac = A.clone();
        assertTrue(Zpotr.zpotf2(BLAS.Uplo.Upper, N, A, 0, N));
        double[] A2 = Ac.clone();
        assertEquals(0, Zpotr.zpotrf(BLAS.Uplo.Upper, N, A2, 0, N));
        for (int i = 0; i < N * N * 2; i++) {
            assertEquals(A2[i], A[i], TOL);
        }
    }

    @Test void testZgetrs() {
        double[] A = makeRandom(N, N, 52);
        double[] B = new double[N * 2];
        java.util.Random r = new java.util.Random(53);
        for (int i = 0; i < N; i++) { B[i * 2] = r.nextGaussian(); B[i * 2 + 1] = r.nextGaussian(); }
        double[] Ao = A.clone(), Bc = B.clone();
        int[] ipiv = new int[N];
        Zgetr.zgetrf(N, N, A, 0, N, ipiv);
        Zgetr.zgetrs(BLAS.Trans.NoTrans, N, 1, A, 0, N, ipiv, B, 0, 1);
        for (int i = 0; i < N; i++) {
            double sr = 0, si = 0;
            for (int j = 0; j < N; j++) {
                double ar = Ao[(i * N + j) * 2], ai = Ao[(i * N + j) * 2 + 1];
                double br = B[j * 2], bi = B[j * 2 + 1];
                sr += ar * br - ai * bi;
                si += ar * bi + ai * br;
            }
            assertEquals(Bc[i * 2], sr, TOL);
            assertEquals(Bc[i * 2 + 1], si, TOL);
        }
    }

    @Test void testZlarftRowWise() {
        int m = 4, n = 5, k = Math.min(m, n);
        double[] A = makeRandom(m, n, 54);
        double[] tau = new double[k * 2];
        double[] work = new double[n * 2];
        Zgelq.zgelq2(m, n, A, 0, n, tau, 0, work, 0);
        double[] T = new double[k * k * 2];
        Zgelq.zlarftRowWise(m, k, A, 0, n, tau, 0, T, 0, k);
        for (int i = 0; i < k; i++) {
            assertEquals(tau[i * 2], T[(i * k + i) * 2], TOL);
            assertEquals(tau[i * 2 + 1], T[(i * k + i) * 2 + 1], TOL);
        }
        for (int i = 0; i < k; i++) for (int j = 0; j < i; j++) {
            assertEquals(0, T[(i * k + j) * 2], TOL);
            assertEquals(0, T[(i * k + j) * 2 + 1], TOL);
        }
    }

    @Test void testZlarfbRowWiseLeft() {
        int m = 5, n = 4, k = Math.min(m, n);
        double[] A = makeRandom(m, n, 55);
        double[] tau = new double[k * 2];
        double[] work = new double[n * 2];
        Zgelq.zgelq2(m, n, A, 0, n, tau, 0, work, 0);
        double[] T = new double[k * k * 2];
        Zgelq.zlarftRowWise(m, k, A, 0, n, tau, 0, T, 0, k);
        double[] C = makeRandom(m, n, 56);
        double[] Cc = C.clone();
        double[] fbWork = new double[k * n * 2];
        Zgelq.zlarfbRowWiseLeft(BLAS.Side.Left, BLAS.Trans.NoTrans, m, n, k,
                A, 0, n, T, 0, k, C, 0, n, fbWork, 0, n);
        boolean changed = false;
        for (int i = 0; i < m * n * 2; i++) {
            if (Math.abs(C[i] - Cc[i]) > TOL) { changed = true; break; }
        }
        assertTrue(changed);
    }

    @Test void testZlarfbRowWiseRight() {
        int m = 4, n = 5, k = Math.min(m, n);
        double[] A = makeRandom(m, n, 57);
        double[] tau = new double[k * 2];
        double[] work = new double[n * 2];
        Zgelq.zgelq2(m, n, A, 0, n, tau, 0, work, 0);
        double[] T = new double[k * k * 2];
        Zgelq.zlarftRowWise(m, k, A, 0, n, tau, 0, T, 0, k);
        double[] C = makeRandom(m, n, 58);
        double[] Cc = C.clone();
        double[] fbWork = new double[k * m * 2];
        Zgelq.zlarfbRowWiseRight(BLAS.Side.Right, BLAS.Trans.NoTrans, m, n, k,
                A, 0, n, T, 0, k, C, 0, n, fbWork, 0, m);
        boolean changed = false;
        for (int i = 0; i < m * n * 2; i++) {
            if (Math.abs(C[i] - Cc[i]) > TOL) { changed = true; break; }
        }
        assertTrue(changed);
    }

    @Test void testZlarfbColWiseForwardLeft() {
        int m = 5, n = 4, k = 2;
        double[] V = makeRandom(m, k, 60);
        for (int i = 0; i < k; i++) for (int j = i + 1; j < k; j++) { V[(i * k + j) * 2] = 0; V[(i * k + j) * 2 + 1] = 0; }
        for (int i = 0; i < k; i++) { V[(i * k + i) * 2] = 1; V[(i * k + i) * 2 + 1] = 0; }
        double[] T = makeRandom(k, k, 61);
        for (int i = 0; i < k; i++) for (int j = 0; j < i; j++) { T[(i * k + j) * 2] = 0; T[(i * k + j) * 2 + 1] = 0; }
        double[] C = makeRandom(m, n, 62);
        double[] Cc = C.clone();
        double[] work = new double[n * n * 2];
        Zlarfb.zlarfb('L', 'N', 'F', 'C', m, n, k, V, 0, k, T, 0, k, C, 0, n, work, 0, n);
        double[] VH = new double[k * m * 2];
        for (int i = 0; i < k; i++) for (int j = 0; j < m; j++) {
            double sr = 0, si = 0;
            for (int p = 0; p < k; p++) {
                double tr = T[(i*k+p)*2], ti = T[(i*k+p)*2+1];
                double vr = V[(j*k+p)*2], vi = V[(j*k+p)*2+1];
                sr += tr*vr - ti*(-vi); si += tr*(-vi) + ti*vr;
            }
            VH[(i*m+j)*2] = sr; VH[(i*m+j)*2+1] = si;
        }
        double[] VTVH = new double[m * m * 2];
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) {
            double sr = 0, si = 0;
            for (int p = 0; p < k; p++) {
                double vr = V[(i*k+p)*2], vi = V[(i*k+p)*2+1];
                double vhr = VH[(p*m+j)*2], vhi = VH[(p*m+j)*2+1];
                sr += vr*vhr - vi*vhi; si += vr*vhi + vi*vhr;
            }
            VTVH[(i*m+j)*2] = sr; VTVH[(i*m+j)*2+1] = si;
        }
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) {
            double sr = 0, si = 0;
            for (int p = 0; p < m; p++) {
                double hr = (p == i) ? 1 - VTVH[(i*m+p)*2] : -VTVH[(i*m+p)*2];
                double hi = -VTVH[(i*m+p)*2+1];
                sr += hr * Cc[(p*n+j)*2] - hi * Cc[(p*n+j)*2+1];
                si += hr * Cc[(p*n+j)*2+1] + hi * Cc[(p*n+j)*2];
            }
            assertEquals(sr, C[(i*n+j)*2], TOL);
            assertEquals(si, C[(i*n+j)*2+1], TOL);
        }
    }

    @Test void testZlarfbRowWiseForwardRight() {
        int m = 4, n = 5, k = 2;
        double[] V = makeRandom(k, n, 63);
        for (int i = 0; i < k; i++) for (int j = i + 1; j < n; j++) { V[(i * n + j) * 2] = 0; V[(i * n + j) * 2 + 1] = 0; }
        for (int i = 0; i < k; i++) { V[(i * n + i) * 2] = 1; V[(i * n + i) * 2 + 1] = 0; }
        double[] T = makeRandom(k, k, 64);
        for (int i = 0; i < k; i++) for (int j = 0; j < i; j++) { T[(i * k + j) * 2] = 0; T[(i * k + j) * 2 + 1] = 0; }
        double[] C = makeRandom(m, n, 65);
        double[] Cc = C.clone();
        double[] work = new double[m * m * 2];
        Zlarfb.zlarfb('R', 'N', 'F', 'R', m, n, k, V, 0, n, T, 0, k, C, 0, n, work, 0, m);
        boolean changed = false;
        for (int i = 0; i < m * n * 2; i++) {
            if (Math.abs(C[i] - Cc[i]) > TOL) { changed = true; break; }
        }
        assertTrue(changed);
    }

    @Test void testZherkLowerConj() {
        double[] A = makeRandom(3, N, 42);
        double[] C = new double[N*N*2];
        Zherk.zherk(BLAS.Uplo.Lower, BLAS.Trans.Conj, N, 3, 1.0, A, 0, 3, 0.0, C, 0, N);
        double[] ref = new double[N*N*2];
        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, N, N, 3, 1.0, 0.0, A, 0, 3, A, 0, 3, 0.0, 0.0, ref, 0, N);
        for (int i = 0; i < N; i++) for (int j = 0; j <= i; j++) {
            assertEquals(ref[(i*N+j)*2], C[(i*N+j)*2], TOL);
            assertEquals(ref[(i*N+j)*2+1], C[(i*N+j)*2+1], TOL);
        }
    }

    @Test void testZtrmmRightLowerTransNonUnit() {
        double[] A = makeRandom(N, N, 43);
        for (int i = 0; i < N; i++) for (int j = i+1; j < N; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        double[] B = makeRandom(N, N, 44), B2 = B.clone();
        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans,
                BLAS.Diag.NonUnit, N, N, 1.0, 0.0, A, 0, N, B2, 0, N);
        double[] ref = new double[N*N*2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, N, N, N, 1.0, 0.0, B, 0, N, A, 0, N, 0.0, 0.0, ref, 0, N);
        for (int i = 0; i < N*N*2; i++) assertEquals(ref[i], B2[i], TOL);
    }

    @Test void testZtrmmRightLowerConjNonUnit() {
        double[] A = makeRandom(N, N, 43);
        for (int i = 0; i < N; i++) for (int j = i+1; j < N; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        double[] B = makeRandom(N, N, 44), B2 = B.clone();
        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj,
                BLAS.Diag.NonUnit, N, N, 1.0, 0.0, A, 0, N, B2, 0, N);
        double[] ref = new double[N*N*2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, N, N, N, 1.0, 0.0, B, 0, N, A, 0, N, 0.0, 0.0, ref, 0, N);
        for (int i = 0; i < N*N*2; i++) assertEquals(ref[i], B2[i], TOL);
    }

    @Test void testZtrmmLeftLowerTransUnit() {
        double[] A = makeRandom(N, N, 43);
        for (int i = 0; i < N; i++) for (int j = i+1; j < N; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        for (int i = 0; i < N; i++) { A[(i*N+i)*2] = 1; A[(i*N+i)*2+1] = 0; }
        double[] B = makeRandom(N, N, 44), B2 = B.clone();
        Ztrmm.ztrmm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Trans,
                BLAS.Diag.Unit, N, N, 1.0, 0.0, A, 0, N, B2, 0, N);
        double[] ref = new double[N*N*2];
        ZLAS.zgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, N, N, N, 1.0, 0.0, A, 0, N, B, 0, N, 0.0, 0.0, ref, 0, N);
        for (int i = 0; i < N*N*2; i++) assertEquals(ref[i], B2[i], TOL);
    }

    @Test void testZtrmmRightUpperTransUnit() {
        double[] A = makeRandom(N, N, 43);
        for (int i = 0; i < N; i++) for (int j = 0; j < i; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        for (int i = 0; i < N; i++) { A[(i*N+i)*2] = 1; A[(i*N+i)*2+1] = 0; }
        double[] B = makeRandom(N, N, 44), B2 = B.clone();
        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans,
                BLAS.Diag.Unit, N, N, 1.0, 0.0, A, 0, N, B2, 0, N);
        double[] ref = new double[N*N*2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, N, N, N, 1.0, 0.0, B, 0, N, A, 0, N, 0.0, 0.0, ref, 0, N);
        for (int i = 0; i < N*N*2; i++) assertEquals(ref[i], B2[i], TOL);
    }

    @Test void testZtrtriLowerNonUnit() {
        double[] A = makePosDef(N);
        Zpotr.zpotrf(BLAS.Uplo.Lower, N, A, 0, N);
        double[] Ai = A.clone();
        assertEquals(0, Ztrtri.ztrtri(BLAS.Uplo.Lower, BLAS.Diag.NonUnit, N, Ai, 0, N));
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            double sr = 0, si = 0;
            for (int k = 0; k < N; k++) {
                double ar = (k <= i) ? A[(i*N+k)*2] : 0;
                double ai = (k <= i) ? A[(i*N+k)*2+1] : 0;
                double br = (k >= j) ? Ai[(k*N+j)*2] : 0;
                double bi = (k >= j) ? Ai[(k*N+j)*2+1] : 0;
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            if (i == j) { assertEquals(1, sr, TOL); assertEquals(0, si, TOL); }
            else { assertEquals(0, sr, TOL); assertEquals(0, si, TOL); }
        }
    }

    @Test void testZtrtriUpperUnit() {
        double[] A = makeRandom(N, N, 43);
        for (int i = 0; i < N; i++) for (int j = 0; j < i; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        for (int i = 0; i < N; i++) { A[(i*N+i)*2] = 1; A[(i*N+i)*2+1] = 0; }
        double[] U = A.clone();
        assertEquals(0, Ztrtri.ztrtri(BLAS.Uplo.Upper, BLAS.Diag.Unit, N, A, 0, N));
        for (int i = 0; i < N; i++) for (int j = i; j < N; j++) {
            double sr = 0, si = 0;
            for (int k = i; k <= j; k++) {
                double ur = (k == i) ? 1 : U[(i*N+k)*2];
                double ui = (k == i) ? 0 : U[(i*N+k)*2+1];
                double vr = (k == j) ? 1 : A[(k*N+j)*2];
                double vi = (k == j) ? 0 : A[(k*N+j)*2+1];
                sr += ur*vr - ui*vi;
                si += ur*vi + ui*vr;
            }
            if (i == j) { assertEquals(1, sr, TOL); assertEquals(0, si, TOL); }
            else { assertEquals(0, sr, TOL); assertEquals(0, si, TOL); }
        }
    }

    @Test void testZlauumUpper() {
        double[] A = makePosDef(N);
        Zpotr.zpotrf(BLAS.Uplo.Upper, N, A, 0, N);
        double[] Uc = A.clone();
        Zlauum.zlauum(BLAS.Uplo.Upper, N, A, 0, N);
        for (int i = 0; i < N; i++) for (int j = i; j < N; j++) {
            double sr = 0, si = 0;
            for (int k = j; k < N; k++) {
                double ur = Uc[(i*N+k)*2], ui = Uc[(i*N+k)*2+1];
                double vr = Uc[(j*N+k)*2], vi = -Uc[(j*N+k)*2+1];
                sr += ur*vr - ui*vi;
                si += ur*vi + ui*vr;
            }
            assertEquals(sr, A[(i*N+j)*2], TOL);
            if (i == j) assertEquals(0, si, TOL);
            else assertEquals(si, A[(i*N+j)*2+1], TOL);
        }
    }

    @Test void testZpotrfLower() {
        double[] A = makePosDef(N), Ac = A.clone();
        assertEquals(0, Zpotr.zpotrf(BLAS.Uplo.Lower, N, A, 0, N));
        for (int i = 0; i < N; i++) for (int j = i+1; j < N; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        double[] ref = new double[N*N*2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, N, N, N, 1.0, 0.0, A, 0, N, A, 0, N, 0.0, 0.0, ref, 0, N);
        for (int i = 0; i < N; i++) for (int j = 0; j <= i; j++) {
            assertEquals(Ac[(i*N+j)*2], ref[(i*N+j)*2], TOL);
            assertEquals(Ac[(i*N+j)*2+1], ref[(i*N+j)*2+1], TOL);
        }
    }

    @Test void testZpotrsLower() {
        double[] A = makePosDef(N);
        double[] B = new double[N*2];
        java.util.Random r = new java.util.Random(45);
        for (int i = 0; i < N; i++) { B[i*2] = r.nextGaussian(); B[i*2+1] = r.nextGaussian(); }
        double[] Bc = B.clone();
        Zpotr.zpotrf(BLAS.Uplo.Lower, N, A, 0, N);
        Zpotr.zpotrs(BLAS.Uplo.Lower, N, 1, A, 0, N, B, 0, 1);
        double[] Ao = makePosDef(N);
        for (int i = 0; i < N; i++) {
            double sr = 0, si = 0;
            for (int j = 0; j < N; j++) {
                double ar = Ao[(i*N+j)*2], ai = Ao[(i*N+j)*2+1];
                double br = B[j*2], bi = B[j*2+1];
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            assertEquals(Bc[i*2], sr, TOL);
            assertEquals(Bc[i*2+1], si, TOL);
        }
    }

    @Test void testZpotriLower() {
        double[] A = makePosDef(N);
        Zpotr.zpotrf(BLAS.Uplo.Lower, N, A, 0, N);
        assertTrue(Zpotr.zpotri(BLAS.Uplo.Lower, N, A, 0, N));
        double[] Ao = makePosDef(N);
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            double sr = 0, si = 0;
            for (int k = 0; k < N; k++) {
                double ar = (k >= i) ? Ao[(i*N+k)*2] : Ao[(k*N+i)*2];
                double ai = (k >= i) ? Ao[(i*N+k)*2+1] : -Ao[(k*N+i)*2+1];
                double br = (k >= j) ? A[(k*N+j)*2] : A[(j*N+k)*2];
                double bi = (k >= j) ? A[(k*N+j)*2+1] : -A[(j*N+k)*2+1];
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            if (i == j) { assertEquals(1, sr, TOL); assertEquals(0, si, TOL); }
            else { assertEquals(0, sr, TOL); assertEquals(0, si, TOL); }
        }
    }

    @Test void testZpoconLower() {
        double[] A = makePosDef(N);
        double anorm = 0;
        for (int j = 0; j < N; j++) {
            double cs = 0;
            for (int i = 0; i < N; i++) {
                double re = A[(i*N+j)*2], im = A[(i*N+j)*2+1];
                cs += Math.sqrt(re*re + im*im);
            }
            anorm = Math.max(anorm, cs);
        }
        Zpotr.zpotrf(BLAS.Uplo.Lower, N, A, 0, N);
        double rc = Zpotr.zpocon(BLAS.Uplo.Lower, N, A, 0, N, anorm, new double[N*8], new int[N*2]);
        assertTrue(rc > 0 && rc <= 1.0);
    }

    @Test void testZgeconInfinityNorm() {
        double[] A = makeRandom(N, N, 48);
        double anorm = 0;
        for (int i = 0; i < N; i++) {
            double rs = 0;
            for (int j = 0; j < N; j++) {
                double re = A[(i*N+j)*2], im = A[(i*N+j)*2+1];
                rs += Math.sqrt(re*re + im*im);
            }
            anorm = Math.max(anorm, rs);
        }
        int[] ipiv = new int[N];
        Zgetr.zgetrf(N, N, A, 0, N, ipiv);
        double rc = Zgetr.zgecon('I', N, A, 0, N, anorm, new double[N*8], new int[N*2]);
        assertTrue(rc > 0 && rc <= 1.0);
    }

    @Test void testZtrconInfNorm() {
        double[] A = makePosDef(N);
        Zpotr.zpotrf(BLAS.Uplo.Upper, N, A, 0, N);
        double rc = Ztrtri.ztrcon('I', BLAS.Uplo.Upper, BLAS.Diag.NonUnit, N, A, 0, N,
                new double[N * 4], new int[N * 2]);
        assertTrue(rc > 0 && rc <= 1.0);
    }

    @Test void testZtrconLower() {
        double[] A = makePosDef(N);
        Zpotr.zpotrf(BLAS.Uplo.Lower, N, A, 0, N);
        double rc = Ztrtri.ztrcon('1', BLAS.Uplo.Lower, BLAS.Diag.NonUnit, N, A, 0, N,
                new double[N * 4], new int[N * 2]);
        assertTrue(rc > 0 && rc <= 1.0);
    }

    @Test void testZtrconUnit() {
        double[] A = makePosDef(N);
        Zpotr.zpotrf(BLAS.Uplo.Upper, N, A, 0, N);
        double rc = Ztrtri.ztrcon('1', BLAS.Uplo.Upper, BLAS.Diag.Unit, N, A, 0, N,
                new double[N * 4], new int[N * 2]);
        assertTrue(rc > 0 && rc <= 1.0);
    }

    @Test void testZlatrsLowerNoTransNonUnit() {
        double[] A = makeRandom(N, N, 49);
        for (int i = 0; i < N; i++) for (int j = i+1; j < N; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        double[] x = makeRandom(N, 1, 50), xc = x.clone();
        Zlatrs.zlatrs(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                false, N, A, 0, N, x, 0, new double[N], 0);
        Ztrsv.ztrsv(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, N, A, 0, N, xc, 0, 1);
        for (int i = 0; i < N; i++) {
            assertEquals(xc[i*2], x[i*2], TOL);
            assertEquals(xc[i*2+1], x[i*2+1], TOL);
        }
    }

    @Test void testZlatrsLowerNoTransTracksInterleavedTailNorm() {
        double smlnum = Dlamch.dlamch('S') / Dlamch.dlamch('P');
        double bignum = 1.0 / smlnum;
        double[] A = {
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 2.0, 0.0, 1.0, 0.0
        };
        double[] rhs = {
                -1e100, 1e280,
                1e100, -1e280,
                0.9 * bignum, 0.9 * bignum
        };
        double[] x = rhs.clone();

        double scale = Zlatrs.zlatrs(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                false, 3, A, 0, 3, x, 0, new double[3], 0);

        assertTrue(scale < 1.0);
        assertTrue(Math.abs(x[4]) + Math.abs(x[5]) < bignum);

        double[] lhs = {
                x[0], x[1],
                x[2], x[3],
                2.0 * x[2] + x[4], 2.0 * x[3] + x[5]
        };
        for (int i = 0; i < lhs.length; i++) {
            double expected = scale * rhs[i];
            double tol = Math.max(1e-12, Math.abs(expected) * 1e-12);
            assertEquals(expected, lhs[i], tol);
        }
    }

    @Test void testZlatrsUpperTransNonUnit() {
        double[] A = makeRandom(N, N, 49);
        for (int i = 0; i < N; i++) for (int j = 0; j < i; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        double[] x = makeRandom(N, 1, 50), xc = x.clone();
        Zlatrs.zlatrs(BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit,
                false, N, A, 0, N, x, 0, new double[N], 0);
        Ztrsv.ztrsv(BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, N, A, 0, N, xc, 0, 1);
        for (int i = 0; i < N; i++) {
            assertEquals(xc[i*2], x[i*2], TOL);
            assertEquals(xc[i*2+1], x[i*2+1], TOL);
        }
    }

    @Test void testZlatrsUpperConjNonUnit() {
        double[] A = makeRandom(N, N, 49);
        for (int i = 0; i < N; i++) for (int j = 0; j < i; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        double[] x = makeRandom(N, 1, 50), xc = x.clone();
        Zlatrs.zlatrs(BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
                false, N, A, 0, N, x, 0, new double[N], 0);
        Ztrsv.ztrsv(BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit, N, A, 0, N, xc, 0, 1);
        for (int i = 0; i < N; i++) {
            assertEquals(xc[i*2], x[i*2], TOL);
            assertEquals(xc[i*2+1], x[i*2+1], TOL);
        }
    }

    @Test void testZlatrsUpperNoTransUnit() {
        double[] A = makeRandom(N, N, 49);
        for (int i = 0; i < N; i++) for (int j = 0; j < i; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        for (int i = 0; i < N; i++) { A[(i*N+i)*2] = 1; A[(i*N+i)*2+1] = 0; }
        double[] x = makeRandom(N, 1, 50), xc = x.clone();
        Zlatrs.zlatrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit,
                false, N, A, 0, N, x, 0, new double[N], 0);
        Ztrsv.ztrsv(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, N, A, 0, N, xc, 0, 1);
        for (int i = 0; i < N; i++) {
            assertEquals(xc[i*2], x[i*2], TOL);
            assertEquals(xc[i*2+1], x[i*2+1], TOL);
        }
    }

    @Test void testZgetrsTrans() {
        double[] A = makeRandom(N, N, 52);
        double[] B = new double[N * 2];
        java.util.Random r = new java.util.Random(53);
        for (int i = 0; i < N; i++) { B[i * 2] = r.nextGaussian(); B[i * 2 + 1] = r.nextGaussian(); }
        double[] Ao = A.clone(), Bc = B.clone();
        int[] ipiv = new int[N];
        Zgetr.zgetrf(N, N, A, 0, N, ipiv);
        Zgetr.zgetrs(BLAS.Trans.Trans, N, 1, A, 0, N, ipiv, B, 0, 1);
        for (int i = 0; i < N; i++) {
            double sr = 0, si = 0;
            for (int j = 0; j < N; j++) {
                double ar = Ao[(j * N + i) * 2], ai = Ao[(j * N + i) * 2 + 1];
                double br = B[j * 2], bi = B[j * 2 + 1];
                sr += ar * br - ai * bi;
                si += ar * bi + ai * br;
            }
            assertEquals(Bc[i * 2], sr, TOL);
            assertEquals(Bc[i * 2 + 1], si, TOL);
        }
    }

    @Test void testZgetrsConj() {
        double[] A = makeRandom(N, N, 52);
        double[] B = new double[N * 2];
        java.util.Random r = new java.util.Random(53);
        for (int i = 0; i < N; i++) { B[i * 2] = r.nextGaussian(); B[i * 2 + 1] = r.nextGaussian(); }
        double[] Ao = A.clone(), Bc = B.clone();
        int[] ipiv = new int[N];
        Zgetr.zgetrf(N, N, A, 0, N, ipiv);
        Zgetr.zgetrs(BLAS.Trans.Conj, N, 1, A, 0, N, ipiv, B, 0, 1);
        for (int i = 0; i < N; i++) {
            double sr = 0, si = 0;
            for (int j = 0; j < N; j++) {
                double ar = Ao[(j * N + i) * 2], ai = -Ao[(j * N + i) * 2 + 1];
                double br = B[j * 2], bi = B[j * 2 + 1];
                sr += ar * br - ai * bi;
                si += ar * bi + ai * br;
            }
            assertEquals(Bc[i * 2], sr, TOL);
            assertEquals(Bc[i * 2 + 1], si, TOL);
        }
    }

    @Test void testZpotf2Lower() {
        double[] A = makePosDef(N), Ac = A.clone();
        assertTrue(Zpotr.zpotf2(BLAS.Uplo.Lower, N, A, 0, N));
        double[] A2 = Ac.clone();
        assertEquals(0, Zpotr.zpotrf(BLAS.Uplo.Lower, N, A2, 0, N));
        for (int i = 0; i < N * N * 2; i++) {
            assertEquals(A2[i], A[i], TOL);
        }
    }

    @Test void testZlauumUpperBlocked() {
        int n = 100;
        double[] A = makePosDef(n);
        Zpotr.zpotrf(BLAS.Uplo.Upper, n, A, 0, n);
        double[] Uc = A.clone();
        Zlauum.zlauum(BLAS.Uplo.Upper, n, A, 0, n);
        for (int i = 0; i < n; i++) for (int j = i; j < n; j++) {
            double sr = 0, si = 0;
            for (int k = j; k < n; k++) {
                double ur = Uc[(i*n+k)*2], ui = Uc[(i*n+k)*2+1];
                double vr = Uc[(j*n+k)*2], vi = -Uc[(j*n+k)*2+1];
                sr += ur*vr - ui*vi;
                si += ur*vi + ui*vr;
            }
            assertEquals(sr, A[(i*n+j)*2], TOL * n);
            if (i == j) assertEquals(0, si, TOL * n);
            else assertEquals(si, A[(i*n+j)*2+1], TOL * n);
        }
    }

    @Test void testZlauumLowerBlocked() {
        int n = 100;
        double[] A = makePosDef(n);
        Zpotr.zpotrf(BLAS.Uplo.Lower, n, A, 0, n);
        double[] Lc = A.clone();
        Zlauum.zlauum(BLAS.Uplo.Lower, n, A, 0, n);
        for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++) {
            double sr = 0, si = 0;
            for (int k = i; k < n; k++) {
                double lr = Lc[(k*n+i)*2], li = Lc[(k*n+i)*2+1];
                double rr = Lc[(k*n+j)*2], ri = -Lc[(k*n+j)*2+1];
                sr += lr*rr - li*ri;
                si += lr*ri + li*rr;
            }
            assertEquals(sr, A[(i*n+j)*2], TOL * n);
            if (i == j) assertEquals(0, si, TOL * n);
            else assertEquals(si, A[(i*n+j)*2+1], TOL * n);
        }
    }

    @Test void testZlauumUpperBlockedOffsetSubmatrixMatchesDirect() {
        int n = 100;
        int lda = n + 5;
        int aOff = 7;
        double[] factor = makePosDef(n);
        assertEquals(0, Zpotr.zpotrf(BLAS.Uplo.Upper, n, factor, 0, n));

        double[] expected = factor.clone();
        double[] actual = new double[(aOff + n * lda) * 2];
        copyMatrixToOffset(factor, n, n, actual, aOff, lda);

        Zlauum.zlauum(BLAS.Uplo.Upper, n, expected, 0, n);
        Zlauum.zlauum(BLAS.Uplo.Upper, n, actual, aOff, lda);

        assertTriangularSubmatrixEquals(BLAS.Uplo.Upper, n, expected, 0, n, actual, aOff, lda, TOL * n);
    }

    @Test void testZlauumLowerBlockedOffsetSubmatrixMatchesDirect() {
        int n = 100;
        int lda = n + 6;
        int aOff = 11;
        double[] factor = makePosDef(n);
        assertEquals(0, Zpotr.zpotrf(BLAS.Uplo.Lower, n, factor, 0, n));

        double[] expected = factor.clone();
        double[] actual = new double[(aOff + n * lda) * 2];
        copyMatrixToOffset(factor, n, n, actual, aOff, lda);

        Zlauum.zlauum(BLAS.Uplo.Lower, n, expected, 0, n);
        Zlauum.zlauum(BLAS.Uplo.Lower, n, actual, aOff, lda);

        assertTriangularSubmatrixEquals(BLAS.Uplo.Lower, n, expected, 0, n, actual, aOff, lda, TOL * n);
    }

    @Test void testZtrtriUpperUnitBlocked() {
        int n = 100;
        double[] A = makeRandom(n, n, 143);
        for (int i = 0; i < n; i++) for (int j = 0; j < i; j++) {
            A[(i*n+j)*2] = 0; A[(i*n+j)*2+1] = 0;
        }
        for (int i = 0; i < n; i++) { A[(i*n+i)*2] = 1; A[(i*n+i)*2+1] = 0; }
        for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) {
            A[(i*n+j)*2] *= 0.1; A[(i*n+j)*2+1] *= 0.1;
        }
        double[] U = A.clone();
        assertEquals(0, Ztrtri.ztrtri(BLAS.Uplo.Upper, BLAS.Diag.Unit, n, A, 0, n));
        for (int i = 0; i < n; i++) for (int j = i; j < n; j++) {
            double sr = 0, si = 0;
            for (int k = i; k <= j; k++) {
                double ur = (k == i) ? 1 : U[(i*n+k)*2];
                double ui = (k == i) ? 0 : U[(i*n+k)*2+1];
                double vr = (k == j) ? 1 : A[(k*n+j)*2];
                double vi = (k == j) ? 0 : A[(k*n+j)*2+1];
                sr += ur*vr - ui*vi;
                si += ur*vi + ui*vr;
            }
            if (i == j) { assertEquals(1, sr, TOL * n * n * 100); assertEquals(0, si, TOL * n * n * 100); }
            else { assertEquals(0, sr, TOL * n * n * 100); assertEquals(0, si, TOL * n * n * 100); }
        }
    }

    @Test void testZtrtriUpperUnitBlockedOffsetSubmatrixMatchesDirect() {
        int n = 100;
        int lda = n + 4;
        int aOff = 9;
        double[] factor = makeRandom(n, n, 143);
        for (int i = 0; i < n; i++) for (int j = 0; j < i; j++) {
            factor[(i*n+j)*2] = 0;
            factor[(i*n+j)*2+1] = 0;
        }
        for (int i = 0; i < n; i++) {
            factor[(i*n+i)*2] = 1;
            factor[(i*n+i)*2+1] = 0;
        }
        for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) {
            factor[(i*n+j)*2] *= 0.1;
            factor[(i*n+j)*2+1] *= 0.1;
        }

        double[] expected = factor.clone();
        double[] actual = new double[(aOff + n * lda) * 2];
        copyMatrixToOffset(factor, n, n, actual, aOff, lda);

        assertEquals(0, Ztrtri.ztrtri(BLAS.Uplo.Upper, BLAS.Diag.Unit, n, expected, 0, n));
        assertEquals(0, Ztrtri.ztrtri(BLAS.Uplo.Upper, BLAS.Diag.Unit, n, actual, aOff, lda));

        assertTriangularSubmatrixEquals(BLAS.Uplo.Upper, n, expected, 0, n, actual, aOff, lda, TOL * n);
    }

    @Test void testZtrtriLowerNonUnitBlocked() {
        int n = 100;
        double[] A = makePosDef(n);
        Zpotr.zpotrf(BLAS.Uplo.Lower, n, A, 0, n);
        double[] L = A.clone();
        assertEquals(0, Ztrtri.ztrtri(BLAS.Uplo.Lower, BLAS.Diag.NonUnit, n, A, 0, n));
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            double sr = 0, si = 0;
            for (int k = 0; k < n; k++) {
                double ar = (k <= i) ? L[(i*n+k)*2] : 0;
                double ai = (k <= i) ? L[(i*n+k)*2+1] : 0;
                double br = (k >= j) ? A[(k*n+j)*2] : 0;
                double bi = (k >= j) ? A[(k*n+j)*2+1] : 0;
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            if (i == j) { assertEquals(1, sr, TOL * n * n * 100); assertEquals(0, si, TOL * n * n * 100); }
            else { assertEquals(0, sr, TOL * n * n * 100); assertEquals(0, si, TOL * n * n * 100); }
        }
    }

    @Test void testZtrtriLowerNonUnitBlockedOffsetSubmatrixMatchesDirect() {
        int n = 100;
        int lda = n + 7;
        int aOff = 13;
        double[] factor = makePosDef(n);
        assertEquals(0, Zpotr.zpotrf(BLAS.Uplo.Lower, n, factor, 0, n));

        double[] expected = factor.clone();
        double[] actual = new double[(aOff + n * lda) * 2];
        copyMatrixToOffset(factor, n, n, actual, aOff, lda);

        assertEquals(0, Ztrtri.ztrtri(BLAS.Uplo.Lower, BLAS.Diag.NonUnit, n, expected, 0, n));
        assertEquals(0, Ztrtri.ztrtri(BLAS.Uplo.Lower, BLAS.Diag.NonUnit, n, actual, aOff, lda));

        assertTriangularSubmatrixEquals(BLAS.Uplo.Lower, n, expected, 0, n, actual, aOff, lda, TOL * n);
    }

    @Test void testZpotriUpperBlocked() {
        int n = 100;
        double[] A = makePosDef(n);
        Zpotr.zpotrf(BLAS.Uplo.Upper, n, A, 0, n);
        assertTrue(Zpotr.zpotri(BLAS.Uplo.Upper, n, A, 0, n));
        double[] Ao = makePosDef(n);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            double sr = 0, si = 0;
            for (int k = 0; k < n; k++) {
                double ar = (k >= i) ? Ao[(i*n+k)*2] : Ao[(k*n+i)*2];
                double ai = (k >= i) ? Ao[(i*n+k)*2+1] : -Ao[(k*n+i)*2+1];
                double br = (k <= j) ? A[(k*n+j)*2] : A[(j*n+k)*2];
                double bi = (k <= j) ? A[(k*n+j)*2+1] : -A[(j*n+k)*2+1];
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            if (i == j) { assertEquals(1, sr, TOL * n * n * 100); assertEquals(0, si, TOL * n * n * 100); }
            else { assertEquals(0, sr, TOL * n * n * 100); assertEquals(0, si, TOL * n * n * 100); }
        }
    }

    @Test void testZpotriLowerBlocked() {
        int n = 100;
        double[] A = makePosDef(n);
        Zpotr.zpotrf(BLAS.Uplo.Lower, n, A, 0, n);
        assertTrue(Zpotr.zpotri(BLAS.Uplo.Lower, n, A, 0, n));
        double[] Ao = makePosDef(n);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            double sr = 0, si = 0;
            for (int k = 0; k < n; k++) {
                double ar = (k >= i) ? Ao[(i*n+k)*2] : Ao[(k*n+i)*2];
                double ai = (k >= i) ? Ao[(i*n+k)*2+1] : -Ao[(k*n+i)*2+1];
                double br = (k >= j) ? A[(k*n+j)*2] : A[(j*n+k)*2];
                double bi = (k >= j) ? A[(k*n+j)*2+1] : -A[(j*n+k)*2+1];
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            if (i == j) { assertEquals(1, sr, TOL * n * n * 100); assertEquals(0, si, TOL * n * n * 100); }
            else { assertEquals(0, sr, TOL * n * n * 100); assertEquals(0, si, TOL * n * n * 100); }
        }
    }

    @Test void testZherkUpperConj() {
        double[] A = makeRandom(3, N, 42);
        double[] C = new double[N*N*2];
        Zherk.zherk(BLAS.Uplo.Upper, BLAS.Trans.Conj, N, 3, 1.0, A, 0, 3, 0.0, C, 0, N);
        double[] ref = new double[N*N*2];
        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, N, N, 3, 1.0, 0.0, A, 0, 3, A, 0, 3, 0.0, 0.0, ref, 0, N);
        for (int i = 0; i < N; i++) for (int j = i; j < N; j++) {
            assertEquals(ref[(i*N+j)*2], C[(i*N+j)*2], TOL);
            assertEquals(ref[(i*N+j)*2+1], C[(i*N+j)*2+1], TOL);
        }
    }

    @Test void testZherkLowerNoTrans() {
        double[] A = makeRandom(N, 3, 42);
        double[] C = new double[N*N*2];
        Zherk.zherk(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, N, 3, 1.0, A, 0, 3, 0.0, C, 0, N);
        double[] ref = new double[N*N*2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, N, N, 3, 1.0, 0.0, A, 0, 3, A, 0, 3, 0.0, 0.0, ref, 0, N);
        for (int i = 0; i < N; i++) for (int j = 0; j <= i; j++) {
            assertEquals(ref[(i*N+j)*2], C[(i*N+j)*2], TOL);
            assertEquals(ref[(i*N+j)*2+1], C[(i*N+j)*2+1], TOL);
        }
    }

    @Test void testZtrmmRightLowerTransUnit() {
        double[] A = makeRandom(N, N, 43);
        for (int i = 0; i < N; i++) for (int j = i+1; j < N; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        for (int i = 0; i < N; i++) { A[(i*N+i)*2] = 1; A[(i*N+i)*2+1] = 0; }
        double[] B = makeRandom(N, N, 44), B2 = B.clone();
        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans,
                BLAS.Diag.Unit, N, N, 1.0, 0.0, A, 0, N, B2, 0, N);
        double[] ref = new double[N*N*2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, N, N, N, 1.0, 0.0, B, 0, N, A, 0, N, 0.0, 0.0, ref, 0, N);
        for (int i = 0; i < N*N*2; i++) assertEquals(ref[i], B2[i], TOL);
    }

    @Test void testZtrmmRightLowerConjUnit() {
        double[] A = makeRandom(N, N, 43);
        for (int i = 0; i < N; i++) for (int j = i+1; j < N; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        for (int i = 0; i < N; i++) { A[(i*N+i)*2] = 1; A[(i*N+i)*2+1] = 0; }
        double[] B = makeRandom(N, N, 44), B2 = B.clone();
        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj,
                BLAS.Diag.Unit, N, N, 1.0, 0.0, A, 0, N, B2, 0, N);
        double[] ref = new double[N*N*2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, N, N, N, 1.0, 0.0, B, 0, N, A, 0, N, 0.0, 0.0, ref, 0, N);
        for (int i = 0; i < N*N*2; i++) assertEquals(ref[i], B2[i], TOL);
    }

    @Test void testZtrtriLowerUnit() {
        double[] A = makeRandom(N, N, 43);
        for (int i = 0; i < N; i++) for (int j = i+1; j < N; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        for (int i = 0; i < N; i++) { A[(i*N+i)*2] = 1; A[(i*N+i)*2+1] = 0; }
        double[] L = A.clone();
        assertEquals(0, Ztrtri.ztrtri(BLAS.Uplo.Lower, BLAS.Diag.Unit, N, A, 0, N));
        for (int i = 0; i < N; i++) for (int j = 0; j <= i; j++) {
            double sr = 0, si = 0;
            for (int k = j; k <= i; k++) {
                double lr = (k == i) ? 1 : L[(i*N+k)*2];
                double li = (k == i) ? 0 : L[(i*N+k)*2+1];
                double vr = (k == j) ? 1 : A[(k*N+j)*2];
                double vi = (k == j) ? 0 : A[(k*N+j)*2+1];
                sr += lr*vr - li*vi;
                si += lr*vi + li*vr;
            }
            if (i == j) { assertEquals(1, sr, TOL); assertEquals(0, si, TOL); }
            else { assertEquals(0, sr, TOL); assertEquals(0, si, TOL); }
        }
    }

    @Test void testZtrconInfNormLowerNonUnit() {
        double[] A = makePosDef(N);
        Zpotr.zpotrf(BLAS.Uplo.Lower, N, A, 0, N);
        double rc = Ztrtri.ztrcon('I', BLAS.Uplo.Lower, BLAS.Diag.NonUnit, N, A, 0, N,
                new double[N * 4], new int[N * 2]);
        assertTrue(rc > 0 && rc <= 1.0);
    }

    @Test void testZtrconInfNormUpperUnit() {
        double[] A = makePosDef(N);
        Zpotr.zpotrf(BLAS.Uplo.Upper, N, A, 0, N);
        double rc = Ztrtri.ztrcon('I', BLAS.Uplo.Upper, BLAS.Diag.Unit, N, A, 0, N,
                new double[N * 4], new int[N * 2]);
        assertTrue(rc > 0 && rc <= 1.0);
    }

    @Test void testZtrcon1NormLowerUnit() {
        double[] A = makePosDef(N);
        Zpotr.zpotrf(BLAS.Uplo.Lower, N, A, 0, N);
        double rc = Ztrtri.ztrcon('1', BLAS.Uplo.Lower, BLAS.Diag.Unit, N, A, 0, N,
                new double[N * 4], new int[N * 2]);
        assertTrue(rc > 0 && rc <= 1.0);
    }

    @Test void testZtrconInfNormLowerUnit() {
        double[] A = makePosDef(N);
        Zpotr.zpotrf(BLAS.Uplo.Lower, N, A, 0, N);
        double rc = Ztrtri.ztrcon('I', BLAS.Uplo.Lower, BLAS.Diag.Unit, N, A, 0, N,
                new double[N * 4], new int[N * 2]);
        assertTrue(rc > 0 && rc <= 1.0);
    }

    @Test void testZlatrsLowerTransNonUnit() {
        double[] A = makeRandom(N, N, 49);
        for (int i = 0; i < N; i++) for (int j = i+1; j < N; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        double[] x = makeRandom(N, 1, 50), xc = x.clone();
        Zlatrs.zlatrs(BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit,
                false, N, A, 0, N, x, 0, new double[N], 0);
        Ztrsv.ztrsv(BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, N, A, 0, N, xc, 0, 1);
        for (int i = 0; i < N; i++) {
            assertEquals(xc[i*2], x[i*2], TOL);
            assertEquals(xc[i*2+1], x[i*2+1], TOL);
        }
    }

    @Test void testZlatrsLowerConjNonUnit() {
        double[] A = makeRandom(N, N, 49);
        for (int i = 0; i < N; i++) for (int j = i+1; j < N; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        double[] x = makeRandom(N, 1, 50), xc = x.clone();
        Zlatrs.zlatrs(BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
                false, N, A, 0, N, x, 0, new double[N], 0);
        Ztrsv.ztrsv(BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.NonUnit, N, A, 0, N, xc, 0, 1);
        for (int i = 0; i < N; i++) {
            assertEquals(xc[i*2], x[i*2], TOL);
            assertEquals(xc[i*2+1], x[i*2+1], TOL);
        }
    }

    @Test void testZlatrsLowerNoTransUnit() {
        double[] A = makeRandom(N, N, 49);
        for (int i = 0; i < N; i++) for (int j = i+1; j < N; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        for (int i = 0; i < N; i++) { A[(i*N+i)*2] = 1; A[(i*N+i)*2+1] = 0; }
        double[] x = makeRandom(N, 1, 50), xc = x.clone();
        Zlatrs.zlatrs(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit,
                false, N, A, 0, N, x, 0, new double[N], 0);
        Ztrsv.ztrsv(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, N, A, 0, N, xc, 0, 1);
        for (int i = 0; i < N; i++) {
            assertEquals(xc[i*2], x[i*2], TOL);
            assertEquals(xc[i*2+1], x[i*2+1], TOL);
        }
    }

    @Test void testZlatrsLowerTransUnit() {
        double[] A = makeRandom(N, N, 49);
        for (int i = 0; i < N; i++) for (int j = i+1; j < N; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        for (int i = 0; i < N; i++) { A[(i*N+i)*2] = 1; A[(i*N+i)*2+1] = 0; }
        double[] x = makeRandom(N, 1, 50), xc = x.clone();
        Zlatrs.zlatrs(BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit,
                false, N, A, 0, N, x, 0, new double[N], 0);
        Ztrsv.ztrsv(BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit, N, A, 0, N, xc, 0, 1);
        for (int i = 0; i < N; i++) {
            assertEquals(xc[i*2], x[i*2], TOL);
            assertEquals(xc[i*2+1], x[i*2+1], TOL);
        }
    }

    @Test void testZlatrsLowerConjUnit() {
        double[] A = makeRandom(N, N, 49);
        for (int i = 0; i < N; i++) for (int j = i+1; j < N; j++) {
            A[(i*N+j)*2] = 0; A[(i*N+j)*2+1] = 0;
        }
        for (int i = 0; i < N; i++) { A[(i*N+i)*2] = 1; A[(i*N+i)*2+1] = 0; }
        double[] x = makeRandom(N, 1, 50), xc = x.clone();
        Zlatrs.zlatrs(BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.Unit,
                false, N, A, 0, N, x, 0, new double[N], 0);
        Ztrsv.ztrsv(BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.Unit, N, A, 0, N, xc, 0, 1);
        for (int i = 0; i < N; i++) {
            assertEquals(xc[i*2], x[i*2], TOL);
            assertEquals(xc[i*2+1], x[i*2+1], TOL);
        }
    }
}
