package com.curioloop.yum4j.linalg.blas.cmplx;

interface Zlarfg {

    static final double DLAMCH_E = 0x1p-53;
    static final double DLAMCH_S = 0x1p-1022;
    static final double SAFMIN = DLAMCH_S / DLAMCH_E;

    static void zlarfg(int n, double[] alpha, int aOff, double[] x, int xOff, int incX,
                       double[] tau, int tOff) {
        if (n <= 0) {
            tau[tOff] = 0.0;
            tau[tOff + 1] = 0.0;
            return;
        }

        double alphar = alpha[aOff];
        double alphai = alpha[aOff + 1];

        double xnorm = 0.0;
        for (int i = 0; i < n - 1; i++) {
            int bytePos = (xOff + i * incX) * 2;
            double re = x[bytePos];
            double im = x[bytePos + 1];
            xnorm += re * re + im * im;
        }
        xnorm = Math.sqrt(xnorm);

        if (xnorm == 0.0 && alphai == 0.0) {
            tau[tOff] = 0.0;
            tau[tOff + 1] = 0.0;
            return;
        }

        double beta = -Math.copySign(Math.hypot(Math.hypot(alphar, alphai), xnorm), alphar);

        double rsafmn = 1.0 / SAFMIN;
        int knt = 0;
        if (Math.abs(beta) < SAFMIN) {
            do {
                knt++;
                for (int i = 0; i < n - 1; i++) {
                    int bytePos = (xOff + i * incX) * 2;
                    x[bytePos] *= rsafmn;
                    x[bytePos + 1] *= rsafmn;
                }
                beta *= rsafmn;
                alphai *= rsafmn;
                alphar *= rsafmn;
            } while (Math.abs(beta) < SAFMIN && knt < 20);

            xnorm = 0.0;
            for (int i = 0; i < n - 1; i++) {
                int bytePos = (xOff + i * incX) * 2;
                double re = x[bytePos];
                double im = x[bytePos + 1];
                xnorm += re * re + im * im;
            }
            xnorm = Math.sqrt(xnorm);
            beta = -Math.copySign(Math.hypot(Math.hypot(alphar, alphai), xnorm), alphar);
        }

        tau[tOff] = (beta - alphar) / beta;
        tau[tOff + 1] = -alphai / beta;

        if (n - 1 > 0) {
            double deltaRe = alphar - beta;
            double deltaIm = alphai;
            double deltaNorm2 = deltaRe * deltaRe + deltaIm * deltaIm;
            double scaleRe = deltaRe / deltaNorm2;
            double scaleIm = -deltaIm / deltaNorm2;
            for (int i = 0; i < n - 1; i++) {
                int bytePos = (xOff + i * incX) * 2;
                double re = x[bytePos];
                double im = x[bytePos + 1];
                x[bytePos] = scaleRe * re - scaleIm * im;
                x[bytePos + 1] = scaleRe * im + scaleIm * re;
            }
        }

        for (int j = 0; j < knt; j++) {
            beta *= SAFMIN;
        }

        alpha[aOff] = beta;
        alpha[aOff + 1] = 0.0;
    }
}
