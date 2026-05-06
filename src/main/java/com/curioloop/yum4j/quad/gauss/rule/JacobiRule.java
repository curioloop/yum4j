/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.quad.gauss.rule;

import com.curioloop.yum4j.math.Gamma;
import com.curioloop.yum4j.quad.gauss.GaussRule;

/**
 * Gauss-Jacobi quadrature rule for
 *   ∫_{−1}^{1} (1−x)^α (1+x)^β f(x) dx,  α,β > −1.
 *
 * <p>Nodes and weights are generated via the Golub-Welsch algorithm applied to
 * the Jacobi three-term recurrence.  The zero-th moment is:
 *   μ₀ = B(α+1, β+1) · 2^(α+β+1)
 *      = Γ(α+1)·Γ(β+1)/Γ(α+β+2) · 2^(α+β+1)
 * computed via {@link Gamma#lgamma(double)}.</p>
 *
 * <p>Jacobi matrix diagonal (αₖ) and off-diagonal (βₖ) entries:
 *   αₖ = (β²−α²) / ((2k+α+β)(2k+α+β+2))
 *   βₖ = √[ 4k(k+α)(k+β)(k+α+β) / ((2k+α+β)²(2k+α+β−1)(2k+α+β+1)) ]
 * with special handling when 2k+α+β ≈ 0 (near the symmetric case α=β=−1/2).</p>
 */
public record JacobiRule(double alpha, double beta) implements GaussRule {

    private static final double LOG_TWO = Math.log(2.0);

    public JacobiRule {
        validateExponent("alpha", alpha);
        validateExponent("beta", beta);
    }

    @Override
    public double zeroMoment() {
        return Math.exp((alpha + beta + 1.0) * LOG_TWO
            + Gamma.lgamma(alpha + 1.0)
            + Gamma.lgamma(beta + 1.0)
            - Gamma.lgamma(alpha + beta + 2.0));
    }

    @Override
    public void fillJacobi(int n, double[] diag, int diagOff, double[] offDiag, int offDiagOff) {
        double sum = alpha + beta;
        for (int i = 0; i < n; i++) {
            double k = i;
            double denom = (2.0 * k + sum) * (2.0 * k + sum + 2.0);
            diag[diagOff + i] = Math.abs(denom) <= Math.ulp(1.0)
                    ? 0.0
                    : (beta * beta - alpha * alpha) / denom;
        }
        for (int i = 1; i < n; i++) {
            double k = i - 1.0;
            double edge = k + 1.0 + sum;
            double d0 = 2.0 * k + sum + 1.0, d1 = 2.0 * k + sum + 2.0, d2 = 2.0 * k + sum + 3.0;
            if (Math.abs(edge) <= Math.ulp(1.0) && Math.abs(d0) <= Math.ulp(1.0)) {
                offDiag[offDiagOff + i - 1] = Math.sqrt(
                        4.0 * (k + 1.0) * (k + 1.0 + alpha) * (k + 1.0 + beta) / (d1 * d1 * d2));
            } else {
                offDiag[offDiagOff + i - 1] = Math.sqrt(
                        4.0 * (k + 1.0) * (k + 1.0 + alpha) * (k + 1.0 + beta) * edge
                                / (d1 * d1 * d0 * d2));
            }
        }
    }

    private static void validateExponent(String name, double value) {
        if (!Double.isFinite(value)) throw new IllegalArgumentException(name + " must be finite");
        if (value <= -1.0) throw new IllegalArgumentException(name + " must be > -1");
    }

}
