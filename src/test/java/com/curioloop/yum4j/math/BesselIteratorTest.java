package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class BesselIteratorTest {

    @TestFactory
    Stream<DynamicTest> boostIteratorSpotCasesRemainAligned() {
        return BOOST_SPOT_CASES.stream().map(testCase -> dynamicTest(testCase.name(), () -> assertBoostSpotCase(testCase)));
    }

    @Test
    void besselJBackwardsBoundaryConstructorsPreserveBoostPairSemantics() {
        double x = 2.0;

        assertJBackwardSequence(adapt(new JBackwardState(0.0, x)), 0.0, x, 5, 5.0e-13);
        assertJBackwardSequence(adapt(new JBackwardState(0.0, x, Bessel.j(0.0, x))), 0.0, x, 5, 5.0e-13);
        assertJBackwardSequence(adapt(new JBackwardState(-1.0, x, Bessel.j(0.0, x), Bessel.j(-1.0, x))), -1.0, x, 5, 5.0e-13);
        assertJBackwardSequence(adapt(new JBackwardState(-0.75, x, Bessel.j(0.25, x), Bessel.j(-0.75, x))), -0.75, x, 5, 5.0e-13);

        assertThrows(IllegalArgumentException.class,
            () -> new JBackwardState(-1.0001, x, Bessel.j(-0.0001, x), Bessel.j(-1.0001, x)));
    }

    @Test
    void besselIBackwardsBoundaryConstructorsPreserveNegativeOrderSemantics() {
        double x = 2.0;

        assertIBackwardSequence(adapt(new IBackwardState(-1.0, x)), -1.0, x, 5, 5.0e-13);
        assertIBackwardSequence(adapt(new IBackwardState(-1.0, x, Bessel.i(-1.0, x))), -1.0, x, 5, 5.0e-13);
        assertIBackwardSequence(adapt(new IBackwardState(-1.0, x, Bessel.i(0.0, x), Bessel.i(-1.0, x))), -1.0, x, 5, 5.0e-13);
        assertIBackwardSequence(adapt(new IBackwardState(-0.75, x, Bessel.i(0.25, x), Bessel.i(-0.75, x))), -0.75, x, 5, 5.0e-13);

        assertThrows(IllegalArgumentException.class,
            () -> new IBackwardState(-1.0001, x, Bessel.i(-0.0001, x), Bessel.i(-1.0001, x)));
    }

    @Test
    void besselIForwardsBoundaryConstructorsPreserveOrderOneSemantics() {
        double x = 2.0;

        assertIForwardSequence(adapt(new IForwardState(1.0, x)), 1.0, x, 5, 5.0e-13);
        assertIForwardSequence(adapt(new IForwardState(1.0, x, Bessel.i(1.0, x))), 1.0, x, 5, 5.0e-13);
        assertIForwardSequence(adapt(new IForwardState(1.0, x, Bessel.i(0.0, x), Bessel.i(1.0, x))), 1.0, x, 5, 5.0e-13);

        assertThrows(IllegalArgumentException.class,
            () -> new IForwardState(1.0001, x, Bessel.i(0.0001, x), Bessel.i(1.0001, x)));
    }

    @Test
    void explicitCurrentCanTrackWeightedBesselISequences() {
        double order = -12.0;
        double x = 2.0;
        IteratorAdapter iterator = adapt(new IForwardState(order, x, Bessel.iExponentiallyWeighted(order, x)));

        for (int step = 0; step < 6; step++) {
            double expectedOrder = order + step;
            double expected = Bessel.iExponentiallyWeighted(expectedOrder, x);
            assertClose(expectedOrder, iterator.order(), 0.0);
            assertClose(expected, iterator.value(), 5.0e-13);
            if (step < 5) {
                iterator.advance();
            }
        }
    }

    @Test
    void explicitPairCanTrackWeightedBesselINegativeOrderBoundarySequences() {
        double order = -0.75;
        double x = 2.0;

        IteratorAdapter backwards = adapt(new IBackwardState(
            order,
            x,
            Bessel.iExponentiallyWeighted(order + 1.0, x),
            Bessel.iExponentiallyWeighted(order, x)));
        for (int step = 0; step < 5; step++) {
            double expectedOrder = order - step;
            double expected = Bessel.iExponentiallyWeighted(expectedOrder, x);
            assertClose(expectedOrder, backwards.order(), 0.0);
            assertClose(expected, backwards.value(), 5.0e-13);
            if (step < 4) {
                backwards.advance();
            }
        }

        IteratorAdapter forwards = adapt(new IForwardState(
            order,
            x,
            Bessel.iExponentiallyWeighted(order - 1.0, x),
            Bessel.iExponentiallyWeighted(order, x)));
        for (int step = 0; step < 5; step++) {
            double expectedOrder = order + step;
            double expected = Bessel.iExponentiallyWeighted(expectedOrder, x);
            assertClose(expectedOrder, forwards.order(), 0.0);
            assertClose(expected, forwards.value(), 5.0e-13);
            if (step < 4) {
                forwards.advance();
            }
        }
    }

    @Test
    void iteratorDomainsMatchBoostContracts() {
        assertThrows(IllegalArgumentException.class, () -> new JBackwardState(-2.3, 5.0));
        assertThrows(IllegalArgumentException.class, () -> new JBackwardState(-2.3, 5.0, 2.0));
        assertThrows(IllegalArgumentException.class, () -> new JBackwardState(-2.3, 5.0, 2.0, 2.0));

        assertThrows(IllegalArgumentException.class, () -> new IBackwardState(-1.1, 2.0));
        assertThrows(IllegalArgumentException.class, () -> new IBackwardState(-1.1, 2.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new IBackwardState(-1.1, 2.0, 1.0, 1.0));

        assertThrows(IllegalArgumentException.class, () -> new IForwardState(1.1, 2.0));
        assertThrows(IllegalArgumentException.class, () -> new IForwardState(1.1, 2.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new IForwardState(1.1, 2.0, 1.0, 1.0));
    }

    private static void assertBoostSpotCase(BoostSpotCase testCase) {
        IteratorAdapter iterator = createIterator(testCase.kind(), testCase.constructorMode(), testCase.order(), testCase.x());
        for (int step = 0; step < testCase.advanceCount(); step++) {
            iterator.advance();
        }

        double expectedOrder = testCase.order() + testCase.kind().orderStep() * testCase.advanceCount();
        double expected = testCase.kind().valueAt(expectedOrder, testCase.x());
        assertClose(expectedOrder, iterator.order(), 0.0);
        assertClose(expected, iterator.value(), testCase.tolerance());

        if (testCase.verifyPostAdvance()) {
            double captured = iterator.value();
            iterator.advance();
            assertClose(expected, captured, testCase.tolerance());

            double nextOrder = expectedOrder + testCase.kind().orderStep();
            double nextValue = testCase.kind().valueAt(nextOrder, testCase.x());
            assertClose(nextOrder, iterator.order(), 0.0);
            assertClose(nextValue, iterator.value(), testCase.tolerance());
        }
    }

    private static IteratorAdapter createIterator(IteratorKind kind,
                                                  ConstructorMode constructorMode,
                                                  double order,
                                                  double x) {
        return switch (kind) {
            case J_BACKWARD -> createJBackwardIterator(constructorMode, order, x);
            case I_BACKWARD -> createIBackwardIterator(constructorMode, order, x);
            case I_FORWARD -> createIForwardIterator(constructorMode, order, x);
        };
    }

    private static IteratorAdapter createJBackwardIterator(ConstructorMode constructorMode, double order, double x) {
        return switch (constructorMode) {
            case AUTO -> adapt(new JBackwardState(order, x));
            case CURRENT -> adapt(new JBackwardState(order, x, Bessel.j(order, x)));
            case PAIR -> adapt(new JBackwardState(order, x, Bessel.j(order + 1.0, x), Bessel.j(order, x)));
        };
    }

    private static IteratorAdapter createIBackwardIterator(ConstructorMode constructorMode, double order, double x) {
        return switch (constructorMode) {
            case AUTO -> adapt(new IBackwardState(order, x));
            case CURRENT -> adapt(new IBackwardState(order, x, Bessel.i(order, x)));
            case PAIR -> adapt(new IBackwardState(order, x, Bessel.i(order + 1.0, x), Bessel.i(order, x)));
        };
    }

    private static IteratorAdapter createIForwardIterator(ConstructorMode constructorMode, double order, double x) {
        return switch (constructorMode) {
            case AUTO -> adapt(new IForwardState(order, x));
            case CURRENT -> adapt(new IForwardState(order, x, Bessel.i(order, x)));
            case PAIR -> adapt(new IForwardState(order, x, Bessel.i(order - 1.0, x), Bessel.i(order, x)));
        };
    }

    private static IteratorAdapter adapt(JBackwardState iterator) {
        return new IteratorAdapter(iterator::order, iterator::value, iterator::advance);
    }

    private static IteratorAdapter adapt(IBackwardState iterator) {
        return new IteratorAdapter(iterator::order, iterator::value, iterator::advance);
    }

    private static IteratorAdapter adapt(IForwardState iterator) {
        return new IteratorAdapter(iterator::order, iterator::value, iterator::advance);
    }

    private static void assertJBackwardSequence(IteratorAdapter iterator,
                                                double startOrder,
                                                double x,
                                                int steps,
                                                double tolerance) {
        for (int step = 0; step < steps; step++) {
            double expectedOrder = startOrder - step;
            double expected = Bessel.j(expectedOrder, x);
            assertClose(expectedOrder, iterator.order(), 0.0);
            assertClose(expected, iterator.value(), tolerance);
            if (step + 1 < steps) {
                iterator.advance();
            }
        }
    }

    private static void assertIBackwardSequence(IteratorAdapter iterator,
                                                double startOrder,
                                                double x,
                                                int steps,
                                                double tolerance) {
        for (int step = 0; step < steps; step++) {
            double expectedOrder = startOrder - step;
            double expected = Bessel.i(expectedOrder, x);
            assertClose(expectedOrder, iterator.order(), 0.0);
            assertClose(expected, iterator.value(), tolerance);
            if (step + 1 < steps) {
                iterator.advance();
            }
        }
    }

    private static void assertIForwardSequence(IteratorAdapter iterator,
                                               double startOrder,
                                               double x,
                                               int steps,
                                               double tolerance) {
        for (int step = 0; step < steps; step++) {
            double expectedOrder = startOrder + step;
            double expected = Bessel.i(expectedOrder, x);
            assertClose(expectedOrder, iterator.order(), 0.0);
            assertClose(expected, iterator.value(), tolerance);
            if (step + 1 < steps) {
                iterator.advance();
            }
        }
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    private static final List<BoostSpotCase> BOOST_SPOT_CASES = List.of(
        new BoostSpotCase("boost-j-backward-auto", IteratorKind.J_BACKWARD, ConstructorMode.AUTO, 12.0, 2.0, 4, true, 5.0e-13),
        new BoostSpotCase("boost-j-backward-current", IteratorKind.J_BACKWARD, ConstructorMode.CURRENT, 12.0, 2.0, 4, false, 5.0e-13),
        new BoostSpotCase("boost-j-backward-pair", IteratorKind.J_BACKWARD, ConstructorMode.PAIR, 12.0, 2.0, 4, false, 5.0e-13),
        new BoostSpotCase("boost-i-backward-auto", IteratorKind.I_BACKWARD, ConstructorMode.AUTO, 12.0, 2.0, 4, true, 5.0e-13),
        new BoostSpotCase("boost-i-backward-current", IteratorKind.I_BACKWARD, ConstructorMode.CURRENT, 12.0, 2.0, 4, false, 5.0e-13),
        new BoostSpotCase("boost-i-backward-pair", IteratorKind.I_BACKWARD, ConstructorMode.PAIR, 12.0, 2.0, 4, false, 5.0e-13),
        new BoostSpotCase("boost-i-forward-auto", IteratorKind.I_FORWARD, ConstructorMode.AUTO, -12.0, 2.0, 4, true, 5.0e-13),
        new BoostSpotCase("boost-i-forward-current", IteratorKind.I_FORWARD, ConstructorMode.CURRENT, -12.0, 2.0, 4, false, 5.0e-13),
        new BoostSpotCase("boost-i-forward-pair", IteratorKind.I_FORWARD, ConstructorMode.PAIR, -12.0, 2.0, 4, false, 5.0e-13)
    );

    private record BoostSpotCase(String name,
                                 IteratorKind kind,
                                 ConstructorMode constructorMode,
                                 double order,
                                 double x,
                                 int advanceCount,
                                 boolean verifyPostAdvance,
                                 double tolerance) {
    }

    private record IteratorAdapter(DoubleSupplier orderSupplier,
                                   DoubleSupplier valueSupplier,
                                   Runnable advanceAction) {
        private double order() {
            return orderSupplier.getAsDouble();
        }

        private double value() {
            return valueSupplier.getAsDouble();
        }

        private void advance() {
            advanceAction.run();
        }
    }

    private static final class JBackwardState {
        private final double x;
        private double order;
        private double next;
        private double current;

        private JBackwardState(double order, double x) {
            this(order, x, Bessel.j(order, x));
        }

        private JBackwardState(double order, double x, double current) {
            Recurrences.validateBesselJBackwardOrder(order);
            this.x = x;
            this.order = order;
            this.current = current;
            this.next = Recurrences.besselJBackwardSeedNext(order, x, current);
        }

        private JBackwardState(double order, double x, double next, double current) {
            Recurrences.validateBesselJBackwardPairOrder(order);
            this.x = x;
            this.order = order;
            this.next = next;
            this.current = current;
        }

        private double order() {
            return order;
        }

        private double value() {
            return current;
        }

        private void advance() {
            double previous = Recurrences.besselJBackwardPrevious(order, x, current, next);
            next = current;
            current = previous;
            order -= 1.0;
        }
    }

    private static final class IBackwardState {
        private final double x;
        private double order;
        private double next;
        private double current;

        private IBackwardState(double order, double x) {
            this(order, x, Bessel.i(order, x));
        }

        private IBackwardState(double order, double x, double current) {
            Recurrences.validateBesselIBackwardOrder(order);
            this.x = x;
            this.order = order;
            this.current = current;
            this.next = Recurrences.besselIBackwardSeedNext(order, x, current);
        }

        private IBackwardState(double order, double x, double next, double current) {
            Recurrences.validateBesselIBackwardOrder(order);
            this.x = x;
            this.order = order;
            this.next = next;
            this.current = current;
        }

        private double order() {
            return order;
        }

        private double value() {
            return current;
        }

        private void advance() {
            double previous = Recurrences.besselIBackwardPrevious(order, x, current, next);
            next = current;
            current = previous;
            order -= 1.0;
        }
    }

    private static final class IForwardState {
        private final double x;
        private double order;
        private double previous;
        private double current;

        private IForwardState(double order, double x) {
            this(order, x, Bessel.i(order, x));
        }

        private IForwardState(double order, double x, double current) {
            Recurrences.validateBesselIForwardOrder(order);
            this.x = x;
            this.order = order;
            this.current = current;
            this.previous = Recurrences.besselIForwardSeedPrevious(order, x, current);
        }

        private IForwardState(double order, double x, double previous, double current) {
            Recurrences.validateBesselIForwardOrder(order);
            this.x = x;
            this.order = order;
            this.previous = previous;
            this.current = current;
        }

        private double order() {
            return order;
        }

        private double value() {
            return current;
        }

        private void advance() {
            double next = Recurrences.besselIForwardNext(order, x, previous, current);
            previous = current;
            current = next;
            order += 1.0;
        }
    }

    private enum ConstructorMode {
        AUTO,
        CURRENT,
        PAIR
    }

    private enum IteratorKind {
        J_BACKWARD(-1.0) {
            @Override
            double valueAt(double order, double x) {
                return Bessel.j(order, x);
            }
        },
        I_BACKWARD(-1.0) {
            @Override
            double valueAt(double order, double x) {
                return Bessel.i(order, x);
            }
        },
        I_FORWARD(1.0) {
            @Override
            double valueAt(double order, double x) {
                return Bessel.i(order, x);
            }
        };

        private final double orderStep;

        IteratorKind(double orderStep) {
            this.orderStep = orderStep;
        }

        abstract double valueAt(double order, double x);

        double orderStep() {
            return orderStep;
        }
    }
}