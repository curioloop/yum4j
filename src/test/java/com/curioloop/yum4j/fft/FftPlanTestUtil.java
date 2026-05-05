package com.curioloop.yum4j.fft;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

final class FftPlanTestUtil {
    private FftPlanTestUtil() {
    }

    // ---- factor extraction ----

    static List<Integer> factors(FftRealPlan plan) {
        return factorsFromPlan(plan);
    }

    static List<Integer> factors(FftComplexPlan plan) {
        return factorsFromPlan(plan);
    }

    static List<Integer> factors(FftRealTransform transform) {
        return switch (transform.strategy()) {
            case FftRealTransform.MixedRadixReal m -> factors(m.plan());
            default -> List.of();
        };
    }

    static List<Integer> factors(FftComplexTransform transform) {
        return switch (transform.strategy()) {
            case FftComplexTransform.MixedRadix m -> factors(m.plan());
            default -> List.of();
        };
    }

    // ---- route / bluestein / convolution length (test-only introspection) ----

    static FftRoute route(FftComplexTransform transform) {
        return switch (transform.strategy()) {
            case FftComplexTransform.SplitRadix _ -> FftRoute.SPLIT_RADIX;
            case FftComplexTransform.MixedRadix _ -> FftRoute.DIRECT;
            case FftComplexTransform.Bluestein _  -> FftRoute.BLUESTEIN;
        };
    }

    static FftRoute route(FftRealTransform transform) {
        return switch (transform.strategy()) {
            case FftRealTransform.SplitRadixReal _ -> FftRoute.SPLIT_RADIX;
            case FftRealTransform.MixedRadixReal _ -> FftRoute.DIRECT;
            case FftRealTransform.BluesteinReal _  -> FftRoute.BLUESTEIN;
        };
    }

    static boolean usesBluestein(FftComplexTransform transform) {
        return transform.strategy() instanceof FftComplexTransform.Bluestein;
    }

    static boolean usesBluestein(FftRealTransform transform) {
        return transform.strategy() instanceof FftRealTransform.BluesteinReal;
    }

    static int convolutionLength(FftComplexTransform transform) {
        return transform.strategy() instanceof FftComplexTransform.Bluestein b
                ? b.plan().convolutionLength() : transform.length();
    }

    static int convolutionLength(FftRealTransform transform) {
        return transform.strategy() instanceof FftRealTransform.BluesteinReal b
                ? b.plan().convolutionLength() : transform.length();
    }

    // ---- internals ----

    private static List<Integer> factorsFromPlan(Object plan) {
        Object[] factors = readField(plan, "factors", Object[].class);
        List<Integer> result = new ArrayList<>(factors.length);
        for (Object factorData : factors) {
            result.add(readField(factorData, "factor", Integer.class));
        }
        return result;
    }

    private static <T> T readField(Object target, String fieldName, Class<T> fieldType) {
        try {
            Field field = target.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            return fieldType.cast(field.get(target));
        } catch (ReflectiveOperationException exception) {
            throw new AssertionError("Unable to read test field " + fieldName, exception);
        }
    }
}
