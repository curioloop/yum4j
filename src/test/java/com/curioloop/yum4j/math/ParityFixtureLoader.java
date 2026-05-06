package com.curioloop.yum4j.math;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public final class ParityFixtureLoader {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper()
        .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
    private static final Set<String> REQUIRED_META_FIELDS = Set.of("module", "source");
    private static final Set<String> REQUIRED_BOOST_META_FIELDS = Set.of(
        "boostVersion",
        "compiler",
        "generatedAt"
    );

    private ParityFixtureLoader() {
    }

    public static <T> T load(String resourcePath, Class<T> fixtureType) {
        InputStream stream = Thread.currentThread()
            .getContextClassLoader()
            .getResourceAsStream(resourcePath);

        if (stream == null) {
            throw new IllegalArgumentException("Fixture resource not found: " + resourcePath);
        }

        try (InputStream inputStream = stream) {
            JsonNode root = OBJECT_MAPPER.readTree(inputStream);
            validateFixtureSchema(resourcePath, root);
            return OBJECT_MAPPER.treeToValue(root, fixtureType);
        } catch (IOException exception) {
            throw new UncheckedIOException("Failed to read fixture resource: " + resourcePath, exception);
        }
    }

    private static void validateFixtureSchema(String resourcePath, JsonNode root) {
        if (!requiresBoostMathSchema(resourcePath)) {
            return;
        }
        if (root == null || !root.isObject()) {
            throw new IllegalArgumentException("Boost math fixture must contain a JSON object root: " + resourcePath);
        }

        JsonNode meta = root.get("meta");
        if (meta == null || !meta.isObject()) {
            throw new IllegalArgumentException("Boost math fixture must contain an object meta block: " + resourcePath);
        }
        validateRequiredTextFields(resourcePath, meta, REQUIRED_META_FIELDS);
        validateRequiredTextFields(resourcePath, meta, REQUIRED_BOOST_META_FIELDS);

        JsonNode cases = root.get("cases");
        if (cases == null || !cases.isArray()) {
            throw new IllegalArgumentException("Boost math fixture must contain a top-level cases array: " + resourcePath);
        }
    }

    private static void validateRequiredTextFields(String resourcePath, JsonNode meta, Set<String> requiredFields) {
        List<String> missingFields = requiredFields.stream()
            .filter(field -> !meta.has(field) || meta.path(field).asText().isBlank())
            .sorted()
            .toList();
        if (!missingFields.isEmpty()) {
            throw new IllegalArgumentException(
                "Boost math fixture is missing required meta fields " + missingFields + ": " + resourcePath
            );
        }
    }

    private static boolean requiresBoostMathSchema(String resourcePath) {
        String normalizedPath = resourcePath.toLowerCase(Locale.ROOT);
        return normalizedPath.startsWith("boost/math/") && normalizedPath.endsWith("-boost-parity-fixture.json");
    }
}