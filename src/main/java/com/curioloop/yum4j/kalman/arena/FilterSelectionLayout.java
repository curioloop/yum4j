package com.curioloop.yum4j.kalman.arena;

public value record FilterSelectionLayout(int selectedDesignBase,
                                    int selectedObsCovBase,
                                    int selectedObsInterceptBase,
                                    int selectedEndogBase,
                                    int totalLength) {

    public static FilterSelectionLayout create(int scalarWidth,
                                               int kEndog,
                                               int kStates) {
        int offset = 0;

        int selectedDesignBase = offset;
        offset += scalarWidth * kEndog * kStates;
        int selectedObsCovBase = offset;
        offset += scalarWidth * kEndog * kEndog;
        int selectedObsInterceptBase = offset;
        offset += scalarWidth * kEndog;
        int selectedEndogBase = offset;
        offset += scalarWidth * kEndog;

        return new FilterSelectionLayout(
            selectedDesignBase,
            selectedObsCovBase,
            selectedObsInterceptBase,
            selectedEndogBase,
            offset);
    }
}