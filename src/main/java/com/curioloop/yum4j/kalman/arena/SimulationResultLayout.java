package com.curioloop.yum4j.kalman.arena;

public value record SimulationResultLayout(int generatedMeasurementDisturbanceBase,
                                     int generatedMeasurementDisturbanceLength,
                                     int generatedStateDisturbanceBase,
                                     int generatedStateDisturbanceLength,
                                     int generatedObsBase,
                                     int generatedObsLength,
                                     int generatedStateBase,
                                     int generatedStateLength,
                                     int simulatedMeasurementDisturbanceBase,
                                     int simulatedMeasurementDisturbanceLength,
                                     int simulatedStateDisturbanceBase,
                                     int simulatedStateDisturbanceLength,
                                     int simulatedStateBase,
                                     int simulatedStateLength,
                                     int totalLength) {

    public static SimulationResultLayout create(int scalarWidth,
                                                int kEndog,
                                                int kStates,
                                                int kPosdef,
                                                int nobs,
                                                boolean storeGenerated,
                                                boolean storeState,
                                                boolean storeDisturbance) {
        int offset = 0;

        int generatedMeasurementDisturbanceBase = offset;
        int generatedMeasurementDisturbanceLength = storeGenerated ? scalarWidth * kEndog * nobs : 0;
        offset += generatedMeasurementDisturbanceLength;
        int generatedStateDisturbanceBase = offset;
        int generatedStateDisturbanceLength = storeGenerated ? scalarWidth * kPosdef * nobs : 0;
        offset += generatedStateDisturbanceLength;
        int generatedObsBase = offset;
        int generatedObsLength = storeGenerated ? scalarWidth * kEndog * nobs : 0;
        offset += generatedObsLength;
        int generatedStateBase = offset;
        int generatedStateLength = storeGenerated ? scalarWidth * kStates * (nobs + 1) : 0;
        offset += generatedStateLength;
        int simulatedMeasurementDisturbanceBase = offset;
        int simulatedMeasurementDisturbanceLength = storeDisturbance ? scalarWidth * kEndog * nobs : 0;
        offset += simulatedMeasurementDisturbanceLength;
        int simulatedStateDisturbanceBase = offset;
        int simulatedStateDisturbanceLength = storeDisturbance ? scalarWidth * kPosdef * nobs : 0;
        offset += simulatedStateDisturbanceLength;
        int simulatedStateBase = offset;
        int simulatedStateLength = storeState ? scalarWidth * kStates * nobs : 0;
        offset += simulatedStateLength;

        return new SimulationResultLayout(
            generatedMeasurementDisturbanceBase,
            generatedMeasurementDisturbanceLength,
            generatedStateDisturbanceBase,
            generatedStateDisturbanceLength,
            generatedObsBase,
            generatedObsLength,
            generatedStateBase,
            generatedStateLength,
            simulatedMeasurementDisturbanceBase,
            simulatedMeasurementDisturbanceLength,
            simulatedStateDisturbanceBase,
            simulatedStateDisturbanceLength,
            simulatedStateBase,
            simulatedStateLength,
            offset);
    }
}