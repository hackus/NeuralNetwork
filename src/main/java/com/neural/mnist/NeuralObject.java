package com.neural.mnist;

import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static java.util.stream.Collectors.toList;

public class NeuralObject implements NeuralType {
    private int resultSize;
    private int rowSize;
    private int colSize;
    private int maxInput;
    private int maxOutput;

    List<Double> result = new ArrayList<>();
    List<Double> input = new ArrayList<>();

    public NeuralObject(final int rowSize, final int colSize, final int resultSize){
        this.rowSize = rowSize;
        this.colSize = colSize;
        this.resultSize = resultSize;
    }

    @Override
    public void setInput(List<Double> input) {
        this.input = input;
    }

    @Override
    public void setResult(List<Double> result) {
        this.result = result;
    }

    @Override
    public void putInput(int rowIndex, int colIndex, double value) {
        input.add(value);
    }

    @Override
    public void putOutput(int value) {
        for(int i=0;i<resultSize;i++){
            result.add(0d);
        }
        result.set(value, 1d);
    }

    @Override
    public List<Double> toInput() {
        return input;
    }

    @Override
    public List<Double> toOutput() {
        return result;
    }
}
