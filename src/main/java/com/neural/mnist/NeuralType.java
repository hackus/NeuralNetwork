package com.neural.mnist;

import java.util.List;

public interface NeuralType {
    void setInput(List<Double> input);

    void setResult(List<Double> result);

    void putInput(int rowIndex, int colIndex, double value);

    void putOutput(int value);

    List<Double> toInput();

    List<Double> toOutput();
}
