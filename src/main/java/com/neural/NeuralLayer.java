package com.neural;

import com.neural.functions.ActivationFunction;

public class NeuralLayer {
    int numberOfItems;
    ActivationFunction<Double> activationFunction;

    public NeuralLayer(int numberOfItems, ActivationFunction<Double> activationFunction){
        this.numberOfItems = numberOfItems;
        this.activationFunction = activationFunction;
    }
}
