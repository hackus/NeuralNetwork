package com.neural;

import com.neural.functions.ActivationFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class Node {

    private static AtomicInteger sequence = new AtomicInteger(Integer.MIN_VALUE);
    private Integer id = sequence.incrementAndGet();
    private Double value;
    private Double activatedValue = null;
    private ActivationFunction<Double> activationFunction;
    private int currentIteration;
    private Double bias;
    private NeuralError error = new NeuralError(0d,0d);
    private int layerId;

    private List<NeuralLink> backwardLinks = new ArrayList<>();
    private List<NeuralLink> forwardLinks = new ArrayList<>();

    private Node(ActivationFunction<Double> activationFunction, int layerId, Double bias) {
        this.activationFunction = activationFunction;
        this.bias = bias;
        this.layerId = layerId;
        if(layerId==0){
            activatedValue = value;
        }
    }

    private Node(Double value, ActivationFunction<Double> activationFunction, Integer layerId, Double bias) {
        this(activationFunction, layerId, bias);
        this.value = value;
    }

    public static Node buildDefaultNode(Double value, ActivationFunction<Double> activationFunction, int layerId, Double bias) {
        return new Node(value, activationFunction, layerId, bias);
    }

    public static Node buildNode(ActivationFunction<Double> activationFunction, int layerId, Double bias) {
        return new Node(activationFunction, layerId, bias);
    }

    public Integer getId() {
        return id;
    }

    public void setBackwardLinks(List<NeuralLink> backwardLinks) {
        this.backwardLinks = backwardLinks;
    }

    public Double getValue() {
        return value;
    }

    public Double getActivatedValue() {
        return activatedValue;
    }

    public ActivationFunction<Double> getActivationFunction() {
        return activationFunction;
    }

    public List<NeuralLink> getBackwardLinks() {
        return backwardLinks;
    }

    public void setValue(Double value) {
        this.value = value;
    }

    public void setCurrentIteration(int currentIteration) {
        this.currentIteration = currentIteration;
    }

    public void setActivatedValue(Double activatedValue) {
        this.activatedValue = activatedValue;
    }

    public void updateActivatedValue() {
        setActivatedValue(getActivationFunction().calculate(value));
    }

    public int getLayerId(){
        return layerId;
    }

    public List<NeuralLink> getForwardLinks() {
        return forwardLinks;
    }

    public Double getBias() {
        return bias;
    }

    public void setBias(Double bias) {
        this.bias = bias;
    }

    public NeuralError getError(){
        return error;
    }

    public int getCurrentIteration(){
        return currentIteration;
    }
}
