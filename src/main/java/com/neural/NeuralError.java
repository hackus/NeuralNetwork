package com.neural;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NeuralError {

    final static Logger logger = LoggerFactory.getLogger(NeuralError.class);

    Double LEARNING_RATE = 0.5d;
    Double totalToOut;
    Double outToNet;
    Double errorCost;
    Double coeficient;

    public NeuralError(Double totalToOut, Double outToNet){
        this.totalToOut = totalToOut;
        this.outToNet = outToNet;
        this.errorCost = totalToOut * outToNet;
        this.coeficient = 1d;
    }

    public Double getCalculatedWeigth(NeuralLink backwardLink, NeuralNetwork network){
        return backwardLink.getWeight() - coeficient * network.getBackwardNodeByLink(backwardLink).getActivatedValue();
    }

    public Double getCalculatedBias(Node node){
        return node.getBias() - coeficient;
    }

    public void setError(Node node, NeuralNetwork network){
        Double nodeError = 0d;
        for (NeuralLink forwardLink : node.getForwardLinks()) {
            nodeError += network.getChildNodeByLink(forwardLink).getError().getErrorCost() * forwardLink.getWeight();
        }
        totalToOut = nodeError;

        Double out = node.getActivatedValue();
        outToNet = out * (1-out);

        if(out.equals(1)){
            logger.debug("************************************");
            logger.debug("out value is 1");
            logger.debug("************************************");
        }

        errorCost = totalToOut*outToNet;
        coeficient = LEARNING_RATE * errorCost;
    }

    public void setErrorOnLastLayer(Double target, Node node){
        Double out = node.getActivatedValue();
        totalToOut = -(target-out);

        outToNet = out * (1-out);
        if(out.equals(1)){
            logger.debug("************************************");
            logger.debug("out value is 1");
            logger.debug("************************************");
        }

        errorCost = totalToOut*outToNet;
        coeficient = LEARNING_RATE * errorCost;
    }

    public Double getErrorCost() {
        return errorCost;
    }
}
