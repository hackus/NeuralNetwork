package com.neural;

import com.neural.functions.ActivationFunction;
import com.neural.mnist.NeuralType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toList;
import static org.assertj.core.api.Assertions.assertThat;

public class NeuralNetwork {

    final static Logger logger = LoggerFactory.getLogger(NeuralNetwork.class);

    Random r = new Random();
    double rangeMin = 0d;
    double rangeMax = 0.00001;

    List<List<Node>> network = new ArrayList<>();

    public NeuralNetwork() {}

    public void buildNetwork(List<NeuralLayer> layers){
        layers.stream().forEach(layer -> {
            addLayer(layer);
        });
    }

    public double getNextDouble(){
       return rangeMin + (rangeMax - rangeMin) * r.nextDouble();
    }

    public void feedForward(NeuralType image) {
        List<Double> values = image.toInput();
        if(values.size() != getFirstLayer().size()){
            String strError = "Input data is out of bound.";
            logger.error(strError);
            throw new RuntimeException(strError);
        }
        List<Node> layer = getFirstLayer();
        for(int i=0;i<values.size();i++){
            layer.get(i).setValue(values.get(i));
        };
    }

    public void addLayer(NeuralLayer layer){
        if(network.size() == 0){
            network.add(buildDefaultLayer(layer.numberOfItems, layer.activationFunction));
        } else {
            int layerIndex = network.size();
            network.add(buildIntermediateLayer(layer.numberOfItems, layer.activationFunction, getOutputLayer(), layerIndex));
        }
    }

    private List<Node> buildIntermediateLayer(int numberOfItems, ActivationFunction<Double> activationFunction, List<Node> previousLayer, int layerIndex) {
        List<Node> newLayer = buildLayer(numberOfItems, activationFunction, layerIndex);

        int newNodeIndex = 0;
        for(Node newNode : newLayer){
            List<NeuralLink> list = new ArrayList<>();
            for(Node prevNode : previousLayer){
                NeuralLink link = new NeuralLink(
                    getLastLayerIndex()
                    , list.size()
                    , prevNode.getId()
                    , network.size()
                    , newNodeIndex
                    , newNode.getId()
                    , Math.random());
                list.add(link);
                prevNode.getForwardLinks().add(link);
            };
            newNodeIndex++;
            newNode.setBackwardLinks(list);
        };
        return newLayer;
    }

    public List<Node> buildDefaultLayer(int n, ActivationFunction<Double> activationFunction) {
        return IntStream.range(0, n)
            .mapToObj(i -> {
                return Node.buildDefaultNode(Math.random()
                    , activationFunction
                    , 0
                    , 0d);
            })
            .collect(toList());
    }

    public List<Node> buildLayer(int n, ActivationFunction<Double> activationFunction, int layerIndex) {
        return IntStream.range(0, n)
                .mapToObj(i -> {
                    return Node.buildNode(activationFunction
                        , layerIndex
                        , Math.random());
                })
                .collect(toList());
    }

    public int getCurrentIteration(){
        return network.get(0).get(0).getCurrentIteration() == 0 ? 1 : 0;
    }

    public void runNetworkForward(NeuralType image){
        feedForward(image);
        int currentIteration = getCurrentIteration();
        for(int i=0;i<network.size();i++) {
            List<Node> layer = network.get(i);
            for(Node node : layer){
                updateNodeValue(node, currentIteration);
                node.updateActivatedValue();
            };
        }
    }

    public void updateNodeValue(Node node, int currentIteration){
        if(currentIteration == node.getCurrentIteration()) return;
        node.setCurrentIteration(currentIteration);

        if(node.getBackwardLinks().size() != 0){
            double calculatedValue = 0d;
            for(NeuralLink link : node.getBackwardLinks()){
                calculatedValue += getBackwardNodeActivatedValueByLink(link);
            };
            node.setValue((calculatedValue + node.getBias()));
        }
    }

    public Double getBackwardNodeActivatedValueByLink(NeuralLink link){
        Node parentNode = getBackwardNodeByLink(link);
        return parentNode.getActivatedValue() * link.getWeight();
    }

    public void runNetworkBackward(List<Double> expectedOutput){
        applyErrorOnTheLastLayer(expectedOutput);
        for(int i=getLastLayerIndex()-1;i>=0;i--){
            for(Node node : getLayer(i)){
                node.getError().setError(node, this);
                updateBias(node);
                updateWeights(node);
            };
        }
    }

    public void updateBias(Node node) {
        if(node.getLayerId()!=0) {
            node.setBias(node.getError().getCalculatedBias(node));
        }
    }

    public void applyErrorOnTheLastLayer(List<Double> expectedOutput){
        List<Node> outputLayer = getOutputLayer();
        for(int i=0;i<outputLayer.size();i++){
            Double target = expectedOutput.get(i);
            Node node = outputLayer.get(i);
            node.getError().setErrorOnLastLayer(target, node);
            updateBiasOnLastLayer(node);
            updateWeights(node);
        };
    }

    public void updateBiasOnLastLayer(Node node) {
        node.setBias(node.getError().getCalculatedBias(node));
    }

    public void updateWeights(Node node){
        for(NeuralLink link : node.getBackwardLinks()){
            link.setWeight(node.getError().getCalculatedWeigth(link,this));
        };
    }

    public Node getBackwardNodeByLink(NeuralLink link){
        return network.get(link.getParentListIndex()).get(link.getParentItemIndex());
    }

    public Node getChildNodeByLink(NeuralLink link){
        return network.get(link.getChildListIndex()).get(link.getChildItemIndex());
    }

    public void learn(List<NeuralType> trainData, List<NeuralType> testData, int epochs) {
        for(int i=0;i<epochs;i++) {
            AtomicLong start = new AtomicLong(System.nanoTime());
            for(int j=0;j<trainData.size();j++){
                NeuralType image = trainData.get(j);
                runNetworkForward(image);
                runNetworkBackward(image.toOutput());
                if(j>0 && j % 1000 == 0){
                    long end = System.nanoTime();
                    logger.debug(String.format("epoch %d image index %d, elapsed time %d", i, j, TimeUnit.MILLISECONDS.convert(end - start.get(), TimeUnit.NANOSECONDS)));
//                    print();
                    start.set(System.nanoTime());
                }
            };
            int numberOfFoundImages = evaluate(testData);
            logger.debug(String.format("number of found images: %d", numberOfFoundImages));
        }
    }

    public int evaluate(List<NeuralType> testData) {
        int foundItems = 0;
        assertThat(testData.size() == 10000);
        for(int i=0;i<testData.size();i++) {
            NeuralType obj = testData.get(i);
            runNetworkForward(obj);
            if(compare(obj.toOutput(), getNetworkOutput())){
                foundItems++;
            }
        };
        return foundItems;
    }

    public boolean compare(List<Double> expected, List<Double> output){
        if(expected.size() != output.size()){
            String strError = String.format("Expected size %d not equal to output size %d", expected.size());
            logger.error(strError);
            throw new RuntimeException(strError);
        }
        int maxExpected = getIndexOfMaxValue(expected);
        int maxOutput = getIndexOfMaxValue(output);
        return maxExpected==maxOutput;
    }

    public int getIndexOfMaxValue(List<Double> lst){
        int maxIndex = 0;
        for(int i=1;i<lst.size();i++){
            if(lst.get(maxIndex).compareTo(lst.get(i)) < 0){
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public List<Double> getNetworkOutput(){
        List<Double> output = new ArrayList<>();
        List<Node> layer = getOutputLayer();
        for(int i=0;i<layer.size();i++){
            output.add(layer.get(i).getActivatedValue());
        }
        return output;
    }

    public List<Node> getFirstLayer(){
        return getLayer(0);
    }

    public List<Node> getOutputLayer(){
        return getLayer(getLastLayerIndex());
    }

    public int getLastLayerIndex(){
        return network.size()-1;
    }

    public List<Node> getLayer(int index){
        return network.get(index);
    }

    public void printWeights(){
        logger.debug("print weights___________________________________");


        for(int i=0;i<network.size();i++){
            StringBuffer buffer = new StringBuffer("{");
            buffer.append("\n");
            List<Node> layer = network.get(i);
            buffer.append("Layer " + i);
            buffer.append("\n");
            for(int j=0;j<layer.size();j++) {
                Node node = layer.get(j);
                buffer.append("Node " + j);
                buffer.append("\n");
                for(int k=0;k<node.getForwardLinks().size();k++){
                    NeuralLink link = node.getForwardLinks().get(k);
                    buffer.append(String.format("w(%d)=%s,",k,link.getWeight().toString()));
                }
                buffer.append("\n");
            }
            buffer.append("}\n");
            logger.debug(buffer.toString());
        }
    }

    public void print(){
        logger.debug("print network___________________________________");

        for(int i=1;i<network.size();i++){
            List<Node> layer = network.get(i);
            StringBuffer buffer = new StringBuffer("{");
            for(int j=0;j<layer.size();j++){
                Node node = layer.get(j);
                buffer.append(String.format("Layer index %d, Node index %d, Node value %s, Node activatedValue %s, bias %s",i,j,node.getValue().toString(),node.getActivatedValue().toString(), node.getBias().toString()));

                buffer.append("\n");
//                for(NeuralLink link : node.getBackwardLinks()){
//                    buffer.append(String.format("    link list index %d, item index %d, weight %f", link.getParentListIndex(), link.getParentItemIndex(), link.getWeight()));
//                    buffer.append("\n");
//                };
            };
            buffer.append("}\n");
            logger.debug(buffer.toString());
        };

        logger.debug("");
    }
}

