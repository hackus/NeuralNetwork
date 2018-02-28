package com.neural;

import java.util.List;

public class NeuralLink {
    int parentListIndex;
    int parentItemIndex;
    Integer parentId;
    int childListIndex;
    int childItemIndex;
    Integer childId;
    private Double weight;

    public NeuralLink(int parentListIndex,
                      int parentItemIndex,
                      int parentId,
                      int childListIndex,
                      int childItemIndex,
                      Integer childId,
                      Double weight){
        this.parentListIndex = parentListIndex;
        this.parentItemIndex = parentItemIndex;
        this.parentId = parentId;
        this.childListIndex = childListIndex;
        this.childItemIndex = childItemIndex;
        this.childId = childId;
        this.weight = weight;
    }

    public Integer getParentId() {
        return parentId;
    }

    public Double getWeight() {
        return weight;
    }

    public int getParentItemIndex() {
        return parentItemIndex;
    }

    public int getParentListIndex() {
        return parentListIndex;
    }

    public void setWeight(Double weight) {
        this.weight = weight;
    }

    public Node getNode(List<List<Node>> network){
        return network.get(getParentListIndex()).get(parentItemIndex);
    }

    public int getChildListIndex() {
        return childListIndex;
    }

    public void setChildListIndex(int childListIndex) {
        this.childListIndex = childListIndex;
    }

    public int getChildItemIndex() {
        return childItemIndex;
    }
}
