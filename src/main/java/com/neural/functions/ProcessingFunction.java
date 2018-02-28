package com.neural.functions;

import com.neural.Node;

import java.util.List;

public interface ProcessingFunction<T> {

    public void calculate(List<List<Node>> network);
}
