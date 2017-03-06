<?php
/**
 * @project: perceptron
 * @package: NeuralNetwork
 * @author: Tom Ganz
 * @date: 23.01.2017
 */

namespace NeuralNetwork;


use InvalidArgumentException;

class Layer
{
    protected $nodes = [];
    protected $weights = [];
    protected $errorSignals = [];
    protected $nodesPure = [];
    /**
     * @return mixed
     */
    public function getNodes()
    {
        return $this->nodes;
    }
    public function getNodesPure()
    {
        return $this->nodesPure;
    }
    public function setError($i,$val){
        if(!is_numeric($val))
            throw new InvalidArgumentException();
        $this->errorSignals[$i] = $val;
    }
    public function getError($i){
        return $this->errorSignals[$i];
    }

    /**
     * @return array
     */
    public function getErrorSignals()
    {
        return $this->errorSignals;
    }

    /**
     * @param array $errorSignals
     */
    public function setErrorSignals($errorSignals)
    {
        $this->errorSignals = $errorSignals;
    }

    /**
     * @param mixed $nodes
     */
    public function setNodes($nodes)
    {
        $this->nodes = $nodes;
    }
    public function setNode($i,$val){
        if(!is_numeric($val))
            throw new InvalidArgumentException();
        $this->nodes[$i] = $val;
    }
    public function setNodePure($i,$val){
        if(!is_numeric($val))
            throw new InvalidArgumentException();
        $this->nodesPure[$i] = $val;
    }
    public function addNode($node)
    {
        array_push($this->nodes, $node);
    }

    /**
     * @return array
     */
    public function getWeights()
    {
        return $this->weights;
    }

    /**
     * @param array $weights
     */
    public function setWeights($weights)
    {
        $this->weights = $weights;
    }


}