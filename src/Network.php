<?php
namespace NeuralNetwork;

use NeuralNetwork\ActivationFunction\ActivationFunction;
use NeuralNetwork\Observer\Observer;

class Network
{
    const startNodeValue = 0;
    protected $layers = [];
    protected $weights = [];
    protected $activationFunction;
    protected $eta = 0.001;

    /**
     * Network constructor.
     * @param $nodeCounts
     * @param ActivationFunction $activationFunction
     */
    public function __construct($nodeCounts, ActivationFunction $activationFunction)
    {
        if (!is_array($nodeCounts))
            throw new InvalidArgumentException();

        $this->activationFunction = $activationFunction;
        foreach ($nodeCounts as $key => $layer) {
            $l = new Layer();
            array_push($this->layers, $l);
            for ($i = 0; $i < $layer; $i++)
                $l->addNode(SELF::startNodeValue);
        }

        $this->initWeights();

    }

    /**
     * (not used) attempt to decrease learning rate
     * @param $val
     */
    public function calculateETA($val)
    {
        $this->eta = (exp(-$val)) / 2;
    }

    /**
     * weight between [-0.25,0.25]
     * @return float
     */
    protected function getRandomWeight()
    {
        return ((mt_rand(0, 1000) / 1000) - 0.5) / 2;
    }


    public function trainData($train, $epochs)
    {

        for ($i = 0; $i < $epochs; $i++) {
            $sum_error = 0;
            $eta = $this->eta;
            shuffle($train);
            for ($j = 0; $j < count($train); $j++) {
                $actualOutput = $this->calculate([$train[$j][0], $train[$j][1], $train[$j][2], $train[$j][3]]);
                $sum_error += pow($actualOutput[0] - $train[$j][4], 2) + pow($actualOutput[1] - $train[$j][5], 2) + pow($actualOutput[2] - $train[$j][6], 2);
                $this->backPropagate($actualOutput, [$train[$j][4], $train[$j][5], $train[$j][6]]);
            }
            printf("epoch=%d lrate=%.5f error=%.3f\r\n", $i, $eta, $sum_error);
        }

    }

    /**
     * inits weights
     */
    public function initWeights()
    {
        for ($i = 1; $i < count($this->layers); $i++) {
            $tempWeights = [];
            foreach ($this->layers[$i]->getNodes() as $nodeID => $node) {
                foreach ($this->layers[$i - 1]->getNodes() as $prevNodeID => $prevNode) {
                    $tempWeights[$prevNodeID][$nodeID] = $this->getRandomWeight();
                }
            }
            $this->layers[$i - 1]->setWeights($tempWeights);
        }
    }

    /**
     * feed forward
     * @param $inputVector
     * @return mixed
     */
    public function calculate($inputVector)
    {
        if (!is_array($inputVector))
            throw new \InvalidArgumentException();

        foreach ($inputVector as $key => $node) {
            $this->layers[0]->setNode($key, $node);
            $this->layers[0]->setNodePure($key, $node);
        }

        for ($i = 1; $i < count($this->layers); $i++) {
            foreach ($this->layers[$i]->getNodes() as $nodeID => $node) {
                $sum = 0;
                foreach ($this->layers[$i - 1]->getNodes() as $prevNodeID => $prevNode) {
                    $sum += $prevNode * $this->layers[$i - 1]->getWeights()[$prevNodeID][$nodeID];
                }
                $this->layers[$i]->setNode($nodeID, $this->activationFunction->activate($sum));
                $this->layers[$i]->setNodePure($nodeID, $sum);
            }
        }

        return $this->layers[count($this->layers) - 1]->getNodes();
    }

    /**
     * @param $output
     * @param $desiredOutput
     */
    public function backPropagate($output, $desiredOutput)
    {

        if (count($output) != count($desiredOutput))
            throw new \InvalidArgumentException();


        for ($i = 0; $i < count($output); $i++) {
            $this->layers[count($this->layers) - 1]->setError($i, ($desiredOutput[$i] - $output[$i])
                * $this->activationFunction->derivativeActivate($output[$i]));
        }

        for ($i = count($this->layers) - 2; $i >= 1; $i--) {
            foreach ($this->layers[$i]->getNodes() as $nodeID => $node) {
                $sum = 0;
                foreach ($this->layers[$i + 1]->getNodes() as $predNodeID => $predNode) {
                    $sum += ($this->layers[$i + 1]->getError($predNodeID) * $this->layers[$i]->getWeights()[$nodeID][$predNodeID]);
                }
                $this->layers[$i]->setError($nodeID, $sum * $this->activationFunction->derivativeActivate($node));
            }

        }


        for ($i = 1; $i < count($this->layers); $i++) {
            $tempWeights = [];
            foreach ($this->layers[$i]->getNodes() as $nodeID => $node) {
                foreach ($this->layers[$i - 1]->getNodes() as $prevNodeID => $prevNode) {
                    $tempWeights[$prevNodeID][$nodeID] = $this->layers[$i - 1]->getWeights()[$prevNodeID][$nodeID] +
                        $this->eta * $this->layers[$i]->getError($nodeID) * $prevNode;
                }
            }
            $this->layers[$i - 1]->setWeights($tempWeights);
        }
        $this->eta *= 0.9999999;
    }

    /**
     * prints layer objects
     */
    public function getStatus()
    {
        print_r($this->layers);
    }
}