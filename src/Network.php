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
    protected $eta = 0.5;

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

    /**
     * maybe remove the while loop to prevent overfitting
     * @param $input
     * @param $output
     */
    public function train($input, $output)
    {
        $actualOutput = $this->calculate($input);
        while (abs($actualOutput[0] - $output[0]) > 0.001 || abs($actualOutput[1] - $output[1]) > 0.001 || abs($actualOutput[2] - $output[2]) > 0.001) {
            $actualOutput = $this->calculate($input);
            $this->backPropagate($actualOutput, $output);
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

        foreach ($inputVector as $key => $node)
            $this->layers[0]->setNode($key, $node);

        for ($i = 1; $i < count($this->layers); $i++) {
            foreach ($this->layers[$i]->getNodes() as $nodeID => $node) {
                $sum = 0;
                foreach ($this->layers[$i - 1]->getNodes() as $prevNodeID => $prevNode) {
                    $sum += $prevNode * $this->layers[$i - 1]->getWeights()[$prevNodeID][$nodeID];
                }
                $this->layers[$i]->setNode($nodeID, $this->activationFunction->activate($sum));
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
            //added: multiplying derivative of current node to difference
            $this->layers[count($this->layers) - 1]->setError($i,
                $this->activationFunction->derivativeActivate($output[$i]) * ($desiredOutput[$i] - $output[$i]));
        }

        for ($i = count($this->layers) - 2; $i >= 0; $i--) {
            foreach ($this->layers[$i]->getNodes() as $nodeID => $node) {
                $sum = 0;
                foreach ($this->layers[$i + 1]->getNodes() as $predNodeID => $predNode) {
                    $sum += $this->layers[$i + 1]->getError($predNodeID) * $this->layers[$i]->getWeights()[$nodeID][$predNodeID];
                }
                //added: multiplying derivative of current node to difference
                $this->layers[$i]->setError($nodeID, $this->activationFunction->derivativeActivate($node) * $sum);
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


    }

    /**
     * prints layer objects
     */
    public function getStatus()
    {
        print_r($this->layers);
    }
}