<?php
namespace NeuralNetwork\ActivationFunction;
/**
 * Created by PhpStorm.
 * User: Tom
 * Date: 22.01.2017
 * Time: 17:28
 *
 * Taken from https://github.com/dkriesel/snipe/blob/master/com/dkriesel/snipe/neuronbehavior/TangensHyperbolicusAnguita.java
 */
class TangensHyperbolicusAnguita implements ActivationFunction
{

    public function activate($value)
    {
        if ($value > 0) {
            if ($value <= 1.92033) {
                return 0.96016 - 0.26037 * ($value - 1.92033) * ($value - 1.92033);
                // c-a*(x-b)^2
            } else {
                return 0.96016;
            }
        } else {
            if ($value >= -1.92033) {
                return 0.26037 * ($value + 1.92033) * ($value + 1.92033) - 0.96016;
            } else {
                return -0.96016;
            }
        }
    }

    public function derivativeActivate($value)
    {
        if ($value > 0) {
            if ($value <= 1.92033) {
                return -2 * 0.26037 * $value + 2 * 0.26037 * 1.92033;
            } else {
                return 0.0781;
            }
        } else {
            if ($value >= -1.92033) {
                return 2 * 0.26037 * $value + 2 * 0.26037 * 1.92033;
            } else {
                return 0.0781;
            }
        }
    }
}