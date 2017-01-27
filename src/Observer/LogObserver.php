<?php
/**
 * Created by PhpStorm.
 * User: Tom
 * Date: 22.01.2017
 * Time: 18:48
 */

namespace NeuralNetwork\Observer;


class LogObserver implements Observer
{

    public function listen($event, $data)
    {
        printf("%s => %s\n", $event, $data);
    }
}