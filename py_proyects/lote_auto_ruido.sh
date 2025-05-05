




#!/bin/bash

bases=("birds" "Corel5k" "emotions" "genbase" "medical" "rcv1subset1" "rcv1subset2" "rcv1subset3" "rcv1subset4" "scene" "yeast")
clasificadores=("MLkNN" "LabelPowerset" "BRkNNaClassifier")
ruidos=("PUMN" "add" "sub" "add-sub" "DAAS")
porcentajes=("0" "10" "20" "30" "40" "50" "60")
umbrales=("0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" )

COMMAND="python3 -u Experiments.py "  # Comando a ejecutar (puede ser un ejecutable o script)


# loop only for swap
for base in "${bases[@]}"
do
    for clasificador in "${clasificadores[@]}"
    do
        for porcentaje in "${porcentajes[@]}"
        do
            echo $COMMAND --bbdd "$base" --case MPG --cls "$clasificador" --noise swap --percen "$porcentaje"  
            $COMMAND --bbdd "$base" --case MPG --cls "$clasificador" --noise swap --percen "$porcentaje"   
        done
    done
done


# Loop for all combinations except swap
for base in "${bases[@]}"
do
    for clasificador in "${clasificadores[@]}"
    do
        for ruido in "${ruidos[@]}"
        do
            for porcentaje in "${porcentajes[@]}"
            do
                for umbral in "${umbrales[@]}"
                do
                    echo $COMMAND --bbdd "$base" --case MPG --cls "$clasificador" --noise "$ruido" --percen "$porcentaje" --prob "$umbral"  
                    $COMMAND --bbdd "$base" --case MPG --cls "$clasificador" --noise "$ruido" --percen "$porcentaje" --prob "$umbral"  
                done
            done
        done
    done
done

echo "Finalizado el procesamiento de todos los archivos."
echo "End of processing all files."


