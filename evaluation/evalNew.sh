set -e
export PYTHONPATH=$PYTHONPATH:/home/ihradis/projects/2016-07-07_JPEG/cnn-image-processing:/home/ihradis/projects/2015-07-29_realCNNDeconvolution/CODE-2/src/

GT_PATH=/home/ihradis/projects/2016-07-07_JPEG/data/orig
SCRIPT_PATH=/home/ihradis/projects/2016-07-07_JPEG/cnn-image/
FCN_SCRIPT=$SCRIPT_PATH/bin/feed_forward_cnn.py
EVAL_SCRIPT_PATH=$SCRIPT_PATH//evaluation/

mkdir -p models
WD=$(pwd)
while true; do
    cd $WD
    MODEL=$(ls | grep caffemodel | sort -V | head -1)
    MODEL=${MODEL/.caffemodel/}
    if [ -n "$MODEL" ]; then
        cd $WD
        for list in *.tst; do
            TEST=${list/.tst/}
            TARGET_DIR=$WD/$TEST/$MODEL

            if [ ! -d "$TARGET_DIR" ]; then
                mkdir -p $TARGET_DIR
                cd $TARGET_DIR
                $FCN_SCRIPT -c $WD/evalConfig.yaml -d $WD/evalDeploy.prototxt -cw $WD/$MODEL.caffemodel -l $WD/$list
                for i in *.png; do
                    mv $i ${i/.jpg_fcn/}
                done
            fi


            cd $WD
            matlab -nodesktop -nojvm -nodisplay -nosplash -r "addpath('$EVAL_SCRIPT_PATH'); evalPSNR('$GT_PATH/$TEST', '$TEST/$MODEL'); exit" | grep PSNR: >> $TEST.results
        done
        mv $MODEL.caffemodel $MODEL.solverstate ./models/
    else
        sleep 60
    fi
done


done
