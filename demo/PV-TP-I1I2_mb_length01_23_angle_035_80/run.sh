export PYTHONPATH=$PYTHONPATH:/home/ihradis/projects/2016-07-07_JPEG/cnn-image:/home/ihradis/projects/2015-07-29_realCNNDeconvolution/CODE-2/src/

SCRIPT=/home/ihradis/projects/2016-07-07_JPEG/cnn-image/bin/feed_forward_cnn.py

#$SCRIPT -c config.yaml -d deploy.prototxt -cw net_iter_400000.caffemodel -l live.q20
$SCRIPT -c config.yaml -d deploy.prototxt -cw S14_19_real_130000.model -l live.q20