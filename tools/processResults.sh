for metric in loss mse psnr
do
    for set in TEXT #skyscraper Urban100 BSDS500 LIVE1 Holidays 10kUSAdultDatabase;
    do
        for net in */
        do
            net=${net/\//}
            echo $net $set $metric $( cat ${net}/*.log | grep $set | grep $metric | tail -n 3 | gawk '{sum+=$NF} END{print sum / NR}' )
        done
    done
done
