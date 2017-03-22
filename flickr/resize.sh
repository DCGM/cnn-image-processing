ORIGIN=./large/

for size in 1200 1000 0833 0700 0578 0482 0400 0334 0279 0232
do  
    mkdir -p $size
    mogrify -resize @${size}000 -format jpg -quality 98 -path ./$size/ -verbose $ORIGIN/*.jpg
done
