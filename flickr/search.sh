call=flickr.photos.search
key="5ea6c4c69dacd0c9eab3c285a8d226e6"
sig="520d57b564e636a8"
text1="api_key"
text2="method$call"
md5=`echo -n $sig$text1$key$text2 | md5sum | sed s/\ \ -//`
curl -F text="cars" -F method=$call -F api_key=$key -F api_sig=$md5 https://flickr.com/services/rest
