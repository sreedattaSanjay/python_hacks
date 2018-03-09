find . -type f -name '*.mp4' | while read mp4; do
    b=`basename $mp4 .mp4`
    d=`dirname $mp4`
    ./DenseTrackStab "$b.mp4" | gzip > "$b.gz"
done
