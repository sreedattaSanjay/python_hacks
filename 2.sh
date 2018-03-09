for DIR in $(ls -d */)
do
    FIRSTFILE=$(ls $DIR | sort -n | head -1)
    STARTNO=$(echo $FIRSTFILE | grep -Eo "[0-9]+")
    DIGITNO=3
    IMGSEQ=$(echo ${FIRSTFILE%.*} | grep -iEo "[a-z]+")
    EXTEN=${FIRSTFILE#*.}
    OUTPUT=${DIR%/}
    ffmpeg -i "$DIR$OUTPUT"-%0"$DIGITNO"d".$EXTEN" $OUTPUT.mp4
done

#ffmpeg -i "adl-02-cam0-d/adl-02-cam0-d-%03d.png" adl-02-cam0-d.mp4fmpeg -i "adl-02-cam0-d/adl-02-cam0-d-%03d.png" adl-02-cam0-d.mp4