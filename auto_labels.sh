#command to add a new line anywhere comma is encountered
#sed 's/, /\n/g' your.file

#remove extra files to avoid duplicates or aggregation
rm -f matched.txt
rm -f one_line.txt
rm -f clean_videos.txt
rm -f clean_videos1.txt
rm -f clean_videos2.txt
rm -f clean_videos3.txt
rm -f matched_videos1.txt
rm -f matched_videos2.txt
rm -f final_matched_videos.txt
rm -f final_matched_videos.csv
rm -f final_matched_videos1.txt
rm -f final_matched_videos2.txt
rm -f final_matched_videos3.txt
rm -f cleaned_final_videos.txt

#awk '!a[$0]++' fc7_list_names.txt > fc7_list_names_no_duplicate.txt 

#search for the keyword in the following files and print the whole line
file="final_links.txt"
while IFS= read -r line
do
	#awk 'NR==FNR{a[$1]; next} {for(k in a) if($0 ~ k){print k,$0; delete a[k]}}' final_links.txt video_names.txt
	printf 'vvvvv\n'
	printf '%s' "$line"
	printf ','
	grep "$line" video_effective.txt | sed 's/^.*: //'
	printf ','
	grep "$line" video_exciting.txt | sed 's/^.*: //'
	printf ','
	grep "$line" video_funny.txt | sed 's/^.*: //'
	printf ','
	grep "$line" video_language.txt | sed 's/^.*: //'
	printf ','
	grep "$line" video_sentiments.txt | sed 's/^.*: //'
	printf ','
	grep "$line" video_topics.txt | sed 's/^.*: //'
	printf ','
	grep "$line" fc7_list_names_no_duplicate.txt
	printf ','
	printf 'sssss\n'
done <"$file" > matched.txt

#print only lines which meet these conditions
tr -s '\n' ' ' <matched.txt > one_line.txt
sed 's/vvvvv/\n&/g' one_line.txt > clean_videos.txt
#sed -e 's/[\t ]//g;/^$/d' clean_videos.txt > clean_videos1.txt
sed -e 's/vvvvv//g' clean_videos.txt > clean_videos2.txt
sed -e 's/sssss//g' clean_videos2.txt > clean_videos3.txt
awk '/sda|debaditya/' clean_videos3.txt > matched_videos1.txt

#add headers
{ echo "VIDEO, EFFECTIVENESS, EXCITING, FUNNY, LANGUAGE, SENTIMENTS, TOPICS, PATH"; cat matched_videos1.txt; } > final_matched_videos.txt
#VIDEO,EFFECTIVENESS,EXCITING,FUNNY,LANGUAGE,SENTIMENTS,TOPICS,PATH

#remove all spaces from the file
sed -r 's/\s+//g' final_matched_videos.txt > final_matched_videos1.txt
#remove all double quotes and double commas
sed 's/"//g' final_matched_videos1.txt > final_matched_videos2.txt
sed 's/,\{2,\}/,/g' final_matched_videos2.txt > final_matched_videos3.csv
#sed -i 's/.$//' final_matched_videos3.txt > final_matched_videos4.txt
#replace spaces with commas
#tr ' ' ',' < final_matched_videos.txt > final_matched_videos2.csv
#sed 's/ \{1,\}/,/g' final_matched_videos.txt > final_matched_videos3.csv

#remove extra files
rm -f matched.txt
rm -f one_line.txt
rm -f clean_videos.txt
rm -f clean_videos1.txt
rm -f clean_videos2.txt
rm -f clean_videos3.txt
rm -f matched_videos1.txt
rm -f final_matched_videos.txt
rm -f final_matched_videos.csv
rm -f final_matched_videos1.txt
rm -f final_matched_videos2.txt
rm -f cleaned_final_videos.txt