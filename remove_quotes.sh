sed "s/'/ /g" file.csv > file_new.txt
sed 's/\"//g' file_new.txt > final_file.txt
sed -r 's/\s+//g' final_file.txt > final_links.txt
awk '$0="https://www.youtube.com/watch?v="$0' final_file.txt > youtube_links.txt
sed -r 's/\s+//g' youtube_links.txt > youtube_links_updated.txt
youtube-dl -a youtube_links_updated.txt --ignore-errors -u crashhacker09@gmail.com -p Vishnu@19