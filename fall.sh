#!/bin/bash

# NAME:         get_all_media.sh
# AUTHOR:       crashhacker
# LICENSE:      bash get_all_media.sh www.website.com

WEBSITE="$1"
echo "Getting jpg list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.jpg$" | awk '{print $2}' | tee jpg_pic_links.txt
echo "Downloading jpg_pics..."    
wget -nc -P ./jpg_pics -i jpg_pic_links.txt

WEBSITE="$1"
echo "Getting png list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.png$" | awk '{print $2}' | tee png_pic_links.txt
echo "Downloading... png_pics..."    
wget -nc -P ./png_pics -i png_pic_links.txt

WEBSITE="$1"
echo "Getting gif list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.gif$" | awk '{print $2}' | tee gif_pic_links.txt
echo "Downloading....gif_pics..."    
wget -nc -P ./gif_pics -i gif_pic_links.txt

WEBSITE="$1"
echo "Getting webm list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.webm$" | awk '{print $2}' | tee webm_links.txt
echo "Downloading...webms..."    
wget -nc -P ./webms -i webm_links.txt

WEBSITE="$1"
echo "Getting mp4 list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.mp4$" | awk '{print $2}' | tee mp4_links.txt
echo "Downloading..mp4s..."    
wget -nc -P ./mp4s -i mp4_links.txt

WEBSITE="$1"
echo "Getting mkv list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.mkv$" | awk '{print $2}' | tee mkv_links.txt
echo "Downloading..mkv_files..."    
wget -nc -P ./mkvs -i mkv_links.txt

WEBSITE="$1"
echo "Getting pdf list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.pdf$" | awk '{print $2}' | tee pdf_links.txt
echo "Downloading..pdfs..."    
wget -nc -P ./pdfs -i pdf_links.txt

WEBSITE="$1"
echo "Getting text list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.txt$" | awk '{print $2}' | tee text_links.txt
echo "Downloading..text_files..."    
wget -nc -P ./txts -i text_links.txt

WEBSITE="$1"
echo "Getting html list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.html$" | awk '{print $2}' | tee html_links.txt
echo "Downloading..html_files..."    
wget -nc -P ./htmls -i html_links.txt

WEBSITE="$1"
echo "Getting ppt list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.ppt$" | awk '{print $2}' | tee ppt_links.txt
echo "Downloading..ppt_files..."    
wget -nc -P ./ppts -i ppt_links.txt

WEBSITE="$1"
echo "Getting pptx list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.pptx$" | awk '{print $2}' | tee pptx_links.txt
echo "Downloading..pptx_files..."    
wget -nc -P ./pptxs -i pptx_links.txt

WEBSITE="$1"
echo "Getting zip list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.zip$" | awk '{print $2}' | tee zip_links.txt
echo "Downloading..zip_files..."    
wget -nc -P ./zips -i zip_links.txt

WEBSITE="$1"
echo "Getting tar list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.tar$" | awk '{print $2}' | tee tar_links.txt
echo "Downloading..tar_files..."    
wget -nc -P ./tars -i tar_links.txt

WEBSITE="$1"
echo "Getting rar list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.rar$" | awk '{print $2}' | tee rar_links.txt
echo "Downloading..rar_files..."    
wget -nc -P ./rars -i rar_links.txt

WEBSITE="$1"
echo "Getting csv list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.csv$" | awk '{print $2}' | tee csv_links.txt
echo "Downloading..csv_files..."    
wget -nc -P ./csvs -i csv_links.txt

WEBSITE="$1"
echo "Getting matlabfiles list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.m$" | awk '{print $2}' | tee matlab_links.txt
echo "Downloading..matlab_files..."    
wget -nc -P ./matlabs -i matlab_links.txt

WEBSITE="$1"
echo "Getting python list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.py$" | awk '{print $2}' | tee python_links.txt
echo "Downloading..matlab_files..."    
wget -nc -P ./pythons -i python_links.txt

WEBSITE="$1"
echo "Getting c list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.c$" | awk '{print $2}' | tee c_links.txt
echo "Downloading..c_files..."    
wget -nc -P ./cs -i c_links.txt

WEBSITE="$1"
echo "Getting cpp list..."
lynx -cache=0 -dump -listonly "$WEBSITE" | grep ".*\.cpp$" | awk '{print $2}' | tee cpp_links.txt
echo "Downloading..cpp_files..."    
wget -nc -P ./cpps -i cpp_links.txt