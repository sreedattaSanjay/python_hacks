clc;
clear all;
close all;

files_path = '/home/cs16mtech11021/fall_detection/depth_files/tst_1/'
files = dir(files_path)
dirFlags = [files.isdir]
% Extract only those that are directories.
subFolders = files(dirFlags)
% Print folder names to command window.
for k = 3 : length(subFolders)
	fprintf('Sub Folder #%d = %s\n', k, subFolders(k).name);
    subf = strcat(files_path, subFolders(k).name)
    files_2 = dir(subf)    
    for i = 3 : length(files_2)
        cd(subf)
        subfilename = files_2(i).name
        fprintf('Bin File #%d = %s\n', k, subFolders(k).name);
        pathDepth = subfilename;
        rowPixel = 240;
        columnPixel = 320;
        M = zeros(rowPixel,columnPixel);
        fid = fopen(pathDepth);
        arrayFrame = fread(fid,'uint16');
        fclose(fid);
        for r=1:rowPixel
            M(r,:) = arrayFrame((r-1)*columnPixel+1:r*columnPixel);
            imwrite(M,sprintf('%d.png',i))
            %figure;imagesc(M);
        end
        cd(files_path)
    end
    
end