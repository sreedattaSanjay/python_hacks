%Place this code in the same folder of the database (Data1,Data2,...Data11)
clear all;close all;clc;

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % Choose:
startFoldIdx = 1;   % Index of the first test folder to load (1->'Data1')
stopFoldIdx = 1;    % Index of the last test folder to load (11->'Data11')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

%The script loads:
% - PckVal_*:       packet from wearable device
% - TStamp_QPC_*:   timestamp assigned by the PC to the packet, when it arrives to the PC [us] 
% - TStamp_Int_*:   internal timestamp of the wearable device [ms]
% - *_acc_raw_*:    acceleration data
% - M:              depth frame
% - KinectTime:     timestamps assigned by the PC to the frame, when it is available to the PC [100ns]/[us]
% - jMatDep:        skeleton joints in depth space
% - jMatSkl:        skeleton joints in skeleton space
% - KinectTimeBody: timestamps assigned by the PC to the skeleton, when it is available to the PC [100ns]/[us]


ADLFolderName = {'sit','grasp','walk','lay'};
FallFolderName = {'front','back','side','EndUpSit'};
%*****Wearable device*****
device1 = '35EE'; %Number of device1
device2 = '36F9'; %Number of device2

%*******Kinect - V2*******
rowPixel = 424; %[pixel] number of row
columnPixel = 512;  %[pixel] number of column
frameIdx = 1; %depth frame number

% Load selected folders
for idx_folder = startFoldIdx:stopFoldIdx
    for groupName = {'ADL' 'Fall'}
        if strcmp(groupName,'ADL')
            subfolder = ADLFolderName;
        else
            subfolder = FallFolderName;
        end
        for name_Subfolder = subfolder
            for idx_test = 1:3
                Folder = strcat('Data',num2str(idx_folder),'/',cell2mat(groupName),'/',cell2mat(name_Subfolder),'/',num2str(idx_test)); %Folder where are stored the data
                
                %*************************
                %Load wearable device data
                %*************************
                %**35EE**
                PckVal_35EE = loadPackets(device1,Folder);
                PckValSize_35EE = size(PckVal_35EE);
                TStamp_QPC_35EE = csvread(strcat(Folder,'/Time/TimeStamps',device1,'.csv'));
                TStamp_Int_35EE = (PckVal_35EE(:,3)+256.*PckVal_35EE(:,4)+256.*256.*PckVal_35EE(:,5));
                %accelerometer values
                X_acc_raw_35EE = PckVal_35EE(:,8)+256*PckVal_35EE(:,9);
                Y_acc_raw_35EE = PckVal_35EE(:,10)+256*PckVal_35EE(:,11);
                Z_acc_raw_35EE = PckVal_35EE(:,12)+256*PckVal_35EE(:,13);
                %36F9
                PckVal_36F9 = loadPackets(device2,Folder);
                PckValSize_36F9 = size(PckVal_36F9);
                TStamp_QPC_36F9 = csvread(strcat(Folder,'/Time/TimeStamps',device2,'.csv'));
                TStamp_Int_36F9 = (PckVal_36F9(:,3)+256.*PckVal_36F9(:,4)+256.*256.*PckVal_36F9(:,5));
                %accelerometer values
                X_acc_raw_36F9 = PckVal_36F9(:,8)+256*PckVal_36F9(:,9);
                Y_acc_raw_36F9 = PckVal_36F9(:,10)+256*PckVal_36F9(:,11);
                Z_acc_raw_36F9 = PckVal_36F9(:,12)+256*PckVal_36F9(:,13);
                
                %*************************
                %****Load depth frame*****
                %*************************
                fid = fopen(strcat(Folder,'/Depth/Filedepth_',num2str(frameIdx-1),'.bin'));
                arrayFrame = fread(fid,'uint16');
                fclose(fid);
                M = zeros(rowPixel, columnPixel);
                for r=1:rowPixel
                    M(r,:) = arrayFrame((r-1)*columnPixel+1:r*columnPixel);
                end
                imwrite(M,sprintf('%d.png',frameIdx))
                frameIdx = frameIdx + 1
                %Load time information
                KinectTime = csvread(strcat(Folder,'/Time/DepthTime.csv'));
                
                %*************************
                %******Load skeleton******
                %*************************
                fileNameSk1DS = strcat(Folder,'/Body','/Fileskeleton.csv'); %joint in the depth frame
                fileNameSk1SS = strcat(Folder,'/Body','/FileskeletonSkSpace.csv'); %joint in 3D space
                Sk1SkDepth = csvread(fileNameSk1DS);
                Sk1SkSpace = csvread(fileNameSk1SS);
                %Find player number
                for idx_player = 1:6
                    if Sk1SkDepth(25*(idx_player-1)+1,1) ~= 0
                        break;
                    end
                end
                %Find row index of the specific player in Sk1SkDepth
                row_idx = find(Sk1SkDepth(:,5) == idx_player-1);
                Sk1SkDepth = Sk1SkDepth(row_idx,:);
                NumFrameSkelDepth = fix(length(Sk1SkDepth(:,1))/25);
                Sk1SkSpace = Sk1SkSpace(row_idx,:);
                NumFrameSkelSpace = fix(length(Sk1SkSpace(:,1))/25);
                %restore 25 joints in groups 
                jMatDep = zeros(25, 3, NumFrameSkelDepth);
                jMatSkl = zeros(25, 3, NumFrameSkelSpace);
                for n = 1:NumFrameSkelDepth
                    jMatDep(:,:,n) = Sk1SkDepth(((n-1)*25+1):n*25,1:3);
                end
                for n = 1:NumFrameSkelSpace
                    jMatSkl(:,:,n) = Sk1SkSpace(((n-1)*25+1):n*25,1:3);
                end
                %Load time information
                KinectTimeBody = csvread(strcat(Folder,'/Time/BodyTime.csv'));
                
                % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
                % % Put here your code! 
                % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
                
                % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
                % Plot raw Acceleration data of each test             
%                 figure; 
%                 subplot 121; title('35EE'); hold on;
%                 plot(X_acc_raw_35EE,'b'); plot(Y_acc_raw_35EE,'r'); plot(Z_acc_raw_35EE,'k');
%                 subplot 122; title('36F9'); hold on;
%                 plot(X_acc_raw_36F9,'b'); plot(Y_acc_raw_36F9,'r'); plot(Z_acc_raw_36F9,'k');
%                 
                % Plot Depth and Skeleton data of each test
%                 figure;
%                 subplot 131;
%                 imagesc(M); title('depth frame');
%                 subplot 132;  hold on;
%                 plot3(jMatSkl(:,1,1),jMatSkl(:,3,1),jMatSkl(:,2,1),'.r','markersize',20); view(0,0); axis equal;
%                 title('skeleton in skeleton space');
%                 subplot 133;
%                 plot3(jMatDep(:,1,1),jMatDep(:,3,1),jMatDep(:,2,1),'.b','markersize',20); view(0,0); axis equal; set(gca,'ZDir','Reverse');
%                 title('skeleton in depth space');
                % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
            end
        end
    end
end