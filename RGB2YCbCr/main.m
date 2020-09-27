clc;
clear all;

%% source images 2 Y channel
% run it to generate the data (Y channel) input into the network, otherwise comment this code.
% for num=1:100
%   I_near=(imread(strcat('.\Test_near\',num2str(num),'.jpg'))); 
%   I_far=(imread(strcat('.\Test_far\',num2str(num),'.jpg'))); 
%   
%   [Y_near,Cb_near,Cr_near]=RGB2YCbCr(I_near); 
%   [Y_far,Cb_far,Cr_far]=RGB2YCbCr(I_far); 
%   imwrite(Y_near, strcat('.\Test_near\',num2str(num),'.jpg'));
%   imwrite(Y_far, strcat('.\Test_far\',num2str(num),'.jpg'));
% end

%% Restore the output of the network to RGB
  for i=1:40
      I_result=double(imread(strcat('.\epoch\',num2str(i),'.jpg')));
      I_init_near=double(imread(strcat('.\source\Near\',num2str(i),'.jpg')));
      I_init_far=double(imread(strcat('.\source\Far\',num2str(i),'.jpg')));    
      [Y1,Cb1,Cr1]=RGB2YCbCr(I_init_near);
      [Y2,Cb2,Cr2]=RGB2YCbCr(I_init_far);  
      [H,W]=size(Cb1);
      Cb=ones([H,W]);
      Cr=ones([H,W]);
      
      for k=1:H
          for n=1:W
           if (abs(Cb1(k,n)-128)==0&&abs(Cb2(k,n)-128)==0)  
              Cb(k,n)=128;
           else
                middle_1= Cb1(k,n)*abs(Cb1(k,n)-128)+Cb2(k,n)*abs(Cb2(k,n)-128);
                middle_2=abs(Cb1(k,n)-128)+abs(Cb2(k,n)-128);
                Cb(k,n)=middle_1/middle_2;
           end   
            if (abs(Cr1(k,n)-128)==0&&abs(Cr2(k,n)-128)==0)      
               Cr(k,n)=128;  
            else
                middle_3=Cr1(k,n)*abs(Cr1(k,n)-128)+Cr2(k,n)*abs(Cr2(k,n)-128);
                middle_4=abs(Cr1(k,n)-128)+abs(Cr2(k,n)-128); 
                Cr(k,n)=middle_3/middle_4;
            end              
          end
      end
      
      I_final_YCbCr=cat(3,I_result,Cb,Cr);    
      I_final_RGB=YCbCr2RGB(I_final_YCbCr);
      imwrite(uint8(I_final_RGB), strcat('.\RGB_result\',num2str(i),'.jpg')); 

    end