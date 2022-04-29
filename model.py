import csv 
import numpy as np
import pickle
import os
from scipy.signal import argrelmax, argrelmin
import numpy as np


def find_row(filename,col_number,value):
     var = str(value)
     coln = str(col_number)
     with open(filename, 'r') as o:
         myData = csv.reader(o) 
         index = 0 

         for index, row in enumerate(myData):
           try:
              if row[col_number] == var:
                 return index
           except IndexError:
              pass


def find_row_new(myData,col_number,value):
    var = str(value)
    coln = str(col_number)
    #  with open(filename, 'r') as o:
    #      myData = csv.reader(o) 
    #      index = 0 

    for index, row in enumerate(myData):
        #print(index,row)
        try:
            if row[col_number] == var:
                return index
        except IndexError:
            pass





def n_d_array(txt_file,row_start2,row_end2,num_rows2):
    ppg_ls=[]
    for i in range(row_start2+1,row_end2):
        if len(txt_file[i])>1:
            ppg_ls.append(txt_file[i][1])
    #print(len(ppg_ls))
    #print("n-d array rows ",num_rows2)
    PPG_dat=np.asarray(ppg_ls, dtype=str, order=None)
    #print(type(PPG_dat))
    #print(np.shape(PPG_dat))
    return PPG_dat



# def getme_x(direcname):
    
#     file_paths = []  # List which will store all of the full filepaths.

#     # Walk the tree.
#     for root, directories, files in os.walk(direcname):
#         for filename in files:
#             # Join the two strings in order to form the full filepath.
#             filepath = filename
#             file_paths.append(filepath)  # Add it to the list.

#     list_of_txt_files = file_paths  # Self-explanatory.

#     ppg_of_txt=[]
#     ppg_ayu_features=[]

#     for i in range(len(list_of_txt_files)):
#         file_name = list_of_txt_files[i]             

#         row_start = find_row(file_name,1," PPG Signal : ")   
#         row_end = find_row(file_name,1," GATI FFT  : ")      

#         num_rows = row_end - row_start -1 ;

#         x = np.loadtxt(file_name, dtype=str, delimiter=',', usecols=(1),skiprows=row_start+1, max_rows=num_rows)

#         ppg_of_txt.append(x)






#         row_start1 = find_row(file_name,1," Calculated : ")   
#         row_end1 = find_row(file_name,1," KPV2 ")      

#         num_rows1 = row_end1 - row_start1 -1 ;

#         PPG_features_from_ayurlite = np.loadtxt(file_name, dtype=str, delimiter=',', usecols=(1),skiprows=row_start1+1, max_rows=num_rows1)


#         PPG_features_from_ayurlite_cleaned=np.delete(PPG_features_from_ayurlite,[1,2,3,4,5,6,7,8,9,10,18])  
#         PPG_features_from_ayurlite_cleaned =PPG_features_from_ayurlite_cleaned.astype('float64')
#         ppg_ayu_features.append(PPG_features_from_ayurlite_cleaned)



#     x_raw = np.array(ppg_of_txt,dtype=object)

#     ppg_features = np.array(ppg_ayu_features,dtype=object)







#     x_train = np.zeros((len(list_of_txt_files),46))

#     j=0 

#     while(j<len(list_of_txt_files)):




#         single_waveform1=x_raw[j]
#         single_waveform1=single_waveform1.astype('float64')
#         single_waveform=single_waveform1.flatten()
#         sample_rate=20



#         maxima_index = argrelmax(np.array(single_waveform))[0]
#         minima_index = argrelmin(np.array(single_waveform))[0]
#         derivative_1 = np.diff(single_waveform, n=1) * float(sample_rate)
#         derivative_1_maxima_index = argrelmax(np.array(derivative_1))[0]
#         derivative_1_minima_index = argrelmin(np.array(derivative_1))[0]
#         derivative_2 = np.diff(single_waveform, n=2) * float(sample_rate)
#         derivative_2_maxima_index = argrelmax(np.array(derivative_2))[0]
#         derivative_2_minima_index = argrelmin(np.array(derivative_2))[0]


#         # x
#         x = single_waveform[maxima_index[0]]
#         x_train[j,0]=x
#         # y
#         y = single_waveform[maxima_index[1]]
#         x_train[j,1]=y
#         # z
#         z = single_waveform[minima_index[0]]
#         x_train[j,2]=z
#         # t_pi
#         t_pi = float(len(single_waveform)) / float(sample_rate)
#         x_train[j,3]=t_pi
#         # y/x
#         x_train[j,4]=(y / x)
#         # (x-y)/x
#         x_train[j,5]=((x - y) / x)
#         # z/x
#         x_train[j,6]=(z / x)
#         # (y-z)/x
#         x_train[j,7]=((y - z) / x)
#         # t_1
#         t_1 = float(maxima_index[0] + 1) / float(sample_rate)
#         x_train[j,8]=(t_1)
#         # t_2
#         t_2 = float(minima_index[0] + 1) / float(sample_rate)
#         x_train[j,9]=(t_2)
#         # t_3
#         t_3 = float(maxima_index[1] + 1) / float(sample_rate)
#         x_train[j,10]=(t_3)
#         # delta_t
#         delta_t = t_3 - t_2
#         x_train[j,11]=(delta_t)

#         # A_2/A_1
#         x_train[j,12]=(sum(single_waveform[:maxima_index[0]]) / sum(single_waveform[maxima_index[0]:]))
#         # t_1/x
#         x_train[j,13]=(t_1 / x)
#         # y/(t_pi-t_3)
#         x_train[j,14]=(y / (t_pi - t_3))
#         # t_1/t_pi
#         x_train[j,15]=(t_1 / t_pi)
#         # t_2/t_pi
#         x_train[j,16]=(t_2 / t_pi)
#         # t_3/t_pi
#         x_train[j,17]=(t_3 / t_pi)
#         # delta_t/t_pi
#         x_train[j,18]=(delta_t / t_pi)
#         # t_a1
#         t_a1 = float(derivative_1_maxima_index[0]) / float(sample_rate)
#         x_train[j,19]=(t_a1)
#         # t_b1
#         t_b1 = float(derivative_1_minima_index[0]) / float(sample_rate)
#         x_train[j,20]=(t_b1)
#         # t_e1
#         t_e1 = float(derivative_1_maxima_index[1]) / float(sample_rate)
#         x_train[j,21]=(t_e1)
#         # t_f1
#         t_f1 = float(derivative_1_minima_index[1]) / float(sample_rate)
#         x_train[j,22]=(t_f1)
#         # b_2/a_2
#         a_2 = derivative_2[derivative_2_maxima_index[0]]
#         b_2 = derivative_2[derivative_2_minima_index[0]]
#         x_train[j,23]=(b_2 / a_2)
#         # e_2/a_2
#         e_2 = derivative_2[derivative_2_maxima_index[1]]
#         x_train[j,24]=(e_2 / a_2)
#         # (b_2+e_2)/a_2
#         x_train[j,25]=((b_2 + e_2) / a_2)
#         # t_a2
#         t_a2 = float(derivative_2_maxima_index[0]) / float(sample_rate)
#         x_train[j,26]=(t_a2)
#         # t_b2
#         t_b2 = float(derivative_2_minima_index[0]) / float(sample_rate)
#         x_train[j,27]=(t_b2)
#         # t_a1/t_pi
#         x_train[j,28]=(t_a1 / t_pi)
#         # t_b1/t_pi
#         x_train[j,29]=(t_b1 / t_pi)
#         # t_e1/t_pi
#         x_train[j,30]=(t_e1 / t_pi)
#         # t_f1/t_pi
#         x_train[j,31]=(t_f1 / t_pi)
#         # t_a2/t_pi
#         x_train[j,32]=(t_a2 / t_pi)
#         # t_b2/t_pi
#         x_train[j,33]=(t_b2 / t_pi)
#         # (t_a1-t_a2)/t_pi
#         x_train[j,34]=((t_a1 - t_a2) / t_pi)
#         # (t_b1-t_b2)/t_pi
#         x_train[j,35]=((t_b1 - t_b2) / t_pi)
#         # (t_e1-t_2)/t_pi
#         x_train[j,36]=((t_e1 - t_2) / t_pi)
#         # (t_f1-t_3)/t_pi
#         x_train[j,37]=((t_f1 - t_3) / t_pi)        

#         x_train[j,38]=ppg_features[j,0]

#         x_train[j,39]=ppg_features[j,1]

#         x_train[j,40]=ppg_features[j,2]

#         x_train[j,41]=ppg_features[j,3]

#         x_train[j,42]=ppg_features[j,4]

#         x_train[j,43]=ppg_features[j,5]

#         x_train[j,44]=ppg_features[j,6]

#         x_train[j,45]=ppg_features[j,7]




#         j+=1         

    

#     return x_train




# def get_labels_from_txt(directory):
#     """
#     This function will generate the file names in a directory 
#     tree by walking the tree either top-down or bottom-up. For each 
#     directory in the tree rooted at directory top (including top itself), 
#     it yields a 3-tuple (dirpath, dirnames, filenames).
#     """
#     file_paths = []  # List which will store all of the full filepaths.

#     # Walk the tree.
#     for root, directories, files in os.walk(directory):
#         for filename in files:
#             # Join the two strings in order to form the full filepath.
#             filepath = filename
#             file_paths.append(filepath)  # Add it to the list.

#     list_of_txt_files = file_paths  # Self-explanatory.

#     label_of_txt=[]
#     labels=[]

#     for i in range(len(list_of_txt_files)):
        
#         file_name = list_of_txt_files[i]             
                   
#         row_start = find_row(file_name,1," User Manual Data : ")   
#         row_end = find_row(file_name,1," Advance Settings ")      

#         num_rows = row_end - row_start -1 ;

#         labels_from_ayurlite = np.loadtxt(file_name, dtype=str, delimiter=',', usecols=(1),skiprows=row_start+1, max_rows=num_rows)

#         labels_from_ayurlite_cleaned=np.delete(labels_from_ayurlite,[0,1,2,3,6,7,8,9]) 

#         labels_from_ayurlite_cleaned =labels_from_ayurlite_cleaned.astype('float64')

        

#         labels.append(labels_from_ayurlite_cleaned)
        
        
#     y_train = np.array(labels,dtype=object)
        
#     i=0    
#     c= np.zeros((len(y_train),1))

#     for i in range(len(y_train)): 

     

#         if(y_train[i,0]>135):
#             c[i,0]=0
#         elif(120<y_train[i,0]<130):
#             c[i,0]=1

#         elif(y_train[i,0]<100 or y_train[i,1]<70):
#             c[i,0]=2

#         else:
#             c[i,0]=3

        
        
#     return c #, y_train       # c is final label list 




def extract_features(file_name):              

    # row_start2 = find_row(file_name,1," PPG Signal : ")   
    # row_end2 = find_row(file_name,1," GATI FFT  : ") 

    row_start2 = find_row_new(file_name,1," PPG Signal : ")   
    row_end2 = find_row_new(file_name,1," GATI FFT  : ")      
    #print("ppg",row_start2)
    #print("gati",row_end2)
    # num_rows2 = row_end2 - row_start2 -1 ;
    # PPG_dat = np.loadtxt(file_name, dtype=str, delimiter=',', usecols=(1),skiprows=row_start2+1, max_rows=num_rows2)

    num_rows2 = row_end2 - row_start2 -1 ;
    PPG_dat = n_d_array(file_name,row_start2,row_end2,num_rows2)
    
    PPG_dat = PPG_dat.astype('float64')
    #print("PPG_dat",PPG_dat)
    
    
    # row_start3 = find_row(file_name,1," Calculated : ")   
    # row_end3 = find_row(file_name,1," KPV2 ") 
    row_start3 = find_row_new(file_name,1," Calculated : ")   
    row_end3 = find_row_new(file_name,1," KPV2 ")     

    #print("calculated",row_start3)
    #print("KPV2",row_end3)
    num_rows3 = row_end3 - row_start3 -1 ;

    #aaa = np.loadtxt(file_name, dtype=str, delimiter=',', usecols=(1),skiprows=row_start3+1, max_rows=num_rows3)
    aaa = n_d_array(file_name,row_start3,row_end3,num_rows3)
    #print("aaa",aaa)
    # bbb=np.delete(aaa,[1,2,3,4,5,6,7,8,9,10,18]) 
    bbb=np.delete(aaa,[1,2,3,4,5,6,7,8,9,10])
    #print("bbb",bbb)
    bbb =bbb.astype('float64')
    ccc = np.array(bbb,dtype=object)
    #print("ccc",ccc)
    


    x_sample=np.zeros((1,46))
    single_waveform=PPG_dat
    sample_rate=15       



    maxima_index = argrelmax(np.array(single_waveform))[0]
    minima_index = argrelmin(np.array(single_waveform))[0]
    derivative_1 = np.diff(single_waveform, n=1) * float(sample_rate)
    derivative_1_maxima_index = argrelmax(np.array(derivative_1))[0]
    derivative_1_minima_index = argrelmin(np.array(derivative_1))[0]
    derivative_2 = np.diff(single_waveform, n=2) * float(sample_rate)
    derivative_2_maxima_index = argrelmax(np.array(derivative_2))[0]
    derivative_2_minima_index = argrelmin(np.array(derivative_2))[0]


    # x
    x = single_waveform[maxima_index[0]]
    x_sample[0,0]=x
    # y
    y = single_waveform[maxima_index[1]]
    x_sample[0,1]=y
    # z
    z = single_waveform[minima_index[0]]
    x_sample[0,2]=z
    # t_pi
    t_pi = float(len(single_waveform)) / float(sample_rate)
    x_sample[0,3]=t_pi
    # y/x
    x_sample[0,4]=(y / x)
    # (x-y)/x
    x_sample[0,5]=((x - y) / x)
    # z/x
    x_sample[0,6]=(z / x)
    # (y-z)/x
    x_sample[0,7]=((y - z) / x)
    # t_1
    t_1 = float(maxima_index[0] + 1) / float(sample_rate)
    x_sample[0,8]=(t_1)
    # t_2
    t_2 = float(minima_index[0] + 1) / float(sample_rate)
    x_sample[0,9]=(t_2)
    # t_3
    t_3 = float(maxima_index[1] + 1) / float(sample_rate)
    x_sample[0,10]=(t_3)
    # delta_t
    delta_t = t_3 - t_2
    x_sample[0,11]=(delta_t)

    # A_2/A_1
    x_sample[0,12]=(sum(single_waveform[:maxima_index[0]]) / sum(single_waveform[maxima_index[0]:]))
    # t_1/x
    x_sample[0,13]=(t_1 / x)
    # y/(t_pi-t_3)
    x_sample[0,14]=(y / (t_pi - t_3))
    # t_1/t_pi
    x_sample[0,15]=(t_1 / t_pi)
    # t_2/t_pi
    x_sample[0,16]=(t_2 / t_pi)
    # t_3/t_pi
    x_sample[0,17]=(t_3 / t_pi)
    # delta_t/t_pi
    x_sample[0,18]=(delta_t / t_pi)
    # t_a1
    t_a1 = float(derivative_1_maxima_index[0]) / float(sample_rate)
    x_sample[0,19]=(t_a1)
    # t_b1
    t_b1 = float(derivative_1_minima_index[0]) / float(sample_rate)
    x_sample[0,20]=(t_b1)
    # t_e1
    t_e1 = float(derivative_1_maxima_index[1]) / float(sample_rate)
    x_sample[0,21]=(t_e1)
    # t_f1
    t_f1 = float(derivative_1_minima_index[1]) / float(sample_rate)
    x_sample[0,22]=(t_f1)
    # b_2/a_2
    a_2 = derivative_2[derivative_2_maxima_index[0]]
    b_2 = derivative_2[derivative_2_minima_index[0]]
    x_sample[0,23]=(b_2 / a_2)
    # e_2/a_2
    e_2 = derivative_2[derivative_2_maxima_index[1]]
    x_sample[0,24]=(e_2 / a_2)
    # (b_2+e_2)/a_2
    x_sample[0,25]=((b_2 + e_2) / a_2)
    # t_a2
    t_a2 = float(derivative_2_maxima_index[0]) / float(sample_rate)
    x_sample[0,26]=(t_a2)
    # t_b2
    t_b2 = float(derivative_2_minima_index[0]) / float(sample_rate)
    x_sample[0,27]=(t_b2)
    # t_a1/t_pi
    x_sample[0,28]=(t_a1 / t_pi)
    # t_b1/t_pi
    x_sample[0,29]=(t_b1 / t_pi)
    # t_e1/t_pi
    x_sample[0,30]=(t_e1 / t_pi)
    # t_f1/t_pi
    x_sample[0,31]=(t_f1 / t_pi)
    # t_a2/t_pi
    x_sample[0,32]=(t_a2 / t_pi)
    # t_b2/t_pi
    x_sample[0,33]=(t_b2 / t_pi)
    # (t_a1-t_a2)/t_pi
    x_sample[0,34]=((t_a1 - t_a2) / t_pi)
    # (t_b1-t_b2)/t_pi
    x_sample[0,35]=((t_b1 - t_b2) / t_pi)
    # (t_e1-t_2)/t_pi
    x_sample[0,36]=((t_e1 - t_2) / t_pi)
    # (t_f1-t_3)/t_pi
    x_sample[0,37]=((t_f1 - t_3) / t_pi)
    
    x_sample[0,38]=ccc[0]

    x_sample[0,39]=ccc[1]

    x_sample[0,40]=ccc[2]

    x_sample[0,41]=ccc[3]

    x_sample[0,42]=ccc[4]

    x_sample[0,43]=ccc[5]
    
    x_sample[0,44]=ccc[6]
    
    x_sample[0,45]=ccc[7]
    
            
    return x_sample

          
    

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


X= np.load('outfiles.npy')
#print("infiles ",X)
Y= np.load('infiles.npy')



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0)

classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)



#Make Pickle File of Our Model

pickle.dump(classifier, open("model.pkl", "wb"))



"""

#print("********************************RESULT USING DECISION TREE CLASSIFIER************************************")


test_result= classifier.predict(X_test)
score=accuracy_score(y_test,test_result)
print("The model is predicting with an accuracy of " + str(score*100)+"%")


x_user= extract_features(user_name)

result= classifier.predict(x_user)
#outcome(result,score)

"""






#35- 58.3
#10- 50

