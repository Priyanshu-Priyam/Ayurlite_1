import boto3
from flask import session
import pandas as pd
import numpy as np
import csv
#from model import extract_features
#import pickle

#classifier = pickle.load(open('model.pkl', 'rb'))


S3_BUCKET = 'ayurlite3'
S3_KEY = 'AKIAZW6W3DLQDXUOBG5M'
S3_SECRET = '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'

def _get_s3_resource():
    if S3_KEY and S3_SECRET:
        return boto3.resource(
            's3',
            aws_access_key_id=S3_KEY,
            aws_secret_access_key=S3_SECRET
        )
    else:
        
        return boto3.resource('s3')


def get_bucket():
    s3_resource = _get_s3_resource()
    if 'bucket' in session:
        bucket = session['bucket']
    else:
        bucket = S3_BUCKET

    return s3_resource.Bucket(bucket)


def get_buckets_list():
    client = boto3.client('s3')
    return client.list_buckets().get('Buckets')





def find_row(myData,col_number,value):
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
    print(len(ppg_ls))
    print("n-d array rows ",num_rows2)
    PPG_dat=np.asarray(ppg_ls, dtype=str, order=None)
    print(type(PPG_dat))
    print(np.shape(PPG_dat))
    return PPG_dat






def read_bucket_objects(filename):
    s3_resource = _get_s3_resource()
    my_bucket=s3_resource.Bucket(S3_BUCKET)
    bucket_list = []
    for file in my_bucket.objects.filter(Prefix = filename):
        file_name=file.key
        bucket_list.append(file.key)
    print("filename ",filename)


    # df =[]
    # for file in bucket_list:
    #     obj = s3_resource.Object(S3_BUCKET,file)
    #     data=obj.get()['Body'].read()
    #     df.append(pd.read_csv(io.BytesIO(data), delimiter="\n", low_memory=False))
    # print(len(df[0]))
    # print(df[0].tail())
            


    # txt_file = s3_resource.Object(S3_BUCKET,'11Jan2022_01_01_18.txt').get()['Body'].read().decode('utf-8').splitlines()
    # print(type(txt_file))
    # for line in txt_file:
    #     arr = line.split()
    # print(type(arr[0]))
    # print(arr[0])



    # txt_file = s3_resource.Object(S3_BUCKET,'11Jan2022_01_01_18.txt').get()['Body'].read()
    # myData = csv.reader(io.BytesIO(txt_file)) 
    # print(myData)
    # for i ,r in enumerate(myData):
    #     print(i,r)
#print(get_buckets_list())
# 11Jan2022_12_59_35.txt  11Jan2022_01_01_18.txt
    txt_file=[]
    s3= _get_s3_resource()
    for line in s3.Object(S3_BUCKET, filename).get()['Body'].iter_lines():
        decoded_line = line.decode('utf-8')
        txt_file.append(decoded_line.split(','))
    return txt_file
  

    

    # row_start2 = find_row(txt_file,1," PPG Signal : ") #" PPG Signal : ")  #" Calculated : ")
    # row_end2 = find_row(txt_file,1, " GATI FFT  : ")#" GATI FFT  : ") #" KPV2 ")
    # num_rows2 = row_end2 - row_start2 -1
    #print(n_d_array(txt_file,row_start2,row_end2,num_rows2))

    # row_start3 = find_row(txt_file,1," Calculated : ") #" PPG Signal : ")  #" Calculated : ")
    # row_end3 = find_row(txt_file,1, " KPV2 ")#" GATI FFT  : ") #" KPV2 ")
    # num_rows3 = row_end3 - row_start3 -1 ;
    

    #print("found at ", find_row(txt_file,1, " PPG Signal : "))  #" PPG Signal : "))
    #print("found at  ",find_row(txt_file,1, " GATI FFT  : "))   #" GATI FFT  : "))
    #print(n_d_array(txt_file,row_start3,row_end3,num_rows3))
    
    # bbb=np.delete(PPG_dat,[1,2,3,4,5,6,7,8,9,10]) 
    # print(bbb)
   




    
#read_bucket_objects()



# def find_row2(filename,col_number,value):
#      var = str(value)
#      coln = str(col_number)
#      with open(filename, 'r') as o:
#          myData = csv.reader(o) 
#          index = 0 

#          for index, row in enumerate(myData):
#            try:
#               if row[col_number] == var:
#                  return index
#            except IndexError:
#               pass

# row_start2 = find_row2('static//11Jan2022_12_59_35.txt',1," Calculated : ")
# row_end2 = find_row2('static//11Jan2022_12_59_35.txt',1," KPV2 ")
# num_rows2 = row_end2 - row_start2 -1
# print("<-----------------------EOF------------------------------------>")
# print(row_start2)
# print(row_end2)
# PPG_dat = np.loadtxt('static//11Jan2022_12_59_35.txt', dtype=str, delimiter=',', usecols=(1),skiprows=row_start2+1, max_rows=num_rows2)
# print(len(PPG_dat))
# print(type(PPG_dat))
# print(np.shape(PPG_dat))
# bbb=np.delete(PPG_dat,[1,2,3,4,5,6,7,8,9,10,18]) 
# print(bbb)