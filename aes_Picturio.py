# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:25:26 2017

@author: SzMike
"""


import requests
#from pprint import pprint
import json
import os
#from PIL import Image
import mimetypes

#
#image_files=[]
#image_files.append(r'd:\DATA\RealEstate\117600\10022.jpg')


class scoring:
    def __init__(self):
         # authentication
        url='https://api.picturio.com/token'

        user='szabolcs.mike@gmail.com'
        pswd='p1ctur3sq3'

        data = {'grant_type':'password',
                'username':user,
                'password':pswd}
        
        response = requests.post(url, data=data)
        print(response)
        
        self.out_token=response.json()
        
        
    def get_scores(self,image_files):       
        
        # create processing session
        out_token=self.out_token
        
        url='https://api.picturio.com/processing' 
        
        data = {'SourceType': 1,  'GroupGranularity': 0, 'GroupingType':2 }
        header = {'Authorization':'Bearer '+out_token['access_token'],
                  'content-type':'application/json',
                  'accept':'application/json'}
        response = requests.post(url, headers=header, data=json.dumps(data))
        print(response)
        session_id=response.json()
        
        # uploading photos
        url='https://api.picturio.com//processing//'+session_id+'//add-image'
        
        headers={}
        headers['Authorization']='Bearer '+out_token['access_token']
        headers['accept']='application/json'

        files={}
        for i, image_files in enumerate(image_files):
            if os.path.exists(image_files):
                fname=os.path.basename(image_files)
                files={}
                files[fname]=(fname,open(image_files, 'rb'),mimetypes.guess_type(image_files)[0])
        #files[fname]=(fname,open(images, 'rb'),mimetypes.guess_type(image_files[0])[0])
            
        #files={'file':('10022.jpg',open(image_files[0], 'rb'),mimetypes.guess_type(image_files[0])[0])}
                response = requests.post(url, headers=headers, files=files)
                print(str(i)+' : ')
                print(response)
            else:
                print(fname+' does not exists')
        #


        ## PROCESS
        
        url='https://api.picturio.com//processing//'+session_id
        header = {'Authorization':'Bearer '+out_token['access_token'],'accept':'application/json'}
        
        pr=requests.get(url,headers=header)
        print(pr)
        

        ## STATUS
        url='https://api.picturio.com//processing//'+session_id+'//status'
        header = {'Authorization':'Bearer '+out_token['access_token'],'accept':'application/json'}

        isCompleted=False
        while not isCompleted:
            st=requests.get(url,headers=header)
            if st.json()['status']=='Completed':
                isCompleted=True
                print(st.json())

        ## results
        url='https://api.picturio.com//processing//'+session_id+'//result'
        header = {'Authorization':'Bearer '+out_token['access_token'],'accept':'application/json'}
        
        res=requests.get(url,headers=header)
        
        print(res.json())
        
        return res.json()['Images']
        