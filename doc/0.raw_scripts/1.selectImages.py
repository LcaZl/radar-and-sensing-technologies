# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 14:40:33 2021

@author: rslab
"""

import csv
import json
import os
from datetime import datetime

import pandas as pd

month = []   
month.append(['0101', '0131'])
month.append(['0201', '0228'])
month.append(['0301', '0331'])
month.append(['0401', '0430'])
month.append(['0501', '0531'])
month.append(['0601', '0630'])
month.append(['0701', '0731'])
month.append(['0801', '0831'])
month.append(['0901', '0930'])
month.append(['1001', '1031'])
month.append(['1101', '1130'])
month.append(['1201', '1231'])

months = []   
months.append(['0101', '0215'])
months.append(['0115', '0315'])
months.append(['0215', '0415'])
months.append(['0315', '0515'])
months.append(['0415', '0615'])
months.append(['0515', '0715'])
months.append(['0615', '0815'])
months.append(['0715', '0915'])
months.append(['0815', '1015'])
months.append(['0915', '1115'])
months.append(['1015', '1215'])
months.append(['1115', '1231'])



def write_csv(metadata_path, csv_path, cloud_threshold = 40.0):

    #print(metadata_path)
    with open(csv_path,'w+') as f:
        writer = csv.writer(f)
        header = ['name', 'cloudCover', 'timestamp','nodata_percentage','orbit']
        writer.writerow(header)

        for root, _, files in os.walk(metadata_path):
            for file in files:
                #print(file)
                if file.endswith('.json'):
                    l2a_tileinfo_path = os.path.join(root,file)
                    xml_tileinfo_path = l2a_tileinfo_path.replace('parsed','original')
                    xml_tileinfo_path = xml_tileinfo_path.replace('tileInfo.json','metadata.xml')

                    try:
                        with open(l2a_tileinfo_path) as json_data, open(xml_tileinfo_path) as xml_data:
                            
                            #load json
                            json_tileinfo = json.load(json_data)

                            # Get Filename
                            filename = json_tileinfo['productName']

                            # Get cloud coverage
                            cloud_line = ''
                            xml_lines = xml_data.readlines()
                            for line in xml_lines:
                                if '<CLOUDY_PIXEL_PERCENTAGE>' in line:
                                    cloud_line = line
                                if '<SENSING_TIME metadataLevel="Standard">' in line:
                                    sensing_time_line = line
                                if '<NODATA_PIXEL_PERCENTAGE>' in line:
                                    nodata_line = line

                            cloud_coverage = float(cloud_line.replace('<CLOUDY_PIXEL_PERCENTAGE>','').replace('</CLOUDY_PIXEL_PERCENTAGE>',''))
                            sensing_time = sensing_time_line.replace('<SENSING_TIME metadataLevel="Standard">','') \
                                                            .replace('</SENSING_TIME>','').replace('"','').replace('    ','').replace('\n','')
                            nodata = nodata_line.replace('<NODATA_PIXEL_PERCENTAGE>','').replace('</NODATA_PIXEL_PERCENTAGE>','') \
                                                .replace('</SENSING_TIME>','').replace('"','').replace('  ','').replace('\n','')
                            orbit = filename.split('_')[4]

                            data = [filename,str(cloud_coverage),sensing_time,str(nodata),orbit]

                            if cloud_coverage <= cloud_threshold:
                                writer.writerow(data)

                    except Exception as e:
                        print ('Exception: ', e)
                        print ('***********************************Skipped******************************')
                        continue



def getImagesInBetween2(df_test, after, before, year):
    imagesInBetween_t = []
    for z in range(len(df_test)):
            img_row = df_test.iloc[[z]]
            img_name = str(img_row.name.tolist()[0])
            img = os.path.basename(img_name)
            img_date = img[img.index(year):img.index(year)+8]
            img_date = datetime.strptime(img_date, "%Y%m%d") 
            if after <= img_date <= before:
                imagesInBetween_t.append(df_test.iloc[[z]])

                
    return imagesInBetween_t


def getImagesInBetween(images, after, before, year):
    imagesInBetween_t = []
    for z in range(len(images)):
            img = os.path.basename(images[z])
            img_date = img[img.index(year):img.index(year)+8]
            img_date = datetime.strptime(img_date, "%Y%m%d") 
            if after <= img_date <= before:
                imagesInBetween_t.append(images[z])

                
    return imagesInBetween_t




def reduce_number_of_images(csv_path):
    
    df_test = pd.read_csv(csv_path)

    df_test= df_test.dropna()
    df_test['nameCopy']= df_test.name.str[4:]
    all_orbits = df_test['orbit'].unique()

    year = '2019'    
    img_final_all = []
    print(df_test.sort_values('nameCopy'))
     
    for s in range(12):   
        month_t = months[s] 
        after = datetime.strptime(year+month_t[0], "%Y%m%d")
        before = datetime.strptime(year+month_t[1], "%Y%m%d")
        imagesInBetween2 = (getImagesInBetween2(df_test, after, before, year))
        if len(imagesInBetween2) > 0:
            imagesInBetween2 = pd.concat(imagesInBetween2)
            imagesInBetween2 = imagesInBetween2.sort_values('nameCopy')
        

        ix1 = 0
        ix2 = 0

        if len(imagesInBetween2)>8:
            month_current = month[s] 
            after = datetime.strptime(year+month_current[0], "%Y%m%d")
            before = datetime.strptime(year+month_current[1], "%Y%m%d")  
            img_month = (getImagesInBetween2(imagesInBetween2, after, before, year))
            img_month = pd.concat(img_month)
            
            missing = 8-len(img_month)
            noImages1 = False
            noImages2 = False
            while missing > 0 :
                if  month_current[1] != "1231":
                    month_after = month[s+1]  
                    after = datetime.strptime(year+month_after[0], "%Y%m%d")
                    before = datetime.strptime(year+month_after[1], "%Y%m%d")  
                    img_monthAfter = getImagesInBetween2(imagesInBetween2, after, before, year)
                    
                    if len(img_monthAfter) > 0 and noImages1 == False:
                        img_monthAfter = pd.concat(img_monthAfter)
                        img_monthAfter = img_monthAfter.sort_values('nameCopy') 
                        temp =[img_monthAfter.iloc[[ix1]], img_month]
                        img_month = pd.concat(temp)
                        ix1 = ix1+1
                        missing = missing-1
                        if ix1 == len(img_monthAfter): # if no more images are avialble
                            noImages1 = True
                    else:
                        noImages1 = True
                        
                if  month_current[0] != "0101":
                    month_before = month[s-1] 
                    after = datetime.strptime(year+month_before[0], "%Y%m%d")
                    before = datetime.strptime(year+month_before[1], "%Y%m%d")  
                    img_monthBefore = getImagesInBetween2(imagesInBetween2, after, before, year)
                    
                    if len(img_monthBefore) > 0 and noImages2 == False:
                        img_monthBefore = pd.concat(img_monthBefore)
                        img_monthBefore = img_monthBefore.sort_values('nameCopy', ascending=False)
                        temp =[img_monthBefore.iloc[[ix2]], img_month]
                        img_month = pd.concat(temp)
                        ix2 = ix2+1
                        missing = missing-1
                        if ix2 == len(img_monthBefore):# if no more images are avialble
                            noImages2 = True
                    else:
                        noImages2 = True  
                        
                if noImages1 == True and noImages2 == True:
                    missing = 0
                        
            if len(img_month) > 8:
                img_month= img_month.sort_values('cloudCover')
                img_month = img_month[0:8]
            #print(img_month) 
            
            #add image from missing orbits
            current_orbits = img_month['orbit'].unique()
            if len(current_orbits) != len(all_orbits):
                for p in range(len(all_orbits)):
                    orbit = all_orbits[p]
                    if orbit not in current_orbits:
                        df_orbit = imagesInBetween2[(imagesInBetween2["orbit"] == orbit)]
                        if len(df_orbit) != 0:
                            df_orbit = df_orbit.sort_values('nodata_percentage')# or sort by cloudCover
                            temp = [df_orbit.iloc[[0]], img_month]
                            img_month = pd.concat(temp)

            img_final_all.append(img_month)          
        else: 
            #print(imagesInBetween2)
            img_final_all.append(imagesInBetween2)    
            
    return img_final_all


if __name__ == '__main__':
    result = reduce_number_of_images('1.selectImages_test_data/20MRA.csv')
    print(result)



