import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://triagecancer.org/congressional-social-media"
page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')


## the following was taken from : https://www.geeksforgeeks.org/convert-html-table-into-csv-file-in-python/

# empty list 
data = [] 
   
# for getting the header from 
# the HTML file 
list_header = [] 
header = soup.find_all("table")[0].find("tr") 
  
for items in header: 
    try: 
        list_header.append(items.get_text()) 
    except: 
        continue
  
# for getting the data  
HTML_data = soup.find_all("table")[0].find_all("tr")[1:] 
  
for element in HTML_data: 
    sub_data = [] 
    for sub_element in element: 
        try: 
            sub_data.append(sub_element.get_text()) 
        except: 
            continue
    data.append(sub_data) 
  
# Storing the data into Pandas 
# DataFrame  
socmed_df = pd.DataFrame(data = data, columns = list_header) 
   
# Converting Pandas DataFrame 
# into CSV file 
socmed_df.to_csv('./data/congress_twitter_handles.csv') 