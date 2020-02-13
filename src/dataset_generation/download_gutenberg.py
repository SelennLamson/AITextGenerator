import urllib.request
import os
from zipfile import ZipFile

if not os.path.exists("treated_files.txt"):
    open("treated_files.txt", 'a').close()
if not os.path.exists("extracted"):
    os.mkdir("extracted")

files = os.listdir("robot/")

with open("treated_files.txt", "r") as file:
    for line in file:
        files = [elt for elt in files if line.strip() != elt]

if len(files) == 0:
    print("All files are already treated!")
else:
    selected = files[0]
    print("Using file: " + selected)

    with open("treated_files.txt", "a+") as file:
        file.write(selected + "\n")

    urls = []
    with open("robot/" + selected, "r") as file:
        for line in file:
            if "href" in line:
                url = line.split('"')[1]
                if "harvest" not in url:
                    urls.append(url)

    for i, url in enumerate(urls):
        print("Downloading: " + url + "...")
        urllib.request.urlretrieve(url, 'temp.zip')
        print("Extracting content...")
        with ZipFile('temp.zip', 'r') as zipObj:
            # Extract all the contents of zip file in different directory
            zipObj.extractall('extracted/')
        os.remove('temp.zip')
