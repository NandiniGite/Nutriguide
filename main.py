import requests

def fetchandsave(url,path):
    r=requests.get(url)
    with open(path,"w") as f:
        f.write(r.text)


url="https://www.canada.ca/en/health-canada/services/food-nutrition/healthy-eating/nutrient-data/nutrient-value-some-common-foods-2008.html"

fetchandsave(url,"data/nutrition.html")