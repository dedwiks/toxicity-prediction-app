import requests

def test_api(name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/property/CanonicalSMILES/JSON"
    
    print("URL:", url)
    
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    
    print("STATUS:", response.status_code)
    print("TEXT:", response.text[:200])
    
    try:
        data = response.json()
        smiles = data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        print("SMILES:", smiles)
    except Exception as e:
        print("ERROR:", e)


# 🔥 THIS LINE IS CRITICAL
test_api("aspirin")