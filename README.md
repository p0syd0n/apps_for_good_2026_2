# To run the code, 
## Optional virtual environment:
```
python3 -m venv venv
```

Activate the venv:
```
source ./venv/bin/activate
```        

## Download the packages:
```
pip install -r requirements.txt
```
## Run the server:
```
python3 app.py
```

Now, /inference will process a query. So far, it will fetch relevant papers and attempt to save PDF's. It does not calculate individual metrics for each PDF yet.
